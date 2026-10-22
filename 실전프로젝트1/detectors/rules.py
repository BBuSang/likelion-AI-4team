"""규칙 기반 탐지기: 정규식(번호·이메일·주소) + Kiwi 형태소 분석·성씨·호칭 규칙(이름).

원칙
    - 데이터셋 인물 목록(personas.json)은 쓰지 않고, 일반 성씨 목록과 호칭·조사 규칙만 사용
    - 특정 템플릿 문구('상담원:' 등)에 기대지 않고, 호칭·조사·콜론 같은 일반 단서로 판단
    - 원문은 고치지 않음 (정규식이 하이픈·점·공백 변형을 허용)

이름 판단 순서
    1. 후보: 한글 어절 앞부분(성+이름 2~3자, 복성 4자)과 뒤에 붙은 조사·호칭을 분리
       띄어 쓴 이름은 '성 한 글자 어절 + 이름 어절'로 후보를 만듦 ('윤 선영')
    2. 근거: 뒤에 호칭(고객·님·씨·직함), 뒤에 콜론(메신저 화자), 앞에 라벨(콜론·고객·담당 등),
       Kiwi가 고유명사(NNP) 한 덩어리로 분석
    3. 문서 안 전파: 근거가 있어 잡힌 이름과 같은 표기(띄어쓰기 무시)·이름 부분은 근거 없이도 잡음
    성+직함('허 부장')은 이름 어절이 직함이면 버림 (팀 기준: 개인정보 아님)
"""
import re
from functools import lru_cache

_kiwi = None

# ------------------------------------------------------------------ 정규식
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
RRN = re.compile(r"(?<![\d-])\d{6}\s?-\s?\d{7}(?![\d-])")
CARD = re.compile(r"(?<![\d-])\d{4}([- ]?)\d{4}\1\d{4}\1\d{4}(?![\d-])")
PHONE = re.compile(r"(?<![\d-])(?:01[016789]|02|0[3-6][1-5]|070)([-. ]?)\d{3,4}\1\d{4}(?![\d-])")
ACCOUNT = re.compile(r"(?<![\d-])\d{2,6}(?:-\d{2,6}){2,3}(?![\d-])")
DATE = re.compile(r"(?:19|20)\d{2}[-./]\d{1,2}[-./]\d{1,2}")
SIDO = ("서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충청북|충청남|충북|충남|"
        "전라북|전라남|전북|전남|경상북|경상남|경북|경남|제주")
ADDRESS = re.compile(
    rf"(?:{SIDO})[가-힣]*"                                  # 시·도
    r"(?:\s[가-힣]+(?:시|군|구))*"                            # 시·군·구
    r"(?:\s[가-힣0-9]+(?:로|길)\s?\d+(?:-\d+)?"               # 도로명 + 건물번호
    r"|(?:\s[가-힣0-9]+(?:읍|면|동|리))+\s\d+(?:-\d+)?)"      # 또는 지번 주소
    r"(?:,?\s?(?:[가-힣A-Za-z0-9]+\s)?\d+(?:동|호|층)(?:\s\d+호)?)?")
PHONE_EXCLUDE_BEFORE = re.compile(r"(?:대표\s?번호|팩스|FAX|fax)[^\n]{0,6}$")

# ------------------------------------------------------------------ 이름 규칙
SURNAMES = set("김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진나지엄채원천방"
               "공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용")
SURNAMES2 = {"남궁", "제갈", "선우", "독고", "황보", "사공", "서문", "동방"}
TITLES = ["고객", "님", "씨", "과장", "대리", "팀장", "부장", "차장", "사원", "주임", "실장", "본부장", "이사",
          "상무", "전무", "대표", "선임", "책임", "수석", "상담원", "매니저", "담당", "드림", "본인", "연구원",
          "위원", "센터장", "지점장", "회원", "선생", "귀하", "변호사", "세무사", "기자"]
JOSA = ["", "은", "는", "이", "가", "을", "를", "의", "에게", "께", "께서", "와", "과", "랑", "이랑", "도", "만",
        "한테", "에게서", "아", "야", "이가", "이는", "이도", "이를", "이의", "이한테", "이와",
        "입니다", "이다", "이고", "이며", "이었", "였", "로", "으로", "에", "부터", "까지"]
HONORIFIC = ["", "님", "씨"]
SUFFIXES = sorted({h + j for h in HONORIFIC for j in JOSA}, key=len, reverse=True)
TITLE_RE = "|".join(sorted(TITLES, key=len, reverse=True))
TITLE_AFTER = re.compile(rf"\s?\(?(?:{TITLE_RE})")
LABEL_BEFORE = re.compile(r"(?:[:：(]|고객|담당자?|상담원|작성자?|발신|수신|성명|이름|예금주|명의자?|문의|대표이사"
                          r"|[가-힣A-Za-z](?:팀|센터|본부|지점))\s*$")
LIST_SEP = re.compile(r"(?:\([^()\n]{0,20}\))?\s*[,，·]\s*")
VOCATIVE = ("아", "야", "이랑", "랑", "이")
ORG_END = set("팀명부실처국청단점층")
OOV_ID_MAX = 10  # Kiwi 사전에 없는 말(미등록어)은 형태소 id가 아주 작게 나옴 (예: '최채원' → id 2)


def load():
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()


@lru_cache(maxsize=None)
def kiwi_tokens(word):
    """후보 문자열만 따로 분석한 (형태, 품사, 사전 등록 여부) 목록."""
    return tuple((t.form, t.tag, t.id > OOV_ID_MAX) for t in _kiwi.tokenize(word))


def name_like(name):
    """Kiwi 기준으로 이름으로 볼 수 있는지: "yes" / "weak" / "no".
    no  : 조사가 섞임('정보를'), 명사 파생 접미사로 끝남('유형별'), 사전에 있는 일반명사 한 단어('신청서')
    weak: 사전에 있는 두 글자 이상 일반명사로 시작('남성호', '허수아', 그러나 '인사팀', '고객명'도 여기),
          용언·어미·접사가 섞임('손하은', '허하은')
          → 근거가 있고 기관·항목명 끝 글자('팀', '명' 등)가 아닐 때만 이름으로 봄
    미등록어 한 덩어리('최채원')는 yes (Kiwi가 처음 보는 이름을 일반명사로 분석하기 때문)"""
    toks = kiwi_tokens(name)
    if any(tag[0] == "J" for _, tag, _ in toks) or toks[-1][1] == "XSN":  # '유형별'
        return "no"
    if any(tag[0] in "EV" or tag.startswith("X") for _, tag, _ in toks):
        return "weak"  # '손하은' → 손 + 하(XSV) + 은(ETM) 처럼 이름 글자가 용언·어미·접사로 분석되는 경우
    form, tag, known = toks[0]
    if known and tag != "NNP":
        if len(toks) == 1:
            return "no"
        if len(form) >= 2:
            return "weak"
    return "yes"


def is_nnp(name):
    return [tag for _, tag, _ in kiwi_tokens(name)] == ["NNP"]


def is_common_word(word):
    """사전에 있는 일반명사 한 단어인지 ('분기' → True, '선영'·'예린' → False)."""
    toks = kiwi_tokens(word)
    return len(toks) == 1 and toks[0][2] and toks[0][1].startswith("NN") and toks[0][1] != "NNP"


def split_suffix(word, n):
    """어절 앞 n글자를 이름으로 보고, 나머지가 조사·호칭이면 그 나머지를 돌려줌 (아니면 None)."""
    rest = word[n:]
    return rest if rest in SUFFIXES else None


def evidence(text, start, end, suffix, like="yes"):
    """이름 근거: 뒤에 님/씨·호칭·콜론, 앞에 라벨. weak 후보는 기관·항목명 끝 글자('인사팀', '고객명')면 제외."""
    if like == "weak" and text[end - 1] in ORG_END:
        return False
    after = text[end + len(suffix):]
    return (suffix[:1] in ("님", "씨") or bool(TITLE_AFTER.match(after)) or after[:1] in (":", "：")
            or bool(LABEL_BEFORE.search(text[max(0, start - 8):start])))


def person_candidates(text):
    """(start, end, 정규화 이름, 종류, 근거 유무) 목록. 종류: full / spaced / given."""
    words = [(m.start(), m.group()) for m in re.finditer(r"[가-힣]+", text)]
    out = []
    for i, (s, w) in enumerate(words):
        # 붙여 쓴 전체 이름: 성 1자 + 이름 2자, 복성 2자 + 이름 2자
        for n in (4, 3):
            if len(w) < n or (n == 4 and w[:2] not in SURNAMES2) or (n == 3 and w[0] not in SURNAMES):
                continue
            suf = split_suffix(w, n)
            name = w[:n]
            like = name_like(name) if suf is not None and name not in TITLES else "no"
            if like == "no":
                continue
            ev = evidence(text, s, s + n, suf, like) or is_nnp(name)
            out.append((s, s + n, name, "full" if like == "yes" else "weak", ev))
            break
        # 띄어 쓴 이름: '윤 선영'
        if len(w) == 1 and w in SURNAMES and i + 1 < len(words):
            s2, w2 = words[i + 1]
            suf = split_suffix(w2, 2) if len(w2) >= 2 else None
            given = w2[:2]
            if (s2 == s + 2 and text[s + 1] == " " and suf is not None
                    and not any(given.startswith(t) or t.startswith(given) for t in TITLES)
                    and not is_common_word(given) and name_like(w + given) != "no"):  # '전 분기' 제외
                out.append((s, s2 + 2, w + given, "spaced", evidence(text, s, s2 + 2, suf)))
        # 이름만: '승현아', '소율이랑', '은영 씨'
        if len(w) >= 2 and w[:2] not in TITLES:
            suf = split_suffix(w, 2)
            if suf is not None:
                after = text[s + len(w):]
                strong = suf.startswith(VOCATIVE) or suf[:1] in ("님", "씨") or bool(re.match(r"\s(?:씨|님)", after))
                out.append((s, s + 2, w[:2], "given", strong and name_like(w[:2]) == "yes"))
    return out


def detect_persons(text, taken):
    cands = [c for c in person_candidates(text) if not any(a < c[1] and c[0] < b for a, b in taken)]
    names = {c[2] for c in cands if c[4] and c[3] != "given"}
    givens = {n[-2:] for n in names} | {c[2] for c in cands if c[4] and c[3] == "given"}
    ok = {c for c in cands if c[4] or (c[3] != "given" and c[2] in names) or (c[3] == "given" and c[2] in givens)}
    # 나열 전파: '박예린(팀장), 남성호, 이소율' 처럼 잡힌 이름 뒤에 쉼표로 이어진 이름
    changed = True
    while changed:
        changed = False
        ends = {c[1] for c in ok}
        for c in cands:
            if c not in ok and c[3] != "given":
                for e in ends:
                    if e < c[0] and LIST_SEP.fullmatch(text[e:c[0]]):
                        ok.add(c)
                        changed = True
                        break
    # 긴 후보 먼저 ('오 서연'이 '서연'보다 우선)
    return [{"label": "person", "start": s, "end": e}
            for s, e, *_ in sorted(ok, key=lambda c: c[0] - c[1])]


# ------------------------------------------------------------------ 탐지
def digits(s):
    return sum(ch.isdigit() for ch in s)


def detect(text):
    load()
    ents, taken = [], []

    def add(label, s, e):
        if not any(a < e and s < b for a, b in taken):
            ents.append({"label": label, "start": s, "end": e})
            taken.append((s, e))

    for m in EMAIL.finditer(text):
        add("email", m.start(), m.end())
    for m in RRN.finditer(text):
        add("resident_registration_number", m.start(), m.end())
    for m in CARD.finditer(text):
        add("card_number", m.start(), m.end())
    for m in PHONE.finditer(text):
        if not PHONE_EXCLUDE_BEFORE.search(text[max(0, m.start() - 12):m.start()]):
            add("phone_number", m.start(), m.end())
    for m in ACCOUNT.finditer(text):
        if 10 <= digits(m.group()) <= 14 and not DATE.fullmatch(m.group()):
            add("account_number", m.start(), m.end())
    for m in ADDRESS.finditer(text):
        add("address", m.start(), m.end())
    for e in detect_persons(text, taken):
        add("person", e["start"], e["end"])
    return ents
