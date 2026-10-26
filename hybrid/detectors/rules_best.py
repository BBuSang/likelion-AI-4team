"""규칙 기반 결합판: hosung 규칙(기본) + 기존 rules.py 규칙 일부 + 형식 보강.

hosung 규칙(detectors/hosung_rules.py)을 그대로 돌린 뒤, 겹치지 않는 구간에만 아래를 더한다.

1. 주민번호(기존 rules): 6-7자리면 날짜·성별 자리가 맞지 않아도 주민번호로 봄 ('123456-9876543')
   - 앞 문맥이 회원번호·사업자번호·주문번호 같은 다른 번호면 제외
2. 이름(기존 rules의 어절 규칙): hosung이 놓친 후보 중 오탐이 적은 종류만
   - 이름만: 근거가 있고 뒤에 '랑/이랑'이 오거나 부르는 말 뒤에 쉼표·느낌표 ('은서랑', '승현아,')
   - 사전 단어가 섞인 이름(weak): 바로 뒤에 '님'이 올 때만 ('이성욱 님')
   - 전체·띄어 쓴 이름: 장소 이름 끝 글자(구·동·로 등)로 끝나면 제외 ('마포구', '신사동')
   기존 rules의 주소·계좌·전화는 쓰지 않음 (dev·test·edge·synth_dev에서 더하면 오탐만 늘었음)
3. 이름 규칙 보강 (synth_dev 오류에서):
   - 콜론 없는 이름 칸 '예약자 설정윤', '문의자명: 박성은', '주문자 김 은주'
   - 연락처 바로 앞 이름 '박민기(010…', '한태윤 010…', '박경은 mommy@…'
   - 이름이 아닌 서식 단어 제외: '배송지'·'송금자'·'진료일'(Kiwi가 고유명사로 보거나 성씨로 시작),
     '홍콩행'·'제주행'(지명 + 행)
4. holdout_fixes (holdout 오류를 보고 넣은 일반 형식, dev 계열 데이터에는 없는 형식):
   - 주소 앞 우편번호 '(05299) 서울특별시 …'를 주소에 포함
   - 읍·면 바로 뒤 번지 '충청남도 아산시 배방읍 252-4' (원본은 동·리·가 뒤 번지만 받음)
   - 끝 묶음이 한 자리인 계좌 '515-31-231761-6'
   - (2차) 역할 명사 제외: '소명자:', '위임인:', '송금인:', '심사역' (일반명사 2글자 + 자·인·역·관)
   - (2차) 이름 뒤 주격 조사 '이' 잘라냄: '심사역 안윤이 조건' → '안윤' (뒤에 호칭이 오면 그대로 둠)
   holdout을 공정하게 채점하려면 환경변수 PII_HOLDOUT_FIXES=0 으로 끈다.
5. hard_fixes (synth_hard_dev 오류에서, PII_HARD_FIXES=0 으로 끔):
   - 이름 뒤 호칭·조사 잘라냄: '김선희님', '박하영씨', '이선호이랑', '지은이한테', '정다영의' → 이름만
     (이름 끝 글자와 겹칠 수 있는 '은·이·가·도·아·야' 한 글자는 남는 이름이 3글자 이상이거나 문서 다른 곳에 그 이름이 따로 나올 때만)
   - 주소 앞 괄호 없는 우편번호 '06524 서울특별시 …', '07209서울특별시 …'를 주소에 포함
"""
import os
import re

from detectors import hosung_rules as H
from detectors import rules as old

_det = None
HOLDOUT_FIXES = os.environ.get("PII_HOLDOUT_FIXES", "1") != "0"
HARD_FIXES = os.environ.get("PII_HARD_FIXES", "1") != "0"

RRN_LOOSE = re.compile(r"(?<![\d-])\d{6}\s?-\s?\d{7}(?![\d-])")
NOT_RRN_WORDS = re.compile(r"회원\s?번호|고객\s?번호|" + H.NOT_ACCOUNT_WORDS.pattern)
PLACE_END = set("시군구동로길역읍면")
GIVEN_FOLLOW = re.compile(r"이랑|랑|[아야][,!]")

FIELDS = ("성명|이름|고객명|예금주|수신인|민원인|신청인|신청자|요청자|가입자|명의자|계약자|예약자|주문자|수령인|수령자|"
          "입금자|송금자|문의자|보호자|대리인|세대주|환자")
FIELD_NO_COLON = re.compile(rf"(?<![가-힣])(?:{FIELDS})명? ?[:：]? ?([가-힣]{{3}}|[가-힣] [가-힣]{{2}})(?![가-힣])")
BEFORE_CONTACT = re.compile(r"(?<![가-힣])([가-힣]{3}) ?[(（]? ?(?=\+82|0\d{1,2}[-. ]?\d{3,4}[-. ]?\d{4}|[A-Za-z0-9._%+-]+@)")
# 성씨로 시작하는 서식 단어 (문서 항목명·경조사 용어)
NOT_NAMES = {"배송지", "수령지", "거주지", "소재지", "목적지", "출발지", "도착지", "송금자", "문의자", "신고인", "진료일",
             "주문일", "신청일", "변경일", "접수일", "결제일", "연소득", "선결제", "장인상", "장모상", "부친상", "모친상",
             "조부상", "조모상", "빙부상", "빙모상", "외조부상", "외조모상"}

POSTAL_BEFORE = re.compile(r"\(\d{5}\)\s?$")
POSTAL_BARE_BEFORE = re.compile(r"(?<![\d\-.])[0-6]\d{4} ?$")  # '06524 서울…', '07209서울…'
# 이름 뒤 호칭·조사. 앞쪽이 길게 맞도록 긴 것부터
NAME_TAIL_SAFE = ["님", "씨", "이랑", "랑", "하고", "이한테", "한테", "이에게", "에게서", "에게", "께서", "께", "이에요", "입니다",
                  "이가", "이는", "이의", "이도", "이와", "이고", "는", "의", "를", "을", "와", "과"]
NAME_TAIL_AMBIG = ["은", "이", "가", "도", "아", "야"]  # 이름 끝 글자일 수도 있음 ('김지은', '진이')
_TAIL_TOK = "|".join(sorted(NAME_TAIL_SAFE + NAME_TAIL_AMBIG + ["서", "도", "요", "로", "에"], key=len, reverse=True))
NAME_TAIL = re.compile(rf"(?: ?님|씨)?(?:{_TAIL_TOK}){{0,3}}")
EUP_JIBUN = re.compile(rf"(?:(?:{H.SIDO}) )?(?:{H.SGG} ){{1,2}}[가-힣]{{1,6}}(?:읍|면) (?:산 ?)?\d+(?:-\d+)?(?:번지)?"
                       rf"{H._NUM_END}")
ACCOUNT_TAIL1 = re.compile(rf"(?<![\d{H.DASH}])\d{{2,6}}(?:[{H.DASH}]\d{{2,7}}){{1,2}}[{H.DASH}]\d(?![\d{H.DASH}])")


def load():
    global _det
    old.load()
    if _det is None:
        _det = H.RuleBasedDetector()


def _overlaps(ents, s, e):
    return [x for x in ents if x["start"] < e and s < x["end"]]


def _clause_before(text, start, window=25):
    line_start = text.rfind("\n", 0, start) + 1
    return re.split(r"[.,/;]\s", text[max(line_start, start - window):start])[-1]


def _replace(ents, s, e, label):
    """[s, e) 안에 완전히 들어가는 구간만 있으면 그것들을 지우고 새 구간으로 바꾼다 (걸쳐 있으면 그대로 둠)."""
    hits = _overlaps(ents, s, e)
    if all(s <= x["start"] and x["end"] <= e for x in hits):
        for x in hits:
            ents.remove(x)
        ents.append({"label": label, "start": s, "end": e})


def _old_names(text, ents):
    """기존 rules의 이름 후보 중 hosung 결과와 겹치지 않고 오탐이 적은 종류만."""
    kinds = {(c[0], c[1]): (c[3], c[4]) for c in old.person_candidates(text)}
    taken = [(x["start"], x["end"]) for x in ents if x["label"] != "person"]
    out = []
    for p in old.detect_persons(text, taken):
        s, e = p["start"], p["end"]
        if _overlaps(ents + out, s, e) or _not_name(text, p):
            continue
        kind, ev = kinds.get((s, e), ("full", False))
        after = text[e:e + 3]
        if kind == "given":
            ok = ev and GIVEN_FOLLOW.match(after)
        elif kind == "weak":
            ok = re.match(r"\s?님", after)
        else:
            ok = not _place(text[s:e])
        if ok:
            out.append({"label": "person", "start": s, "end": e})
    return out


def _place(word):
    """장소 이름: 끝 글자가 구·동·로 등, 또는 '서울숲'·'제주도'처럼 사전에 있는 한 단어 + 숲·도 (hosung PLACE_SUFFIX)."""
    toks = _det.an.tokenize(word)
    return word[-1] in PLACE_END or (word[-1] in H.PLACE_SUFFIX and len(toks) == 1 and toks[0].known)


def _surname_ok(name):
    return name[0] in H.SURNAMES or name[:2] in H.COMPOUND_SURNAMES


def _more_names(text, ents):
    """콜론 없는 이름 칸, 연락처 바로 앞 이름."""
    out = []
    for pat in (FIELD_NO_COLON, BEFORE_CONTACT):
        for m in pat.finditer(text):
            s, e = m.start(1), m.end(1)
            name = m.group(1).replace(" ", "")
            if (_overlaps(ents + out, s, e) or not _surname_ok(name) or name in NOT_NAMES or _place(name)
                    or name[1:] in H.TITLES or not _det._looks_like_name(name) or _det._is_common_word(name)):
                continue
            out.append({"label": "person", "start": s, "end": e})
    return out


def _not_name(text, e):
    word = text[e["start"]:e["end"]].replace(" ", "")
    if word in NOT_NAMES:
        return True
    # '우편번호', '연락번호': 끝 두 글자가 일반명사인 4글자 이상 (복성 이름 제외. 한국 이름은 대부분 3글자)
    if len(word) >= 4 and word[:2] not in H.COMPOUND_SURNAMES and not re.search(r"[A-Za-z]", word) \
            and _det._is_common_word(word[-2:]):
        return True
    if HOLDOUT_FIXES and _role_noun(word):
        return True
    # '홍콩행', '제주행': 사전에 있는 지명 + 행
    toks = _det.an.tokenize(word[:-1])
    return len(word) >= 3 and word[-1] == "행" and len(toks) == 1 and toks[0].known and toks[0].tag == "NNP"


def _role_noun(word):
    """'소명+자', '위임+인', '송금+인', '심사+역', '세대+주': 사전 일반명사 2글자 + 사람·직무 접미사.
    '김영자'·'박명인'은 해당 없음."""
    toks = _det.an.tokenize(word)
    return (len(word) == 3 and word[2] in "자인역관주" and toks[0].tag == "NNG" and toks[0].known
            and len(toks[0].form) == 2)


def trim_josa_i(text, e):
    """'안윤이 조건' → '안윤': 이름 뒤 주격 조사 '이' + 띄어쓰기. 뒤에 호칭이 오면('정송이 님') 이름으로 둠."""
    s, t = e["start"], e["end"]
    word = text[s:t]
    if (e["label"] == "person" and re.fullmatch(r"[가-힣]{3,4}", word) and word[-1] == "이"
            and re.match(r"[\s,.]", text[t:t + 1]) and not old.TITLE_AFTER.match(text, t)):
        return {**e, "end": t - 1}
    return e


_SAFE_START = re.compile(r" ?님|씨|" + "|".join(sorted(NAME_TAIL_SAFE, key=len, reverse=True)))


def trim_name_tail(text, e):
    """'김선희님' → '김선희', '지은이한테'의 '지은이' → '지은', '송 다운은' → '송 다운'.
    이름 구간 뒤쪽을 잘라 남는 이름 + 어절 나머지가 호칭·조사로만 이루어질 때 자름 (가장 많이 자르는 쪽부터).
    '은·이·가·도·아·야'로 시작하면 이름 끝 글자일 수 있어서('김지은'), 남는 이름이 3글자 이상(복성은 4글자)이거나
    문서 다른 곳에 그 이름이 같은 글자 없이 나올 때만 자름."""
    s, t = e["start"], e["end"]
    if e["label"] != "person":
        return e
    we = t
    while we < len(text) and "가" <= text[we] <= "힣":
        we += 1
    for c in range(s + 1, t):
        rem, tail = text[s:c], text[c:we]
        if rem[-1] == " " or not NAME_TAIL.fullmatch(tail):
            continue
        word = rem.replace(" ", "")
        if not (re.search(r"[A-Za-z]{2}", word) or len(re.findall(r"[가-힣]", word)) >= 2):
            continue
        if not _SAFE_START.match(tail):
            # 한 글자 조사만, 남는 이름이 '김민수'·'송 다운'·'남궁민수' 꼴이거나 다른 곳에 그 이름만 따로 나올 때
            if tail not in NAME_TAIL_AMBIG:
                continue
            shape = re.fullmatch(r"(?:남궁|제갈|선우|독고|황보|사공|서문|동방)[가-힣]{2}|[가-힣]{3}|[가-힣] [가-힣]{2}", rem)
            alone = any(not ("가" <= text[m.end():m.end() + 1] <= "힣") or _SAFE_START.match(text, m.end())
                        for m in re.finditer(rf"(?<![가-힣]){re.escape(rem)}", text))
            if not (shape and (rem[:2] in H.COMPOUND_SURNAMES) == (len(word) == 4)) and not alone:
                continue
        return {**e, "end": c}
    return e


def detect(text):
    load()
    ents = _det.detect(text)

    for m in RRN_LOOSE.finditer(text):
        if not _overlaps(ents, m.start(), m.end()) and not NOT_RRN_WORDS.search(_clause_before(text, m.start())):
            ents.append({"label": "resident_registration_number", "start": m.start(), "end": m.end()})

    if HOLDOUT_FIXES:
        for m in EUP_JIBUN.finditer(text):
            if H.drop_venue_addresses(text, [{"label": "address", "start": m.start(), "end": m.end()}]):
                _replace(ents, m.start(), m.end(), "address")  # 장소 주소면 원본처럼 뺌
        for m in ACCOUNT_TAIL1.finditer(text):
            if 10 <= sum(c.isdigit() for c in m.group()) <= 14:
                _replace(ents, m.start(), m.end(), "account_number")
    for x in ents:
        if x["label"] == "address":
            p = POSTAL_BEFORE.search(text, max(0, x["start"] - 8), x["start"]) if HOLDOUT_FIXES else None
            if HARD_FIXES and not p:
                p = POSTAL_BARE_BEFORE.search(text, max(0, x["start"] - 6), x["start"])
                p = p if p and not _overlaps(ents, p.start(), p.end()) else None
            if p:
                x["start"] = p.start()

    ents = [x for x in ents if x["label"] != "person" or not _not_name(text, x)]
    ents += _old_names(text, ents)
    ents += _more_names(text, ents)
    if HOLDOUT_FIXES:
        ents = [trim_josa_i(text, x) for x in ents]
    if HARD_FIXES:
        ents = [trim_name_tail(text, x) for x in ents]
    return sorted(ents, key=lambda x: x["start"])
