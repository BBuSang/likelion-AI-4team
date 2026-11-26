"""받아쓴 상담 글의 개인정보 탐지 = 기존 적재 탐지기 + 통화에만 있는 경우 보완.

문서는 '주민번호: 700815-…'처럼 항목 이름 옆에 값이 있지만, 통화는 질문(상담원)과 답(고객)이 나뉘어
답만 보면 "700815 입니다", "배수지요." 처럼 무엇인지 알 수 없다. 그래서
  1) 대화 맥락: 개인정보를 묻는 문장 바로 다음 답(같은 화자가 아닌 문장)에서 그 종류의 값을 찾아 가림
  2) 받아쓴 이메일: '골뱅이·닷컴'을 말로 하거나 @ 가 '-'로 받아써진 모양
  3) 과한 가림 줄이기: 기존 탐지기가 잘못 받아쓴 은행·부서 이름, 일반 명사를 이름으로 잡은 것 빼기
입력은 transcribe.to_text 형식 ('[분:초] 화자: 말' 한 줄에 한 문장). 화자가 없어도 됨(모노 녹음).
생년월일은 따로 라벨이 없어 주민번호 앞자리와 같은 resident_registration_number 로 가린다.
"""
import re

from app.pii import pii_pipeline

_LINE = re.compile(r"^\[\d\d:\d\d\]\s*(?:(상담원|고객):\s*)?", re.M)

# 묻는 말 → 원하는 종류 (한 문장이 여러 개를 물을 수 있음: "성함과 생년월일")
_ASK = [
    # '네 자리'는 '내 자리'로 받아써지는 일이 많음
    ("card", re.compile(r"카드\s*(?:번호|.{0,6}(?:끝|뒷|뒤)\s*(?:번호|자리|[네내]\s*자리))")),
    ("rrn", re.compile(r"주민\s*(?:등록)?\s*번호")),
    ("birth", re.compile(r"생년\s*월일|생일|생년월")),
    ("account", re.compile(r"계좌")),
    ("email", re.compile(r"이메일|메일")),
    ("address", re.compile(r"주소")),
    ("person", re.compile(r"성함|이름")),
    ("phone", re.compile(r"휴대폰|핸드폰|전화\s*번호|연락처|연락\s*(?:가능한|받으실)?\s*번호|번호")),
]
# 받아쓰기는 묻는 말 끝에 ? 대신 . 을 찍기도 함 ("집주소는요.")
_ASKING = re.compile(r"\?|주세요|주시겠|부탁|말씀|알려|불러|어떻게\s*되|(?:[은는이가도]|번호|자리|성함|주소|이름|월일|메일)요\s*[.?]?$")
_NUM = re.compile(r"\d[\d\- ]*\d|\d")
_DATE = re.compile(r"\d{2,4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일")
_SIDO = re.compile(r"(?:서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충청|충북|충남|전라|전북|전남|경상|경북|경남|제주)\S*"
                   r"|[가-힣]+(?:시|군|구)(?=\s)")
_ADDR_TAIL = re.compile(r"\s*(?:(?:으?로|에)\s*보내.*|번지.*|이?요\.?|입니다\.?|[.,])\s*$")
_NAME_LEAD = re.compile(r"^(?:네|예|저|제\s*이름은|저는|이름은)\s*,?\s*")
_NAME_TAIL = re.compile(r"(?:이고요|이구요|이고|고요|이요|이에요|예요|입니다|이라고\s*합니다|요|고)?\s*[.,!?]?$")
_HANGUL_NAME = re.compile(r"[가-힣]{2,4}")

# 받아쓴 이메일: 아이디 + (골뱅이|@|-) + 도메인 + (닷|.) + 컴/넷
_DOMAINS = r"(?:naver|gmail|daum|hanmail|kakao|nate|outlook|네이버|지메일|다음|한메일|카카오|네이트)"
_EMAIL = re.compile(r"(?:[a-z0-9._]+|[가-힣a-z0-9]+(?:\s[가-힣a-z0-9]+){0,3})\s*(?:@|골뱅이|-)\s*" + _DOMAINS +
                    r"\s*(?:\.|닷|점)\s*(?:com|net|co\.kr|컴|넷|씨오\s*닷?\s*케이알)", re.I)

# 통화 첫머리의 '○○은행 고객센터' 처럼 이름 뒤에 부서가 오면 이름 아님
_ORG_AFTER = re.compile(r"\s*(?:고객센터|상담센터|콜센터|금융사고|대출|외환|인터넷뱅킹|연금|은행|상담(?![원사]))")
# 답 안에서 도메인이 보이면 아이디째 이메일로 봄 ('골뱅이'가 엉뚱하게 받아써져도: syoon90goldanynaver.com)
_EMAIL_TOKEN = re.compile(r"[A-Za-z0-9._\-가-힣]*" + _DOMAINS + r"\s*(?:\.|닷|점)\s*(?:com|net|co\.kr|컴|넷)", re.I)
_TITLE_AFTER = re.compile(r"\s*(?:고객님|님|씨|상담원|입니다|이고|고요|이구요|이라고|라고|이요|요[.,]|,)")


def _lines(text):
    """→ [(줄 시작, 말 시작, 줄 끝, 화자)]"""
    out, pos = [], 0
    for line in text.split("\n"):
        m = _LINE.match(line)
        body = m.end() if m else 0
        out.append((pos, pos + body, pos + len(line), m.group(1) if m else None))
        pos += len(line) + 1
    return out


def _wants(sentence):
    if not _ASKING.search(sentence):
        return set()
    w = set()
    rest = sentence
    for kind, pat in _ASK:
        if pat.search(rest):
            w.add(kind)
            rest = pat.sub(" ", rest)  # '카드 끝 번호' 가 다시 '번호'(전화)로 잡히지 않게
    return w


def _answer_spans(text, b, e, wants):
    s = text[b:e]
    out = []
    taken = []
    if "birth" in wants or "rrn" in wants:
        for m in _DATE.finditer(s):
            out.append(("resident_registration_number", b + m.start(), b + m.end()))
            taken.append((m.start(), m.end()))
    if "address" in wants:
        m = _SIDO.search(s)
        if m:
            tail = _ADDR_TAIL.search(s[m.start():])
            end = m.start() + (tail.start() if tail else len(s) - m.start())
            if end - m.start() >= 4:
                out.append(("address", b + m.start(), b + end))
                taken.append((m.start(), end))
    nums = [m for m in _NUM.finditer(s) if not any(x < m.end() and m.start() < y for x, y in taken)]
    for m in nums:
        d = re.sub(r"\D", "", m.group())
        if len(d) < 4:
            continue
        if "rrn" in wants and (5 <= len(d) <= 7 or len(d) == 13):  # 앞자리 6자리가 하나 빠지거나 붙어 받아써지기도 함
            lab = "resident_registration_number"
        elif "card" in wants and (4 <= len(d) <= 6 or len(d) >= 12):  # 끝 네 자리에 숫자가 하나 더 붙어 받아써지기도 함
            lab = "card_number"
        elif "account" in wants and len(d) >= 6:
            lab = "account_number"
        elif "phone" in wants and len(d) >= 7:
            lab = "phone_number"
        elif "birth" in wants and len(d) in (6, 8):
            lab = "resident_registration_number"
        else:
            continue
        out.append((lab, b + m.start(), b + m.end()))
        taken.append((m.start(), m.end()))
    if "email" in wants:
        for m in _EMAIL_TOKEN.finditer(s):
            out.append(("email", b + m.start(), b + m.end()))
        # 말로 불러 준 이메일 ('두심 고 70 골뱅이 다음 닷네…') — 받아쓰기가 끝을 망가뜨려도 '골뱅이'·@ 가 있으면 답 전체
        if re.search(r"골뱅이|@", s):
            head = _NAME_LEAD.sub("", s)
            out.append(("email", b + len(s) - len(head), e))
    if "person" in wants:
        head = _NAME_LEAD.sub("", s)
        off = len(s) - len(head)
        first = re.split(r"[\s,.]", head, maxsplit=1)[0]
        cand = _NAME_TAIL.sub("", first)
        tags = pii_pipeline._word_tags(cand) if _HANGUL_NAME.fullmatch(cand) else []
        # 조사·어미가 붙은 말('영어로는')은 이름 아님
        if tags and not re.search(r"\d", first) and not any(t[0] in "JE" for t in tags):
            out.append(("person", b + off, b + off + len(cand)))
    return out


def _context_spans(text):
    lines = _lines(text)
    out = []
    for i, (_, b, e, spk) in enumerate(lines):
        if spk == "고객":
            continue
        wants = _wants(text[b:e])
        if not wants:
            continue
        # 답: 묻는 사람이 다시 말하기 전까지 상대의 문장들 (최대 3문장, 답을 끊어 말하는 경우)
        #     화자 정보가 없으면 바로 다음 2문장 (또 묻는 문장이 나오면 멈춤)
        got = 0
        for _, b2, e2, spk2 in lines[i + 1:i + 5]:
            if spk is not None and spk2 == spk:
                if got:
                    break
                continue
            if spk is None and _wants(text[b2:e2]):
                break
            out += _answer_spans(text, b2, e2, wants)
            got += 1
            if got >= (3 if spk is not None else 2):
                break
    return out


# 흔한 성 (통계청 인구 많은 순 + 복성). 품사만 보고 빼면 '나은'·'하은'·'이국민' 같은 진짜 이름이 지워져서,
# 첫 글자가 성이면 품사 규칙으로는 빼지 않음
_SURNAMES = set("김이박최정강조윤장임한오서신권황안송류유전홍고문양손배백허남심노하곽성차주우구민진나지엄변채원천방공현함염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용예봉경사부가복태피")
_COMPOUND_SURNAMES = {"남궁", "제갈", "선우", "독고", "황보", "사공", "서문", "동방", "어금"}
_MONEY = re.compile(r"[십백천만억]*원|[십백천만억]+")
_BANK_LIAISON = re.compile(r"[가-힣]+[은문]행$")  # '새봄은행'이 '세보문행'처럼 소리 나는 대로 받아써짐 (받침 ㅁ + 은 → 문)


def _drop_reason(text, s):
    """기존 탐지기가 이름으로 잡은 구간을 뺄 까닭. 없으면 None (= 남김).
    기본은 남김 — 확실히 이름이 아닌 증거가 있을 때만 뺀다 (가리는 쪽 실수는 유출이 아니지만 빼는 쪽 실수는 유출).
    규칙마다 실전1·2 문서 세트의 정답 이름 10,719개를 지우지 않는지 확인함"""
    w0 = text[s["start"]:s["end"]]
    after = text[s["end"]:s["end"] + 8]
    w = _NAME_TAIL.sub("", w0) or w0  # '임하준이요' 처럼 끝말이 같이 잡힌 경우 떼고 봄
    # 1) 뒤에 무엇이 와도 이름이 아닌 것
    if _BANK_LIAISON.fullmatch(w):
        return "bank"
    if re.search(r"\d", w):
        return "digit"
    if _MONEY.fullmatch(w) and re.search(r"\d\s*$", text[max(0, s["start"] - 3):s["start"]]):  # '12만 4천원' 의 '천원'
        return "money"
    if re.fullmatch(r"골뱅이?", w):  # 말로 한 이메일의 @
        return "at"
    if re.search(r"\s", w0):
        # 띄어 쓴 이름('노 예린', '남궁 재희')은 남기고, 조사·어미가 섞인 말('보내신 분 계좌번호', '미국에 있는')만 뺌
        first = w0.split()[0]
        if (len(first) == 1 and first in _SURNAMES) or first in _COMPOUND_SURNAMES:
            return None
        if any(t[0] in "JE" for part in w0.split() for t in pii_pipeline._word_tags(part)):
            return "phrase"
        return None
    if re.fullmatch(r"[가-힣]{5,}", w):
        # 다섯 글자 이상 한글이 아는 명사 여럿으로 쪼개지면 이름 아님 ('인터넷뱅킹' = 인터넷+뱅킹).
        # 한 덩어리로 남는 말은 외국 이름일 수 있어 남김 ('마크토마스')
        tags = pii_pipeline._word_tags(w)
        if len(tags) >= 2 and all(t.startswith("NN") for t in tags):
            return "long"
    # 2) 호칭·직함이 붙으면 이름
    if _TITLE_AFTER.match(after):
        return None
    if _ORG_AFTER.match(after) and w[0] not in _SURNAMES:  # '외환 상담' (성으로 시작하면 '위하람 고객센터' 처럼 이름일 수 있음)
        return "org_after"
    # 3) 품사 — 성으로 시작하거나 두 글자 이하면 성 없는 이름('아린', '아라')일 수 있어 안 씀
    if w[0] in _SURNAMES or len(w) < 3:
        return None
    tags = pii_pipeline._word_tags(w)
    if tags and tags[0] in ("IC", "NP"):  # 감탄사·대명사 ('저기요', '저희')
        return "ic_np"
    if tags and tags[0][:2] in ("VV", "VA", "VX") and tags[-1][0] == "E":  # 활용한 동사·형용사 ('좋으나', '곧으신')
        return "verb"
    return None


def _keep_person(text, s):
    return s["label"] != "person" or _drop_reason(text, s) is None


def _fix_label(text, s):
    """기존 탐지기가 주소('충청남도 천안시 …')를 이름으로 잡은 경우 주소로 바꿈"""
    w = text[s["start"]:s["end"]]
    if s["label"] == "person" and _SIDO.match(text, s["start"]) and (re.search(r"\s", w) or len(w) >= 5):
        return {**s, "label": "address"}
    return s


def _merge(spans):
    """겹치면 더 넓은 쪽 (앞 구간을 늘림)"""
    out = []
    for s in sorted(spans, key=lambda s: (s["start"], -(s["end"] - s["start"]))):
        if out and s["start"] < out[-1]["end"]:
            if s["end"] > out[-1]["end"]:
                out[-1] = {**out[-1], "end": s["end"]}
            continue
        out.append(dict(s))
    return out


def _extra(text):
    extra = [{"label": lab, "start": a, "end": z} for lab, a, z in _context_spans(text)]
    extra += [{"label": "email", "start": m.start(), "end": m.end()} for m in _EMAIL.finditer(text)]
    return extra


def detect(text):
    base = [s for s in map(lambda s: _fix_label(text, s), pii_pipeline.detect(text)) if _keep_person(text, s)]
    return _merge(base + _extra(text))


def detect_scored(text):
    """적재용: pii_pipeline.detect_scored 와 같은 모양 {spans, hints, needs_review}.
    통화 규칙으로 더한 구간은 flag='rule_added' (검수 화면에 '보완 규칙만 찾았어요'로 보임).
    받아쓰기가 틀릴 수 있어 녹음 문서는 늘 검수로 보냄 (needs_review=True)"""
    sc = pii_pipeline.detect_scored(text)
    base = [s for s in (_fix_label(text, s) for s in sc["spans"]) if _keep_person(text, s)]
    extra = [{**s, "src": "call", "p": None, "flag": "rule_added"} for s in _extra(text)
             if not any(b["start"] <= s["start"] and s["end"] <= b["end"] for b in base)]
    spans = _merge(base + extra)
    hints = [h for h in sc["hints"] if not any(s["start"] < h["end"] and h["start"] < s["end"] for s in spans)]
    return {"spans": spans, "hints": hints, "needs_review": True}
