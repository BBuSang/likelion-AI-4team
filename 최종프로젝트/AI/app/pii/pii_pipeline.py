"""탐지 + 가명처리 모듈.

  - detect()       : 실전 1에서 고른 운영 탐지기(규칙 + atonlee 합집합, app/pii/detector/pii_detector.py, 실전프로젝트1에서 복사)를 그대로 부릅니다.
  - pseudonymize() : 라벨별로 정규화한 값을 HMAC-SHA256 으로 토큰화합니다 (<PERSON_03f0d420>).
                     같은 값이면 문서·질의를 가리지 않고 항상 같은 토큰이 됩니다 (결정적 토큰).
                     이름은 기본적으로 문자열 기준. 적재 때 person_resolve.resolve() 가 구간에 person_key 를 붙이면
                     HMAC("person|이름|key") 로 사람 단위 토큰을 만들어 동명이인을 나눕니다.
  - search_tokens(): 본문에는 넣지 않고 청크의 tokens 배열(검색 필터)에만 넣는 보조 토큰.
                     이름 토큰 <NAME_…>(동명이인 전체), 이름만 토큰 <GIVEN_…>('시우' 질의용), 주소 조각 <ADDRP_…>(일부 주소 질의용).
                     질의 쪽은 vault 없이 이 토큰들을 직접 계산할 수 있습니다.

지켜야 할 약속 (다른 코드가 이 형식에 의존합니다)
  1. detect(text) -> [{"label": str, "start": int, "end": int}, ...]
       - 위치는 원문 기준 [start, end).
  2. pseudonymize(text, spans) -> (masked_text, records)
       - masked_text : 탐지 구간을 토큰으로 바꾼 텍스트
       - records     : [{"label", "start", "end", "original", "normalized", "token"}, ...]
                       start/end 는 '원문' 기준 위치 (토큰 일관성 채점에 사용)
  3. 토큰은 TOKEN_PATTERN 에 맞아야 합니다. (적재 시 청크별 토큰 목록을 이 패턴으로 뽑습니다)
  4. 비밀키는 config.HMAC_KEY (.env) 에서 읽고 코드에 쓰지 않습니다.
"""
import hashlib
import hmac
import re
import sys
import unicodedata
from pathlib import Path

from app.core import config

TOKEN_PATTERN = re.compile(r"<[A-Z_]+_[0-9A-Za-z]+>")
SEARCH_ONLY_PREFIXES = ("NAME", "GIVEN", "ADDRP")  # tokens 배열에만 있고 본문에는 없는 보조 토큰 (복원 대상 아님)

# 실전 1 탐지기 코드 (app/pii/detector 로 복사해 옴. 내부가 평평한 import 라 sys.path 에 넣어 씀)
DETECTOR_DIR = Path(__file__).resolve().parent / "detector"

# ---------------------------------------------------------------- 탐지
_detector = None


def _get_detector():
    """모델 로딩이 오래 걸리므로(첫 실행 1~2분) 처음 부를 때 한 번만 만든다."""
    global _detector
    if _detector is None:
        if str(DETECTOR_DIR) not in sys.path:
            sys.path.insert(0, str(DETECTOR_DIR))
        from pii_detector import PIIDetector
        _detector = PIIDetector()
    return _detector


def detect(text):
    raw = [{"label": e["label"], "start": e["start"], "end": e["end"]} for e in _get_detector().detect(text)]
    return complete_spans(text, raw)


# ---------------------------------------------------------------- 탐지 보완 (worklog 20장)
# 실전 1 탐지기(app/pii/detector)는 그대로 두고, 그 결과에 최종 프로젝트에서 찾은 두 가지를 더한다.
# 적재가 2주차 탐지 캐시를 읽을 때도(load.load) 같은 함수를 지나므로 어디서 탐지하든 결과가 같다.
#  ① 라벨 없는 긴 숫자: guard 가 <MASKED> 로 가리던 9자리 이상 숫자를 번호 토큰으로 → 그 번호로 검색·실명 복원 가능
#     (synth3 '(464039625067 신규 정보…' 같은 계좌 13개). 라벨은 모양으로 고름 — 틀려도 가려지는 건 같음
#     사업자번호·접수번호·사건번호처럼 개인정보가 아닌 번호는 빼고 guard 에 맡김 (실전1 정답 기준)
#  ② 이름 오탐: 법 줄임말('신정법', '금소법' — 에이전트가 사람 기록으로 보냄)과 상태·업무 말('보류', '상태', '이수')
#     — 뒤에 호칭·직함이 붙으면('이수 님') 이름으로 남김
_TITLE_AFTER = re.compile(r"\s?(님|씨|고객|대리|과장|차장|부장|팀장|주임|사원|선임|책임|이사|상무|전무|대표|매니저|프로|원장|지점장)")
_LAW_ABBR = re.compile(r"[가-힣]{2}법")  # 세 글자 + '법' 끝 (신정법·금소법·전금법·여전법·개보법·특금법 …)
_NOT_NAME_WORDS = {"보류", "상태", "현 상태", "접수", "완료", "처리", "이수", "이미", "서술형"}
# 업무 말 + 된/될 (반송된·연체된·누락될). 탐지 모델이 뒤의 '고객'을 보고 이름으로 잡음 (UI 질문에서 발견, 25장)
# 정답 라벨 19,162개 이름 중 된/될 로 끝나는 것 0개
_VERB_FORM = re.compile(r"[가-힣]{2}[된될]")
# 개인정보가 아닌 번호 (라벨이 앞에 오거나 모양이 정해진 것) — 이건 guard 가 전처럼 <MASKED> 로만 가림
_ID_LABEL_BEFORE = re.compile(r"(사업자(?:등록)?\s?번호|(?:접수|주문|사건|민원|송장|운송장|예약|회원|계약|문서|관리|승인|거래|신청|"
                              r"요청|티켓|등록|증권|고객|사원|직원)\s?(?:번호|건)|운송장|송장|대표\s?번호|경찰청|내선|사번|REQ|VOC)"
                              r"[^\d\n]{0,8}$")
_ID_LABEL_AFTER = re.compile(r"^\s?(?:\((?:대표|운송장|송장|접수|주문|사업자)|\([^)]{0,8}아님|번\s?참고)")
_ID_PREFIX = re.compile(r"[A-Za-z]{1,5}-$")  # SD-20240715-88, CAN-20240610-22 같은 영문 머리 번호
_ID_SHAPE = re.compile(r"\d{3}-\d{2}-\d{5}|(?:19|20)\d{2}(?:\d{2}){0,2}-\d{3,8}")  # 사업자번호 3-2-5, 날짜로 시작하는 접수번호


# ③ 성 없는 두 글자 이름 + 호칭·직함 ('보람대리님', '다운고객님', '하늘 고객이') — worklog 21장
#    실전 1 규칙은 '우수고객님'·'신규 고객' 같은 일반 말 때문에 두 글자 + 고객·직함을 일부러 안 잡음.
#    그 일반 말을 '고객·직원 앞에 오는 꾸밈말' 목록으로 빼고 잡는다. 목록은 데이터에서 뽑지 않고 국어 쓰임으로 만듦
_GIVEN_TITLE = re.compile(
    r"(?:(?<![가-힣])|(?<=회원)|(?<=창구)|(?<=담당))([가-힣]{2})( ?)"
    r"(?=(고객|대리|과장|차장|부장|팀장|주임|선임|책임)(?:님)?"
    r"(?:께서|에게|한테|이랑|이나|께|이|가|은|는|의|과|와|도|을|를|만|나|이고|이며|까지|부터)?(?![가-힣]))")
_TITLE_MODIFIERS = {
    # 범위·지시
    "해당", "기존", "신규", "전체", "모든", "일부", "다수", "소수", "각각", "상기", "위의", "다른", "같은", "이번", "지난",
    "다음", "본인", "우리", "저희", "상대", "대상", "관련", "주요", "핵심", "일반", "특정", "개별", "여러", "어느", "모두",
    # 고객 종류
    "우수", "우량", "우대", "단골", "장기", "휴면", "연체", "고령", "청년", "노인", "성인", "해외", "국내", "외국", "개인",
    "법인", "기업", "가입", "탈퇴", "이탈", "잠재", "불만", "민원", "악성", "거래", "카드", "대출", "예금", "보험", "회원",
    "골드", "실버", "가족", "부모", "자녀", "남성", "여성", "남자", "여자", "직장", "학생", "주부", "사업", "상인", "미납",
    "신청", "피해", "분실", "명의", "실제", "최초", "최근", "과거", "현재", "당일", "익일", "예비", "임시", "정식", "내부",
    # 직원 쪽
    "담당", "직원", "부서", "선배", "후배", "동기", "신입", "경력", "전임", "후임", "현임", "책임", "선임", "본사", "지점",
    "영업", "인사", "총무", "재무", "감사", "심사", "상담", "창구", "콜센", "센터", "본부", "팀의", "부의",
}
_DEPT_END = re.compile(r"[팀부과실국청원]$")


_kiwi = None


def _word_tags(w):
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return [t.tag for t in _kiwi.analyze(w, top_n=1)[0][0]]


def _given_title_spans(text, taken):
    """붙여 쓴 꼴('다운고객님')은 꾸밈말만 빼고 잡음. 띄어 쓴 '○○ 고객'은 꾸밈말('거주 고객', '방문 고객')이 많아
    형태소 분석이 고유명사(NNP)로 볼 때만. 동사·형용사 꼴('따라 고객님')은 모두 뺌"""
    out = []
    for m in _GIVEN_TITLE.finditer(text):
        w, spaced, title = m.group(1), m.group(2), m.group(3)
        if w in _TITLE_MODIFIERS or _DEPT_END.search(w) or w in _NOT_NAME_WORDS:
            continue
        tags = _word_tags(w)
        if any(t.startswith(("V", "E", "J", "X")) for t in tags):
            continue
        if spaced and title == "고객" and tags != ["NNP"]:
            continue
        if any(s["start"] < m.end(1) and m.start(1) < s["end"] for s in taken):
            continue
        out.append({"label": "person", "start": m.start(1), "end": m.end(1)})
    return out


# ④ 문서 안 퍼뜨리기 ('김하늘' 이 탐지된 문서의 '하늘은', '하늘이랑') — worklog 22장
#    실전 1 규칙도 같은 문서의 이름만 쓴 표기를 퍼뜨리지만 '하늘'·'보람'처럼 일반 단어면 건너뜀.
#    같은 문서에 성 포함 이름이 실제로 있고, 사람 자리(뒤에 사람에게 붙는 조사)일 때만 일반 단어도 잡는다
_SURNAME_FULL = re.compile(r"[가-힣]{3}")  # 성 1 + 이름 2 (복성·외자 이름은 실전 1 규칙이 따로 다룸)
_PERSON_FOLLOW = re.compile(r"(?:께서|에게서|에게|한테|이랑|하고|님|씨|께|이가|이는|이를|이도|이의|이와|이한테|이에게|"
                            r"이나|이만|이|가|은|는|의|와|과|도|랑|를|을|나|만|까지|부터)(?![가-힣])")


def _given_in_doc_spans(text, taken):
    givens = {}
    for s in taken:
        w = text[s["start"]:s["end"]]
        if s["label"] == "person" and _SURNAME_FULL.fullmatch(w):
            givens[w[1:]] = True
    out = []
    for g in givens:
        if g in _TITLE_MODIFIERS or g in _NOT_NAME_WORDS:
            continue
        for m in re.finditer(rf"(?:(?<![가-힣])|(?<=[팀부실])){re.escape(g)}", text):
            if not _PERSON_FOLLOW.match(text, m.end()):
                continue
            if any(s["start"] < m.end() and m.start() < s["end"] for s in taken + out):
                continue
            out.append({"label": "person", "start": m.start(), "end": m.end()})
    return out


# ⑤ 한 글자 이름·성 ('린님', '최씨', '장께서', '진이한테') — worklog 23장
#    실전 1 규칙은 한 글자를 이름으로 안 봄. 한 글자 + 님·씨·높임 조사, 또는 외자 이름 부르는 꼴('진이한테')만 잡음.
#    '박 대리'처럼 성 + 직함은 팀 라벨 기준에서 대부분 개인정보로 안 봐서(성만으로는 사람을 못 특정) 안 잡음.
#    빼는 말은 국어 쓰임으로: 한 글자 + 님 일반 말(손님·형님…), 대명사(그에게·나한테), 높임 명사·낱말(님께·함께)
_ONE_NAME = re.compile(
    r"(?<![가-힣A-Za-z0-9])([가-힣])(?:"
    r"(?P<hon>님|\s?씨)"
    r"|(?P<prt>께서|께|에게|한테)"
    r"|(?P<call>이(?:한테|에게|랑|하고|께서))"
    r")(?:께서|에게|한테|이랑|이나|께|이|가|은|는|의|과|와|도|을|를|만|나|랑|하고|이고)?(?![가-힣])")
_ONE_NOT_HON = set("아손형누따해달별주임")        # 아님·손님·형님·누님·따님·해님·달님·별님·주님·임님
_ONE_PRONOUN = set("그이나너저제내네걔쟤얘누")    # 그에게·이한테·나한테·누구
_ONE_NOT_PRT = set("함님씨분")                    # 함께(낱말)·님께·씨께서·분께 (높임 명사 뒤 조사)
_ONE_NOT_CALL = set("아누오고어사나많깊높같")      # 아이한테·누이에게·오이랑 …


def _one_char_spans(text, taken):
    out = []
    for m in _ONE_NAME.finditer(text):
        c = m.group(1)
        if m.group("hon") and c in _ONE_NOT_HON:
            continue
        if m.group("prt") and c in _ONE_PRONOUN | _ONE_NOT_PRT:
            continue
        if m.group("call") and c in _ONE_NOT_CALL | _ONE_PRONOUN:
            continue
        if any(s["start"] < m.end(1) and m.start(1) < s["end"] for s in taken):
            continue
        out.append({"label": "person", "start": m.start(1), "end": m.end(1)})
    return out


def _num_label(digits):
    if len(digits) == 13 and digits[2:4] <= "12" and digits[2:4] != "00" and "01" <= digits[4:6] <= "31" and digits[6] in "1234":
        return "resident_registration_number"
    if digits.startswith("0") and 9 <= len(digits) <= 11:
        return "phone_number"
    if 14 <= len(digits) <= 16:
        return "card_number"
    return "account_number"


def complete_spans(text, spans):
    """탐지 결과 보완: 이름 오탐 빼기 + 성 없는 이름 + 호칭 + 라벨 없는 긴 숫자. 여러 번 불러도 결과가 같음"""
    out = []
    for s in spans:
        if s["label"] == "person" and _VERB_FORM.fullmatch(text[s["start"]:s["end"]]):
            continue  # '반송된 고객', '연체된 고객' — 뒤에 호칭이 있어도 이름 아님 (worklog 25장)
        if s["label"] == "person" and not _TITLE_AFTER.match(text[s["end"]:s["end"] + 6]):
            w = text[s["start"]:s["end"]]
            if _LAW_ABBR.fullmatch(w) or (text[s["end"]:s["end"] + 1] == "법" and len(w) == 2) or w in _NOT_NAME_WORDS:
                continue
        out.append(s)
    out += _given_title_spans(text, out)
    out += _given_in_doc_spans(text, out)
    out += _one_char_spans(text, out)
    for m in _GUARD_NUM.finditer(text):
        v, digits = m.group(), re.sub(r"\D", "", m.group())
        if len(digits) < 9 or _GUARD_DATE.fullmatch(v):
            continue
        if any(s["start"] < m.end() and m.start() < s["end"] for s in out):
            continue
        before, after = text[max(0, m.start() - 16):m.start()], text[m.end():m.end() + 12]
        if (_ID_SHAPE.fullmatch(v) or _ID_LABEL_BEFORE.search(before) or _ID_LABEL_AFTER.match(after)
                or _ID_PREFIX.search(before)):
            continue
        out.append({"label": _num_label(digits), "start": m.start(), "end": m.end()})
    return sorted(out, key=lambda s: (s["start"], s["end"]))


def detect_output(text):
    """출력 재검사용 탐지 (LLM 답변). 적재용 detect() 와 다르게 정상 답변을 가리지 않는 쪽을 우선한다.
      - 이름: 토큰 분류 모델만. 규칙 기반은 LLM 답변의 '위임인'·'송금인'·'인사팀' 같은 항목 이름을 이름으로 잡음
      - 번호·이메일·주소: 모델 + 규칙(정규식) — 형식 변형에 강하고 오탐이 거의 없음
    holdout 질의 88개 답변 기준: 적재용 탐지기 오탐 9건 → 0건, 일부러 섞은 원문 개인정보 12/12 잡음 (worklog 10장)."""
    det = _get_detector()  # detector 폴더를 sys.path 에 넣으므로 아래 import 보다 먼저
    from common import dedupe
    from make_union import union_entities
    model = dedupe(text, det.token.detect(text))
    rules = [e for e in dedupe(text, det.rules.detect(text)) if e["label"] != "person"]
    return [{"label": e["label"], "start": e["start"], "end": e["end"]}
            for e in dedupe(text, union_entities([rules, model]))]


# ---------------------------------------------------------------- API 로 보내기 직전 마지막 차단 (worklog 16장)
# 탐지기가 놓친 값이 임베딩·LLM API 로 가지 않게, 토큰 밖에 남은 개인정보 '모양'을 탐지기와 상관없이 가린다.
#   숫자 9자리 이상(하이픈·점·공백 하나로 나뉜 것 포함) = 전화·계좌·카드·주민번호 모양. 날짜(8자리)·금액(쉼표)·접수번호는 남음
#   이메일 모양
# 영문·숫자 경계만 본다 — 한글 바로 앞뒤('620053981703관련')도 잡고, 토큰 안의 16진수('<ACCOUNT_c007d0f3>')는 건드리지 않음
GUARD_MARK = "<MASKED>"
_GUARD_NUM = re.compile(r"(?<![A-Za-z0-9_<])\d+(?:[-. ]\d+)*(?![A-Za-z0-9_])")
# 날짜 + 시각의 시('2026-09-16 18:14' 의 '2026-09-16 18')는 10자리라도 남김. 경찰청 사건번호('2026-623736')는 가려짐 (지나치게 가리는 쪽)
_GUARD_DATE = re.compile(r"(?:19|20)\d{2}[-.]\d{1,2}[-.]\d{1,2}(?: \d{1,2})?")
_GUARD_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")


def guard(text):
    """(가린 글, 가린 개수). config.API_GUARD 가 꺼져 있으면 그대로."""
    if not config.API_GUARD:
        return text, 0
    n = 0

    def num(m):
        nonlocal n
        if sum(c.isdigit() for c in m.group()) < 9 or _GUARD_DATE.fullmatch(m.group()):
            return m.group()
        n += 1
        return GUARD_MARK

    text = _GUARD_NUM.sub(num, text)
    text, k = _GUARD_EMAIL.subn(GUARD_MARK, text)
    return text, n + k


# ---------------------------------------------------------------- 정규화
# 토큰 접두어. 길면 LLM 프롬프트가 지저분해지므로 짧게
PREFIX = {
    "person": "PERSON", "phone_number": "PHONE", "email": "EMAIL", "address": "ADDRESS",
    "resident_registration_number": "RRN", "account_number": "ACCOUNT", "card_number": "CARD",
    "organization": "ORG",
}

# 시·도 정식 명칭 → 줄임말 ('서울특별시 종로구' 와 '서울 종로구' 를 같게)
_SIDO = [
    ("서울특별시", "서울"), ("서울시", "서울"), ("부산광역시", "부산"), ("대구광역시", "대구"),
    ("인천광역시", "인천"), ("광주광역시", "광주"), ("대전광역시", "대전"), ("울산광역시", "울산"),
    ("세종특별자치시", "세종"), ("경기도", "경기"), ("강원특별자치도", "강원"), ("강원도", "강원"),
    ("충청북도", "충북"), ("충청남도", "충남"), ("전북특별자치도", "전북"), ("전라북도", "전북"),
    ("전라남도", "전남"), ("경상북도", "경북"), ("경상남도", "경남"), ("제주특별자치도", "제주"),
]


def _digits(s):
    return re.sub(r"\D", "", s)


def normalize(label, value):
    """표기만 다른 같은 값을 하나로 모은다. 너무 공격적으로 지우면 다른 값끼리 합쳐지므로 라벨별로 필요한 만큼만."""
    v = unicodedata.normalize("NFKC", value).strip()
    if label == "person":
        return re.sub(r"\s+", "", v)                      # '윤 선영' → '윤선영' (조사는 탐지기가 이미 뗌)
    if label == "phone_number":
        d = _digits(v)                                    # 010-1234-5678 / 010.1234.5678 / 01012345678
        return "0" + d[2:] if d.startswith("82") else d   # +82 10-… → 010…
    if label in ("resident_registration_number", "account_number", "card_number"):
        return _digits(v)                                 # 하이픈·공백 차이 제거
    if label == "email":
        return v.lower()
    if label == "address":
        return _normalize_address(v)
    return re.sub(r"\s+", "", v)


def _normalize_address(v):
    """주소: 띄어쓰기는 남기고(공백 하나로) 표기 차이만 맞춘다.
    띄어쓰기를 다 지우면 '종로3길 18 3층'과 '종로3길 1 83층'이 같은 토큰이 되므로 숫자 사이 구분은 유지.
    원래 표기는 vault.token_map.originals·display 에 그대로 있으므로 복원과는 무관하다."""
    v = re.sub(r"^\s*[(（]\d{5}[)）]\s*", "", v)        # 우편번호 '(05299) ' 제거 (있든 없든 같은 주소)
    for long, short in _SIDO:                          # 시·도 정식 명칭 → 줄임말 (우편번호를 뗀 뒤)
        if v.startswith(long):
            v = short + v[len(long):]
            break
    v = re.sub(r"[\s,]+", " ", v).strip()              # 쉼표 → 공백, 연속 공백 → 하나
    v = re.sub(r"(?<=[로길])\s+(?=\d)", "", v)          # '도담길 125' → '도담길125' (도로명+건물번호)
    v = re.sub(r"(?<=\d)\s+(?=(?:동|호|층)(?:\s|$))", "", v)  # '3 층' → '3층'
    v = re.sub(r"(\d+동)(?=\d)", r"\1 ", v)             # '110동905호' → '110동 905호'
    return v


# 두 글자 성 (이름이 4글자 이상일 때만 두 글자 성으로 봄)
_SURNAME2 = ("남궁", "황보", "제갈", "선우", "독고", "사공", "서문", "동방")


def split_name(name):
    """'문시우' → ('문', '시우'), '남궁하율' → ('남궁', '하율'). 2글자 이하는 성 없는 이름으로 봄 → ('', name).
    외자 이름('안윤')도 2글자라 성 없는 이름과 구별할 수 없으므로 검색 토큰은 양쪽 모두 만든다."""
    if len(name) >= 4 and name[:2] in _SURNAME2:
        return name[:2], name[2:]
    if len(name) >= 3:
        return name[:1], name[1:]
    return "", name


# 주소 조각: 시군구, 도로명+번호, 동·호·층 (띄어쓰기 차이 없이 비교하려고 공백 제거)
_ADDR_PART = re.compile(r"[가-힣]+(?:시|군|구)(?=\s|$|,)|[가-힣0-9]+(?:로|길)\s*\d+(?:-\d+)?|\d+동(?=\s|$|,)|\d+호|\d+층")


def address_parts(value):
    v = unicodedata.normalize("NFKC", value)  # 시·도 줄임말 통일 후 조각 추출
    for long, short in _SIDO:
        if v.startswith(long):
            v = short + v[len(long):]
            break
    return sorted({re.sub(r"\s+", "", m.group()) for m in _ADDR_PART.finditer(v)})


# ---------------------------------------------------------------- 가명처리
def _hmac(msg):
    if not config.HMAC_KEY:
        raise RuntimeError("HMAC_KEY 가 비어 있습니다. .env 에 팀 공용 키를 넣으세요.")
    return hmac.new(config.HMAC_KEY.encode(), msg.encode(), hashlib.sha256).hexdigest()[:8]


def make_token(label, normalized, person_key=None):
    msg = f"{label}|{normalized}" + (f"|{person_key}" if label == "person" and person_key else "")
    return f"<{PREFIX.get(label, label.upper())}_{_hmac(msg)}>"


def name_token(name):
    return f"<NAME_{_hmac('name|' + name)}>"


def given_token(given):
    return f"<GIVEN_{_hmac('given|' + given)}>"


def addr_part_token(part):
    return f"<ADDRP_{_hmac('addrp|' + part)}>"


def search_tokens(record):
    """청크 tokens 배열에만 넣는 보조 토큰 (본문 masked_text 에는 넣지 않음)."""
    if record["label"] == "person":
        return [name_token(record["normalized"]), given_token(split_name(record["normalized"])[1])]
    if record["label"] == "address":
        return [addr_part_token(p) for p in address_parts(record["original"])]
    return []


def pseudonymize(text, spans):
    """spans 에 person_resolve 가 붙인 "normalized"(이름만 쓴 표기를 성 포함 이름으로 펼친 값), "person_key" 가 있으면 사용."""
    records, out, last = [], [], 0
    for s in _drop_overlaps(spans):
        original = text[s["start"]:s["end"]]
        normalized = s.get("normalized") or normalize(s["label"], original)
        if not normalized:  # 숫자 라벨인데 숫자가 하나도 없는 경우 등: 원문 기준으로라도 가린다
            normalized = original
        token = make_token(s["label"], normalized, s.get("person_key"))
        out += [text[last:s["start"]], token]
        last = s["end"]
        rec = {"label": s["label"], "start": s["start"], "end": s["end"],
               "original": original, "normalized": normalized, "token": token}
        for k in ("person_key", "basis", "role", "aff"):
            if k in s:
                rec[k] = s[k]
        rec["search_tokens"] = search_tokens(rec)
        records.append(rec)
    out.append(text[last:])
    return "".join(out), records


def _drop_overlaps(spans):
    """겹치는 구간은 먼저 시작하고 더 긴 것을 남깁니다."""
    kept, end = [], -1
    for s in sorted(spans, key=lambda s: (s["start"], -(s["end"] - s["start"]))):
        if s["start"] >= end:
            kept.append(s)
            end = s["end"]
    return kept


IS_DEMO = False
