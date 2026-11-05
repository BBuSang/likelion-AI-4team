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
    return [{"label": e["label"], "start": e["start"], "end": e["end"]}
            for e in _get_detector().detect(text)]


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
