"""v2 가명처리: 우리 탐지기 → 정규화 → HMAC 토큰 → 치환 + 언급 정보 (plan_v2 P2).

    from pseudo.pipeline import process
    r = process("윤 선영 고객(01003492063)이 …")
    r["masked"]    가명처리 본문 (외부로 나가도 되는 텍스트)
    r["mentions"]  [{label, token, o_start, o_end, m_start, m_end, line_no, raw, normalized,
                     form, role, own_role, title, org, sido, sigungu}, ...]
                   role = 문서 안 같은 토큰끼리 공유한 역할 / own_role = 그 언급 자체의 단서로 정한 역할
                   o_* = 원문 기준 위치 (vault.mention_raw, 채점 형식) / m_* = 가명 본문 기준 위치 (rag.mentions)
                   raw·normalized 는 vault 에만 저장. 나머지는 rag 에 저장해도 되는 비개인정보 값

탐지: 실전 1 운영 탐지기 PIIDetector (규칙 + atonlee). 정답 라벨은 쓰지 않음
정규화 (같은 값의 표기 차이만 없앰. 조사·호칭은 탐지기 경계 보정이 맡음)
  person   공백 제거                       윤 선영 → 윤선영
  phone    숫자만, +82 → 0                 +82 10-0349-2063 → 01003492063
  RRN·계좌·카드  숫자만
  email    소문자
  address  연속 공백 1칸, 앞 우편번호 제거, 맨 앞 시·도 약칭 → 정식 명칭   대구 수성구 … → 대구광역시 수성구 …
           (우편번호·시·도 규칙은 holdout 분석에서 찾은 규칙. dev/test에는 해당 변형 없음 — worklog 19번)
토큰: HMAC-SHA256(HMAC_KEY, "label:정규화값") 앞 8자리 → <PERSON_xxxxxxxx>. 키 버전은 KEY_VER
"""
import hashlib
import hmac
import re
import sys

from . import roles

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config  # noqa: E402

KEY_VER = 1
TOKEN_LEN = 8
TOKEN_RE = re.compile(r"<([A-Z]+)_[0-9a-f]{8}>")
PREFIX = {"person": "PERSON", "phone_number": "PHONE", "email": "EMAIL", "address": "ADDRESS",
          "resident_registration_number": "RRN", "account_number": "ACCOUNT", "card_number": "CARD"}

SIDO = {"서울특별시": ["서울시", "서울"], "부산광역시": ["부산시", "부산"], "대구광역시": ["대구시", "대구"],
        "인천광역시": ["인천시", "인천"], "광주광역시": ["광주시", "광주"], "대전광역시": ["대전시", "대전"],
        "울산광역시": ["울산시", "울산"], "세종특별자치시": ["세종시", "세종"], "경기도": ["경기"],
        "강원특별자치도": ["강원도", "강원"], "충청북도": ["충북"], "충청남도": ["충남"],
        "전북특별자치도": ["전라북도", "전북"], "전라남도": ["전남"], "경상북도": ["경북"], "경상남도": ["경남"],
        "제주특별자치도": ["제주도", "제주"]}
SIDO_ALIAS = {a: full for full, al in SIDO.items() for a in [full, *al]}

# 이름만 부른 표기(form=given) 판정용 성씨. 저장만 하고 연결에는 쓰지 않음 (이름만 부른 표기 연결 안 함 결정)
SURNAMES = set("김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구원나진지엄채"
               "염여변추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용봉경천방공현함예태")
SURNAMES_2 = ("남궁", "황보", "제갈", "선우", "독고", "사공", "서문")

_detector = None


def detector():
    """모델 로딩이 수십 초 걸려 처음 부를 때 한 번만 만듦."""
    global _detector
    if _detector is None:
        sys.path.insert(0, str(config.ROOT))     # 실전프로젝트2/pii_detector.py
        from pii_detector import PIIDetector
        _detector = PIIDetector()
    return _detector


def detect(text):
    return [{"label": e["label"], "start": e["start"], "end": e["end"]} for e in detector().detect(text)]


# ───────────────────────── 정규화·토큰 ─────────────────────────
def normalize_address(v):
    v = re.sub(r"\s+", " ", v.strip())
    v = re.sub(r"^\(?\d{5}\)?\s*", "", v)
    head, _, rest = v.partition(" ")
    head = SIDO_ALIAS.get(head, head)
    return f"{head} {rest}" if rest else head


def normalize(label, value):
    v = value.strip()
    if label == "person":
        return re.sub(r"\s+", "", v)
    if label == "phone_number":
        d = re.sub(r"\D", "", v)
        return "0" + d[2:] if v.startswith("+82") else d
    if label in ("resident_registration_number", "account_number", "card_number"):
        return re.sub(r"\D", "", v)
    if label == "email":
        return v.lower()
    if label == "address":
        return normalize_address(v)
    return v


def make_token(label, normalized, key=None):
    key = config.HMAC_KEY if key is None else key
    if not key:
        raise RuntimeError("HMAC_KEY 가 비어 있습니다. .env 를 확인하세요.")
    h = hmac.new(key.encode(), f"{label}:{normalized}".encode("utf-8"), hashlib.sha256).hexdigest()
    return f"<{PREFIX.get(label, label.upper())}_{h[:TOKEN_LEN]}>"


# ───────────────────────── 언급 정보 ─────────────────────────
def name_form(original, normalized, after=""):
    if re.search(r"\s", original.strip()):
        return "spaced"
    # 2글자 이름 뒤에 부르는 말(서연아, 성호 씨, 민준이랑)이 붙으면 이름만 부른 표기 (첫 글자가 성씨여도)
    if len(normalized) == 2 and re.match(r"\s?(씨|(아|야|이랑|랑)(?![가-힣]))", after):
        return "given"
    if normalized.startswith(SURNAMES_2):
        return "full" if len(normalized) >= 3 else "given"
    if len(normalized) >= 3 or (len(normalized) == 2 and normalized[0] in SURNAMES):
        return "full"
    return "given"


def region(normalized):
    """주소 일반화: 시·도 + 시·군·구까지만 (읍·면·동 이하 버림)."""
    words = normalized.split(" ")
    if not words or words[0] not in SIDO:
        return None, None
    sgg = []
    for w in words[1:3]:
        if re.fullmatch(r"[가-힣]+[시군구]", w):
            sgg.append(w)
        else:
            break
    return words[0], " ".join(sgg) or None


def drop_overlaps(spans):
    """겹치는 구간은 먼저 시작하고 더 긴 것을 남김."""
    kept, end = [], -1
    for s in sorted(spans, key=lambda s: (s["start"], -(s["end"] - s["start"]))):
        if s["start"] >= end:
            kept.append(s)
            end = s["end"]
    return kept


def process(text, spans=None, key=None):
    """spans 를 주지 않으면 우리 탐지기로 찾음 (테스트에서만 직접 넣음)."""
    spans = drop_overlaps(detect(text) if spans is None else spans)
    out, mentions, last, pos = [], [], 0, 0
    for s in spans:
        raw = text[s["start"]:s["end"]]
        norm = normalize(s["label"], raw)
        token = make_token(s["label"], norm, key)
        out.append(text[last:s["start"]])
        pos += s["start"] - last
        m = {"label": s["label"], "token": token, "o_start": s["start"], "o_end": s["end"],
             "m_start": pos, "m_end": pos + len(token), "line_no": text.count("\n", 0, s["start"]),
             "raw": raw, "normalized": norm, "form": None, "role": None, "own_role": None, "title": None,
             "org": None, "sido": None, "sigungu": None}
        if s["label"] == "person":
            m["form"] = name_form(raw, norm, text[s["end"]:s["end"] + 4])
        elif s["label"] == "address":
            m["sido"], m["sigungu"] = region(norm)
        out.append(token)
        pos += len(token)
        last = s["end"]
        mentions.append(m)
    out.append(text[last:])

    persons = [m for m in mentions if m["label"] == "person"]
    info = roles.extract(text, [{"start": m["o_start"], "end": m["o_end"], "key": m["token"]} for m in persons])
    for m, r in zip(persons, info):
        m["role"], m["own_role"], m["title"], m["org"] = r["role"], r["own_role"], r["title"], r["org"]
    return {"masked": "".join(out), "mentions": mentions}
