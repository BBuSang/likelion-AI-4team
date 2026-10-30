"""탐지 + 가명처리 모듈 (실전 1 운영 탐지기 + 정규화 + HMAC 결정적 토큰).

  - detect()       : 실전 1 운영 탐지기 PIIDetector (규칙 + atonlee 합집합, v10). 라벨 7종
  - pseudonymize() : 라벨별 정규화 → HMAC-SHA256(HMAC_KEY, "label:정규화값") 앞 8자리 → <PERSON_xxxxxxxx>
                     같은 값이면 문서·질의를 가리지 않고 항상 같은 토큰이 됨

지켜야 할 약속 (다른 코드가 이 형식에 의존합니다)
  1. detect(text) -> [{"label": str, "start": int, "end": int}, ...]   위치는 원문 기준 [start, end)
  2. pseudonymize(text, spans) -> (masked_text, records)
       - records : [{"label", "start", "end", "original", "normalized", "token"}, ...]  start/end 는 원문 기준
  3. 토큰은 TOKEN_PATTERN 에 맞아야 합니다.
  4. 비밀키는 config.HMAC_KEY (.env) 에서 읽고 코드에 쓰지 않습니다. 키를 바꾸면 전체를 다시 적재해야 합니다.

E2 비교 조건 (환경변수로 전환, 기본값이 제안 방식)
  PSEUDO_NORMALIZE=0    : B1  정규화 없이 HMAC
  PSEUDO_TOKEN=label    : B0  라벨로만 가림 ([PERSON]) → TOKEN_PATTERN 에 안 맞아 토큰 필터가 꺼지고 의미 검색만 남음
  PSEUDO_DETECT=gold    : 상한선  탐지기 대신 정답 라벨 위치 사용 (holdout 코퍼스·질의 정답 파일). 비교용으로만 사용
"""
import hashlib
import hmac
import os
import re
import sys

from . import config

TOKEN_PATTERN = re.compile(r"<[A-Z_]+_[0-9A-Za-z]+>")

PREFIX = {"person": "PERSON", "phone_number": "PHONE", "email": "EMAIL", "address": "ADDRESS",
          "resident_registration_number": "RRN", "account_number": "ACCOUNT", "card_number": "CARD"}
TOKEN_LEN = 8
NORMALIZE = os.getenv("PSEUDO_NORMALIZE", "1") != "0"
TOKEN_MODE = os.getenv("PSEUDO_TOKEN", "hmac")     # hmac | label
DETECT_MODE = os.getenv("PSEUDO_DETECT", "model")  # model | gold

# ---------------------------------------------------------------- 탐지
_detector = None


def _get_detector():
    """모델 로딩이 몇 초 걸려서 처음 부를 때 한 번만 만듭니다."""
    global _detector
    if _detector is None:
        sys.path.insert(0, str(config.ROOT))  # 실전프로젝트2/pii_detector.py
        from pii_detector import PIIDetector
        _detector = PIIDetector()
    return _detector


_gold = None


def _gold_spans(text):
    """상한선 비교용: 원문이 같은 정답 문서·질의의 정답 개체 위치. 정답이 없는 텍스트는 오류."""
    global _gold
    if _gold is None:
        import json
        _gold = {}
        for name, text_key, ents_key in (("holdout_labeled.jsonl", "text", "entities"),
                                         ("holdout_retrieval_queries_gold.jsonl", "query", "query_entities")):
            with open(config.DATA_DIR / name, encoding="utf-8") as f:
                for d in (json.loads(l) for l in f if l.strip()):
                    _gold[d[text_key]] = [{"label": e["label"], "start": e["start"], "end": e["end"]}
                                          for e in d.get(ents_key) or []]
    if text not in _gold:
        raise KeyError("PSEUDO_DETECT=gold 는 holdout 문서·질의에만 쓸 수 있습니다.")
    return _gold[text]


def detect(text):
    if DETECT_MODE == "gold":
        return _gold_spans(text)
    return [{"label": e["label"], "start": e["start"], "end": e["end"]}
            for e in _get_detector().detect(text)]


# ---------------------------------------------------------------- 정규화
# 주소 맨 앞 시·도 약칭 → 정식 명칭 (대구 수성구 … 와 대구광역시 수성구 … 를 같은 값으로)
_SIDO = {"서울특별시": ["서울시", "서울"], "부산광역시": ["부산시", "부산"], "대구광역시": ["대구시", "대구"],
         "인천광역시": ["인천시", "인천"], "광주광역시": ["광주시", "광주"], "대전광역시": ["대전시", "대전"],
         "울산광역시": ["울산시", "울산"], "세종특별자치시": ["세종시", "세종"], "경기도": ["경기"],
         "강원특별자치도": ["강원도", "강원"], "충청북도": ["충북"], "충청남도": ["충남"],
         "전북특별자치도": ["전라북도", "전북"], "전라남도": ["전남"], "경상북도": ["경북"], "경상남도": ["경남"],
         "제주특별자치도": ["제주도", "제주"]}
SIDO_ALIAS = {a: full for full, aliases in _SIDO.items() for a in aliases}


def _normalize_address(v):
    v = re.sub(r"\s+", " ", v)
    v = re.sub(r"^\(?\d{5}\)?\s*", "", v)  # 앞에 붙은 우편번호 제거: (05299) 서울 … → 서울 …
    head, _, rest = v.partition(" ")
    head = SIDO_ALIAS.get(head, head)
    return f"{head} {rest}" if rest else head


def normalize(label, value):
    """같은 값의 표기 차이만 없앱니다. 조사·호칭 제거는 탐지기 경계 보정(postprocess.py)이 이미 맡습니다."""
    if not NORMALIZE:
        return value
    v = value.strip()
    if label == "person":
        return re.sub(r"\s+", "", v)                      # 윤 선영 → 윤선영
    if label == "phone_number":
        d = re.sub(r"\D", "", v)
        return "0" + d[2:] if v.startswith("+82") else d  # +82 10-… → 010…
    if label in ("resident_registration_number", "account_number", "card_number"):
        return re.sub(r"\D", "", v)
    if label == "email":
        return v.lower()
    if label == "address":
        return _normalize_address(v)                     # 대구 수성구 … → 대구광역시 수성구 …
    return v


# ---------------------------------------------------------------- 토큰
def make_token(label, normalized):
    if TOKEN_MODE == "label":
        return f"[{PREFIX.get(label, label.upper())}]"
    if not config.HMAC_KEY:
        raise RuntimeError("HMAC_KEY 가 비어 있습니다. .env 를 확인하세요.")
    h = hmac.new(config.HMAC_KEY.encode(), f"{label}:{normalized}".encode("utf-8"), hashlib.sha256).hexdigest()
    return f"<{PREFIX.get(label, label.upper())}_{h[:TOKEN_LEN]}>"


def pseudonymize(text, spans):
    records, out, last = [], [], 0
    for s in _drop_overlaps(spans):
        original = text[s["start"]:s["end"]]
        normalized = normalize(s["label"], original)
        token = make_token(s["label"], normalized)
        out += [text[last:s["start"]], token]
        last = s["end"]
        records.append({"label": s["label"], "start": s["start"], "end": s["end"],
                        "original": original, "normalized": normalized, "token": token})
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
