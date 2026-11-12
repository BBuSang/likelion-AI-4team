"""실전 1에서 고른 운영 탐지기(규칙 + atonlee 합집합)를 문서 1건 단위로 부르는 입구. 실전 2 가명처리·질의 처리에서 사용.

사용 예
  from pii_detector import PIIDetector
  det = PIIDetector()                       # 모델 로딩 몇 초 (한 번만)
  det.detect("윤 선영 고객(01003492063)이 …")
  → [{"label": "person", "start": 0, "end": 4, "text": "윤 선영"}, {"label": "phone_number", ...}, ...]

결과는 run_detect.py + make_union.py 로 만든 preds/combined/rules+atonlee_*.jsonl 과 같다 (같은 탐지기·같은 합치기).
라벨 7종: person, phone_number, email, address, resident_registration_number, account_number, card_number
"""
from common import dedupe
from make_union import union_entities
from postprocess import drop_field_like_names


class PIIDetector:
    def __init__(self, token_model: str = "atonlee"):
        from detectors.rule_based import RuleBasedDetector
        from detectors.token_cls import HFNerDetector
        import os
        import sys
        self.rules = RuleBasedDetector()
        self.token = HFNerDetector(token_model)
        # 최종 프로젝트 31장: 추가 학습한 모델을 '이름 보태기'로만 씀 — 원래 모델·규칙이 안 잡은 자리의 이름만 더함.
        # 단독으로 바꾸면 holdout 형식에서 오탐·미탐이 늘고, 보태기는 holdout 은 그대로 두고 어려운 문서 미탐만 줄임
        extra = (os.getenv("PII_NAME_MODEL") or "").strip()  # 빈 값이면 끔
        if extra and not os.path.isabs(extra):  # 'outputs/models/…' 는 최종프로젝트/AI 기준
            extra = os.path.join(os.path.dirname(__file__), "..", "..", "..", extra)
        if extra and not os.path.isdir(extra):
            print(f"[pii] PII_NAME_MODEL 폴더 없음 → 이름 보태기 없이 탐지: {extra}", file=sys.stderr)
            extra = None
        self.extra = HFNerDetector(token_model, model_path=extra) if extra else None

    def detect(self, text: str) -> list:
        return self.detect_detailed(text)["spans"]

    def detect_detailed(self, text: str) -> dict:
        """detect() 와 같은 결과 + 규칙·모델 각각의 결과와 모델의 토큰별 확률 (최종 프로젝트 검수 대상 고르기, worklog 29장)"""
        rules = dedupe(text, self.rules.detect(text))
        token = dedupe(text, self.token.detect(text))
        probs = getattr(self.token.model, "last_probs", [])  # 보태기 모델을 돌리기 전에 원래 모델 확률을 챙김
        extra = [e for e in dedupe(text, self.extra.detect(text)) if e["label"] == "person"] if self.extra else []
        merged = dedupe(text, union_entities([rules, token, extra]))  # 겹치면 앞쪽(규칙 → 모델 → 보태기) 우선, 주소는 긴 쪽
        # v11: 모델의 항목 이름·호칭 오탐. 일반 단어 거르기(is_common_word)는 이름 재현율을 떨어뜨려 쓰지 않음 (worklog 35장)
        return {"spans": drop_field_like_names(text, merged), "rules": rules, "model": token, "extra": extra,
                "probs": probs}


if __name__ == "__main__":
    import sys
    det = PIIDetector()
    text = " ".join(sys.argv[1:]) or "윤 선영 고객(01003492063)이 추가 한도를 문의함. 윤선영은 sy.yoon07@example.net로 서류를 받음."
    for e in det.detect(text):
        print(f"{e['label']:<30} {e['text']}")
