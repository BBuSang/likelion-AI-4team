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


class PIIDetector:
    def __init__(self, token_model: str = "atonlee"):
        from detectors.rule_based import RuleBasedDetector
        from detectors.token_cls import HFNerDetector
        self.rules = RuleBasedDetector()
        self.token = HFNerDetector(token_model)

    def detect(self, text: str) -> list:
        rules = dedupe(text, self.rules.detect(text))
        token = dedupe(text, self.token.detect(text))
        return dedupe(text, union_entities([rules, token]))  # 겹치면 규칙 우선, 주소는 긴 쪽


if __name__ == "__main__":
    import sys
    det = PIIDetector()
    text = " ".join(sys.argv[1:]) or "윤 선영 고객(01003492063)이 추가 한도를 문의함. 윤선영은 sy.yoon07@example.net로 서류를 받음."
    for e in det.detect(text):
        print(f"{e['label']:<30} {e['text']}")
