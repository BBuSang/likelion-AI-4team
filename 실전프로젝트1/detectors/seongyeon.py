"""모델 단독 탐지기: seongyeon1/ko-pii-ner-roberta-base (klue/roberta-base, CC-BY-SA-4.0)."""
from detectors.ner import TokenClassifier

_model = TokenClassifier("seongyeon")
load = _model.load
detect = _model.predict
