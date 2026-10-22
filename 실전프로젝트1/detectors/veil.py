"""모델 단독 탐지기: 1T/veil-pii-ko-lite (KoELECTRA-base, Apache-2.0)."""
from detectors.ner import TokenClassifier

_model = TokenClassifier("veil")
load = _model.load
detect = _model.predict
