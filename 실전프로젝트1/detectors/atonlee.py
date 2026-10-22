"""모델 단독 탐지기: atonlee/koelectra-ko-pii-ner (KoELECTRA-small, Apache-2.0)."""
from detectors.ner import TokenClassifier

_model = TokenClassifier("atonlee")
load = _model.load
detect = _model.predict
