"""모델 단독 탐지기: FrameByFrame/korean-pii-e5-base (multilingual-e5-base, MIT)."""
from detectors.ner import TokenClassifier

_model = TokenClassifier("e5")
load = _model.load
detect = _model.predict
