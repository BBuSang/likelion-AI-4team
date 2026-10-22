"""모델 단독 탐지기: YakuzaNeko/kr-dlp-ner-roberta-large (klue/roberta-large, Apache-2.0)."""
from detectors.ner import TokenClassifier

_model = TokenClassifier("dlp")
load = _model.load
detect = _model.predict
