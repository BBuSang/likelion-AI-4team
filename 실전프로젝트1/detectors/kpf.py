"""모델 단독 탐지기: townboy/kpfbert-kdpii (KPF-bert-ner, MIT)."""
from detectors.ner import TokenClassifier

_model = TokenClassifier("kpf")
load = _model.load
detect = _model.predict
