"""하이브리드 탐지기: 정규식 + 규칙 이름 ∪ veil 이름 ∪ seongyeon 이름 (재현율 우선)."""
from detectors import rules
from detectors.hybrid import combine, model_names
from detectors.seongyeon import _model as seong
from detectors.veil import _model as veil


def load():
    rules.load()
    veil.load()
    seong.load()


def detect(text):
    return combine(text, model_names(text, seong.predict(text)) + model_names(text, veil.predict(text)))
