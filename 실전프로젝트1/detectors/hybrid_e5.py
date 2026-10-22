"""하이브리드 탐지기: 정규식 + 규칙 이름 ∪ e5 모델 이름."""
from detectors import rules
from detectors.hybrid import combine
from detectors.e5 import _model


def load():
    rules.load()
    _model.load()


def detect(text):
    return combine(text, _model.predict(text))
