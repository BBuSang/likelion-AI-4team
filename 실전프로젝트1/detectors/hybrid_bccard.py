"""하이브리드 탐지기: 정규식 + 규칙 이름 ∪ bccard 모델 이름."""
from detectors import bccard, rules
from detectors.hybrid import combine


def load():
    rules.load()
    bccard.load()


def detect(text):
    return combine(text, bccard.detect(text))
