"""하이브리드 탐지기: 정규식 + 규칙 이름 ∪ LLM 이름."""
from detectors import llm, rules
from detectors.hybrid import combine


def load():
    rules.load()
    llm.load()


def detect(text):
    return combine(text, llm.detect(text))
