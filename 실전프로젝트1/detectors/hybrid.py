"""하이브리드 탐지 공통 로직: 정형 라벨은 정규식, 이름은 모델 + 규칙 합집합.

- 정형 라벨(전화·이메일·주소·주민번호·계좌·카드)과 규칙 기반 이름은 detectors/rules.py 결과를 그대로 씀
- 모델이 찾은 이름 중 규칙 결과와 겹치지 않는 것만 더함 (겹치면 규칙 쪽 경계를 유지)
- 모델이 찾은 한 글자 이름은 버림: '정 부장'의 '정'처럼 성+직함의 성씨만 잡은 경우 (팀 기준: 개인정보 아님)
- 모델이 찾은 정형 라벨은 쓰지 않음
"""
from detectors import rules


def model_names(text, ents):
    """모델 결과 중 두 글자 이상 이름만."""
    return [e for e in ents if e["label"] == "person" and len(text[e["start"]:e["end"]].strip()) > 1]


def agree(a, b):
    """두 모델이 모두 이름으로 본 구간 (겹치는 것끼리, 경계는 짧은 쪽 = 조사가 덜 붙은 쪽)."""
    out = []
    for x in a:
        for y in b:
            if x["start"] < y["end"] and y["start"] < x["end"]:
                out.append(min(x, y, key=lambda e: e["end"] - e["start"]))
                break
    return out


def combine(text, model_ents):
    ents = rules.detect(text)
    for m in model_ents:
        if (m["label"] == "person" and len(text[m["start"]:m["end"]].strip()) > 1
                and not any(m["start"] < e["end"] and e["start"] < m["end"] for e in ents)):
            ents.append(m)
    return ents
