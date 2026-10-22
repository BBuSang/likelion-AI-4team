"""탐지기 공통 형식과 후처리.

모든 탐지기 모듈(detectors/<이름>.py)은 아래 함수를 제공합니다.
    detect(text) -> [{"label": "person", "start": 53, "end": 57}, ...]
    - start/end: 원문 기준 글자 위치, [start, end) 반열린 구간 (run_compare.py와 같음)
    - 입력은 원문 text만 사용. 정답 필드(entities, persons, roles, hard_cases)는 쓰지 않음
    - 원문을 고쳐서(공백·하이픈 제거 등) 찾았다면 위치를 원문 기준으로 되돌려 반환
    - (선택) load(): 모델 로딩처럼 한 번만 할 일. 문서별 처리 시간에서 빠짐

팀 라벨링 기준 (데이터셋 정답 기준을 따름)
    잡는다   : 전체 이름, 이름만(승현), 띄어 쓴 이름(공백 포함 '윤 선영'), 고객·직원 전화(지역번호 포함),
              이메일, 주소, 주민번호, 계좌번호, 카드번호
    범위 제외: 조사·호칭 ('윤선영은' → '윤선영', '김은영 상담원' → '김은영')
    잡지 않음: 성+직함('허 부장'), 회사 대표번호(1599-0000), 내선번호, 날짜, 민원번호,
              회사·부서·상품명 (organization 라벨 없음)
"""
import re

from run_compare import LABELS

# 성+직함 ('김 대리', '허 부장님이' → '허 부장'). 팀 기준을 바꿔 잡을 때만 사용 (run_detect.py --surname-title)
JOB_TITLES = ["과장", "대리", "팀장", "부장", "차장", "사원", "주임", "실장", "본부장", "이사", "상무", "전무",
              "대표", "선임", "책임", "수석", "상담원", "매니저", "연구원", "위원", "센터장", "지점장"]
_ST = re.compile(r"(?<![가-힣])(?:남궁|제갈|선우|독고|황보|사공|서문|동방|[가-힣])\s?(?:"
                 + "|".join(sorted(JOB_TITLES, key=len, reverse=True)) + r")([가-힣]*)")


def surname_title(text):
    """성+직함 구간. 뒤에 붙은 말이 조사·님·씨일 때만 ('대표번호', '이사회'는 제외)."""
    from detectors.rules import SUFFIXES, SURNAMES, SURNAMES2
    out = []
    for m in _ST.finditer(text):
        span = m.group()[:len(m.group()) - len(m.group(1))]
        sur = span.split()[0] if " " in span else span[:2] if span[:2] in SURNAMES2 else span[0]
        if (sur in SURNAMES or sur in SURNAMES2) and m.group(1) in SUFFIXES:
            out.append({"label": "person", "start": m.start(), "end": m.start() + len(span)})
    return out


def postprocess(text, ents):
    """탐지 결과 정리: 7종 밖 라벨·범위 오류 제거, 앞뒤 공백 제거, 중복·겹침 정리.
    겹치면 같은 라벨은 합치고, 다른 라벨은 먼저 시작하는(같으면 더 긴) 구간을 남김."""
    clean = []
    for e in ents:
        s, t = int(e["start"]), int(e["end"])
        if e.get("label") not in LABELS or not (0 <= s < t <= len(text)):
            continue
        while s < t and text[s].isspace():
            s += 1
        while t > s and text[t - 1].isspace():
            t -= 1
        if s < t:
            clean.append({**e, "start": s, "end": t})

    out = []
    for e in sorted(clean, key=lambda e: (e["start"], -(e["end"] - e["start"]))):
        if out and e["start"] < out[-1]["end"]:
            if e["label"] == out[-1]["label"]:
                out[-1]["end"] = max(out[-1]["end"], e["end"])
            continue
        out.append(e)
    return out
