"""데이터셋에 없는 문장 구조로 쓴 예문에서 탐지기별 이름·번호 검출 확인 (holdout 대용 소규모 점검).

사용법: python probe_unseen.py rules,veil,seongyeon,llm,hybrid_seongyeon
"""
import importlib
import sys
from detectors.base import postprocess

CASES = [
    ("어제 배다온 손님이 지점에 들러 체크카드 재발급을 요청하셨어요.", ["배다온"]),
    ("송하람한테 들은 얘기인데, 그 건은 다음 주로 미뤄졌대.", ["송하람"]),
    ("작성: 리스크관리팀 / 검토: 옥 지후", ["옥 지후"]),
    ("고객 연락처는 공일공-일이삼사-오육칠팔 이 아니라 010 1234 5678 입니다.", ["010 1234 5678"]),
    ("문의 사항은 hr-help@saebom.co.kr 또는 담당 예서윤에게 연락 바랍니다.", ["hr-help@saebom.co.kr", "예서윤"]),
    ("이번 건은 도경수 차장과 나눈 통화 내용을 정리한 것입니다.", ["도경수"]),
    ("지난번 미팅에서 하윤이가 말한 대로 진행하자.", ["하윤"]),
    ("받는 사람: 제갈민호 (서울특별시 마포구 월드컵북로 21, 5층)", ["제갈민호", "서울특별시 마포구 월드컵북로 21, 5층"]),
    ("김 과장 말로는 한결 플러스 카드 한도가 상향됐다고 합니다.", []),
    ("환불 계좌 신한 110-482-339201, 예금주 표은비", ["110-482-339201", "표은비"]),
]


def main():
    names = sys.argv[1].split(",")
    for n in names:
        m = importlib.import_module(f"detectors.{n}")
        if hasattr(m, "load"):
            m.load()
        hit = tot = fp = 0
        lines = []
        for text, gold in CASES:
            found = [text[e["start"]:e["end"]] for e in postprocess(text, m.detect(text))]
            miss = [g for g in gold if not any(g in f or f in g for f in found)]
            extra = [f for f in found if not any(g in f or f in g for g in gold)]
            tot += len(gold); hit += len(gold) - len(miss); fp += len(extra)
            if miss or extra:
                lines.append(f"    미탐 {miss} 오탐 {extra} | {text}")
        print(f"[{n}] 재현 {hit}/{tot}, 오탐 {fp}")
        print("\n".join(lines))


main()
