"""데이터셋에 없는 문장 구조로 쓴 예문에서 탐지기별 이름·번호 검출 확인 (holdout 대용 소규모 점검).

사용법: python probe_unseen.py rules,veil,seongyeon,llm,hybrid_seongyeon [세트 번호: 1(기본) 또는 2]
    세트 1: 처음 만든 예문. 조사·호칭 규칙 수정(손님 등)에 참고했으므로 더 이상 '처음 보는 문장'이 아님
    세트 2: 규칙 수정 뒤, 결과를 보기 전에 새로 쓴 예문
재현은 구간이 겹치면 인정하고, '정확'은 범위까지 정답과 같은 것만 셈
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

CASES2 = [
    ("오늘 오전에 방문한 류하진 가입자분이 한도 상향을 문의했습니다.", ["류하진"]),
    ("채린이가 보낸 파일 확인해 줘.", ["채린"]),
    ("지수야, 내일 보고 자료 준비됐어?", ["지수"]),
    ("신청서는 국하늘 님이 직접 제출하셨습니다.", ["국하늘"]),
    ("보고: 감사팀 주예림 / 승인: 부서장", ["주예림"]),
    ("연락처 010-2233-4455로 회신 바랍니다(담당 강 대리).", ["010-2233-4455"]),
    ("변경 전 주소 대구광역시 수성구 달구벌대로 2450, 1203호", ["대구광역시 수성구 달구벌대로 2450, 1203호"]),
    ("카드 5310 8812 0034 7719 분실 신고 접수", ["5310 8812 0034 7719"]),
    ("문의 주신 하나은행 계좌 352-0912-4471-03 확인했습니다.", ["352-0912-4471-03"]),
    ("어머니 성함은 황보진숙이시고 보호자로 등록돼 있어요.", ["황보진숙"]),
    ("팀원 명단: 금다인, 육서진, 탁민호", ["금다인", "육서진", "탁민호"]),
    ("다음 주에 서아랑 같이 지점 방문 예정", ["서아"]),
    ("이번 분기 고객 만족도 조사 결과를 공유합니다. 대표번호 1588-1234로 문의 바랍니다.", []),
]
SETS = {"1": CASES, "2": CASES2}


def main():
    names = sys.argv[1].split(",")
    cases = SETS[sys.argv[2] if len(sys.argv) > 2 else "1"]
    for n in names:
        m = importlib.import_module(f"detectors.{n}")
        if hasattr(m, "load"):
            m.load()
        hit = exact = tot = fp = 0
        lines = []
        for text, gold in cases:
            found = [text[e["start"]:e["end"]] for e in postprocess(text, m.detect(text))]
            miss = [g for g in gold if not any(g in f or f in g for f in found)]
            extra = [f for f in found if not any(g in f or f in g for g in gold)]
            loose = [f for f in found if f not in gold and any(g in f or f in g for g in gold)]
            tot += len(gold); hit += len(gold) - len(miss); fp += len(extra)
            exact += sum(g in found for g in gold)
            if miss or extra or loose:
                lines.append(f"    미탐 {miss} 오탐 {extra} 범위 다름 {loose} | {text}")
        print(f"[{n}] 재현 {hit}/{tot} (정확 {exact}), 오탐 {fp}")
        print("\n".join(lines))


main()
