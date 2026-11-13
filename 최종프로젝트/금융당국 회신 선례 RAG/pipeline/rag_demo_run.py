"""LLM 포함 RAG를 질문 8개로 시험 실행해 답변·검색 결과를 출력한다(읽기 전용, worklog 61번).
(원래 임시 폴더에서 만든 스크립트를 보관용으로 옮긴 것이다. 경로가 이 PC 기준 절대 경로로 들어 있다.)"""
import json
import sys
import time

sys.path.insert(0, r"C:\Users\USER\Desktop\likelion-AI-4team\최종프로젝트\금융당국 회신 선례 RAG\pipeline")
sys.stdout.reconfigure(encoding="utf-8")
import rag
import time_marks as tm
from common import DATA

EVAL = DATA / "eval"
load = lambda n: [json.loads(l) for l in open(EVAL / f"{n}.jsonl", encoding="utf-8")]
dev3, dev2, una = load("dev3"), load("dev2"), load("unanswerable_20")
pick = lambda rows, tag: next(r for r in rows if r["tag"] == tag)
qs = [("dev3 우회 서술형", pick(dev3, "우회 서술형")), ("dev3 구어체형", pick(dev3, "구어체형")), ("dev3 경계 조건형", pick(dev3, "경계 조건형")), ("dev2 규제 효력형", pick(dev2, "규제 효력형")),
      ("직접 쓴 질문 1", {"question": "적금 가입 고객에게 제휴 카드사 캐시백을 주는 이벤트를 하려고 하는데, 특별이익 제공으로 문제된 회신이 있나요?"}),
      ("직접 쓴 질문 2", {"question": "비대면으로 법인 고객 계좌를 열어줄 때 실명확인을 어떻게 해야 하는지 당국 의견이 있나요?"}),
      ("답변불가", una[0]), ("시점 확인용", {"question": "저축은행 대주주에 대한 신용공여 한도 관련 회신 알려주세요"})]
conn_dummy = rag.connect()
rep = {r["representative_case_key"]: r["group_id"] for r in conn_dummy.execute("select group_id, representative_case_key from interp.case_groups").fetchall()}
grp_rep = {v: k for k, v in rep.items()}
rag.embed("준비")
tot_in = tot_out = 0
for label, r in qs:
    t = time.time()
    res = rag.answer(r["question"])
    sec = time.time() - t
    ids = [h["doc_id"] for h in res["hits"]]
    gold = r.get("gold_group_id")
    gk = grp_rep.get(gold) if gold else None
    tot_in += res["usage"]["input"]
    tot_out += res["usage"]["output"]
    print("=" * 100)
    print(f"[{label}] ({sec:.1f}초)  Q: {res['question']}")
    if gold:
        print(f"  (정답 묶음 대표 {gk}: {'상위 5에 있음 ' + str(ids.index(gk) + 1) + '위' if gk in ids else '상위 5에 없음'})")
    print("  검색 상위 5:", " | ".join(f"{h['doc_id']}({h['chunk_type']},{h['score']:.2f})" for h in res["hits"]))
    print("  --- 답변 ---")
    print("  " + res["answer"].replace("\n", "\n  "))
    print("  --- 검사 ---", res["check"])
    for k, c in list(res["cases"].items())[:3]:
        print(f"  [{k}] {c['title'][:45]} ({c['reply_date']}) 시점 {c['time_mark']}: {c['time_label'][:70]}")
print("=" * 100)
print(f"토큰 합계 입력 {tot_in} 출력 {tot_out}")
