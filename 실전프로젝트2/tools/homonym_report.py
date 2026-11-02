"""동명이인 질의에서 RAG 가 어떻게 처리했는지 두 실행 결과를 나란히 비교한다 (정답은 채점·분석에만).

    python tools/homonym_report.py outputs/rag_e2e_hcx007.jsonl outputs/rag_e2e_v3b_homonym.jsonl

질의마다: 정답 문서 / 혼동 문서(동명이인), LLM 에 넘긴 문서, 답변이 인용한 문서, 답변에 나온 사람 토큰 수
  - 혼동 인용 : 답변이 동명이인(다른 사람)의 문서를 인용 → 두 사람을 섞어 답했을 위험
  - 사람 토큰 : 답변에 나온 서로 다른 <PERSON_…> 수 (동명이인 구분을 켜면 두 사람이 다른 토큰으로 보임)
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QIDS = ["Q052", "Q053", "Q083", "Q084", "Q085", "Q086", "Q087", "Q088", "Q071", "Q072", "Q082"]
_CITE = re.compile(r"[\[<](H\d{4})[\]>]")
_PERSON = re.compile(r"<PERSON_[0-9a-f]{8}>")


def load(path):
    return {r["qid"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}


def main():
    gold = load(ROOT / "data/holdout_retrieval_queries_gold.jsonl")
    runs = [(Path(p).stem, load(p)) for p in sys.argv[1:]]
    totals = {name: {"근거": 0, "인용": 0, "혼동 인용": 0, "n": 0} for name, _ in runs}
    for qid in QIDS:
        q = gold[qid]
        rel, conf = set(q["relevant_doc_ids"]), set(q.get("confusable_doc_ids", []))
        print(f"\n## {qid} [{q['type']}] {q['query']}")
        print(f"   정답 {sorted(rel)}  |  혼동(동명이인) {sorted(conf)[:6]}")
        for name, rows in runs:
            r = rows.get(qid)
            if not r or not r.get("answer"):
                print(f"   - {name}: (결과 없음)")
                continue
            docs, cited = r["doc_ids"], set(_CITE.findall(r["answer"]))
            mark = lambda d: f"{d}✓" if d in rel else f"{d}✗" if d in conf else d
            t = totals[name]
            t["n"] += 1
            t["근거"] += bool(rel & set(docs))
            t["인용"] += bool(rel & cited)
            t["혼동 인용"] += bool(conf & cited)
            print(f"   - {name}: 넘긴 문서 {[mark(d) for d in docs]} → 인용 {[mark(d) for d in sorted(cited)]}, "
                  f"사람 토큰 {len(set(_PERSON.findall(r['answer'])))}종")
            print("       " + r["answer"].replace("\n", " ")[:220])
    print("\n## 합계 (✓ 정답 문서, ✗ 동명이인 문서)")
    for name, t in totals.items():
        print(f"   {name}: 질의 {t['n']} · 근거에 정답 {t['근거']} · 정답 인용 {t['인용']} · 동명이인 문서 인용 {t['혼동 인용']}")


if __name__ == "__main__":
    main()
