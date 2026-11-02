"""유출 검사만 (LLM 답변 없이). API 로 나가는 글 — 적재된 문서(임베딩·LLM 문맥), 마스킹된 질의, LLM 에 넘길 문맥 —
에 정답 라벨의 원문 값이 그대로 있는지 센다. 0 이어야 함. (worklog 15장)

    RERANK=1 python tools/leak_check.py --gold data/synth2/synth2_retrieval_queries_gold.jsonl \
        --corpus data/synth2/synth2_labeled.jsonl --out outputs/synth2/leak_check_v10.jsonl

정답 라벨은 채점에만 쓴다. 검색·문맥은 chat.answer 와 같은 순서(search → select_docs → _context)로 만든다.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
import eval_rag_e2e  # noqa: E402  pii_values() 재사용 (유출 기준을 e2e 채점과 같게)
from starter import chat, db  # noqa: E402
from starter.search import search  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("-k", type=int, default=3)
    a = ap.parse_args()
    eval_rag_e2e.CORPUS = Path(a.corpus)
    vals = eval_rag_e2e.pii_values()

    leaks, cases = collections.Counter(), []
    with db.rag_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id, masked_text FROM rag.chunks")
        for doc_id, text in cur.fetchall():
            for lab, v in vals:
                if v in text:
                    leaks["적재 문서"] += 1
                    cases.append(("적재 문서", doc_id, lab, v))

    queries = eval_rag_e2e.load(a.gold)
    with open(a.out, "w", encoding="utf-8") as f:
        for i, q in enumerate(queries, 1):
            masked_q, q_tokens, cands = search(q["query"], a.k * 2, embed_retries=3)
            docs = chat.select_docs(cands, a.k)
            llm_input = chat._context(docs) + "\n\n" + masked_q
            f.write(json.dumps({"qid": q["qid"], "masked_query": masked_q, "doc_ids": [d["doc_id"] for d in docs],
                                "llm_input": llm_input}, ensure_ascii=False) + "\n")
            for where, text in (("질의", masked_q), ("LLM 입력", llm_input)):
                for lab, v in vals:
                    if v in text:
                        leaks[where] += 1
                        cases.append((where, q["qid"], lab, v))
            if i % 50 == 0:
                print(f"[{i}/{len(queries)}]", flush=True)

    print(f"질의 {len(queries)}개, 정답 원문 값 {len(vals)}종")
    print("유출 (0 이어야 함):", {w: leaks.get(w, 0) for w in ("적재 문서", "질의", "LLM 입력")})
    for c in cases[:20]:
        print("  ", c)


if __name__ == "__main__":
    main()
