"""v2 검색 평가: 기능을 하나씩 켜며 holdout 질의 88개로 비교 (plan_v2 P4). 정답은 채점에만.

    python -X utf8 v2/eval/eval_search.py --tag rule      (DB의 현재 entities 상태를 씀)

질의 임베딩은 outputs/emb_cache.json 캐시 사용 (가명 질의만 전송).
결과: outputs/runs/run_<tag>_<변형>.jsonl, outputs/search_<tag>.json
"""
import argparse
import collections
import json
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
sys.path.insert(0, str(V2))
sys.path.insert(0, str(ROOT / "eval"))
import config  # noqa: E402
from eval_holdout_retrieval import score  # noqa: E402
from ingest import embed_cached  # noqa: E402
from search import Searcher  # noqa: E402

OUT = V2 / "outputs"
VARIANTS = [
    ("S0 토큰+의미", {}),
    ("S1 +문서종류", {"intent": 1}),
    ("S2 +지역", {"intent": 1, "region": 1}),
    ("S3 +상태", {"intent": 1, "region": 1, "status": 1}),
    ("S4 +역할필터", {"intent": 1, "region": 1, "status": 1, "role": 1}),
    ("S5 +개체", {"intent": 1, "region": 1, "status": 1, "role": 1, "entity": 1}),
    ("S6 +같은줄", {"intent": 1, "region": 1, "status": 1, "role": 1, "entity": 1, "line": 1}),
    ("S7 +1위개체", {"intent": 1, "region": 1, "status": 1, "role": 1, "entity": 1, "line": 1, "anchor": 1}),
]


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def confuse_split(gold, runs, doc_persons):
    same, hom = [], []
    for g in gold:
        conf = set(g.get("confusable_doc_ids", []))
        if not conf:
            continue
        top = runs[g["qid"]][:10]
        pid = {e["person_id"] for e in g.get("query_entities", []) if e.get("person_id")}
        s = sum(1 for d in top if d in conf and pid & doc_persons.get(d, set()))
        same.append(s / len(top))
        hom.append((sum(1 for d in top if d in conf) - s) / len(top))
    return sum(same) / len(same), sum(hom) / len(hom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rule")
    a = ap.parse_args()
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    queries = jl(OUT / "pseudo_queries.jsonl")
    gold = jl(config.DATA_DIR / "holdout_retrieval_queries_gold.jsonl")
    doc_persons = collections.defaultdict(set)
    for d in jl(config.DATA_DIR / "holdout_persons_gold.jsonl"):
        doc_persons[d["id"]] |= {p["person_id"] for p in d["persons"]}
    qv = embed_cached([q["masked_query"] for q in queries])

    base = Searcher({})
    sems = {}
    for q in queries:
        sems[q["qid"]] = {d: float(s) for d, s in base.conn.execute(
            "SELECT doc_id, 1 - (embedding <=> %s) FROM rag.chunks WHERE kind = 'doc'", (qv[q["masked_query"]],))}

    result, types = {}, sorted({g["type"] for g in gold})
    for name, opts in VARIANTS:
        s = Searcher(opts, conn=base.conn)
        runs = {q["qid"]: s.search_masked(q, k=20, sem=sems[q["qid"]])[0] for q in queries}
        tag = name.split()[0]
        with open(OUT / "runs" / f"run_{a.tag}_{tag}.jsonl", "w", encoding="utf-8") as f:
            for qid, r in runs.items():
                f.write(json.dumps({"qid": qid, "ranked_doc_ids": r}, ensure_ascii=False) + "\n")
        m = score(gold, [{"qid": k, "ranked_doc_ids": v} for k, v in runs.items()], ks=(1, 3, 5, 10))
        avg = lambda g, c: sum(m[g][c]) / len(m[g][c]) if m[g][c] else None  # noqa: E731
        cs, ch = confuse_split(gold, runs, doc_persons)
        result[name] = {"ALL": {c: avg("ALL", c) for c in ("Hit@1", "Hit@3", "Hit@5", "Recall@10", "MRR", "Confuse@10")},
                        "confuse_same": cs, "confuse_homonym": ch,
                        "types": {t: avg(t, "Hit@1") for t in types}}
    (OUT / f"search_{a.tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'변형':<14}{'Hit@1':>7}{'Hit@3':>7}{'Hit@5':>7}{'R@10':>7}{'MRR':>7}{'Conf@10':>8}{'같은사람':>8}{'동명이인':>8}")
    for name, _ in VARIANTS:
        x = result[name]["ALL"]
        print(f"{name:<14}{x['Hit@1']:>7.3f}{x['Hit@3']:>7.3f}{x['Hit@5']:>7.3f}{x['Recall@10']:>7.3f}{x['MRR']:>7.3f}"
              f"{x['Confuse@10']:>8.3f}{result[name]['confuse_same']:>8.3f}{result[name]['confuse_homonym']:>8.3f}")
    print("\n유형별 Hit@1 (S0 → S5)")
    for t in types:
        print(f"  {t:<26}{result[VARIANTS[0][0]]['types'][t]:.3f} → {result[VARIANTS[-1][0]]['types'][t]:.3f}")


if __name__ == "__main__":
    main()
