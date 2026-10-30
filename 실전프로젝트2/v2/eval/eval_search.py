"""v2 검색 평가. 정답은 채점에만.

    python -X utf8 v2/eval/eval_search.py --tag rule                       holdout 88개, 기능 누적(ladder)
    python -X utf8 v2/eval/eval_search.py --tag rules --set rules          holdout, 위험 규칙 버전 비교
    python -X utf8 v2/eval/eval_search.py --corpus devtest --split tune --set rules --tag tune
                                                                           새 질의 세트 조정용 절반 (DB에 dev/test 적재 상태)

DB의 현재 적재 상태(문서·entities)를 그대로 씀. 질의 임베딩은 outputs/emb_cache.json 캐시 (가명 질의만 전송).
결과: outputs/runs/run_<tag>_<변형>.jsonl, outputs/search_<tag>.json
"""
import argparse
import ast
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
BASE = {"intent": 1, "region": 1, "status": 1, "role": 1, "entity": 1}
SETS = {
    "ladder": [
        ("S0 토큰+의미", {}),
        ("S1 +문서종류", {"intent": 1}),
        ("S2 +지역", {"intent": 1, "region": 1}),
        ("S3 +상태", {"intent": 1, "region": 1, "status": 1}),
        ("S4 +역할필터", {"intent": 1, "region": 1, "status": 1, "role": 1}),
        ("S5 +개체", BASE),
        ("S6 +같은줄", {**BASE, "line": 1}),
        ("S7 +1위개체", {**BASE, "line": 1, "anchor": 1}),
    ],
    # 과적합 점검: holdout 질의를 보고 만든 부분을 일반 버전으로 바꿈 (worklog 43번)
    # 최종 모델 평가: 기준선 vs 최종(search.ALL_OPTS = G6)
    "final": [
        ("S0 토큰+의미", {}),
        ("최종 모델(G6)", None),
    ],
    "rules": [
        ("S0 토큰+의미", {}),
        ("S5 같은줄 없음", BASE),
        ("S6 현재(질의어휘·어간·부서제외)", {**BASE, "line": 1}),
        ("G1 상태 일반어휘", {**BASE, "line": 1, "status_vocab": "general"}),
        ("G2 같은줄 2-gram", {**BASE, "line": 1, "line_mode": "bigram"}),
        ("G3 부서제외 끔", {**BASE, "line": 1, "line_org": 0}),
        ("G4 일반어휘+2-gram", {**BASE, "line": 1, "status_vocab": "general", "line_mode": "bigram"}),
        ("G5 모두 일반", {**BASE, "line": 1, "status_vocab": "general", "line_mode": "bigram", "line_org": 0}),
        ("G6 일반어휘+같은줄없음", {**BASE, "status_vocab": "general"}),
    ],
}


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def P(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def doc_persons_of(corpus):
    dp = collections.defaultdict(set)
    if corpus == "holdout":
        for d in jl(config.DATA_DIR / "holdout_persons_gold.jsonl"):
            dp[d["id"]] |= {p["person_id"] for p in d["persons"]}
    else:
        for split in ("dev", "test"):
            for d in jl(config.DATA_DIR / f"pii_synth_finance_{split}.jsonl"):
                dp[d["id"]] |= {e["person_id"] for e in P(d["entities"]) if e["label"] == "person" and e.get("person_id")}
    return dp


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
    return (sum(same) / len(same), sum(hom) / len(hom)) if same else (0.0, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rule")
    ap.add_argument("--set", choices=SETS, default="ladder")
    ap.add_argument("--corpus", choices=("holdout", "devtest"), default="holdout")
    ap.add_argument("--split", choices=("all", "tune", "eval"), default="all")
    ap.add_argument("--qset", default="devtest", help="devtest 코퍼스의 질의 세트 이름")
    a = ap.parse_args()
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    if a.corpus == "holdout":
        queries = jl(OUT / "pseudo_queries.jsonl")
        gold = jl(config.DATA_DIR / "holdout_retrieval_queries_gold.jsonl")
    else:
        queries = jl(OUT / f"pseudo_queries_{a.qset}.jsonl")
        gold = jl(V2 / "eval_sets" / f"{a.qset}_queries_gold.jsonl")
        if a.split != "all":
            keep = set(json.loads((V2 / "eval_sets" / f"{a.qset}_split.json").read_text(encoding="utf-8"))[a.split])
            queries = [q for q in queries if q["qid"] in keep]
            gold = [g for g in gold if g["qid"] in keep]
    doc_persons = doc_persons_of(a.corpus)
    qv = embed_cached([q["masked_query"] for q in queries])

    base = Searcher({})
    sems = {q["qid"]: {d: float(s) for d, s in base.conn.execute(
        "SELECT doc_id, 1 - (embedding <=> %s) FROM rag.chunks WHERE kind = 'doc'", (qv[q["masked_query"]],))}
        for q in queries}

    variants = SETS[a.set]
    result, types = {}, sorted({g["type"] for g in gold})
    for name, opts in variants:
        s = Searcher(opts, conn=base.conn)
        runs = {q["qid"]: s.search_masked(q, k=20, sem=sems[q["qid"]])[0] for q in queries}
        with open(OUT / "runs" / f"run_{a.tag}_{name.split()[0]}.jsonl", "w", encoding="utf-8") as f:
            for qid, r in runs.items():
                f.write(json.dumps({"qid": qid, "ranked_doc_ids": r}, ensure_ascii=False) + "\n")
        m = score(gold, [{"qid": k, "ranked_doc_ids": v} for k, v in runs.items()], ks=(1, 3, 5, 10))
        avg = lambda g, c: sum(m[g][c]) / len(m[g][c]) if m[g][c] else None  # noqa: E731
        cs, ch = confuse_split(gold, runs, doc_persons)
        result[name] = {"ALL": {c: avg("ALL", c) for c in ("Hit@1", "Hit@3", "Hit@5", "Recall@10", "MRR", "Confuse@10")},
                        "confuse_same": cs, "confuse_homonym": ch, "n": len(queries),
                        "types": {t: avg(t, "Hit@1") for t in types}}
    (OUT / f"search_{a.tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"질의 {len(queries)}개 ({a.corpus}, {a.split})")
    print(f"{'변형':<26}{'Hit@1':>7}{'Hit@3':>7}{'Hit@5':>7}{'R@10':>7}{'MRR':>7}{'Conf@10':>8}{'동명이인':>8}")
    for name, _ in variants:
        x = result[name]["ALL"]
        conf = f"{x['Confuse@10']:>8.3f}" if x["Confuse@10"] is not None else f"{'-':>8}"
        print(f"{name:<26}{x['Hit@1']:>7.3f}{x['Hit@3']:>7.3f}{x['Hit@5']:>7.3f}{x['Recall@10']:>7.3f}{x['MRR']:>7.3f}"
              f"{conf}{result[name]['confuse_homonym']:>8.3f}")
    if a.set == "ladder":
        print("\n유형별 Hit@1 (S0 → 마지막)")
        for t in types:
            print(f"  {t:<26}{result[variants[0][0]]['types'][t]:.3f} → {result[variants[-1][0]]['types'][t]:.3f}")


if __name__ == "__main__":
    main()
