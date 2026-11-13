"""D3 후속 비교: 같은 D3 청크(8,973개)와 같은 질문 세트로 임베딩 모델을 비교한다.
text-embedding-3-small(API, 기존) vs KURE-v1 vs bge-m3 (로컬). 채점은 d3_eval.py와 같은 방식(numpy 정확 검색, 묶음 최고 유사도, 사례 묶음 단위).
테스트셋은 읽지 않는다.  사용: python d3b_eval.py     출력: data/eval/d3b_eval.json
"""
import json
import sqlite3
import sys

import numpy as np

import rag
from common import OUT, manifest
from d2_baseline import EVAL, SETS, by, load, metrics
from d3_eval import auc, load_arm, paired
from d3b_local_embed import key

sys.stdout.reconfigure(encoding="utf-8")
K = 10
LOCAL = {"kure-v1": "nlpai-lab/KURE-v1", "bge-m3": "BAAI/bge-m3"}
cache_local = sqlite3.connect(OUT / "embedding_cache_local.sqlite")
cache_openai = sqlite3.connect(EVAL / "query_embed_cache.sqlite")


def unit(v):
    v = np.frombuffer(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def qvec_fn(arm_name):
    import hashlib
    def f(text):
        masked = rag.mask_pii(text)
        if arm_name == "openai":
            row = cache_openai.execute("select v from q where k = ?", (hashlib.sha256((rag.EMBED_MODEL + "\0" + masked).encode()).hexdigest(),)).fetchone()
        else:
            row = cache_local.execute("select vec from cache where hash = ?", (key(LOCAL[arm_name], masked),)).fetchone()
        if not row:
            raise SystemExit(f"{arm_name}: 질문 임베딩이 캐시에 없음")
        return unit(row[0])
    return f


def rank_all(arm, rows, qvec):
    gold_idx = {g: i for i, g in enumerate(arm["uniq"])}
    out = []
    for r in rows:
        sim = arm["E"] @ qvec(r["question"])
        best = np.full(len(arm["uniq"]), -1.0, dtype=np.float32)
        np.maximum.at(best, arm["gi"], sim)
        gi = gold_idx[r["gold_group_id"]]
        rank = int((best > best[gi]).sum()) + 1
        gc = np.flatnonzero(arm["gi"] == gi)
        out.append({"qid": r["qid"], "rank": rank if rank <= K else None, "rank_full": rank, "top1_score": float(best.max()), "gold_score": float(best[gi]),
                    "gold_chunk_type": arm["types"][gc[np.argmax(sim[gc])]], "tag": r.get("tag"), "gubun": r.get("gubun")})
    return out


def main():
    files = {"openai": "embeddings_d3.npy", "kure-v1": "embeddings_d3_kure-v1.npy", "bge-m3": "embeddings_d3_bge-m3.npy"}
    arms = {a: load_arm("chunks_d3.jsonl", f) for a, f in files.items() if (OUT / f).exists()}
    data = {n: load(n) for n in SETS}
    una = load("unanswerable_20")
    res, summary, una_top = {}, {}, {}
    for a, arm in arms.items():
        qv = qvec_fn(a)
        res[a] = {n: rank_all(arm, data[n], qv) for n in SETS}
        una_top[a] = [float((arm["E"] @ qv(r["question"])).max()) for r in una]
    for a in arms:
        s = {n: {"label": SETS[n], **metrics(res[a][n]), "by_tag": by(res[a][n], "tag"), "by_gubun": by(res[a][n], "gubun"),
                 "gold_chunk_type_top5": {t: sum(1 for x in res[a][n] if x["gold_chunk_type"] == t and x["rank"] and x["rank"] <= 5) for t in ("question_card", "judgment", "rule_card")}} for n in SETS}
        dev = res[a]["dev1"] + res[a]["dev2"]
        s["dev1+dev2"] = {"label": "dev1+dev2", **metrics(dev), "by_tag": by(dev, "tag")}
        ans = [x["top1_score"] for x in dev]
        s["unanswerable"] = {"top1_median": round(float(np.median(una_top[a])), 4), "top1_max": round(max(una_top[a]), 4), "answerable_top1_min": round(min(ans), 4),
                             "answerable_top1_median": round(float(np.median(ans)), 4), "auc": round(auc(ans, una_top[a]), 4)}
        summary[a] = s
    cmp = {}
    for a in arms:
        for b in arms:
            if a < b:
                cmp[f"{a}→{b}"] = {n: paired(res[a][n], res[b][n]) for n in SETS}
                cmp[f"{a}→{b}"]["dev1+dev2"] = paired(res[a]["dev1"] + res[a]["dev2"], res[b]["dev1"] + res[b]["dev2"])
                cmp[f"{a}→{b}"]["all"] = paired(sum((res[a][n] for n in SETS), []), sum((res[b][n] for n in SETS), []))
    with open(EVAL / "d3b_eval.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "paired": cmp, "detail": res}, f, ensure_ascii=False, indent=1)
    manifest("d3b_eval", arms=list(arms))
    for n in list(SETS) + ["dev1+dev2"]:
        for a in arms:
            m = summary[a][n]
            print(f"{n:11s} {a:8s} n={m['n']:3d} Hit@1 {m['hit1']:.3f} Hit@3 {m['hit3']:.3f} Hit@5 {m['hit5']:.3f} Hit@10 {m['hit10']:.3f} MRR@10 {m['mrr10']:.3f} 10위 밖 {m['not_in_top10']}")
    for k, v in cmp.items():
        print(k, {n: (x["better"], x["worse"], x["same"], x["p_sign"]) for n, x in v.items()})
    for a in arms:
        print(a, json.dumps(summary[a]["unanswerable"], ensure_ascii=False))


if __name__ == "__main__":
    main()
