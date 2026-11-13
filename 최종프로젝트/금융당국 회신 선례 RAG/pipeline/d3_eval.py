"""D3 비교: 베이스라인(칸별 청크) vs 후보(질의 카드 + 판단 본문)를 D2와 같은 질문 세트·같은 지표로 채점한다.
검색은 전체 청크를 numpy로 정확히 검색하고 묶음마다 가장 비슷한 조각으로 순위를 매긴다 (D2에서 DB 검색 결과와 일치함을 확인한 방식).
베이스라인 순위가 d2_baseline.json과 같은지 먼저 확인한다. 테스트셋은 읽지 않는다.
사용: python d3_eval.py     출력: data/eval/d3_eval.json
"""
import hashlib
import json
import sqlite3
import sys
from math import comb

import numpy as np

import rag
from common import DATA, OUT, manifest
from d2_baseline import EVAL, SETS, by, load, metrics

sys.stdout.reconfigure(encoding="utf-8")
K = 10
db = sqlite3.connect(EVAL / "query_embed_cache.sqlite")


def qvec(text):
    key = hashlib.sha256((rag.EMBED_MODEL + "\0" + rag.mask_pii(text)).encode()).hexdigest()
    row = db.execute("select v from q where k = ?", (key,)).fetchone()
    if not row:
        raise SystemExit("질문 임베딩 캐시에 없음: d2_baseline.py를 먼저 실행")
    v = np.frombuffer(row[0], dtype=np.float32)
    return v / np.linalg.norm(v)


def load_arm(chunks_file, emb_file):
    chunks = [json.loads(l) for l in open(OUT / chunks_file, encoding="utf-8")]
    E = np.load(OUT / emb_file).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    gids = np.array([c["group_id"] for c in chunks])
    uniq, gi = np.unique(gids, return_inverse=True)
    return {"chunks": chunks, "E": E, "gi": gi, "uniq": uniq, "types": [c["chunk_type"] for c in chunks]}


def rank_all(arm, rows):
    """질문마다 묶음별 최고 유사도를 구해 정답 묶음의 순위(11이상은 None)와 최고 점수를 돌려준다."""
    out = []
    gold_idx = {g: i for i, g in enumerate(arm["uniq"])}
    for r in rows:
        sim = arm["E"] @ qvec(r["question"])
        best = np.full(len(arm["uniq"]), -1.0, dtype=np.float32)
        np.maximum.at(best, arm["gi"], sim)
        gi = gold_idx[r["gold_group_id"]]
        rank = int((best > best[gi]).sum()) + 1
        gold_chunks = np.flatnonzero(arm["gi"] == gi)
        gtype = arm["types"][gold_chunks[np.argmax(sim[gold_chunks])]]
        out.append({"qid": r["qid"], "gold_group_id": r["gold_group_id"], "rank": rank if rank <= K else None, "rank_full": rank, "top1_score": float(best.max()),
                    "gold_score": float(best[gi]), "gold_chunk_type": gtype, "tag": r.get("tag"), "gubun": r.get("gubun"),
                    "doc_class": r.get("doc_class"), "has_filter": bool(r.get("filter"))})
    return out


def auc(pos, neg):
    """pos(답이 있는 질문)의 최고 점수가 neg(답변불가)보다 큰 쌍의 비율."""
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def sign_test(a, b):
    """같은 문항의 순위가 좋아진 수 a, 나빠진 수 b. 양측 정확 이항 검정."""
    n = a + b
    if not n:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(res_a, res_b):
    fa = lambda x: x["rank_full"]
    better = sum(1 for x, y in zip(res_a, res_b) if fa(y) < fa(x))
    worse = sum(1 for x, y in zip(res_a, res_b) if fa(y) > fa(x))
    return {"better": better, "worse": worse, "same": len(res_a) - better - worse, "p_sign": round(sign_test(better, worse), 4)}


def main():
    arms = {"baseline": load_arm("chunks.jsonl", "embeddings.npy"), "d3": load_arm("chunks_d3.jsonl", "embeddings_d3.npy")}
    d2 = json.load(open(EVAL / "d2_baseline.json", encoding="utf-8"))["detail"]
    data = {n: load(n) for n in SETS}
    res = {a: {n: rank_all(arms[a], data[n]) for n in SETS} for a in arms}
    # 검증: numpy 베이스라인이 D2(DB 검색) 순위와 같은가
    diff = [(n, x["qid"]) for n in SETS for x, y in zip(res["baseline"][n], d2[n]) if x["rank"] != y["rank"]]
    print(f"D2 순위와 불일치: {len(diff)}건 {diff[:5]}")
    una = load("unanswerable_20")
    una_top = {a: [float((arms[a]["E"] @ qvec(r["question"])).max()) for r in una] for a in arms}
    summary = {}
    for a in arms:
        s = {n: {"label": SETS[n], **metrics(res[a][n]), "by_tag": by(res[a][n], "tag"), "by_gubun": by(res[a][n], "gubun"),
                 "gold_chunk_type_top5": {t: sum(1 for x in res[a][n] if x["gold_chunk_type"] == t and x["rank"] and x["rank"] <= 5) for t in sorted(set(arms[a]["types"]))}}
             for n in SETS}
        s["dev1+dev2"] = {"label": "dev1+dev2", **metrics(res[a]["dev1"] + res[a]["dev2"]), "by_tag": by(res[a]["dev1"] + res[a]["dev2"], "tag")}
        ans = [x["top1_score"] for n in ("dev1", "dev2") for x in res[a][n]]
        s["unanswerable"] = {"top1_median": round(float(np.median(una_top[a])), 4), "top1_max": round(max(una_top[a]), 4), "answerable_top1_min": round(min(ans), 4),
                             "answerable_top1_median": round(float(np.median(ans)), 4), "auc": round(auc(ans, una_top[a]), 4),
                             "grid": [{"t": t, "answerable_pass": round(float(np.mean([v >= t for v in ans])), 3), "unanswerable_reject": round(float(np.mean([v < t for v in una_top[a]])), 3)} for t in (0.5, 0.54, 0.58, 0.62)]}
        summary[a] = s
    cmp = {n: paired(res["baseline"][n], res["d3"][n]) for n in SETS}
    cmp["dev1+dev2"] = paired(res["baseline"]["dev1"] + res["baseline"]["dev2"], res["d3"]["dev1"] + res["d3"]["dev2"])
    cmp["all"] = paired(sum((res["baseline"][n] for n in SETS), []), sum((res["d3"][n] for n in SETS), []))
    out = {"summary": summary, "paired_baseline_to_d3": cmp, "n_chunks": {a: len(arms[a]["chunks"]) for a in arms}, "d2_mismatch": len(diff)}
    with open(EVAL / "d3_eval.json", "w", encoding="utf-8") as f:
        json.dump({**out, "detail": res}, f, ensure_ascii=False, indent=1)
    manifest("d3_eval", d2_mismatch=len(diff), **{a: len(arms[a]["chunks"]) for a in arms})
    for n in list(SETS) + ["dev1+dev2"]:
        for a in arms:
            m = summary[a][n]
            print(f"{n:11s} {a:8s} n={m['n']:3d} Hit@1 {m['hit1']:.3f} Hit@3 {m['hit3']:.3f} Hit@5 {m['hit5']:.3f} Hit@10 {m['hit10']:.3f} MRR@10 {m['mrr10']:.3f} 10위 밖 {m['not_in_top10']}")
        print("  순위 변화(베이스라인→D3):", cmp[n])
    print("전체 문항 순위 변화:", cmp["all"])
    for a in arms:
        print(a, json.dumps(summary[a]["unanswerable"], ensure_ascii=False))


if __name__ == "__main__":
    main()
