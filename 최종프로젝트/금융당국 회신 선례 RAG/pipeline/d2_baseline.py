"""D2 베이스라인 점수 측정: rag.py의 벡터 검색(칸별 청크, 상위 50조각 → 사례 묶음)을 질문 세트로 돌려 사례 묶음 단위로 채점한다.
대상: 합성 질의셋(rewrite_150, 참고용), 개발셋 dev1·dev2, 답변불가셋(최고 점수 분포).
테스트셋(test1·test2)은 마지막에 한 번만 쓰므로 여기서 읽지 않는다.
질문 임베딩은 모델명+글 해시로 캐시해서 D3 이후 비교 때 같은 질문을 다시 호출하지 않는다.
사용: python d2_baseline.py     출력: data/eval/d2_baseline.json     키·접속 값은 출력하지 않는다.
"""
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict

import numpy as np

import rag
from common import DATA, manifest

EVAL = DATA / "eval"
K = 10                                   # 순위는 상위 10개 묶음까지 본다 (Hit@10, MRR@10)
SETS = {"rewrite_150": "합성 질의셋", "dev1": "개발셋 1", "dev2": "개발셋 2"}
sys.stdout.reconfigure(encoding="utf-8")

_orig_embed = rag.embed_openai           # 주의: D2는 옛 베이스라인 DB(칸별 청크 + OpenAI 벡터)에서 쟀다. 지금 DB(KURE)에서는 d3_eval.py의 baseline 팔로 재현한다
_db = sqlite3.connect(EVAL / "query_embed_cache.sqlite")
_db.execute("create table if not exists q (k text primary key, v blob)")
n_calls = 0


def cached_embed(text):
    """rag.embed와 같은 결과를 돌려주되, 같은 질문은 캐시에서 읽는다."""
    global n_calls
    key = hashlib.sha256((rag.EMBED_MODEL + "\0" + text).encode()).hexdigest()
    row = _db.execute("select v from q where k = ?", (key,)).fetchone()
    if row:
        return np.frombuffer(row[0], dtype=np.float32).tolist()
    vec = np.asarray(_orig_embed(text), dtype=np.float32)
    n_calls += 1
    _db.execute("insert into q values (?, ?)", (key, vec.tobytes()))
    _db.commit()
    return vec.tolist()


def load(name):
    with open(EVAL / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(rows, conn, rep_to_group):
    out = []
    for r in rows:
        hits = rag.search(r["question"], K, conn)
        groups = [rep_to_group[h["doc_id"]] for h in hits]
        gold = r.get("gold_group_id")
        rank = groups.index(gold) + 1 if gold in groups else None
        out.append({"qid": r["qid"], "gold_group_id": gold, "rank": rank, "top1_score": hits[0]["score"] if hits else None,
                    "gold_score": hits[rank - 1]["score"] if rank else None, "gold_chunk_type": hits[rank - 1]["chunk_type"] if rank else None,
                    "n_groups": len(hits), "top_groups": groups[:5], "tag": r.get("tag"), "gubun": r.get("gubun"),
                    "doc_class": r.get("doc_class"), "has_filter": bool(r.get("filter")), "has_article": bool(r.get("cited_article"))})
    return out


def metrics(res):
    n = len(res)
    if not n:
        return {"n": 0}
    rk = [x["rank"] for x in res]
    hit = lambda k: round(sum(1 for r in rk if r and r <= k) / n, 4)
    return {"n": n, "hit1": hit(1), "hit3": hit(3), "hit5": hit(5), "hit10": hit(10),
            "mrr10": round(sum(1 / r for r in rk if r) / n, 4), "not_in_top10": sum(1 for r in rk if not r)}


def by(res, key):
    g = defaultdict(list)
    for x in res:
        g[str(x[key])].append(x)
    return {k: metrics(v) for k, v in sorted(g.items())}


def main():
    rag.embed = cached_embed             # 이 스크립트를 실행할 때만 OpenAI 질문 임베딩(캐시)으로 바꾼다. 다른 스크립트가 import해도 rag.embed는 그대로다
    with rag.connect() as conn:
        rep_to_group = {r["representative_case_key"]: r["group_id"] for r in conn.execute("select group_id, representative_case_key from interp.case_groups").fetchall()}
        valid = set(rep_to_group.values())
        summary, detail = {}, {}
        for name in SETS:
            rows = load(name)
            missing = [r["qid"] for r in rows if r["gold_group_id"] not in valid]
            assert not missing, f"{name}: 정답 묶음이 DB에 없음 {missing}"
            res = run(rows, conn, rep_to_group)
            detail[name] = res
            summary[name] = {"label": SETS[name], **metrics(res), "by_tag": by(res, "tag"), "by_gubun": by(res, "gubun"),
                             "by_doc_class": by(res, "doc_class") if name != "rewrite_150" else None,
                             "by_filter": by(res, "has_filter") if name != "rewrite_150" else None,
                             "gold_chunk_type": {t: sum(1 for x in res if x["gold_chunk_type"] == t and x["rank"] and x["rank"] <= 5) for t in ("question", "answer", "reason", "rule_card")}}
        # 답변불가셋: 최고 점수가 답이 있는 질문과 얼마나 갈리는지 (LLM 판단 전 단계의 유사도 기준값 가능성)
        una = load("unanswerable_20")
        una_top = [rag.search(r["question"], 1, conn)[0]["score"] for r in una]
    ans_top = [x["top1_score"] for n in ("dev1", "dev2") for x in detail[n]]
    grid = []
    for t in np.arange(0.30, 0.76, 0.02):
        grid.append({"threshold": round(float(t), 2), "answerable_pass": round(float(np.mean([s >= t for s in ans_top])), 3),
                     "unanswerable_reject": round(float(np.mean([s < t for s in una_top])), 3)})
    summary["unanswerable"] = {"n": len(una_top), "top1_min": round(min(una_top), 4), "top1_median": round(float(np.median(una_top)), 4), "top1_max": round(max(una_top), 4),
                               "answerable_top1_min": round(min(ans_top), 4), "answerable_top1_median": round(float(np.median(ans_top)), 4),
                               "threshold_grid": grid}
    summary["config"] = {"retrieval": "벡터 검색만, 칸별 청크 상위 50조각 → 묶음 합치기", "k": K, "embed_model": rag.EMBED_MODEL, "n_embed_calls": n_calls,
                         "excluded": ["test1", "test2", "article_lookup_50"], "filters_applied": False}
    with open(EVAL / "d2_baseline.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detail": detail, "unanswerable_top1": una_top}, f, ensure_ascii=False, indent=1)
    manifest("d2_baseline", **{n: summary[n]["n"] for n in SETS})
    return summary


if __name__ == "__main__":
    s = main()
    for n in SETS:
        m = s[n]
        print(f"{m['label']:8s} n={m['n']:3d} Hit@1 {m['hit1']:.3f} Hit@3 {m['hit3']:.3f} Hit@5 {m['hit5']:.3f} Hit@10 {m['hit10']:.3f} MRR@10 {m['mrr10']:.3f} 10위 밖 {m['not_in_top10']}")
    print(json.dumps({k: s[k] for k in ("unanswerable", "config")}, ensure_ascii=False, indent=1))
