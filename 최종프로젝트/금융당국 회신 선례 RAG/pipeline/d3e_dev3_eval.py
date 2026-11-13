"""개발셋(기본 dev3, 인자로 dev4 등)으로 임베딩 모델 2개(OpenAI, KURE-v1)와 BM25 시험 점검을 비교한다. D3 청크(8,973개) 기준, 사례 묶음 단위 채점, numpy 정확 검색.
- OpenAI text-embedding-3-small(API) / KURE-v1(로컬 CPU, 최대 512토큰, 정규화, 질문 접두어 없음). 질문은 개인정보 마스킹 후 임베딩, 같은 질문은 캐시
- BM25 시험 점검: 청크의 형태소 색인(제목 + 본문)으로 즉석 구현한 거친 BM25와, KURE + BM25 합치기(RRF). D4 구현이 아니다
- 시험 세트(test*, final_test)는 읽지 않는다: 인자는 dev로 시작하는 세트만 받는다.
사용: python d3e_dev3_eval.py [dev3|dev4]     출력: data/eval/<세트>_eval.json
"""
import hashlib
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict

import numpy as np
from kiwipiepy import Kiwi

import rag
from common import DATA, OUT, manifest, read_jsonl
from d2_baseline import by, load, metrics
from d3_eval import load_arm, paired
from d3b_local_embed import key as local_key

sys.stdout.reconfigure(encoding="utf-8")
EVAL = DATA / "eval"
K = 10
KEEP = ("NNG", "NNP", "NNB", "NR", "SN", "SL", "SH", "XR", "VV", "VA", "MM")
MODELS = {"kure-v1": "nlpai-lab/KURE-v1"}
FILES = {"openai": "embeddings_d3.npy", "kure-v1": "embeddings_d3_kure-v1.npy"}


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def query_vectors(texts):
    """모델별 질문 벡터. 캐시: OpenAI는 query_embed_cache.sqlite, 로컬 모델은 embedding_cache_local.sqlite."""
    out = {}
    oc = sqlite3.connect(EVAL / "query_embed_cache.sqlite")
    out["openai"], miss = {}, 0
    for t in texts:
        k = hashlib.sha256((rag.EMBED_MODEL + "\0" + t).encode()).hexdigest()
        row = oc.execute("select v from q where k = ?", (k,)).fetchone()
        if row:
            out["openai"][t] = unit(np.frombuffer(row[0], dtype=np.float32))
        else:
            v = np.asarray(rag.embed_openai(t), dtype=np.float32)
            oc.execute("insert into q values (?, ?)", (k, v.tobytes()))
            oc.commit()
            out["openai"][t] = unit(v)
            miss += 1
    print(f"OpenAI 질문 임베딩: 캐시 {len(texts) - miss}, 새로 호출 {miss}", flush=True)
    lc = sqlite3.connect(OUT / "embedding_cache_local.sqlite")
    from sentence_transformers import SentenceTransformer
    for short, name in MODELS.items():
        out[short], todo = {}, []
        for t in texts:
            row = lc.execute("select vec from cache where hash = ?", (local_key(name, t),)).fetchone()
            if row:
                out[short][t] = unit(np.frombuffer(row[0], dtype=np.float32))
            else:
                todo.append(t)
        if todo:
            m = SentenceTransformer(name, device="cpu")
            m.max_seq_length = 512
            vecs = m.encode(todo, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
            for t, v in zip(todo, vecs):
                lc.execute("insert or replace into cache values (?,?,?,?)", (local_key(name, t), name, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes()))
                out[short][t] = unit(v)
            lc.commit()
        print(f"{short} 질문 임베딩: 캐시 {len(texts) - len(todo)}, 새로 계산 {len(todo)}", flush=True)
    return out


def group_best(arm, sim):
    best = np.full(len(arm["uniq"]), -1e9, dtype=np.float32)
    np.maximum.at(best, arm["gi"], sim)
    return best


def rank_of(arm, best, gold):
    gi = int(np.flatnonzero(arm["uniq"] == gold)[0])
    return int((best > best[gi]).sum()) + 1


def main():
    NAME = sys.argv[1] if len(sys.argv) > 1 else "dev3"
    if not NAME.startswith("dev"):
        sys.exit("개발셋(dev*)만 평가할 수 있습니다. 시험 세트는 f2_test_eval.py로 한 번만 평가합니다.")
    rows = load(NAME)
    kmed = float(np.median([r["confusable_sim_kure"] for r in rows if r.get("confusable_sim_kure") is not None])) if any(r.get("confusable_sim_kure") is not None for r in rows) else None
    texts = [rag.mask_pii(r["question"]) for r in rows]
    qv = query_vectors(texts)
    arms = {a: load_arm("chunks_d3.jsonl", f) for a, f in FILES.items()}
    chunks = arms["kure-v1"]["chunks"]

    # BM25 시험 점검 (청크의 bm25_tokens = 제목 + 본문 형태소)
    docs = [c["bm25_tokens"].split() for c in chunks]
    N, avg = len(docs), sum(map(len, docs)) / len(docs)
    lens = np.array([len(d) for d in docs], dtype=np.float32)
    post = defaultdict(list)
    for i, d in enumerate(docs):
        for t, f in Counter(d).items():
            post[t].append((i, f))
    idf = {t: math.log(1 + (N - len(p) + .5) / (len(p) + .5)) for t, p in post.items()}
    kiwi = Kiwi()
    qtoks = [[t.form.lower() for t in r if t.tag.split("-")[0] in KEEP] for r in kiwi.tokenize(texts)]

    def bm25(toks):
        s = np.zeros(N, dtype=np.float32)
        for t in set(toks):
            for i, f in post.get(t, ()):
                s[i] += idf[t] * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * lens[i] / avg))
        return s

    res = {a: [] for a in list(arms) + ["bm25", "kure+bm25"]}
    kure = arms["kure-v1"]
    titles = {r["case_key"]: r["title"] for r in read_jsonl("cases_enriched.jsonl")}
    rep = {g["group_id"]: g["representative_case_key"] for g in read_jsonl("case_groups.jsonl")}
    top1_title = {}
    for r, t, toks in zip(rows, texts, qtoks):
        bests = {}
        for a, arm in arms.items():
            bests[a] = group_best(arm, arm["E"] @ qv[a][t])
        bests["bm25"] = group_best(kure, bm25(toks))
        rrf = np.zeros(len(kure["uniq"]))
        for b in (bests["kure-v1"], bests["bm25"]):
            for pos, gi in enumerate(np.argsort(-b)):
                rrf[gi] += 1 / (60 + pos + 1)
        bests["kure+bm25"] = rrf
        for a, b in bests.items():
            rk = rank_of(kure, b, r["gold_group_id"])
            res[a].append({"qid": r["qid"], "rank": rk if rk <= K else None, "rank_full": rk, "tag": r["tag"], "gubun": r["gubun"],
                           "confusable": r["confusable_sim"] >= 0.80, "confusable_kure": (r.get("confusable_sim_kure") is not None and r["confusable_sim_kure"] >= kmed), "top1_group": int(kure["uniq"][int(np.argmax(b))]), "gold_group_id": r["gold_group_id"]})

    summary = {}
    for a, rr in res.items():
        summary[a] = {**metrics(rr), "by_tag": by(rr, "tag"), "by_gubun": by(rr, "gubun"), "by_confusable": by(rr, "confusable"), "by_confusable_kure": by(rr, "confusable_kure") if kmed is not None else None}
    cmp = {f"{a}→{b}": paired(res[a], res[b]) for a, b in (("openai", "kure-v1"), ("kure-v1", "kure+bm25"), ("kure-v1", "bm25"))}
    # KURE가 1위로 못 맞힌 문항: 정답과 1위 사례의 제목 비교
    misses = []
    for x in res["kure-v1"]:
        if x["rank_full"] > 1:
            misses.append({"qid": x["qid"], "tag": x["tag"], "rank": x["rank_full"], "gold": titles.get(rep.get(x["gold_group_id"]), "?"), "top1": titles.get(rep.get(x["top1_group"]), "?"),
                           "same_title": (titles.get(rep.get(x["gold_group_id"])) or "").strip() == (titles.get(rep.get(x["top1_group"])) or "").strip()})
    with open(EVAL / f"{NAME}_eval.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "paired": cmp, "misses_kure": misses, "detail": res}, f, ensure_ascii=False, indent=1)
    manifest("d3e_dev_eval", set=NAME, n=len(rows), arms=list(res))
    for a in res:
        m = summary[a]
        print(f"{a:10s} n={m['n']} Hit@1 {m['hit1']:.3f} Hit@3 {m['hit3']:.3f} Hit@5 {m['hit5']:.3f} Hit@10 {m['hit10']:.3f} MRR@10 {m['mrr10']:.3f} 10위 밖 {m['not_in_top10']}")
    for k, v in cmp.items():
        print(k, "(앞 방식에서 뒤 방식으로 바꿨을 때) 좋아짐/나빠짐/같음", v["better"], v["worse"], v["same"], "p", v["p_sign"])
    for a in ("openai", "kure-v1"):
        print(a, "유형별 (n, Hit@1, Hit@5, MRR):", {k: (v["n"], v["hit1"], v["hit5"], v["mrr10"]) for k, v in summary[a]["by_tag"].items()})
        if kmed is not None:
            print(a, f"KURE 기준 혼동(최근접 유사도 {kmed:.3f} 이상) 여부별:", {k: (v["n"], v["hit1"], v["hit5"]) for k, v in summary[a]["by_confusable_kure"].items()})
        print(a, "혼동 여부별:", {k: (v["n"], v["hit1"], v["hit5"]) for k, v in summary[a]["by_confusable"].items()}, "구분별:", {k: (v["n"], v["hit1"], v["hit5"]) for k, v in summary[a]["by_gubun"].items()})
    print(f"KURE 1위 아님 {len(misses)}문항, 정답·1위 제목이 같은 것 {sum(m['same_title'] for m in misses)}")
    for m in misses:
        print(f"  [{m['qid']}] {m['tag']} {m['rank']}위 | 정답: {m['gold'][:45]} | 1위: {m['top1'][:45]}{' (제목 같음)' if m['same_title'] else ''}")


if __name__ == "__main__":
    main()
