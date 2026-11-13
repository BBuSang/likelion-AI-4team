"""D3 후속 진단 (읽기 전용): KURE-v1이 1위로 못 맞힌 문항의 질문·정답 제목·1위 제목·순위·질문어 포함률과 BM25 순위,
그리고 KURE / BM25만 / RRF 합치기를 같은 세트로 비교하는 시험 점검. D4 구현이 아니다 (형태소 BM25를 즉석에서 만든 거친 구현).
사용: python d3c_kure_misses.py   (embeddings_d3_kure-v1.npy와 질문 캐시가 필요). 테스트셋은 읽지 않는다."""
import json
import math
import sys
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from kiwipiepy import Kiwi

from common import OUT, read_jsonl
from d2_baseline import SETS, load
from d3b_eval import load_arm, qvec_fn

KEEP = ("NNG", "NNP", "NNB", "NR", "SN", "SL", "SH", "XR", "VV", "VA", "MM")
kiwi = Kiwi()
tok = lambda texts: [[t.form.lower() for t in r if t.tag.split("-")[0] in KEEP] for r in kiwi.tokenize(texts)]

arm = load_arm("chunks_d3.jsonl", "embeddings_d3_kure-v1.npy")
chunks = arm["chunks"]
titles = {r["case_key"]: r["title"] for r in read_jsonl("cases_enriched.jsonl")}
rep = {g["group_id"]: g["representative_case_key"] for g in read_jsonl("case_groups.jsonl")}
gtitle = lambda g: titles.get(rep.get(g), "?")

# BM25 (제목 + 본문 형태소, 청크 단위 → 묶음 최고점)
docs = tok([titles[c["case_key"]] + "\n" + c["text"] for c in chunks])
N = len(docs); avg = sum(map(len, docs)) / N
df = Counter(t for d in docs for t in set(d))
idf = {t: math.log(1 + (N - n + .5) / (n + .5)) for t, n in df.items()}
tfs = [Counter(d) for d in docs]
lens = np.array([len(d) for d in docs])


def bm25(qtoks):
    s = np.zeros(N, dtype=np.float32)
    for t in set(qtoks):
        if t not in idf:
            continue
        for i, tf in enumerate(tfs):
            f = tf.get(t)
            if f:
                s[i] += idf[t] * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * lens[i] / avg))
    return s


def group_rank(sim, gold):
    best = np.full(len(arm["uniq"]), -1e9, dtype=np.float32)
    np.maximum.at(best, arm["gi"], sim)
    gi = {g: i for i, g in enumerate(arm["uniq"])}[gold]
    return int((best > best[gi]).sum()) + 1, best


qv = qvec_fn("kure-v1")
rows = []
for n in SETS:
    for r in load(n):
        rows.append((n, r))
miss = []
allres = []
for n, r in rows:
    sim = arm["E"] @ qv(r["question"])
    rk, best = group_rank(sim, r["gold_group_id"])
    qt = tok([r["question"]])[0]
    b = bm25(qt)
    brk, bbest = group_rank(b, r["gold_group_id"])
    allres.append((n, r["qid"], rk, brk, best, bbest, r, qt))
    if rk > 1:
        top = arm["uniq"][int(np.argmax(best))]
        gi = np.flatnonzero(arm["gi"] == np.flatnonzero(arm["uniq"] == r["gold_group_id"])[0])
        goldtxt = set(sum([docs[i] for i in gi], []))
        toptxt = set(sum([docs[i] for i in np.flatnonzero(arm["gi"] == np.flatnonzero(arm["uniq"] == top)[0])], []))
        qs = set(qt)
        miss.append({"set": n, "qid": r["qid"], "tag": r.get("tag"), "kure_rank": rk, "bm25_rank": brk, "q": r["question"], "gold": gtitle(r["gold_group_id"]), "top1": gtitle(top),
                     "cover_gold": round(len(qs & goldtxt) / max(len(qs), 1), 2), "cover_top1": round(len(qs & toptxt) / max(len(qs), 1), 2)})
print(f"KURE 1위 아닌 문항 {len(miss)}개 / 전체 {len(rows)}")
for m in miss:
    print(f"\n[{m['qid']}] {m['tag']} KURE {m['kure_rank']}위 / BM25 {m['bm25_rank']}위 | 질문어 포함률 정답 {m['cover_gold']} vs 1위 {m['cover_top1']}")
    print("  Q:", m["q"][:150].replace("\n", " "))
    print("  정답:", m["gold"][:70], "\n  1위 :", m["top1"][:70])

# 시험용 합치기(RRF) 점검: KURE + BM25
def rrf_eval(w=60):
    out = []
    for n, qid, rk, brk, best, bbest, r, qt in allres:
        order_v = np.argsort(-best); order_b = np.argsort(-bbest)
        sc = np.zeros(len(best))
        for pos, i in enumerate(order_v): sc[i] += 1 / (w + pos + 1)
        for pos, i in enumerate(order_b): sc[i] += 1 / (w + pos + 1)
        gi = int(np.flatnonzero(arm["uniq"] == r["gold_group_id"])[0])
        out.append((n, int((sc > sc[gi]).sum()) + 1))
    return out

def hit(ranks, k): return round(sum(1 for x in ranks if x <= k) / len(ranks), 3)
for label, sel in (("dev1+dev2", ("dev1", "dev2")), ("합성150", ("rewrite_150",))):
    k_r = [x[2] for x in allres if x[0] in sel]; b_r = [x[3] for x in allres if x[0] in sel]; f_r = [r for n, r in rrf_eval() if n in sel]
    for name, rr in (("KURE", k_r), ("BM25만", b_r), ("KURE+BM25(RRF)", f_r)):
        print(f"{label:10s} {name:15s} Hit@1 {hit(rr,1)} Hit@5 {hit(rr,5)} MRR {round(sum(1/x for x in rr)/len(rr),3)}")
