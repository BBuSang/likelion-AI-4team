"""14-1 응답 시간 측정 — 질문 하나가 들어와 답변이 나올 때까지, 단계별로

사용: python analysis/s34_latency.py [질문 수=40]   → data/eval/latency.jsonl + 요약 출력
방법
  - 지금 서비스 경로 그대로 (s27 답변 흐름, 답변 원칙 v9, 검색 rerank_score + 섞기, 근거 5개 / 비교는 상품마다 3개). 채점은 하지 않음
  - **캐시 없이**: 질문 벡터·상품명 고르기·재순위 캐시를 비우고, 캐시 파일 쓰기는 임시 폴더로 돌림 (평가용 캐시는 건드리지 않음)
  - 질문은 한 번에 하나씩 (동시 처리 없이 한 사람이 물었을 때의 시간)
  - 시작 준비(색인·임베딩 모델 불러오기)는 서버를 띄울 때 한 번이라 따로 잼
  - 질문: 최종 검증 세트에서 유형별로 고름 (seed 고정)
단계
  embed   질문·주제 문장 임베딩 (로컬 KURE, MPS)
  pick    상품명 고르기 LLM (별칭이 애매하거나 못 찾을 때만)
  rerank  점수형 재순위 LLM (비교 질문은 상품마다 한 번)
  answer  답변 생성 LLM
  other   나머지 (BM25·결합 순위·근거 머리 만들기 등 코드)
"""
import json
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
EVAL = ROOT / "data" / "eval"
OUT = EVAL / "latency.jsonl"
SEED = 20261201
PLAN = [("retrieval", "A", 14), ("retrieval", "B-common", 3), ("retrieval", "B-shared", 3), ("absent", "A-own", 6),
        ("status", "disc", 3), ("status", "onsale", 2), ("gap", "disc_nick", 2), ("gap", "fake_full", 3), ("compare", "compare", 4)]


def main():
    n_total = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    import csv
    import openai
    from openai.resources.responses import Responses

    t0 = time.perf_counter()
    import s12_retrieval as R
    import s15_answer_eval as A
    from s8_extract import load_env
    from s16_rerank_variants import SCORE_INSTR
    t_import = time.perf_counter() - t0
    t0 = time.perf_counter()
    idx = R.Index()
    t_index = time.perf_counter() - t0
    t0 = time.perf_counter()
    idx._name_index()
    t_names = time.perf_counter() - t0
    t0 = time.perf_counter()
    from s10_chunks import local_model
    idx._lm = local_model(idx.model)
    idx._lm.encode(["준비"], normalize_embeddings=True)
    t_model = time.perf_counter() - t0
    print(f"시작 준비: 불러오기 {t_import:.1f}s · 색인 {t_index:.1f}s · 상품명 색인 {t_names:.1f}s · 임베딩 모델 {t_model:.1f}s")

    # 캐시 비우기 + 캐시 파일 쓰기를 임시 폴더로
    tmp = Path(tempfile.mkdtemp(prefix="latency_"))
    R.EVAL, R.PICK_CACHE, R.CH = tmp, tmp / "pick.jsonl", tmp
    idx.cache = {}
    idx._rs, idx._pick_cache = {}, {}

    # 단계별 시간 재기
    cur = defaultdict(float)
    enc = idx._lm.encode

    def timed_encode(*a, **k):
        s = time.perf_counter()
        try:
            return enc(*a, **k)
        finally:
            cur["embed"] += time.perf_counter() - s
    idx._lm.encode = timed_encode
    orig_parse = Responses.parse

    def stage_of(instr):
        if instr == SCORE_INSTR:
            return "rerank"
        if instr and instr.startswith(tuple(p[:30] for p in (R.PICK_INSTR, getattr(R, "PICK_INSTR_DISC", R.PICK_INSTR)))):
            return "pick"
        return "llm_other"

    def parse(self, *a, **k):
        s = time.perf_counter()
        try:
            return orig_parse(self, *a, **k)
        finally:
            cur[stage_of(k.get("instructions"))] += time.perf_counter() - s
            cur["n_llm"] += 1

    Responses.parse = parse   # 답변은 아래에서 스트리밍으로 직접 잼 (stream이 내부에서 create를 부르므로 create는 감싸지 않음)

    # 질문 고르기
    rnd = random.Random(SEED)
    qs = []
    scale = n_total / sum(n for *_, n in PLAN)
    for f, typ, n in PLAN:
        pool = [q for q in map(json.loads, open(EVAL / f"{f}_final.jsonl")) if q["type"] == typ]
        rnd.shuffle(pool)
        qs += pool[:max(1, round(n * scale))]

    env = load_env()
    client = openai.OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
    docs = {r["doc_id"]: r for r in csv.DictReader(open(ROOT / "data" / "meta" / "documents.csv"))}
    pnames_all = {r["product_id"]: r["name"] for r in csv.DictReader(open(ROOT / "data" / "meta" / "products.csv"))}
    version, prompt = "v9", A.prompt_of("v9")
    recs = []
    for n, q in enumerate(qs):
        cur.clear()
        s = time.perf_counter()
        qv = idx.text_vec(q["question"])
        ctx_ids, groups = A.context_ids(idx, q["question"], qv, version, k=5)
        extra = A.extra_context(q["question"], idx)
        pids = set().union(*groups) if len(groups) >= 2 else idx.detect_products(q["question"])
        linked = set().union(*(idx.links[p] for p in pids)) if pids else set()
        lcl = set().union(*(idx.doc_clause[d] for d in linked)) if linked else set()
        pn = ", ".join(sorted(pnames_all[p] for p in pids))[:120]
        if len(groups) >= 2:
            ctx = "\n\n".join(f"[{k}] {A.multi_header(idx, i, docs, groups, pnames_all)}\n{idx.items[i]['text'][:2000]}" for k, i in enumerate(ctx_ids, 1))
        else:
            ctx = "\n\n".join(f"[{k}] {A.context_header(idx, i, docs, linked, lcl, pn)}\n{idx.items[i]['text'][:2000]}" for k, i in enumerate(ctx_ids, 1))
        inp = extra + "\n\n" + A.answer_input(version, ctx, q["question"], pn, inferred=bool(pids) and idx.detect_how(q["question"]) != "exact")
        t_first = None
        stream_s = time.perf_counter()
        ans = []
        with client.responses.stream(model=env["CHAT_MODEL"], instructions=prompt, input=inp) as st:
            for ev in st:
                if ev.type == "response.output_text.delta":
                    if t_first is None:
                        t_first = time.perf_counter() - s
                    ans.append(ev.delta)
        cur["answer"] += time.perf_counter() - stream_s
        cur["n_llm"] += 1
        total = time.perf_counter() - s
        llm = cur["pick"] + cur["rerank"] + cur["answer"] + cur["llm_other"]
        rec = {"qid": q["qid"], "type": q["type"], "style": q.get("style"), "question": q["question"], "total": round(total, 2),
               "first_token": round(t_first or total, 2), **{k: round(cur[k], 2) for k in ("embed", "pick", "rerank", "answer", "llm_other")},
               "other": round(total - llm - cur["embed"], 2), "n_llm": int(cur["n_llm"]), "groups": len(groups),
               "in_chars": len(inp), "out_chars": len("".join(ans))}
        recs.append(rec)
        print(f"{n + 1:2d}/{len(qs)} {q['qid']} {q['type']:10s} 전체 {total:5.1f}s · 첫 글자 {rec['first_token']:5.1f}s · "
              f"재순위 {rec['rerank']:4.1f} · 상품명 {rec['pick']:4.1f} · 답변 {rec['answer']:4.1f} · 임베딩 {rec['embed']:4.2f} · LLM {rec['n_llm']}회")
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    summary(recs, {"import": t_import, "index": t_index, "names": t_names, "model": t_model})


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summary(recs, start):
    import statistics as st
    print(f"\n[{len(recs)}개 질문] 중앙값 / p90 / 최대 (초)")
    for k in ("total", "first_token", "rerank", "pick", "answer", "embed", "other"):
        xs = [r[k] for r in recs]
        print(f"  {k:12s} {st.median(xs):5.1f} / {pct(xs, 90):5.1f} / {max(xs):5.1f}")
    by = defaultdict(list)
    for r in recs:
        by[r["type"]].append(r)
    print("유형별 전체 시간 중앙값 (LLM 호출 수)")
    for t, rs in by.items():
        print(f"  {t:10s} n={len(rs):2d} {st.median(r['total'] for r in rs):5.1f}s ({st.median(r['n_llm'] for r in rs):.0f}회)")
    tot = sum(r["total"] for r in recs)
    print("전체 시간 중 비중: " + " · ".join(f"{k} {sum(r[k] for r in recs) / tot:.0%}" for k in ("rerank", "pick", "answer", "embed", "other")))


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summary([json.loads(l) for l in open(OUT)], {})
    else:
        main()
