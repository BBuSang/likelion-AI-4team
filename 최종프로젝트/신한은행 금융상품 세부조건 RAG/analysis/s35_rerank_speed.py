"""14-2 재순위 시간 줄이기 시험 (dev만) — 추론 끄기 · 후보 글자 줄이기

사용: python analysis/s35_rerank_speed.py   → 방식별 dev 정확도(보완 Hit@1·5)와 재순위 호출 시간, data/eval/rerank_speed.json
- 정확도: 검색 dev 105 · 공통 약관 dev 45 · 실제 말투 dev 34, 지금 검색 경로(rerank_score + 섞기). 결과는 재순위 캐시에 방식 표시를 붙여 저장
- 시간: 검색 dev에서 고정한 30문항의 재순위 후보로 캐시 없이 한 번씩 차례로 호출해 중앙값 (방식 순서를 질문마다 돌려 시간대 차이를 고르게)
"""
import json
import random
import statistics as st
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import s12_retrieval as R  # noqa: E402
from s16_rerank_variants import SCORE_INSTR, Scored  # noqa: E402
from s8_extract import load_env  # noqa: E402

EVAL = ROOT / "data" / "eval"
VARIANTS = {"지금": (1500, ""), "추론 끔": (1500, "none"), "짧게 800": (800, ""), "추론 끔 + 짧게": (800, "none")}
VARIANTS_LOW = {"지금": (1500, ""), "추론 low": (1500, "low")}   # 14-2 추가 시험: python analysis/s35_rerank_speed.py low
SETS = ("dev", "common_dev", "real_dev")


def main():
    global VARIANTS
    if "low" in sys.argv:
        VARIANTS = VARIANTS_LOW
    idx = R.Index()
    idx._name_index()
    data = {s: [json.loads(l) for l in open(EVAL / f"retrieval_{s}.jsonl")] for s in SETS}
    vecs = {s: R.embed_questions(qs, idx) for s, qs in data.items()}
    env = load_env()
    from openai import OpenAI
    idx._client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
    out = {"accuracy": {}, "latency": {}}

    # 정확도
    for name, (chars, eff) in VARIANTS.items():
        R.Index.RERANK_CHARS, R.Index.RERANK_EFFORT = chars, eff
        out["accuracy"][name] = {}
        for s, qs in data.items():
            def one(arg):
                q, v = arg
                order = idx.rank("rerank_score", q["question"], v)[:5]
                return next((k + 1 for k, i in enumerate(order) if idx.is_gold_any(i, q)), None)
            with ThreadPoolExecutor(6) as ex:
                ranks = list(ex.map(one, zip(qs, vecs[s])))
            h1 = sum(r == 1 for r in ranks)
            h5 = sum(bool(r) for r in ranks)
            out["accuracy"][name][s] = {"n": len(qs), "h1": h1, "h5": h5}
            print(f"{name:10s} {s:10s} n={len(qs):3d} 보완 H@1 {h1}/{len(qs)} ({h1 / len(qs):.3f}) · H@5 {h5}/{len(qs)} ({h5 / len(qs):.3f})", flush=True)

    # 시간: 같은 후보로 캐시 없이 차례로
    R.Index.RERANK_CHARS, R.Index.RERANK_EFFORT = 1500, ""
    rnd = random.Random(20261202)
    sample = rnd.sample(list(zip(data["dev"], vecs["dev"])), 30)
    cands = []
    for q, v in sample:
        base = [int(i) for i in idx.rank("routed", q["question"], v)]
        by_topic = [int(i) for i in idx.rank("routed", q["question"], v, topic=True)]
        head = base[:7]
        cands.append((q["question"], head + [i for i in by_topic if i not in set(head)][:3]))
    names = list(VARIANTS)
    times = {n: [] for n in names}
    for k, (q, cand) in enumerate(cands):
        for name in names[k % len(names):] + names[:k % len(names)]:
            chars, eff = VARIANTS[name]
            body = "\n\n".join(f"<{n}>\n{idx.items[i]['text'][:chars]}" for n, i in enumerate(cand))
            extra = {"reasoning": {"effort": eff}} if eff else {}
            s = time.perf_counter()
            idx._client.responses.parse(model=env["CHAT_MODEL"], instructions=SCORE_INSTR, text_format=Scored, input=f"질문: {q}\n\n후보:\n{body}", **extra)
            times[name].append(time.perf_counter() - s)
        print(f"  시간 {k + 1}/30", flush=True) if (k + 1) % 10 == 0 else None
    for n in names:
        xs = sorted(times[n])
        out["latency"][n] = {"median": round(st.median(xs), 2), "p90": round(xs[int(0.9 * (len(xs) - 1))], 2)}
        print(f"{n:10s} 재순위 시간 중앙값 {st.median(xs):.2f}s · p90 {xs[int(0.9 * (len(xs) - 1))]:.2f}s")
    (EVAL / ("rerank_speed_low.json" if "low" in sys.argv else "rerank_speed.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
