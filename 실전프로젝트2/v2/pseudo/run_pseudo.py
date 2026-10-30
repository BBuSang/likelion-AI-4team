"""holdout 문서 187건·질의 88개를 v2 가명처리로 처리해 파일로 저장 (DB 적재는 P3).

    python -X utf8 v2/pseudo/run_pseudo.py
    python -X utf8 v2/pseudo/run_pseudo.py --corpus devtest    dev+test 문서 300건 + 새 질의 세트 (과적합 점검)
                                                                → mentions_devtest.jsonl, pseudo_queries_devtest.jsonl

결과 (v2/outputs/, 깃 제외)
  pseudo_holdout.jsonl   채점 형식 {"id", "entities": [{label, start, end, token}]} (원문 기준 위치)
  mentions_holdout.jsonl 문서별 가명 본문 + 언급 전체 (raw·normalized 포함 → vault 로 갈 값, 로컬 전용)
  pseudo_queries.jsonl   질의별 가명 질의 + 토큰 + 언급 (원문 값 제외)
  pseudo_timing.json     처리 시간
"""
import json
import statistics
import sys
import time
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
import config  # noqa: E402
from pseudo import pipeline  # noqa: E402

OUT = V2 / "outputs"


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=("holdout", "devtest"), default="holdout")
    ap.add_argument("--qset", default="devtest", help="devtest 코퍼스의 질의 세트 이름 (eval_sets/<qset>_queries.jsonl)")
    ap.add_argument("--queries-only", action="store_true", help="문서는 건너뛰고 질의만 가명처리")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    pipeline.detector()
    load_sec = time.perf_counter() - t0
    print(f"탐지기 로딩 {load_sec:.1f}초")

    if a.corpus == "holdout":
        docs = jl(config.DATA_DIR / "holdout_labeled.jsonl")
        names = ("pseudo_holdout.jsonl", "mentions_holdout.jsonl", "pseudo_queries.jsonl", "pseudo_timing.json")
        queries = jl(config.DATA_DIR / "holdout_retrieval_queries.jsonl")
    else:
        docs = jl(config.DATA_DIR / "pii_synth_finance_dev.jsonl") + jl(config.DATA_DIR / "pii_synth_finance_test.jsonl")
        names = ("pseudo_devtest.jsonl", "mentions_devtest.jsonl", f"pseudo_queries_{a.qset}.jsonl",
                 f"pseudo_timing_{a.qset}.json")
        qpath = V2 / "eval_sets" / f"{a.qset}_queries.jsonl"
        queries = jl(qpath) if qpath.exists() else []
    if a.queries_only:
        docs = []
    doc_ms = [0.0]
    with open(OUT / (names[0] if docs else "_skip.jsonl"), "w", encoding="utf-8") as fp, \
         open(OUT / (names[1] if docs else "_skip2.jsonl"), "w", encoding="utf-8") as fm:
        for d in docs:                                   # 정답 라벨(entities)은 읽지 않음
            t = time.perf_counter()
            r = pipeline.process(d["text"])
            doc_ms.append((time.perf_counter() - t) * 1000)
            fp.write(json.dumps({"id": d["id"], "entities": [
                {"label": m["label"], "start": m["o_start"], "end": m["o_end"], "token": m["token"]}
                for m in r["mentions"]]}, ensure_ascii=False) + "\n")
            fm.write(json.dumps({"id": d["id"], "doc_type": d["doc_type"], "masked": r["masked"],
                                 "mentions": r["mentions"]}, ensure_ascii=False) + "\n")
    print(f"문서 {len(docs)}건 처리")

    q_ms = [0.0]
    with open(OUT / names[2], "w", encoding="utf-8") as fq:
        for q in queries:
            t = time.perf_counter()
            r = pipeline.process(q["query"])
            q_ms.append((time.perf_counter() - t) * 1000)
            ments = [{k: v for k, v in m.items() if k not in ("raw", "normalized")} for m in r["mentions"]]
            fq.write(json.dumps({"qid": q["qid"], "masked_query": r["masked"],
                                 "query_tokens": sorted({m["token"] for m in ments}), "mentions": ments},
                                ensure_ascii=False) + "\n")
    print(f"질의 {len(queries)}개 처리")

    p95 = lambda xs: sorted(xs)[max(int(len(xs) * 0.95) - 1, 0)]  # noqa: E731
    timing = {"detector_load_sec": round(load_sec, 1),
              "doc_ms_mean": round(statistics.mean(doc_ms), 1), "doc_ms_p95": round(p95(doc_ms), 1),
              "query_ms_mean": round(statistics.mean(q_ms), 1), "query_ms_p95": round(p95(q_ms), 1)}
    (OUT / names[3]).write_text(json.dumps(timing, indent=1), encoding="utf-8")
    print(timing)


if __name__ == "__main__":
    main()
