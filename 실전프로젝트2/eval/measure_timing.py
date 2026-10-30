"""처리 시간을 단계별로 따로 측정합니다 (탐지 · 가명처리 · 임베딩 API · 검색 · 답변 LLM).

    python -X utf8 eval/measure_timing.py            (실전프로젝트2 폴더에서, DB 적재 후)
    python -X utf8 eval/measure_timing.py --no-api   (API 호출 단계 제외)

결과: outputs/timing.json. 다른 작업을 돌리지 않은 상태에서 측정하세요.
API 단계에는 호출 한도(분당 60회)를 지키기 위한 대기 1.05초가 포함되지 않도록, 호출 전 대기와 호출 자체를 나눠 기록합니다.
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from starter import config, db, embed as embed_mod, pii_pipeline, search as search_mod  # noqa: E402


def stats(ms):
    ms = sorted(ms)
    return {"n": len(ms), "mean_ms": round(statistics.mean(ms), 1), "p50_ms": round(ms[len(ms) // 2], 1),
            "p95_ms": round(ms[min(len(ms) - 1, int(len(ms) * 0.95))], 1), "max_ms": round(ms[-1], 1)}


def timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return out, (time.perf_counter() - t) * 1000


def env_info():
    info = {"os": platform.platform(), "python": platform.python_version(), "cpu": platform.processor(),
            "cpu_count": os.cpu_count(), "embed_model": config.EMBED_MODEL, "llm_model": config.LLM_MODEL}
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
    except ImportError:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--api-n", type=int, default=10, help="API 단계 측정 횟수")
    ap.add_argument("--chat-n", type=int, default=3)
    a = ap.parse_args()

    docs = [json.loads(l) for l in open(config.DATA_DIR / "holdout_labeled.jsonl", encoding="utf-8")]
    queries = [json.loads(l)["query"] for l in open(config.DATA_DIR / "holdout_retrieval_queries.jsonl", encoding="utf-8")]
    res = {"env": env_info()}

    _, res["detector_load_ms"] = timed(pii_pipeline._get_detector)
    pii_pipeline.detect("워밍업 문장 윤선영 01012345678")  # 첫 호출 지연 제외

    det, pse = [], []
    for d in docs:
        spans, t1 = timed(pii_pipeline.detect, d["text"])
        _, t2 = timed(pii_pipeline.pseudonymize, d["text"], spans)
        det.append(t1)
        pse.append(t2)
    res["doc_detect"], res["doc_pseudonymize"] = stats(det), stats(pse)
    res["doc_chars_mean"] = round(statistics.mean(len(d["text"]) for d in docs), 1)

    qdet = [timed(search_mod.mask_query, q)[1] for q in queries]
    res["query_detect_pseudonymize"] = stats(qdet)

    with db.rag_conn() as conn:
        tok = [timed(search_mod.search, q, 10, conn, "token")[1] for q in queries]
        res["search_token_only_88"] = stats(tok)  # API 없음: 질의 가명처리 + DB

        if not a.no_api:
            emb = []
            for q in queries[:a.api_n]:
                masked, _ = search_mod.mask_query(q)
                time.sleep(max(0.0, embed_mod._last_call + embed_mod._MIN_INTERVAL - time.monotonic()))
                _, t = timed(embed_mod.embed, [masked])
                emb.append(t)
            res["embed_api_call"] = stats(emb)

            hyb = []
            for q in queries[:a.api_n]:
                time.sleep(max(0.0, embed_mod._last_call + embed_mod._MIN_INTERVAL - time.monotonic()))
                hyb.append(timed(search_mod.search, q, 10, conn, "hybrid")[1])
            res["search_hybrid"] = stats(hyb)

    if not a.no_api and config.LLM_API_KEY:
        from starter.chat import answer
        cq = ["보이스피싱 피해 접수 기록 알려줘", "한별 위임장", "윤 선영 님 한도 상향 심사 결과 알려줘"][:a.chat_n]
        res["chat_answer_total"] = stats([timed(answer, q, 5)[1] for q in cq])  # 검색 + LLM

    config.OUT_DIR.mkdir(exist_ok=True)
    out = config.OUT_DIR / "timing.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
