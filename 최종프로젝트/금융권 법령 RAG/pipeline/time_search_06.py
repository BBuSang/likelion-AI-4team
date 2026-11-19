"""검색 구성별 단계 시간 (LLM 없음) + 저장된 측정 결과의 LLM 시간

    python pipeline/time_search_06.py [--sets dev7 dev8] [--repeat 1]

단계: ① 질문 임베딩(KURE-v1, CPU) ② DB 벡터 검색(pgvector HNSW, 조각 50 → 조문) ③ 연결 조문 붙이기(메모리) ④ 조문 글 읽기(DB)
구성: 검색만(1~5위) / ①만 1~10위 / ①+③ 1~5위+연결 / 합친 구성 / 합친 구성+조문 글(채택)
모델 로드는 미리 해서 시간에서 뺀다. 결과 → eval/timing_search.json
"""
import argparse
import csv
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retriever.search import _neighbors, attach_links, embed, search_vec, union_candidates, with_full_text  # noqa: E402

EVAL = ROOT / "eval"


def summ(v):
    v = sorted(v)
    return {"평균": round(st.mean(v) * 1000, 1), "중앙": round(v[len(v) // 2] * 1000, 1),
            "95%": round(v[max(int(len(v) * .95) - 1, 0)] * 1000, 1), "최대": round(v[-1] * 1000, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["dev7", "dev8"])
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    qs = [q["question"] for q in csv.DictReader(open(EVAL / "queries.csv", encoding="utf-8-sig")) if q["set"] in a.sets]
    t0 = time.perf_counter()
    search_vec(embed(["모델 미리 올리기"])[0], k=1)
    load = time.perf_counter() - t0
    t0 = time.perf_counter()
    _neighbors()
    links_load = time.perf_counter() - t0
    T = {k: [] for k in ("① 임베딩", "② DB 검색", "③ 연결 붙이기(①+③)", "③ 연결 붙이기(합친 구성)", "④ 조문 글 읽기")}
    for _ in range(a.repeat):
        for q in qs:
            t = time.perf_counter(); v = embed([q])[0]; T["① 임베딩"].append(time.perf_counter() - t)
            t = time.perf_counter(); arts = search_vec(v, k=0); T["② DB 검색"].append(time.perf_counter() - t)
            t = time.perf_counter(); top, extra = attach_links(arts); T["③ 연결 붙이기(①+③)"].append(time.perf_counter() - t)
            t = time.perf_counter(); u = union_candidates(arts); T["③ 연결 붙이기(합친 구성)"].append(time.perf_counter() - t)
            t = time.perf_counter(); with_full_text(u); T["④ 조문 글 읽기"].append(time.perf_counter() - t)
    S = {k: summ(v) for k, v in T.items()}
    n = len(T["① 임베딩"])
    base = [T["① 임베딩"][i] + T["② DB 검색"][i] for i in range(n)]
    cfg = {"검색만 (1~5위)": base, "①만 (1~10위)": base,
           "①+③ (1~5위 + 연결)": [base[i] + T["③ 연결 붙이기(①+③)"][i] for i in range(n)],
           "합친 구성 (1~10위 + 연결)": [base[i] + T["③ 연결 붙이기(합친 구성)"][i] for i in range(n)],
           "합친 구성 + 조문 글 (채택)": [base[i] + T["③ 연결 붙이기(합친 구성)"][i] + T["④ 조문 글 읽기"][i] for i in range(n)]}
    C = {k: summ(v) for k, v in cfg.items()}

    # 저장된 측정 결과의 LLM·전체 시간 (같은 문항, 추론 low)
    runs = {"①만 (1~10위)": "select_only1_low_ofit78.jsonl", "합친 구성 (1~10위 + 연결)": "select_union_low_ofit78.jsonl",
            "합친 구성 + 조문 글 (채택)": "select_union_low_full_ofit78.jsonl", "①+③ (1~5위 + 연결) [dev9·10]": "select_both_low_set910.jsonl",
            "채택 구성 test10 [최종]": "select_union_low_final10.jsonl"}
    LLM = {}
    for k, f in runs.items():
        p = EVAL / f
        if not p.exists():
            continue
        rs = [json.loads(l) for l in open(p, encoding="utf-8")]
        LLM[k] = {"문항": len(rs), "LLM 평균": round(st.mean(r["LLM 초"] for r in rs), 2),
                  "LLM 최대": round(max(r["LLM 초"] for r in rs), 2), "전체 평균": round(st.mean(r["전체 초"] for r in rs), 2),
                  "전체 최대": round(max(r["전체 초"] for r in rs), 2), "입력 글자 평균": round(st.mean(r["입력 글자"] for r in rs)),
                  "입력 토큰 평균": round(st.mean(r["토큰"].get("prompt_tokens", 0) for r in rs)),
                  "10초 초과": sum(r["전체 초"] > 10 for r in rs)}
    res = {"문항": n, "세트": a.sets, "준비(서버 시작 때 한 번)": {"모델 로드+첫 검색 초": round(load, 2), "연결 자료 읽기 초": round(links_load, 2)},
           "단계별 ms": S, "구성별 검색 단계 합 ms": C, "구성별 LLM·전체 초 (저장된 측정)": LLM}
    json.dump(res, open(EVAL / "timing_search.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
