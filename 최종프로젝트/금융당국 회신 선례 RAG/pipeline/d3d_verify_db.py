"""DB 재적재(D3 청크 + KURE-v1) 검증: 실제 `rag.search()`(로컬 KURE-v1 질문 임베딩 + DB 벡터 검색)의 순위를 numpy 정확 검색(d3b_eval.json의 kure-v1)과 비교한다.
차이가 날 수 있는 이유는 세 가지다: HNSW 근사, 질문 벡터의 정밀도(Colab fp16 vs 로컬 fp32), 상위 50조각을 묶음으로 합치는 방식(numpy는 전체 청크 기준).
대상: 합성 150, dev1, dev2 (테스트셋은 읽지 않는다). 사용: python d3d_verify_db.py     출력: data/eval/d3d_verify_db.json
"""
import json
import sys
import time

import rag
from common import DATA, manifest
from d2_baseline import SETS, load, metrics

sys.stdout.reconfigure(encoding="utf-8")
EVAL = DATA / "eval"
K = 10


def main():
    ref = json.load(open(EVAL / "d3b_eval.json", encoding="utf-8"))["detail"]["kure-v1"]
    t0 = time.time()
    rag.embed("모델 불러오기 시험")
    print(f"모델 로드 {time.time() - t0:.0f}초", flush=True)
    out, diffs, times = {}, [], []
    with rag.connect() as conn:
        rep = {r["representative_case_key"]: r["group_id"] for r in conn.execute("select group_id, representative_case_key from interp.case_groups").fetchall()}
        for name in SETS:
            rows, res = load(name), []
            for r, want in zip(rows, ref[name]):
                t = time.time()
                hits = rag.search(r["question"], K, conn)
                times.append(time.time() - t)
                groups = [rep[h["doc_id"]] for h in hits]
                rank = groups.index(r["gold_group_id"]) + 1 if r["gold_group_id"] in groups else None
                res.append({"qid": r["qid"], "rank": rank, "n_groups": len(hits)})
                if rank != want["rank"]:
                    diffs.append({"qid": r["qid"], "db_rank": rank, "numpy_rank": want["rank"], "numpy_rank_full": want["rank_full"], "n_groups": len(hits)})
            out[name] = {"db": metrics(res), "numpy": metrics(ref[name]), "min_groups": min(x["n_groups"] for x in res)}
            print(name, "DB", {k: out[name]["db"][k] for k in ("hit1", "hit5", "mrr10", "not_in_top10")}, "| numpy", {k: out[name]["numpy"][k] for k in ("hit1", "hit5", "mrr10", "not_in_top10")}, "| 최소 묶음 수", out[name]["min_groups"], flush=True)
    print(f"순위가 다른 문항 {len(diffs)}개 / {sum(len(load(n)) for n in SETS)}:", json.dumps(diffs, ensure_ascii=False))
    print(f"검색 1회(질문 임베딩 + DB) 중앙값 {sorted(times)[len(times) // 2]:.2f}초")
    with open(EVAL / "d3d_verify_db.json", "w", encoding="utf-8") as f:
        json.dump({"summary": out, "diffs": diffs, "search_seconds_median": sorted(times)[len(times) // 2]}, f, ensure_ascii=False, indent=1)
    manifest("d3d_verify_db", diffs=len(diffs))


if __name__ == "__main__":
    main()
