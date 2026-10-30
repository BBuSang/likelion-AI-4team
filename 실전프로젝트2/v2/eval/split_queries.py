"""새 질의 세트를 조정용(tune)·평가용(eval)으로 나눔. 유형별 층화, seed 고정.
질의 문장은 출력하지 않음 (평가용을 미리 보지 않기 위해). 결과: v2/eval_sets/<qset>_split.json

    python -X utf8 v2/eval/split_queries.py --qset devtest2 --ratio 0.8  devtest2 100개, 8:2

주의: devtest(40개)의 devtest_split.json 은 이전 방식(유형 사이까지 번갈아 배정)으로 만든 것이라
이 스크립트로 다시 만들면 구분이 바뀜. 이미 쓴 구분이므로 다시 실행하지 말 것 (worklog 46번)
"""
import argparse
import collections
import json
import random
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
SEED = 20261030


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qset", required=True)
    ap.add_argument("--ratio", type=float, default=0.5, help="조정용 비율")
    a = ap.parse_args()
    gold = [json.loads(l) for l in open(V2 / "eval_sets" / f"{a.qset}_queries_gold.jsonl", encoding="utf-8") if l.strip()]
    by_type = collections.defaultdict(list)
    for g in gold:
        by_type[g["type"]].append(g["qid"])
    rng = random.Random(SEED)
    tune, ev, acc = [], [], 0.0
    for t in sorted(by_type):
        qids = sorted(by_type[t])
        rng.shuffle(qids)
        for q in qids:                      # 누적 비율을 맞추며 배정 → 유형마다 비율이 고르게
            acc += a.ratio
            if acc >= 1:
                tune.append(q)
                acc -= 1
            else:
                ev.append(q)
    (V2 / "eval_sets" / f"{a.qset}_split.json").write_text(
        json.dumps({"seed": SEED, "tune": sorted(tune), "eval": sorted(ev)}, indent=1), encoding="utf-8")
    print(f"조정용 {len(tune)} · 평가용 {len(ev)}")
    print("유형별 (조정/평가):", {t: (sum(q in tune for q in v), sum(q in ev for q in v)) for t, v in sorted(by_type.items())})


if __name__ == "__main__":
    main()
