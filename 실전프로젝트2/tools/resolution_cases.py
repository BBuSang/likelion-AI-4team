"""eval_resolution.py 결과에서 쪼개짐·합쳐짐 사례를 판단 근거(basis)와 함께 보여준다 (정답 인물 ID 로 채점·분석만).

    python tools/resolution_cases.py dev
"""
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = {"dev": "data/week1/dev_persons_gold.jsonl", "test": "data/week1/test_persons_gold.jsonl",
        "holdout": "data/holdout_persons_gold.jsonl"}


def main():
    split = sys.argv[1]
    gold = {}
    for l in open(ROOT / GOLD[split], encoding="utf-8"):
        r = json.loads(l)
        gold[r["id"]] = r["persons"]
    pred = {}
    for l in open(ROOT / f"outputs/pseudo_resolve_{split}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        pred[r["id"]] = r["entities"]
    info = collections.defaultdict(list)
    for r in csv.DictReader(open(ROOT / f"outputs/resolution_{split}.csv", encoding="utf-8-sig")):
        info[(r["doc_id"], r["token"])].append(r)

    per = collections.defaultdict(lambda: collections.defaultdict(list))
    tok_persons = collections.defaultdict(set)
    for doc, ps in gold.items():
        for m in ps:
            for p in pred[doc]:
                if p["start"] < m["end"] and m["start"] < p["end"]:
                    r = info[(doc, p["token"])][0]
                    per[m["person_id"]][p["token"]].append(
                        (doc, r["original"], m["form"], r["role"], r["aff"], r["basis"], r["person_key"]))
                    tok_persons[p["token"]].add(m["person_id"])
    kinds = collections.Counter()
    for pid, toks in per.items():
        if len(toks) < 2:
            continue
        full = {t for t, l in toks.items() if any(x[2] != "given" for x in l)}
        kind = "이름만 쓴 표기" if len(full) <= 1 else "성 포함 이름끼리"
        kinds[kind] += 1
        if kind == "성 포함 이름끼리":
            print(f"쪼개짐 {pid}")
            for t, l in toks.items():
                print(f"   {t} ×{len(l)}  " + "  ".join(f"{x[0]}:{x[1]}/{x[3] or '-'}/{x[4] or '-'}/{x[5]}" for x in l[:4]))
    for t, ps in tok_persons.items():
        if len(ps) > 1:
            print(f"합쳐짐 {t} {sorted(ps)}  " + "  ".join(
                f"{x[0]}:{x[1]}/{x[3] or '-'}/{x[5]}" for p in sorted(ps) for x in per[p][t][:2]))
    print("쪼개짐 종류:", dict(kinds))


if __name__ == "__main__":
    main()
