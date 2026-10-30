"""역할 추출 정확도 (dev/test 정답 roles 기준).

정답 역할 코드 → 3종: C 고객 → customer / A·M·B·D·E·E1~E4·H 직원 → staff / V·VC 외부 → external
언급 위치는 정답 위치를 씀 (탐지 오류와 분리해 역할 추출만 측정). 정답 역할이 없는 인물은 채점에서 제외.

    python -X utf8 v2/eval/eval_roles.py --split dev [--show 30]
"""
import argparse
import ast
import collections
import json
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from pseudo.roles import extract  # noqa: E402

ROOT = V2.parent
MAP = {"C": "customer", "V": "external", "VC": "external",
       **{c: "staff" for c in ("A", "M", "B", "D", "E", "E1", "E2", "E3", "E4", "H")}}


def P(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def evaluate(split, show=0):
    docs = [json.loads(l) for l in open(ROOT / f"data/pii_synth_finance_{split}.jsonl", encoding="utf-8")]
    ment, person, errors = collections.Counter(), collections.Counter(), []
    for d in docs:
        gold = {pid: MAP[c] for c, pid in P(d["roles"]).items() if c in MAP}
        ents = [e for e in P(d["entities"]) if e["label"] == "person"]
        res = extract(d["text"], [{"start": e["start"], "end": e["end"], "key": e["text"].replace(" ", "")} for e in ents])
        seen = {}
        for e, r in zip(ents, res):
            g = gold.get(e.get("person_id"))
            if g is None:
                continue
            ment[(g, r["role"])] += 1
            seen[e["person_id"]] = r["role"]
            if r["role"] != g:
                errors.append((g, r["role"], d["doc_type"], d["text"][max(0, e["start"] - 14):e["end"] + 8].replace("\n", "⏎")))
        for pid, r in seen.items():
            person[(gold[pid], r)] += 1

    def summary(c):
        n = sum(c.values())
        pred = sum(v for (g, r), v in c.items() if r)
        right = sum(v for (g, r), v in c.items() if r == g)
        return {"n": n, "coverage": pred / n, "precision": right / pred if pred else 0, "accuracy": right / n}

    return {"mention": summary(ment), "doc_person": summary(person),
            "confusion": {f"{g}->{r}": v for (g, r), v in sorted(person.items(), key=lambda x: -x[1])},
            "errors": errors[:show]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    r = evaluate(a.split, a.show)
    for k in ("mention", "doc_person"):
        s = r[k]
        print(f"{k:<11} n={s['n']:<4} 역할 붙임 {s['coverage']:.3f}  붙인 것 중 정확 {s['precision']:.3f}  전체 정확 {s['accuracy']:.3f}")
    print("문서별 인물 (정답->예측):", r["confusion"])
    for e in r["errors"]:
        print("  ", e)


if __name__ == "__main__":
    main()
