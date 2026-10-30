"""개체 묶기 일반화 확인: dev/test 문서를 같은 파이프라인(우리 탐지기 → 가명처리 → 역할 → 규칙 개체 묶기)으로
처리하고 정답 인물 ID로 채점. DB에 넣지 않고 메모리에서만 처리 (holdout DB 그대로).

    python -X utf8 v2/eval/eval_entities_split.py --split test

규칙은 holdout·dev를 보며 만들었고 test는 보지 않았으므로 test 결과가 "새 파일" 성능에 가까움.
"""
import argparse
import ast
import collections
import json
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
import config  # noqa: E402
from entity.resolve import profiles, rule_clusters  # noqa: E402
from pseudo import pipeline  # noqa: E402


def P(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    docs = [json.loads(l) for l in open(config.DATA_DIR / f"pii_synth_finance_{a.split}.jsonl", encoding="utf-8")]
    rows, gold = [], {}
    mid = 0
    for d in docs:
        r = pipeline.process(d["text"])                       # 정답 entities 는 채점에만
        for m in r["mentions"]:
            mid += 1
            rows.append((mid, d["id"], d["id"] + "#d", m["token"], m["label"], m["line_no"], m["role"], m["org"],
                         m["form"], m["own_role"]))
            for g in P(d["entities"]):
                if g["label"] == "person" and g.get("person_id") and g["start"] < m["o_end"] and m["o_start"] < g["end"]:
                    gold[mid] = g["person_id"]
    prof = profiles(rows)
    ent = {}
    for t, by in prof.items():
        for n, c in enumerate(rule_clusters(by)):
            for dd in c["docs"]:
                ent[(t, dd)] = n

    items = collections.defaultdict(lambda: collections.defaultdict(set))   # token → doc → {person}
    for mid_, d, _, t, lab, *_ in rows:
        if lab == "person" and mid_ in gold:
            items[t][d].add(gold[mid_])
    pairs, hom, split = collections.Counter(), [], 0
    for t, by in items.items():
        ds = list(by)
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                pairs[(bool(by[ds[i]] & by[ds[j]]), ent[(t, ds[i])] == ent[(t, ds[j])])] += 1
        persons = set().union(*by.values())
        ents = {ent[(t, d)] for d in ds}
        if len(persons) > 1:
            pure = all(len(set().union(*[by[d] for d in ds if ent[(t, d)] == e])) == 1 for e in ents)
            hom.append({"persons": len(persons), "docs": len(ds), "entities": len(ents), "pure": pure})
        elif len(ents) > 1:
            split += 1
    tp, fp, fn = pairs[(True, True)], pairs[(False, True)], pairs[(True, False)]
    r = {"split": a.split, "docs": len(docs), "name_tokens": len(items),
         "pair_precision": round(tp / (tp + fp), 3) if tp + fp else None,
         "pair_recall": round(tp / (tp + fn), 3) if tp + fn else None, "wrong_merge_pairs": fp,
         "homonym_tokens": len(hom), "homonym_pure": sum(h["pure"] for h in hom), "homonym": hom,
         "single_person_tokens_split": split, "single_person_tokens": len(items) - len(hom)}
    print(json.dumps(r, ensure_ascii=False))
    (V2 / "outputs" / f"entities_{a.split}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
