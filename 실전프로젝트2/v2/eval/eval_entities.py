"""개체 묶기 채점 (rag.mentions.entity_id vs 정답 인물 ID). 정답은 채점에만.

    python -X utf8 v2/eval/eval_entities.py [--tokens <PERSON_…> ...] [--save name]

원문 위치는 vault.mention_raw 에서 읽어 holdout_persons_gold 와 맞춤 (채점 전용 접근).
- 쌍 정밀도: 같은 개체로 묶은 (토큰, 문서) 쌍 중 실제 같은 사람
- 쌍 재현율: 실제 같은 사람인 쌍 중 같은 개체로 묶은 비율
- 동명이인 토큰: 정답 인물 수 vs 개체 수, 완전 분리 여부 (개체마다 한 사람만, 사람마다 한 개체만)
"""
import argparse
import collections
import json
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
import config  # noqa: E402
import db  # noqa: E402


def evaluate(tokens=None):
    with db.ingest_conn() as conn:
        ments = conn.execute("SELECT mention_id, doc_id, token, entity_id FROM rag.mentions "
                             "WHERE label = 'person'").fetchall()
    with db.vault_conn() as conn:
        pos = {m: (o_s, o_e) for m, o_s, o_e in conn.execute("SELECT mention_id, o_start, o_end FROM vault.mention_raw")}
    gold = {}
    for line in open(config.DATA_DIR / "holdout_persons_gold.jsonl", encoding="utf-8"):
        d = json.loads(line)
        for p in d["persons"]:
            gold[(d["id"], p["start"], p["end"])] = p["person_id"]
    items = collections.defaultdict(dict)        # token → doc → (entity, {person_id})
    for mid, d, t, e in ments:
        if tokens and t not in tokens:
            continue
        pid = gold.get((d, *pos[mid]))
        ent, pids = items[t].get(d, (e, set()))
        if pid:
            pids.add(pid)
        items[t][d] = (e, pids)
    pairs, hom, wrong_split = collections.Counter(), [], 0
    for t, by in items.items():
        ds = [d for d in by if by[d][1]]
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                pairs[(bool(by[ds[i]][1] & by[ds[j]][1]), by[ds[i]][0] == by[ds[j]][0])] += 1
        persons = set().union(*[by[d][1] for d in ds]) if ds else set()
        ents = {by[d][0] for d in ds}
        if len(persons) > 1:
            pure = all(len(set().union(*[by[d][1] for d in ds if by[d][0] == e])) == 1 for e in ents)
            single = all(len({by[d][0] for d in ds if p in by[d][1]}) == 1 for p in persons)
            hom.append({"token": t, "persons": len(persons), "entities": len(ents), "pure": pure, "separated": pure and single})
        elif len(ents) > 1:
            wrong_split += 1
    tp, fp, fn = pairs[(True, True)], pairs[(False, True)], pairs[(True, False)]
    return {"pair_precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "pair_recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "wrong_merge_pairs": fp, "missed_pairs": fn,
            "homonym": hom, "homonym_pure": sum(h["pure"] for h in hom), "homonym_separated": sum(h["separated"] for h in hom),
            "single_person_tokens_split": wrong_split}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", nargs="*")
    ap.add_argument("--save")
    a = ap.parse_args()
    r = evaluate(a.tokens)
    print(json.dumps(r, ensure_ascii=False))
    if a.save:
        (V2 / "outputs" / f"entities_{a.save}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
