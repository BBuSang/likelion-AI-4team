"""동명이인 쌍별로 언급이 어떤 토큰·판단 근거(basis)로 갔는지 본다 (정답 인물 ID 는 분석에만). worklog 16장.

    python tools/homonym_cases.py synth2
출력: 쌍마다 사람별 {토큰: 언급 수}, 근거 분포, 같은 토큰을 나눠 가진 경우(합쳐짐)
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from starter import pii_pipeline, person_resolve  # noqa: E402

SPLITS = {"synth2": "data/synth2/synth2_labeled.jsonl", "synth3": "data/synth3/synth3_labeled.jsonl"}


def main():
    split = sys.argv[1]
    path = ROOT / SPLITS[split]
    docs = [json.loads(l) for l in open(path, encoding="utf-8")]
    cache = {json.loads(l)["id"]: json.loads(l)["spans"]
             for l in open(ROOT / f"outputs/detect_{path.stem}.jsonl", encoding="utf-8")}
    spans, _ = person_resolve.resolve(docs, {d["id"]: cache[d["id"]] for d in docs})
    pairs = json.load(open(path.parent / "_work/personas.json", encoding="utf-8"))["homonyms"]
    roles = {p["pid"]: p["role"] for p in json.load(open(path.parent / "_work/personas.json", encoding="utf-8"))["people"]}
    by_pid = collections.defaultdict(list)  # 정답 인물 → (문서, 토큰, 근거, 역할 판단)
    for d in docs:
        _, recs = pii_pipeline.pseudonymize(d["text"], spans[d["id"]])
        for e in d["entities"]:
            if e["label"] != "person":
                continue
            r = next((r for r in recs if r["label"] == "person" and r["start"] < e["end"] and e["start"] < r["end"]), None)
            by_pid[e["person_id"]].append((d["id"], r["token"] if r else None, r.get("basis") if r else "미탐",
                                           r.get("role") if r else None))
    total = collections.Counter()
    for a, b in pairs:
        ta = collections.Counter(t for _, t, _, _ in by_pid[a])
        tb = collections.Counter(t for _, t, _, _ in by_pid[b])
        shared = set(ta) & set(tb)
        kind = f"{roles[a]}↔{roles[b]}"
        total[kind, "merged" if shared else "separated"] += 1
        print(f"== {a}({roles[a]}) / {b}({roles[b]})  {'합쳐짐 ' + str(len(shared)) + '토큰' if shared else '구분됨'}")
        for pid, tc in ((a, ta), (b, tb)):
            basis = collections.Counter(x[2] for x in by_pid[pid])
            role = collections.Counter(x[3] for x in by_pid[pid])
            print(f"   {pid}: 언급 {sum(tc.values())}, 토큰 {len(tc)}개 {dict(tc.most_common(3))}")
            print(f"        근거 {dict(basis)}  역할 판단 {dict(role)}")
    print(dict(total))


if __name__ == "__main__":
    main()
