"""직원 목록 + 문서 메타데이터 만들기 (app/org/infer.py) 와 확인.

    python scripts/build_org.py holdout            # 2주차 탐지 캐시 사용
    python scripts/build_org.py synth2 --sample 15 # 문서 15건 담당자·팀을 눈으로 확인

결과: outputs/org_<split>/employees.jsonl, doc_meta.jsonl (이름 원문이 들어 있음 → outputs 는 git 제외)
채점 (담당자·팀 정답은 없음 → 사람 묶기만 2주차 인물 정답으로 확인, 정답은 채점에만 씀)
  - 합쳐짐: 직원 1명에 정답 인물이 2명 이상 (다른 사람을 한 직원으로)
  - 쪼개짐: 정답 인물 1명이 직원 2명 이상으로
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
W2 = ROOT.parents[1] / "실전프로젝트2"

from app.core import config  # noqa: E402
from app.org import infer  # noqa: E402

SPLITS = {
    "dev": ("data/week1/dev_docs.jsonl", "data/week1/dev_persons_gold.jsonl", "detect_dev_docs.jsonl"),
    "test": ("data/week1/test_docs.jsonl", "data/week1/test_persons_gold.jsonl", "detect_test_docs.jsonl"),
    "holdout": ("data/holdout_labeled.jsonl", "data/holdout_persons_gold.jsonl", "detect_holdout_labeled.jsonl"),
    "synth2": ("data/synth2/synth2_labeled.jsonl", "data/synth2/synth2_persons_gold.jsonl", "detect_synth2_labeled.jsonl"),
    "synth3": ("data/synth3/synth3_labeled.jsonl", "data/synth3/synth3_persons_gold.jsonl", "detect_synth3_labeled.jsonl"),
}


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=SPLITS)
    ap.add_argument("--sample", type=int, default=0, help="담당자·팀 판단을 눈으로 볼 문서 수")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    data, gold_path, cache = SPLITS[a.split]
    docs = load(W2 / data)
    spans = {r["id"]: r["spans"] for r in load(W2 / "outputs" / cache)}
    employees, meta, mentions, emp_key, emp_id = infer.build(docs, spans)

    out = config.OUT_DIR / f"org_{a.split}"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("employees", employees), ("doc_meta", meta)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(docs)
    vis = collections.Counter(m["visibility"] for m in meta)
    ob = collections.Counter(m["basis"]["owner"] for m in meta)
    tb = collections.Counter(m["basis"]["team"] for m in meta)
    print(f"[{a.split}] 문서 {n}건 → 직원 {len(employees)}명 (팀 앎 {sum(e['team'] is not None for e in employees)}), "
          f"역할 {dict(collections.Counter(e['role'] for e in employees))}")
    print(f"  담당자 있음 {sum(m['owner_emp_id'] is not None for m in meta)}/{n}  근거 {dict(ob.most_common())}")
    print(f"  팀 있음     {sum(m['team'] is not None for m in meta)}/{n}  근거 {dict(tb.most_common())}")
    print(f"  공개 범위   {dict(vis.most_common())}")
    print(f"  팀별 직원   {dict(collections.Counter(e['team'] for e in employees).most_common())}")

    # 문서 종류별: 팀 못 정한 비율
    by_type = collections.defaultdict(lambda: [0, 0])
    for d, m in zip(docs, meta):
        by_type[d.get("doc_type")][0] += 1
        by_type[d.get("doc_type")][1] += m["team"] is None
    worst = sorted(((t, x, y) for t, (y, x) in by_type.items() if x), key=lambda r: -r[1])[:8]
    print("  팀 모름이 많은 종류:", ", ".join(f"{t} {x}/{y}" for t, x, y in worst))

    # 채점: 직원 ↔ 정답 인물
    gold = {r["id"]: r["persons"] for r in load(W2 / gold_path)}
    emp_pids, pid_emps = collections.defaultdict(set), collections.defaultdict(set)
    unmatched = 0
    for doc_id, ms in mentions.items():
        for m in ms:
            hit = [g["person_id"] for g in gold.get(doc_id, []) if g["start"] < m["end"] and m["start"] < g["end"]]
            if not hit:
                unmatched += 1  # 정답에 사람이 아닌데 탐지기가 이름으로 잡은 것 등
                continue
            e = emp_id[emp_key[(doc_id, m["start"])]]
            emp_pids[e].add(hit[0])
            pid_emps[hit[0]].add(e)
    merged = {e: p for e, p in emp_pids.items() if len(p) > 1}
    split = {p: e for p, e in pid_emps.items() if len(e) > 1}
    print(f"  채점: 직원 언급 {sum(len(v) for v in mentions.values())}회 (정답 인물 없음 {unmatched}), "
          f"합쳐짐 {len(merged)}명, 쪼개짐 {len(split)}명 / 정답 인물 {len(pid_emps)}명")
    by_id = {e["emp_id"]: e for e in employees}
    for e, p in list(merged.items())[:5]:
        print(f"    합쳐짐 {e} {by_id[e]['name']}·{by_id[e]['team']} ← {sorted(p)}")
    for p, es in list(split.items())[:5]:
        print(f"    쪼개짐 {p} → " + ", ".join(f"{x}({by_id[x]['team']})" for x in sorted(es)))

    if a.sample:
        import random
        print("\n--- 표본 (담당자·팀 판단 눈으로 확인)")
        for d, m in random.Random(a.seed).sample(list(zip(docs, meta)), a.sample):
            o = by_id.get(m["owner_emp_id"])
            head = " / ".join(l.strip() for l in d["text"].strip().split("\n")[:3])[:110]
            print(f"[{d['id']} {d.get('doc_type')}] {head}")
            print(f"    → 담당 {o['name'] + ' ' + (o['position'] or '') if o else '-'} ({m['basis']['owner']}), "
                  f"팀 {m['team']} ({m['basis']['team']}), {m['visibility']}")


if __name__ == "__main__":
    main()
