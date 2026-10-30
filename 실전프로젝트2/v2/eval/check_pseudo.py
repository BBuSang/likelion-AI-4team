"""v2 가명처리 결과 점검 (run_pseudo.py 다음에 실행). 정답은 채점에만 사용.

    python -X utf8 v2/eval/check_pseudo.py

1. E1 토큰 일관성: 강사 키트 채점 도구(eval/check_token_consistency.py) 그대로
2. 유출·과잉 마스킹: 정답 개체 대비 (문서·질의)
3. 이름 형태(form) 정확도: 정답 form 과 비교
4. 주소 일반화: 시·도 / 시·군·구를 뽑은 비율
5. 역할: holdout 분포 + dev/test 정답 대비 정확도 (탐지기 위치 기준)
"""
import ast
import collections
import json
import subprocess
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
sys.path.insert(0, str(V2))
from pseudo import pipeline  # noqa: E402

OUT = V2 / "outputs"
ROLE_MAP = {"C": "customer", "V": "external", "VC": "external",
            **{c: "staff" for c in ("A", "M", "B", "D", "E", "E1", "E2", "E3", "E4", "H")}}


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def P(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def coverage(gold_items, pred_items):
    """gold_items/pred_items: [(id, [spans])] → 미탐·과잉·경계 차이."""
    pred = dict(pred_items)
    miss, over, boundary, n_gold = [], [], 0, 0
    for i, gs in gold_items:
        ps = pred.get(i, [])
        for g in gs:
            n_gold += 1
            hit = [p for p in ps if overlap(g, p)]
            if not hit:
                miss.append((i, g["label"], g["text"]))
            elif not any(p["start"] == g["start"] and p["end"] == g["end"] for p in hit):
                boundary += 1
        for p in ps:
            if not any(overlap(g, p) for g in gs):
                over.append((i, p["label"], p.get("text", "")))
    return {"gold": n_gold, "miss": len(miss), "over": len(over), "boundary": boundary,
            "miss_cases": miss, "over_cases": over}


def main():
    result = {}

    # 1. E1
    tc = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "eval/check_token_consistency.py"),
                         "--gold", str(ROOT / "data/holdout_persons_gold.jsonl"),
                         "--pred", str(OUT / "pseudo_holdout.jsonl"), "--out", str(OUT / "tc_v2.csv")],
                        capture_output=True, text=True, encoding="utf-8")
    result["E1"] = tc.stdout.strip()
    print("── E1 토큰 일관성\n" + tc.stdout)

    # 2. 유출·과잉
    docs = {d["id"]: d for d in jl(ROOT / "data/holdout_labeled.jsonl")}
    ments = {d["id"]: d for d in jl(OUT / "mentions_holdout.jsonl")}
    doc_cov = coverage([(i, d["entities"]) for i, d in docs.items()],
                       [(i, [{"label": m["label"], "start": m["o_start"], "end": m["o_end"], "text": m["raw"]}
                             for m in ments[i]["mentions"]]) for i in docs])
    qgold = jl(ROOT / "data/holdout_retrieval_queries_gold.jsonl")
    qpred = {q["qid"]: q for q in jl(OUT / "pseudo_queries.jsonl")}
    qtext = {q["qid"]: q["query"] for q in qgold}
    q_cov = coverage([(q["qid"], [{**e, "text": e["text"]} for e in q.get("query_entities", [])]) for q in qgold],
                     [(i, [{"label": m["label"], "start": m["o_start"], "end": m["o_end"],
                            "text": qtext[i][m["o_start"]:m["o_end"]]} for m in q["mentions"]]) for i, q in qpred.items()])
    result["leak_docs"], result["leak_queries"] = doc_cov, q_cov
    for name, c in (("문서", doc_cov), ("질의", q_cov)):
        print(f"── {name}: 정답 개체 {c['gold']} · 미탐(원문 노출) {c['miss']} · 과잉 마스킹 {c['over']} · 경계 차이 {c['boundary']}")
        for x in c["miss_cases"]:
            print("   미탐", x)
        for x in collections.Counter((l, t) for _, l, t in c["over_cases"]).most_common(10):
            print("   과잉", x)
    # 가명 본문에 정답 원문 문자열이 그대로 남았는지 (문자열 기준 이중 확인)
    left = [(i, e["text"]) for i, d in docs.items() for e in d["entities"] if e["text"] in ments[i]["masked"]]
    result["raw_string_left"] = left
    print(f"   가명 본문에 정답 원문 문자열이 남은 곳: {len(left)} {left[:5]}")

    # 3. 이름 형태
    pg = {d["id"]: d["persons"] for d in jl(ROOT / "data/holdout_persons_gold.jsonl")}
    form = collections.Counter()
    for i, ps in pg.items():
        by = {(m["o_start"], m["o_end"]): m for m in ments[i]["mentions"] if m["label"] == "person"}
        for p in ps:
            m = by.get((p["start"], p["end"]))
            if m:
                form[(p["form"], m["form"])] += 1
    right = sum(v for (g, p), v in form.items() if g == p)
    result["form"] = {"accuracy": right / sum(form.values()), "confusion": {f"{g}->{p}": v for (g, p), v in form.items()}}
    print(f"── 이름 형태 정확도 {result['form']['accuracy']:.3f}  {result['form']['confusion']}")

    # 4. 주소 일반화
    addr = [m for d in ments.values() for m in d["mentions"] if m["label"] == "address"]
    result["region"] = {"address": len(addr), "sido": sum(1 for m in addr if m["sido"]),
                        "sigungu": sum(1 for m in addr if m["sigungu"])}
    print(f"── 주소 일반화 {result['region']}")
    for m in addr:
        if not m["sigungu"]:
            print("   시·군·구 없음:", m["normalized"][:30])

    # 5. 역할
    hold = collections.Counter(m["role"] for d in ments.values() for m in d["mentions"] if m["label"] == "person")
    result["roles_holdout"] = dict(hold)
    print(f"── holdout 역할 분포 {dict(hold)}")
    for split in ("dev", "test"):
        c, person = collections.Counter(), collections.Counter()
        for d in jl(ROOT / f"data/pii_synth_finance_{split}.jsonl"):
            gold = {pid: ROLE_MAP[k] for k, pid in P(d["roles"]).items() if k in ROLE_MAP}
            r = pipeline.process(d["text"])
            seen = {}
            for g in P(d["entities"]):
                if g["label"] != "person" or g.get("person_id") not in gold:
                    continue
                hit = [m for m in r["mentions"] if m["label"] == "person"
                       and m["o_start"] < g["end"] and g["start"] < m["o_end"]]
                role = hit[0]["role"] if hit else "미탐"
                c[(gold[g["person_id"]], role)] += 1
                seen[g["person_id"]] = role
            for pid, role in seen.items():
                person[(gold[pid], role)] += 1
        n = sum(person.values())
        pred = sum(v for (g, r), v in person.items() if r not in (None, "미탐"))
        right = sum(v for (g, r), v in person.items() if g == r)
        result[f"roles_{split}"] = {"n": n, "coverage": pred / n, "precision": right / pred, "accuracy": right / n,
                                    "confusion": {f"{g}->{r}": v for (g, r), v in person.items()}}
        s = result[f"roles_{split}"]
        print(f"── 역할 {split} (탐지기 위치, 문서별 인물 {n}) 붙임 {s['coverage']:.3f} · 붙인 것 중 정확 "
              f"{s['precision']:.3f} · 전체 정확 {s['accuracy']:.3f}  {s['confusion']}")

    for c in (doc_cov, q_cov):
        c.pop("over_cases")
    (OUT / "check_pseudo.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
