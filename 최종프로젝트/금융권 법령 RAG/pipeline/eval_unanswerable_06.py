"""기획안 5-8-1 답변불가셋과 기준점수

    python pipeline/eval_unanswerable_06.py --scores     # 답변 가능 dev1~10 500 + 답변불가 40 의 검색 1위 점수 (LLM 없음)
    python pipeline/eval_unanswerable_06.py --answer     # 답변불가 40 문항 answer() (LLM 40회, 이어서 실행 가능)
    python pipeline/eval_unanswerable_06.py --report     # 기준점수 정하기(dev1~5 + na_dev) → 확인(dev6~10 + na_test)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retriever.search import embed, search_vec  # noqa: E402

EVAL = ROOT / "eval"
NA = EVAL / "unanswerable.csv"
SCORES = EVAL / "na_scores.jsonl"
ANSWERS = EVAL / "na_answers.jsonl"
MARGIN = 0.01  # 기획안 5-8-1: t = 답변 가능 dev1~5 1위 점수 최솟값 − 0.01
PICK_SETS = [f"dev{i}" for i in range(1, 6)]
CHECK_SETS = [f"dev{i}" for i in range(6, 11)]


def load_na():
    return list(csv.DictReader(open(NA, encoding="utf-8-sig")))


def scores():
    qs = [q for q in csv.DictReader(open(EVAL / "queries.csv", encoding="utf-8-sig")) if q["split"] == "dev"]
    items = [(q["qid"], q["set"], "답변 가능", q["question"]) for q in qs] + [(q["qid"], q["set"], q["category"], q["question"]) for q in load_na()]
    vecs = embed([x[3] for x in items[:1]])  # 모델 올리기
    with open(SCORES, "w", encoding="utf-8") as f:
        for n in range(0, len(items), 32):
            batch = items[n:n + 32]
            vecs = embed([x[3] for x in batch])
            for (qid, st, cat, q), v in zip(batch, vecs):
                arts = search_vec(v, k=3)
                f.write(json.dumps({"qid": qid, "set": st, "category": cat, "top1": round(arts[0]["score"], 4),
                                    "top1_title": arts[0]["title"], "top2": round(arts[1]["score"], 4)}, ensure_ascii=False) + "\n")
    print(SCORES.name, len(items))


def answers_path(prompt):
    """5-8-1 의 첫 결과(기준점수 넣기 전, 지시문 v1)는 na_answers.jsonl. 5-8-2 부터는 na_answers_<지시문>.jsonl (기준점수 거침)"""
    return ANSWERS if not prompt else EVAL / f"na_answers_{prompt}.jsonl"


def answers(prompt=None, sets=None):
    from retriever.answer import answer, warmup
    out = answers_path(prompt)
    done = {json.loads(l)["qid"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    todo = [q for q in load_na() if q["qid"] not in done and (not sets or q["set"] in sets)]
    print(f"{out.name}: 남은 {len(todo)} (이미 {len(done)}) · 준비 {warmup()}초", flush=True)
    with open(out, "a", encoding="utf-8") as f:
        for q in todo:
            r = answer(q["question"], prompt=prompt)
            f.write(json.dumps({"qid": q["qid"], "set": q["set"], "category": q["category"], "found": r["found"],
                                "answer": r["answer"], "sources": [s["title"] for s in r["sources"]],
                                "top1": r["candidates"][0]["score"] if r["candidates"] else None,
                                "timing": r["timing"], "error": r["error"], "below_min_score": r.get("below_min_score")},
                               ensure_ascii=False) + "\n")
            f.flush()


def report():
    S = [json.loads(l) for l in open(SCORES, encoding="utf-8")]
    A = {r["qid"]: r for r in map(json.loads, open(ANSWERS, encoding="utf-8"))} if ANSWERS.exists() else {}
    ans = lambda sets: [r for r in S if r["set"] in sets]
    na = lambda st: [r for r in S if r["set"] == st]
    pick = ans(PICK_SETS)
    t = round(min(r["top1"] for r in pick) - MARGIN, 4)
    res = {"기준점수 t": t, "정한 근거": f"답변 가능 {'·'.join(PICK_SETS)} {len(pick)}문항 1위 점수 최솟값 {min(r['top1'] for r in pick)} − {MARGIN}"}

    def dist(rs):
        v = sorted(r["top1"] for r in rs)
        return {"n": len(v), "최소": v[0], "5%": v[max(int(len(v) * .05) - 1, 0)], "중앙": v[len(v) // 2], "최대": v[-1]}

    res["1위 점수 분포"] = {"답변 가능 dev1~5": dist(pick), "답변 가능 dev6~10": dist(ans(CHECK_SETS)),
                       "na_dev": dist(na("na_dev")), "na_test": dist(na("na_test"))}
    res["유형별 1위 점수 (40)"] = {c: dist([r for r in S if r["category"] == c]) for c in sorted({r["category"] for r in S if r["set"].startswith("na")})}
    chk = ans(CHECK_SETS)
    res["확인: 답변 가능 dev6~10 중 막힘"] = [(r["qid"], r["top1"]) for r in chk if r["top1"] < t]
    for st in ("na_dev", "na_test", "na_new"):
        rs = na(st)
        res[f"{st} 기준점수로 막힘"] = f"{sum(r['top1'] < t for r in rs)}/{len(rs)}"
    if A:
        rows = []
        for r in S:
            if r["set"] not in ("na_dev", "na_test"):  # 5-8-1 범위 (na_new 는 5-8-2)
                continue
            a = A.get(r["qid"])
            blocked = r["top1"] < t
            rows.append({"qid": r["qid"], "set": r["set"], "유형": r["category"], "1위 점수": r["top1"], "1위 조문": r["top1_title"],
                         "기준점수로 막힘": blocked, "LLM found": a["found"] if a else None,
                         "결과": "막힘(LLM 안 부름)" if blocked else ("LLM이 없다고 함" if a and not a["found"] else "LLM이 근거를 대며 답함(환각)"),
                         "LLM 답변": a["answer"] if a else "", "LLM 근거": a["sources"] if a else [], "LLM 시간": a["timing"]["total"] if a else None})
        from collections import Counter
        res["답변불가 40 결과 (기준점수 + LLM)"] = dict(Counter(x["결과"] for x in rows))
        res["답변불가 40 결과 (LLM만, 기준점수 없이)"] = dict(Counter("LLM이 없다고 함" if not A[x["qid"]]["found"] else "LLM이 근거를 대며 답함(환각)" for x in rows))
        res["환각 문항"] = [(x["qid"], x["LLM 근거"], x["LLM 답변"][:160]) for x in rows if A[x["qid"]]["found"]]
        with open(EVAL / "na_review.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    json.dump(res, open(EVAL / "summary_unanswerable.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", action="store_true")
    ap.add_argument("--answer", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--prompt", default=None, help="지시문 (v1·v2). 주면 na_answers_<지시문>.jsonl 에 기준점수를 거친 answer() 결과")
    ap.add_argument("--sets", nargs="+", default=None, help="na_dev na_test na_new 중 일부만")
    a = ap.parse_args()
    if a.scores:
        scores()
    if a.answer:
        answers(a.prompt, a.sets)
    if a.report:
        report()
