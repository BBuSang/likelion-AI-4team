"""기획안 5-6 리트리버 평가 (LLM 없음)

    python pipeline/evaluate_06.py --sets dev1 test1 --run baseline

지표 (조문 단위): Hit@1, Hit@5, (참고) 법령 Hit@5 = 상위 5개 조문 중 정답 법령이 있는 비율
출력: eval/results_<run>.csv (질문별 1~5위, 정답 순위, 실패 원인 추정), eval/summary_<run>.json, reports/5-6_평가_<run>.md
정답은 eval/gold.csv (지금은 status=AI초안, 사용자 확인 전 → 점수는 잠정)
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict

from common import REPORTS, ROOT, TODAY

sys.path.insert(0, str(ROOT))
from retriever.search import EF_SEARCH, POOL, embed, search_vec  # noqa: E402

EVAL = ROOT / "eval"
SUBS = (" 시행령", " 시행규칙")


def base_law(name):
    for s in SUBS:
        if name.endswith(s):
            return name[: -len(s)]
    return name


def failure_hint(gold, top5, rank):
    """Hit@5 실패만: 원인 추정 (사람이 확인할 칸). 기획안 5-6-3 분류 중 자동으로 알 수 있는 것만"""
    glaws = {g["law_name"] for g in gold}
    tlaws = [a["law_name"] for a in top5]
    if glaws & set(tlaws):
        return "같은 법 다른 조문 (법은 맞힘)"
    if {base_law(x) for x in glaws} & {base_law(x) for x in tlaws}:
        return "위계 혼동 (법률↔시행령·시행규칙)"
    if rank:
        return f"정답 조문이 {rank}위 (6위 이하)"
    return f"정답이 조각 상위 {POOL}개 안에 없음"


LOCK = EVAL / "test_lock.json"


def check_lock(sets):
    """기획안 5-6-4: 테스트셋은 세트마다 한 번만. 이미 잰 test 세트는 다시 재지 않는다 (풀 방법은 없음: 잠금 파일을 사람이 직접 지워야 함)"""
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    done = [s for s in sets if s.startswith("test") and s in lock]
    if done:
        for s in done:
            print(f"🔒 {s} 는 이미 {lock[s]['measured_on']} `{lock[s]['run']}` 로 측정함 (Hit@1 {lock[s]['Hit@1']}, Hit@5 {lock[s]['Hit@5']}) → 다시 재지 않음")
        raise SystemExit("잠긴 테스트셋이 있어 멈춥니다. dev 세트로 개선하고, 새 test 세트(test2·test3)는 마지막에 한 번씩만.")
    return lock


def lock_tests(lock, sets, by_set, run, qs, gold):
    """잰 test 세트를 잠금 파일에 기록 (질문·정답 지문 포함: 나중에 정답을 고쳐도 '잰 때의 셋'이 무엇이었는지 남김)"""
    import hashlib
    for s in sets:
        if not s.startswith("test"):
            continue
        items = [(q["qid"], q["question"], sorted(g["article_uid"] for g in gold[q["qid"]])) for q in qs if q["set"] == s]
        lock[s] = {"measured_on": TODAY, "run": run, "Hit@1": by_set[s]["Hit@1"], "Hit@5": by_set[s]["Hit@5"],
                   "n": by_set[s]["문항"], "set_sha256": hashlib.sha256(json.dumps(items, ensure_ascii=False).encode()).hexdigest(),
                   "note": "기획안 5-6-4: 세트마다 한 번만. 이 결과로 검색을 고치지 않는다"}
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["dev1"])
    ap.add_argument("--run", default="baseline")
    args = ap.parse_args()
    lock = check_lock(args.sets)

    qs = [q for q in csv.DictReader(open(EVAL / "queries.csv", encoding="utf-8-sig")) if q["set"] in args.sets]
    gold = defaultdict(list)
    for g in csv.DictReader(open(EVAL / "gold.csv", encoding="utf-8-sig")):
        gold[g["qid"]].append(g)
    status = Counter(q["status"] for q in qs)
    t0 = time.time()
    qv = embed([q["question"] for q in qs]) if len(qs) <= 64 else \
        __import__("numpy").concatenate([embed([q["question"] for q in qs[i:i + 32]]) for i in range(0, len(qs), 32)])
    t_emb = time.time() - t0
    rows, t_db = [], 0.0
    for q, v in zip(qs, qv):
        t1 = time.time()
        arts = search_vec(v, k=0)  # 조각 50개를 조문으로 합친 전체 순위
        t_db += time.time() - t1
        g = gold[q["qid"]]
        gids = {x["article_uid"] for x in g}
        rank = next((a["rank"] for a in arts if a["doc_id"] in gids), None)
        top5 = arts[:5]
        hit1, hit5 = rank == 1, rank is not None and rank <= 5
        law5 = bool({x["law_name"] for x in g} & {a["law_name"] for a in top5})
        r = {"qid": q["qid"], "set": q["set"], "유형": q["q_type"], "질문": q["question"],
             "정답 조문": " / ".join(f'{x["law_name"]} {x["article_label"]}({x["title"]})' + ("" if x["primary_gold"] == "True" else " [부]")
                                  for x in g),
             "정답 구분": "/".join(x["law_level"] for x in g)}
        for i in range(5):
            a = top5[i] if i < len(top5) else None
            r[f"{i + 1}위"] = f'{a["score"]:.3f} {a["title"]}' + (" ✔" if a["doc_id"] in gids else "") if a else ""
        r.update({"정답 순위": rank or f">{len(arts)}", "Hit@1": "O" if hit1 else "X", "Hit@5": "O" if hit5 else "X",
                  "법령 Hit@5": "O" if law5 else "X", "실패 원인(추정)": "" if hit5 else failure_hint(g, top5, rank),
                  "실패 원인(확인)": ""})
        rows.append(r)

    def score(rs):
        n = len(rs) or 1
        return {"문항": len(rs), "Hit@1": round(sum(r["Hit@1"] == "O" for r in rs) / n, 3),
                "Hit@5": round(sum(r["Hit@5"] == "O" for r in rs) / n, 3),
                "법령 Hit@5": round(sum(r["법령 Hit@5"] == "O" for r in rs) / n, 3)}

    by_set = {s: score([r for r in rows if r["set"] == s]) for s in args.sets}
    by_type = {s: {t: score([r for r in rows if r["set"] == s and r["유형"] == t])
                   for t in ["상황형", "용어형", "조문지정형", "하위법령형", "복수법령형"]} for s in args.sets}
    fails = {s: Counter(r["실패 원인(추정)"].split(" (")[0] if "위" not in r["실패 원인(추정)"] else "정답이 6위 이하"
                        for r in rows if r["set"] == s and r["Hit@5"] == "X") for s in args.sets}
    summary = {"run": args.run, "date": TODAY, "sets": args.sets, "정답 상태": dict(status),
               "설정": {"검색": "벡터만 (KURE-v1, pgvector HNSW)", "조각 pool": POOL, "ef_search": EF_SEARCH, "조문 합치기": "최고점"},
               "세트별": by_set, "유형별": by_type, "Hit@5 실패 원인(추정)": {s: dict(v) for s, v in fails.items()},
               "질문 임베딩 평균(초)": round(t_emb / len(qs), 3), "DB 검색 평균(ms)": round(1000 * t_db / len(qs), 1)}
    out = EVAL / f"results_{args.run}.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (EVAL / f"summary_{args.run}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    lock_tests(lock, args.sets, by_set, args.run, qs, gold)
    print(json.dumps({k: summary[k] for k in ("세트별", "Hit@5 실패 원인(추정)", "질문 임베딩 평균(초)", "DB 검색 평균(ms)")},
                     ensure_ascii=False, indent=1))
    report(summary, rows, args)


def report(s, rows, args):
    L = []
    p = L.append
    p(f"# 5-6 리트리버 평가: `{s['run']}` ({', '.join(args.sets)})\n")
    p(f"{s['date'][:4]}-{s['date'][4:6]}-{s['date'][6:]} · 검색 {s['설정']['검색']} · 조각 {s['설정']['조각 pool']}개 → 조문으로 합침(최고점) · "
      f"ef_search {s['설정']['ef_search']} · 코드 `retriever/search.py`, `pipeline/evaluate_06.py`\n")
    p(f"> ⚠️ 정답은 **AI 초안**(`status`: {s['정답 상태']}) — 사용자 확인(`eval/gold_review.csv`) 전이라 점수는 잠정. "
      "test 세트는 기획안상 **세트마다 한 번만** 재고 그 결과로 고치지 않는다.\n")
    p("## 점수 (조문 단위)\n")
    p("| 세트 | 문항 | Hit@1 | Hit@5 | (참고) 법령 Hit@5 | 기준 Hit@5 ≥ 0.80 |\n|---|---|---|---|---|---|")
    for k, v in s["세트별"].items():
        p(f"| {k} | {v['문항']} | {v['Hit@1']:.2f} | **{v['Hit@5']:.2f}** | {v['법령 Hit@5']:.2f} | "
          f"{'✅' if v['Hit@5'] >= 0.8 else '❌'}{' (dev 기준)' if k.startswith('dev') else ' (참고: test)'} |")
    p(f"\n질문 임베딩 평균 {s['질문 임베딩 평균(초)']}초 (CPU) · DB 검색 평균 {s['DB 검색 평균(ms)']}ms\n")
    p("## 유형별\n")
    for k, tv in s["유형별"].items():
        p(f"**{k}**\n")
        p("| 유형 | 문항 | Hit@1 | Hit@5 | 법령 Hit@5 |\n|---|---|---|---|---|")
        for t, v in tv.items():
            p(f"| {t} | {v['문항']} | {v['Hit@1']:.2f} | {v['Hit@5']:.2f} | {v['법령 Hit@5']:.2f} |")
        p("")
    p("## Hit@5 실패 질문 (원인은 자동 추정, 확인 칸은 `eval/results_<run>.csv`의 `실패 원인(확인)`)\n")
    for k in args.sets:
        fs = [r for r in rows if r["set"] == k and r["Hit@5"] == "X"]
        p(f"### {k} — {len(fs)}문항: " + ", ".join(f"{a} {b}" for a, b in s["Hit@5 실패 원인(추정)"][k].items()) + "\n")
        p("| qid | 유형 | 질문 | 정답 | 정답 순위 | 1위 | 2위 | 추정 원인 |\n|---|---|---|---|---|---|---|---|")
        for r in fs:
            esc = lambda x: x.replace("|", "\\|")
            p(f"| {r['qid']} | {r['유형']} | {esc(r['질문'])} | {esc(r['정답 조문'])} | {r['정답 순위']} | {esc(r['1위'])} | {esc(r['2위'])} | {r['실패 원인(추정)']} |")
        p("")
    p(f"## 질문별 전체 표\n\n`eval/results_{s['run']}.csv` — 컬럼: qid, 세트, 유형, 질문, 정답 조문, 1~5위(점수·조문, ✔ = 정답), 정답 순위, Hit@1, Hit@5, 법령 Hit@5, 실패 원인(추정·확인)")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"5-6_평가_{s['run']}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"reports/5-6_평가_{s['run']}.md")


if __name__ == "__main__":
    main()
