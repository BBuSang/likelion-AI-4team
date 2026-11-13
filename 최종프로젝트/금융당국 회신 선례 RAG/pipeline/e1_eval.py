"""E1 ② 조문 조회 평가: 조문 조회셋 50(정답 450건)으로 조회 결과의 정밀도·재현율과 법령·조문 해석 정확도를 잰다.
- 정식 표기 50문항 + 표기 변형(약칭, '제' 생략, 띄어쓰기, 문장 속 표현)을 DB의 이름에서 만들어 해석 정확도를 본다
- 법령 이름이 없거나 없는 조문 같은 부정 사례의 동작을 확인한다
정답은 d1_articles.py가 만든 것(추출 결과 일부 포함, 수기 확인)이라 자기 검증의 한계가 있다.
사용: python e1_eval.py     출력: data/eval/e1_eval.json
"""
import json
import sys
import time
from collections import defaultdict

from articles_lookup import Lookup, connect
from common import DATA, manifest

sys.stdout.reconfigure(encoding="utf-8")
EVAL = DATA / "eval"


def aliases_for(lk, law_id):
    """질문 표기용 약칭: 약칭 중 가장 짧은 것. 시행령·시행규칙은 모법 약칭 + 종류."""
    names = sorted((n for n, ids in lk.names.items() if ids == {law_id}), key=len)
    return [n for n in names if n != "".join(lk.laws[law_id]["law_name"].split())]


def spell(article, sub):
    return f"{article}조" + (f"의{sub}" if sub else "")


def main():
    gold = [json.loads(l) for l in open(EVAL / "article_lookup_50.jsonl", encoding="utf-8")]
    with connect() as conn:
        lk = Lookup(conn)
        by_name = {r["law_name"]: lid for lid, r in lk.laws.items()}
        main_keys = lambda res: {k for r in res["results"] for k in r["case_keys"]}
        main_groups = lambda res: {r["group_id"] for r in res["results"] if r["case_keys"]}
        forms = {"정식": lambda g, a, n: f"{g['law_name']} 제{spell(a, n)}", "'제' 생략": lambda g, a, n: f"{g['law_name']} {spell(a, n)}",
                 "띄어쓰기": lambda g, a, n: f"{g['law_name']} 제 {a} 조" + (f" 의 {n}" if n else ""),
                 "문장 속": lambda g, a, n: f"{g['law_name']} 제{spell(a, n)} 관련 회신 사례 최근 것부터 보여 주세요"}
        stats = defaultdict(lambda: defaultdict(int))
        misses = []
        times = []
        for g in gold:
            lid = by_name[g["law_name"]]
            als = aliases_for(lk, lid)
            qs = {k: f(g, g["article"], g["article_sub"]) for k, f in forms.items()}
            if als:
                qs["약칭"] = f"{als[0]} 제{spell(g['article'], g['article_sub'])}"
            gk, gg = set(g["gold_case_keys"]), set(g["gold_group_ids"])
            for form, q in qs.items():
                t0 = time.time()
                res = lk.lookup(q)
                times.append(time.time() - t0)
                m = res["parsed"][0] if res["parsed"] else {}
                ok = len(res["parsed"]) == 1 and m.get("law_id") == lid and m.get("article") == g["article"] and m.get("sub") == g["article_sub"]
                pk, pg = main_keys(res) if ok else set(), main_groups(res) if ok else set()
                s = stats[form]
                s["n"] += 1
                s["parse_ok"] += ok
                s["tp"] += len(pk & gk); s["fp"] += len(pk - gk); s["fn"] += len(gk - pk)
                s["gtp"] += len(pg & gg); s["gfp"] += len(pg - gg); s["gfn"] += len(gg - pg)
                if form == "정식" and (not ok or pk != gk):
                    misses.append({"aid": g["aid"], "query": q, "parse_ok": ok, "fn": sorted(gk - pk), "fp": sorted(pk - gk)})
        summary = {}
        for form, s in stats.items():
            summary[form] = {"n": s["n"], "parse_ok": round(s["parse_ok"] / s["n"], 4),
                             "case_precision": round(s["tp"] / max(s["tp"] + s["fp"], 1), 4), "case_recall": round(s["tp"] / max(s["tp"] + s["fn"], 1), 4),
                             "group_precision": round(s["gtp"] / max(s["gtp"] + s["gfp"], 1), 4), "group_recall": round(s["gtp"] / max(s["gtp"] + s["gfn"], 1), 4),
                             "tp_fp_fn": [s["tp"], s["fp"], s["fn"]]}
        # 부정·경계 사례
        neg = [("제27조의2 알려줘", "need_law"), ("없는법 제1조", "need_law"), ("은행법 제9999조", "no_cases"), ("은행법에 대해 알려줘", "no_article"),
               ("은행법 제27조의2와 제28조", "ok"), ("2조 원 규모 증자", "no_article"), ("은행법 제27조의2제1항 사례", "ok"), ("은행법 제 27 조의 2 와 보험업법 제 5 조", "ok")]
        negres = []
        for q, want in neg:
            r = lk.lookup(q)
            negres.append({"query": q, "want": want, "got": r["status"], "ok": r["status"] == want,
                           "parsed": [(p.get("law_name") or p.get("law_name_text"), p["article"], p["sub"], p["status"]) for p in r["parsed"]]})
        amb = {k: [lk.laws[i]["law_name"] for i in v] for k, v in lk.ambiguous.items()}
    out = {"summary": summary, "misses_canonical": misses, "negative": negres, "ambiguous_names": amb, "lookup_seconds": {"median": sorted(times)[len(times) // 2], "max": max(times)}}
    with open(EVAL / "e1_eval.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    manifest("e1_eval", forms={k: v["n"] for k, v in summary.items()})
    for form, v in summary.items():
        print(f"{form:8s} n={v['n']:2d} 해석 {v['parse_ok']:.3f} | 사례 정밀도 {v['case_precision']:.4f} 재현율 {v['case_recall']:.4f} (tp/fp/fn {v['tp_fp_fn']}) | 묶음 정밀도 {v['group_precision']:.4f} 재현율 {v['group_recall']:.4f}")
    print("정식 표기 불일치:", json.dumps(misses, ensure_ascii=False))
    for r in negres:
        print(("OK " if r["ok"] else "NG "), r["query"], "→", r["got"], "(기대", r["want"] + ")", r["parsed"])
    print("모호한 이름(법령 둘 이상):", amb)
    print("조회 시간", out["lookup_seconds"])


if __name__ == "__main__":
    main()
