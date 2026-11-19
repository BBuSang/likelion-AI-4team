"""F21: 직접성 판정(G) + 기존 답변 검증(C) 결합 (worklog 121번, 사용자 "판정+검증 결합 측정", 2026-11-17). 개발셋만 쓴다.

f20의 G 답변(정상 300, 정답 제외 300, 답변불가 20)을 그대로 두고 `verify.check`(99번 구성: 인용 사례가 질문을 직접 다루지 않으면 거절)만 붙인다.
답변을 다시 만들지 않는다. 거절하지 않은 답은 f20 심사 결과를 그대로 쓴다. 데모에서는 사이드바 "답변 검증" 토글을 켠 것과 같다.

주의: 이 조합은 f20 결과(G와 C가 막은 문항이 다르다는 것)를 본 뒤에 고른 것이다. 통과해도 새 세트(118번 A4)로 다시 확인해야 한다.

사전 기준 (돌리기 전에 정함. 결과를 보고 고치지 않는다). 기준선 = f20 G(정답 제외 불일치 20 / 불일치 + 부분 56, 정상 일치 274 / 불일치 0, 답변불가 20/20)
- K1 정답 제외 불일치 ≤ 10 (G의 절반 이하)
- K2 정답 제외 불일치 + 부분일치 ≤ 56 (G보다 늘지 않음)
- K3 정상 일치 ≥ 268 (f20과 같은 하한: f19 V3 283에서 5%p까지만 잃음)
- K4 정상 불일치 0
- K5 답변불가 20 모두 거절
- K6 검증 호출 실패 3건 이하
모두 통과하면 데모의 "답변 검증" 토글 기본값을 켬으로 바꾼다. 하나라도 못 넘으면 G만 쓰고 그대로 보고한다.
사용: python pipeline/f21_gate_verify_eval.py     캐시: data/eval/_f21_check.jsonl   결과: data/eval/f21_gate_verify_eval.json
"""
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import f19_a1_a2_eval as f19  # noqa: E402

service = f19.service
import verify as vf  # noqa: E402

EVAL = f19.EVAL
SETS, UNANS = f19.SETS, f19.UNANS
CHK, OUT = EVAL / "_f21_check.jsonl", EVAL / "f21_gate_verify_eval.json"


def main():
    data = {n: [json.loads(x) for x in (EVAL / f"{n}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] for n in SETS + [UNANS]}
    qtext = {r["qid"]: r["question"] for n in data for r in data[n]}
    ans = {(x["arm"], x["set"], x["qid"]): x for x in f19.read(EVAL / "_f20_answers.jsonl")}
    jdg = {(x["arm"], x["set"], x["qid"]): x for x in f19.read(EVAL / "_f20_judge.jsonl")}
    done = {(x["arm"], x["set"], x["qid"]): x for x in f19.read(CHK)}
    todo = [k for k in ans if k[0] in ("gate", "gate_holdout") and k not in done]

    def run(k):
        a = ans[k]
        if a.get("error") or a["not_found"]:
            return {"arm": k[0], "set": k[1], "qid": k[2], "checked": False, "abstained": False}
        _, info = vf.check(service.mask(qtext[k[2]]), service.HEADING.sub("", a["shown"]), a["cited_context"])
        return {"arm": k[0], "set": k[1], "qid": k[2], "checked": info["checked"], "abstained": info["abstained"], "direct": info.get("direct"),
                "flagged": len(info.get("flagged") or []), "error": info.get("error")}
    with ThreadPoolExecutor(6) as ex:
        for i, c in enumerate(ex.map(run, todo), 1):
            f19.append(CHK, c)
            done[(c["arm"], c["set"], c["qid"])] = c
            if i % 50 == 0 or i == len(todo):
                print(f"검증 {i}/{len(todo)}", flush=True)

    def verdict(arm, n, q):
        return "답변없음" if done[(arm, n, q)]["abstained"] else jdg[(arm, n, q)]["correctness"]
    rows = [(n, r["qid"], r["corpus"]) for n in SETS for r in data[n]]
    norm = [verdict("gate", n, q) for n, q, _ in rows]
    hold = [verdict("gate_holdout", n, q) for n, q, _ in rows]
    g_norm = [jdg[("gate", n, q)]["correctness"] for n, q, _ in rows]
    g_hold = [jdg[("gate_holdout", n, q)]["correctness"] for n, q, _ in rows]
    un = sum(ans[("gate", UNANS, r["qid"])]["not_found"] or done[("gate", UNANS, r["qid"])]["abstained"] for r in data[UNANS])
    fails = sum(bool(c.get("error")) for c in done.values())
    nc, hc = Counter(norm), Counter(hold)
    crit = {"K1_hold_wrong": {"value": hc["불일치"], "limit": "≤ 10", "pass": hc["불일치"] <= 10},
            "K2_hold_other": {"value": hc["불일치"] + hc["부분일치"], "limit": "≤ 56", "pass": hc["불일치"] + hc["부분일치"] <= 56},
            "K3_norm_match": {"value": nc["일치"], "limit": "≥ 268", "pass": nc["일치"] >= 268},
            "K4_norm_wrong": {"value": nc["불일치"], "limit": "= 0", "pass": nc["불일치"] == 0},
            "K5_unanswerable": {"value": un, "limit": "= 20", "pass": un == 20},
            "K6_failures": {"value": fails, "limit": "≤ 3", "pass": fails <= 3}}
    moved = lambda a, b: dict(Counter(f"{x}→답변없음" for x, y in zip(a, b) if x != y))          # 검증이 거절로 바꾼 문항의 원래 판정
    res = {"criteria": crit, "all_pass": all(c["pass"] for c in crit.values()), "norm": dict(nc), "hold": dict(hc),
           "abstained_from": {"norm": moved(g_norm, norm), "hold": moved(g_hold, hold)},
           "by_corpus": {cp: {"norm": dict(Counter(v for v, (_, _, c) in zip(norm, rows) if c == cp)), "hold": dict(Counter(v for v, (_, _, c) in zip(hold, rows) if c == cp))} for cp in ("interp", "exmnt")},
           "flagged_answers": {"norm": sum(1 for n, q, _ in rows if done[("gate", n, q)].get("flagged")), "hold": sum(1 for n, q, _ in rows if done[("gate_holdout", n, q)].get("flagged"))},
           "updated_at": f19.now()}
    f19.save(OUT, res)
    print("통과" if res["all_pass"] else "미달", json.dumps({k: (v["value"], v["pass"]) for k, v in crit.items()}, ensure_ascii=False))
    print("정상", res["norm"], "| 정답 제외", res["hold"])
    print("거절로 바뀐 문항", json.dumps(res["abstained_from"], ensure_ascii=False), "| 회신·현장건의", json.dumps(res["by_corpus"], ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
