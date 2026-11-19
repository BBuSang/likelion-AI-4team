"""F20: 불일치·부분일치 줄이기 (worklog 120번, 사용자 2026-11-17: "불일치가 있으면 안 된다. 과적합 방지 차원에서 줄일 방법을 찾아라.
부분도 안 된다. 답이 없다고 비슷한 사례를 정답처럼 말하면 안 된다"). 개발셋만 쓴다. 시험 세트는 읽지 않는다.

후보 두 가지를 같은 기준으로 잰다. 둘 다 특정 문항·주제에 맞춘 규칙·예시·기준값이 없다.
- G 직접성 판정(`app/gate.py`, 새로 만듦): 답하기 전에 검색된 사례마다 대상·행위·쟁점이 질문과 같은지 판정해 '직접'인 사례만으로 답한다. 없으면 "찾지 못했습니다".
  `service.ask(..., use_gate=True)`. 질문당 LLM 호출 1회 추가.
- C 기존 답변 검증(`verify.check`, 99번 구성: 인용 사례가 질문을 직접 다루지 않으면 거절). 99번에서는 정답 제외 문항이 없어 효과를 잴 수 없었다.
  f19의 V3 답변을 그대로 두고 검증만 붙인다(답변을 다시 만들지 않음). 거절하지 않은 답은 f19 심사 결과를 그대로 쓴다.

문항: f19와 같다. 정상 = dev7~10 300문항, 정답 제외 = 같은 300문항에서 정답 묶음을 검색 대상에서 뺀 것, 답변불가 20. 심사는 f19와 같은 전문 심사.
기준선: f19 V3(정상 일치 283 / 부분 13 / 불일치 4, 답변불가 19/20), f19 정답 제외(불일치 74, 불일치 + 부분 166).

사전 기준 (돌리기 전에 정함. 결과를 보고 판정·기준·프롬프트를 고치지 않는다)
- H1 정답 제외 불일치 ≤ 24 (기준선 74의 1/3 이하)
- H2 정답 제외 불일치 + 부분일치 ≤ 83 (기준선 166의 절반 이하)
- N1 정상 불일치 + 부분일치 ≤ 17 (기준선 이하)
- N2 정상 일치 ≥ 268 (기준선 283에서 5%p(15문항)까지만 잃음. 사용자 우선순위: 틀린 답보다 "찾지 못했습니다"가 낫다)
- U 답변불가 20 모두 거절
- E 잘못된 사례번호 0, 판정·검증 호출 실패 3건(1%) 이하
고르는 법: 둘 다 통과하면 정답 제외 불일치 + 부분일치가 적은 쪽, 같으면 정상 일치가 많은 쪽. 통과한 후보는 데모 기본값을 켬으로 바꾼다.
둘 다 통과하지 못하면 그대로 보고한다. 같은 개발셋으로 두 후보를 비교해 고른 것이므로, 채택한 것은 나중에 새 세트(118번 A4)로 한 번 더 확인해야 한다.

한계: 질문은 정답 회신을 보고 LLM이 만든 합성 질문, 생성·판정·심사가 모두 같은 모델. 정상 세트의 부분일치 중 정답이 문맥에 있는데 조건을 빠뜨린 유형은 두 후보 모두 직접 다루지 않는다.
사용: python pipeline/f20_direct_gate_eval.py [--smoke]
캐시: data/eval/_f20_answers.jsonl, _f20_judge.jsonl, _f20_check.jsonl   결과: data/eval/f20_direct_gate_eval.json
"""
import argparse
import functools
import hashlib
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import f19_a1_a2_eval as f19  # noqa: E402  (정답 제외 감싸기·답변 실행·심사·집계를 그대로 쓴다)

service, rag, es = f19.service, f19.rag, f19.es
import verify as vf  # noqa: E402

EVAL = f19.EVAL
SETS, UNANS = f19.SETS, f19.UNANS
BASE = {"hold_wrong": 74, "hold_other": 166, "norm_other": 17, "norm_match": 283}


def fingerprint():
    paths = sorted((ROOT / "app").glob("*.py")) + [ROOT / "pipeline" / n for n in ("rag.py", "f4_rag_answer_eval.py", "f19_a1_a2_eval.py", Path(__file__).name)]
    data = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    data["model"] = rag.CHAT_MODEL
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16], data


def criteria(norm, hold, unans, invalid, failures):
    nc, hc = Counter(norm), Counter(hold)
    c = {"H1_hold_wrong": {"value": hc["불일치"], "limit": 24, "pass": hc["불일치"] <= 24},
         "H2_hold_other": {"value": hc["불일치"] + hc["부분일치"], "limit": 83, "pass": hc["불일치"] + hc["부분일치"] <= 83},
         "N1_norm_other": {"value": nc["불일치"] + nc["부분일치"], "limit": 17, "pass": nc["불일치"] + nc["부분일치"] <= 17},
         "N2_norm_match": {"value": nc["일치"], "limit": 268, "pass": nc["일치"] >= 268},
         "U_unanswerable": {"value": unans, "limit": 20, "pass": unans == 20},
         "E_errors": {"invalid": invalid, "failures": failures, "pass": invalid == 0 and failures <= 3}}
    return {"criteria": c, "all_pass": all(x["pass"] for x in c.values()),
            "norm": dict(nc), "hold": dict(hc)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = "_f20s" if args.smoke else "_f20"
    ANS, JDG, CHK, RUN = (EVAL / f"{tag}_{x}" for x in ("answers.jsonl", "judge.jsonl", "check.jsonl", "run.json"))
    OUT = EVAL / ("f20_smoke.json" if args.smoke else "f20_direct_gate_eval.json")
    fp, parts = fingerprint()
    if RUN.exists():
        assert json.loads(RUN.read_text(encoding="utf-8"))["fingerprint"] == fp, "설정이 바뀌었다: 캐시와 섞지 않는다"
    else:
        f19.save(RUN, {"started_at": f19.now(), "fingerprint": fp, "parts": parts})
    f19.install()
    service.ask = functools.partial(service.ask, use_gate=True)          # f19.run_one이 부르는 service.ask를 G로 바꾼다
    data = {n: [json.loads(x) for x in (EVAL / f"{n}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] for n in SETS + [UNANS]}
    if args.smoke:
        data = {n: v[:2] for n, v in data.items()}
    rows = [r for n in SETS for r in data[n]]
    with rag.connect() as conn:
        groups = {r["case_key"]: r["group_id"] for r in conn.execute("select case_key, group_id from interp.cases union all select case_key, group_id from interp.exmnt_cases").fetchall()}
        refs = f19.gold_refs(conn, rows)
    rag.embed("준비")

    # ---------- G: 직접성 판정으로 다시 답한다
    answers = {(x["arm"], x["set"], x["qid"]): x for x in f19.read(ANS)}
    for arm in ("gate", "gate_holdout"):
        es.MAX_N = f19.ES_MAX + 1 if arm.endswith("holdout") else f19.ES_MAX
        todo = [(n, r) for n in SETS + ([UNANS] if arm == "gate" else []) for r in data[n] if (arm, n, r["qid"]) not in answers]
        t0 = time.time()
        with ThreadPoolExecutor(f19.WORKERS) as ex:
            for i, a in enumerate(ex.map(lambda x: f19.run_one(arm, x[0], x[1], groups), todo), 1):
                f19.append(ANS, a)
                answers[(arm, a["set"], a["qid"])] = a
                if i % 20 == 0 or i == len(todo):
                    print(f"[{arm}] 답변 {i}/{len(todo)} ({time.time() - t0:.0f}초)", flush=True)
    es.MAX_N = f19.ES_MAX
    judged = {(x["arm"], x["set"], x["qid"]): x for x in f19.read(JDG)}
    todo = [(r, answers[(arm, n, r["qid"])], refs[r["qid"]]) for arm in ("gate", "gate_holdout") for n in SETS for r in data[n] if (arm, n, r["qid"]) not in judged]
    with ThreadPoolExecutor(6) as ex:
        for i, j in enumerate(ex.map(f19.judge, todo), 1):
            f19.append(JDG, j)
            judged[(j["arm"], j["set"], j["qid"])] = j
            if i % 50 == 0 or i == len(todo):
                print(f"[G] 심사 {i}/{len(todo)}", flush=True)

    # ---------- C: f19 V3 답변에 기존 검증만 붙인다
    f19_ans = {(x["arm"], x["set"], x["qid"]): x for x in f19.read(EVAL / ("_f19s_answers.jsonl" if args.smoke else "_f19_answers.jsonl"))}
    f19_jdg = {(x["arm"], x["set"], x["qid"]): x for x in f19.read(EVAL / ("_f19s_judge.jsonl" if args.smoke else "_f19_judge.jsonl"))}
    checked = {(x["arm"], x["set"], x["qid"]): x for x in f19.read(CHK)}
    qtext = {r["qid"]: r["question"] for n in SETS + [UNANS] for r in data[n]}
    todo = [k for k in f19_ans if k[0] in ("v3", "holdout") and k[2] in qtext and k not in checked]

    def run_check(k):
        a = f19_ans[k]
        if a.get("error") or a["not_found"]:
            return {"arm": k[0], "set": k[1], "qid": k[2], "checked": False, "abstained": False}
        _, info = vf.check(service.mask(qtext[k[2]]), service.HEADING.sub("", a["shown"]), a["cited_context"])
        return {"arm": k[0], "set": k[1], "qid": k[2], "checked": info["checked"], "abstained": info["abstained"], "direct": info.get("direct"), "error": info.get("error")}
    with ThreadPoolExecutor(6) as ex:
        for i, c in enumerate(ex.map(run_check, todo), 1):
            f19.append(CHK, c)
            checked[(c["arm"], c["set"], c["qid"])] = c
            if i % 50 == 0 or i == len(todo):
                print(f"[C] 검증 {i}/{len(todo)}", flush=True)

    # ---------- 집계
    res = {"fingerprint": fp, "updated_at": f19.now(), "baseline": BASE, "candidates": {}}
    g_norm = [judged[("gate", n, r["qid"])]["correctness"] for n in SETS for r in data[n]]
    g_hold = [judged[("gate_holdout", n, r["qid"])]["correctness"] for n in SETS for r in data[n]]
    g_ans = [answers[(arm, n, r["qid"])] for arm in ("gate", "gate_holdout") for n in SETS for r in data[n]]
    g_un = [answers[("gate", UNANS, r["qid"])] for r in data[UNANS]]
    g_fail = sum(bool(a.get("error")) or bool((a.get("gate") or {}).get("error")) for a in g_ans + g_un)          # 시스템 오류 + 판정 실패(판정 없이 답함)
    res["candidates"]["G"] = criteria(g_norm, g_hold, sum(a.get("not_found", False) for a in g_un), sum(len(a.get("invalid") or []) for a in g_ans), g_fail)

    def c_verdict(arm, n, q):
        k = (arm, n, q)
        return "답변없음" if checked[k]["abstained"] else f19_jdg[k]["correctness"]
    c_norm = [c_verdict("v3", n, r["qid"]) for n in SETS for r in data[n]]
    c_hold = [c_verdict("holdout", n, r["qid"]) for n in SETS for r in data[n]]
    c_un = sum(f19_ans[("v3", UNANS, r["qid"])]["not_found"] or checked[("v3", UNANS, r["qid"])]["abstained"] for r in data[UNANS])
    c_fail = sum(bool(c.get("error")) for c in checked.values())
    c_inv = sum(len(f19_ans[("v3", n, r["qid"])].get("invalid") or []) for n in SETS for r in data[n])
    res["candidates"]["C"] = criteria(c_norm, c_hold, c_un, c_inv, c_fail)

    # 세부: 정상 세트에서 잃은 일치, 정답 제외에서 막은 불일치, 회신·현장건의별, 판정 비용
    corpus = {r["qid"]: r["corpus"] for r in rows}
    base_norm = {r["qid"]: f19_jdg[("v3", n, r["qid"])]["correctness"] for n in SETS for r in data[n]}
    base_hold = {r["qid"]: f19_jdg[("holdout", n, r["qid"])]["correctness"] for n in SETS for r in data[n]}
    for name, norm, hold in (("G", g_norm, g_hold), ("C", c_norm, c_hold)):
        q = [r["qid"] for r in rows]
        d = {"norm_match_lost": sum(base_norm[x] == "일치" and v != "일치" for x, v in zip(q, norm)),
             "norm_match_lost_to_notfound": sum(base_norm[x] == "일치" and v == "답변없음" for x, v in zip(q, norm)),
             "norm_other_fixed": sum(base_norm[x] in ("불일치", "부분일치") and v in ("일치", "답변없음") for x, v in zip(q, norm)),
             "hold_wrong_blocked": sum(base_hold[x] == "불일치" and v == "답변없음" for x, v in zip(q, hold)),
             "by_corpus": {cp: {"norm": dict(Counter(v for x, v in zip(q, norm) if corpus[x] == cp)), "hold": dict(Counter(v for x, v in zip(q, hold) if corpus[x] == cp))}
                           for cp in ("interp", "exmnt")}}
        res["candidates"][name]["detail"] = d
    gate_logs = [a for a in g_ans]
    res["candidates"]["G"]["detail"]["gate_fail_open"] = sum(1 for a in gate_logs if (a.get("gate") or {}).get("error"))
    passed = [k for k, v in res["candidates"].items() if v["all_pass"]]
    if passed:
        def other(k):
            h = res["candidates"][k]["hold"]
            return (h.get("불일치", 0) + h.get("부분일치", 0), -res["candidates"][k]["norm"].get("일치", 0))
        res["chosen"] = min(passed, key=other)
    else:
        res["chosen"] = None
    f19.save(OUT, res)
    for k, v in res["candidates"].items():
        print(k, "통과" if v["all_pass"] else "미달", json.dumps({c: (x.get("value"), x["pass"]) for c, x in v["criteria"].items()}, ensure_ascii=False))
        print("  정상", v["norm"], "| 정답 제외", v["hold"], "|", json.dumps(v["detail"], ensure_ascii=False))
    print("선택:", res["chosen"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
