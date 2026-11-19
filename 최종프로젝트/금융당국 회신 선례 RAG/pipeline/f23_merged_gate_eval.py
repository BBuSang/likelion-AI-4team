"""F23: 판정을 답변 호출에 합친 방식 (worklog 123번, 사용자: 판정·검증을 켜면 시간이 너무 길다 → 추천안 "판정을 답변 호출에 합치고 검증은 끔").
개발셋만 쓴다(dev7~10 300 + 정답 제외 300 + 답변불가 20, f19와 같은 방식·같은 심사). 시험 세트는 읽지 않는다. test11은 잠겨 있어 쓰지 않는다.

갈래: M = `service.ask(q, use_gate="merged", use_verify=False)`. 답변 모델이 한 번의 호출에서 사례마다 '직접/간접'을 적고 '직접'인 사례만으로 답한다
(`gate.merged`, 판정 기준 문구는 120번 GATE_SYSTEM과 같다). 질문당 호출: 리랭킹 + 답변 2회(지금 데모는 리랭킹 + 판정 + 답변 + 검증 4회).
비교 기준(같은 4개 동시 실행): f19 끔(정상 불일치 4, 정답 제외 불일치 74, 중앙값 5.4초), f20 판정(0, 20, 10.5초), f21 판정 + 검증(0, 8, 12~14초: f22).

사전 기준 (돌리기 전에 정함. 결과를 보고 프롬프트·규칙·기준을 고치지 않는다)
- M1 정상 불일치 0
- M2 정상 일치 ≥ 268 (f20·f21과 같은 하한)
- M3 정답 제외 불일치 ≤ 20 (판정 단독 수준 이하)
- M4 정답 제외 불일치 + 부분일치 ≤ 56 (판정 단독 수준 이하)
- M5 답변불가 20 모두 거절
- M6 정상 응답시간 중앙값 ≤ 8.0초 (4개 동시 실행, f19 끔 5.4초와 f20 판정 10.5초 사이)
- M7 시스템 오류 0, 잘못된 사례번호 0, 형식 오류로 판정 없이 답한 경우 3건 이하
모두 통과하면 데모 기본값을 M(합친 방식 + 검증 끔)으로 바꾼다. 하나라도 못 넘으면 지금 설정(판정 + 검증)을 그대로 두고 보고한다.
한계: 합성 단일 정답 질문, 생성·판정·심사가 같은 모델. dev7~10은 120~121번에서 판정·검증을 고를 때 쓴 세트다. 통과해도 새 세트로 다시 확인해야 한다.
시험 실행(--smoke)에서 고친 것: 모델이 '직접' 사례만 적고 나머지를 판정 목록에서 빼 형식 오류가 18건 중 7건 났다. 목록에 없는 사례를 '간접'으로 읽도록 바꿨다(`gate.merged`). 판정 기준·문구·사전 기준은 그대로다.
사용: python pipeline/f23_merged_gate_eval.py [--smoke]
"""
import argparse
import functools
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import f19_a1_a2_eval as f19  # noqa: E402

service, rag, es = f19.service, f19.rag, f19.es
EVAL, SETS, UNANS = f19.EVAL, f19.SETS, f19.UNANS
_orig_merged = service._answer_merged


def _merged(*a, **k):
    out = _orig_merged(*a, **k)
    f19._tl.ctx, f19._tl.ids = a[1], a[2]          # 심사에 넘길 인용 사례 전문을 잡는다(f19 `_answer` 감싸기와 같은 일)
    return out


def fingerprint():
    paths = sorted((ROOT / "app").glob("*.py")) + [ROOT / "pipeline" / n for n in ("rag.py", "f4_rag_answer_eval.py", "f19_a1_a2_eval.py", Path(__file__).name)]
    data = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    data["model"] = rag.CHAT_MODEL
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16], data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tag = "_f23s" if args.smoke else "_f23"
    ANS, JDG, RUN = (EVAL / f"{tag}_{x}" for x in ("answers.jsonl", "judge.jsonl", "run.json"))
    OUT = EVAL / ("f23_smoke.json" if args.smoke else "f23_merged_gate_eval.json")
    fp, parts = fingerprint()
    if RUN.exists():
        assert json.loads(RUN.read_text(encoding="utf-8"))["fingerprint"] == fp, "설정이 바뀌었다: 캐시와 섞지 않는다"
    else:
        f19.save(RUN, {"started_at": f19.now(), "fingerprint": fp, "parts": parts})
    f19.install()
    service._answer_merged = _merged
    service.ask = functools.partial(service.ask, use_gate="merged", use_verify=False)
    data = {n: [json.loads(x) for x in (EVAL / f"{n}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] for n in SETS + [UNANS]}
    if args.smoke:
        data = {n: v[:2] for n, v in data.items()}
    rows = [r for n in SETS for r in data[n]]
    with rag.connect() as conn:
        groups = {r["case_key"]: r["group_id"] for r in conn.execute("select case_key, group_id from interp.cases union all select case_key, group_id from interp.exmnt_cases").fetchall()}
        refs = f19.gold_refs(conn, rows)
    rag.embed("준비")
    answers = {(x["arm"], x["set"], x["qid"]): x for x in f19.read(ANS)}
    for arm in ("merged", "merged_holdout"):
        es.MAX_N = f19.ES_MAX + 1 if arm.endswith("holdout") else f19.ES_MAX
        todo = [(n, r) for n in SETS + ([UNANS] if arm == "merged" else []) for r in data[n] if (arm, n, r["qid"]) not in answers]
        t0 = time.time()
        with ThreadPoolExecutor(f19.WORKERS) as ex:
            for i, a in enumerate(ex.map(lambda x: f19.run_one(arm, x[0], x[1], groups), todo), 1):
                f19.append(ANS, a)
                answers[(arm, a["set"], a["qid"])] = a
                if i % 20 == 0 or i == len(todo):
                    print(f"[{arm}] 답변 {i}/{len(todo)} ({time.time() - t0:.0f}초)", flush=True)
    es.MAX_N = f19.ES_MAX
    judged = {(x["arm"], x["set"], x["qid"]): x for x in f19.read(JDG)}
    todo = [(r, answers[(arm, n, r["qid"])], refs[r["qid"]]) for arm in ("merged", "merged_holdout") for n in SETS for r in data[n] if (arm, n, r["qid"]) not in judged]
    with ThreadPoolExecutor(6) as ex:
        for i, j in enumerate(ex.map(f19.judge, todo), 1):
            f19.append(JDG, j)
            judged[(j["arm"], j["set"], j["qid"])] = j
            if i % 50 == 0 or i == len(todo):
                print(f"심사 {i}/{len(todo)}", flush=True)

    def col(arm):
        A = [answers[(arm, n, r["qid"])] for n in SETS for r in data[n]]
        J = [judged[(arm, n, r["qid"])] for n in SETS for r in data[n]]
        C, G = Counter(j["correctness"] for j in J), Counter(j["grounded"] for j in J)
        answered = sum(v for k, v in C.items() if k not in ("답변없음", "시스템오류"))
        corpus = [r["corpus"] for n in SETS for r in data[n]]
        return {"correctness": dict(C), "faithful_answered": round(G["충실"] / answered, 4) if answered else None,
                "by_corpus": {cp: dict(Counter(j["correctness"] for j, c in zip(J, corpus) if c == cp)) for cp in ("interp", "exmnt")},
                "gold_in_context": sum(a.get("gold_in_context", False) for a in A), "invalid": sum(len(a.get("invalid") or []) for a in A),
                "errors": sum(bool(a.get("error")) for a in A), "format_fallback": sum(bool((a.get("gate") or {}).get("error")) for a in A),
                "inconsistent": sum((a.get("gate") or {}).get("inconsistent", 0) for a in A),
                "unlisted": sum((a.get("gate") or {}).get("unlisted", 0) for a in A),
                "seconds_p50": statistics.median(a["seconds"] for a in A if "seconds" in a), "seconds_p90": sorted(a["seconds"] for a in A if "seconds" in a)[int(len(A) * 0.9) - 1]}
    norm, hold = col("merged"), col("merged_holdout")
    un = [answers[("merged", UNANS, r["qid"])] for r in data[UNANS]]
    nc, hc = Counter(norm["correctness"]), Counter(hold["correctness"])
    refused = sum(a.get("not_found", False) for a in un)
    errs = norm["errors"] + hold["errors"] + sum(bool(a.get("error")) for a in un)
    crit = {"M1_norm_wrong0": {"value": nc["불일치"], "pass": nc["불일치"] == 0},
            "M2_norm_match": {"value": nc["일치"], "limit": 268, "pass": nc["일치"] >= 268},
            "M3_hold_wrong": {"value": hc["불일치"], "limit": 20, "pass": hc["불일치"] <= 20},
            "M4_hold_other": {"value": hc["불일치"] + hc["부분일치"], "limit": 56, "pass": hc["불일치"] + hc["부분일치"] <= 56},
            "M5_unanswerable": {"value": refused, "limit": 20, "pass": refused == len(un)},
            "M6_seconds_p50": {"value": norm["seconds_p50"], "limit": 8.0, "pass": norm["seconds_p50"] <= 8.0},
            "M7_errors": {"errors": errs, "invalid": norm["invalid"] + hold["invalid"], "format_fallback": norm["format_fallback"] + hold["format_fallback"],
                          "pass": errs == 0 and norm["invalid"] + hold["invalid"] == 0 and norm["format_fallback"] + hold["format_fallback"] <= 3}}
    res = {"fingerprint": fp, "updated_at": f19.now(), "criteria": crit, "all_pass": all(c["pass"] for c in crit.values()),
           "normal": norm, "holdout": hold, "unanswerable": {"n": len(un), "refused": refused}}
    f19.save(OUT, res)
    print("통과" if res["all_pass"] else "미달", json.dumps({k: (v.get("value", v.get("errors")), v["pass"]) for k, v in crit.items()}, ensure_ascii=False))
    for k in ("normal", "holdout"):
        print(" ", k, json.dumps(res[k], ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
