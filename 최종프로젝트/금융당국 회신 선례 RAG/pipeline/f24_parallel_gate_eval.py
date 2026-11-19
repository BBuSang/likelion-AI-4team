"""F24: 판정을 답변과 동시에 + 임베딩 중복 제거 (worklog 124번, 사용자: "이 답변 시간이 늘어나지 않는 방향으로 개선할 방법은 없어?" → 추천안 "판정·답변 동시 실행 + 임베딩 중복 제거, 검증 끔").
개발셋만 쓴다(dev7~10 300 + 정답 제외 300 + 답변불가 20, f19와 같은 방식·같은 심사). 시험 세트는 읽지 않는다.

갈래 P = `service.ask(q, use_gate="parallel", use_verify=False)` (`service._answer_parallel`): 판정과 답변을 동시에 시작하고,
'직접' 사례가 없으면 찾지 못했습니다, 먼저 만든 답이 '직접' 사례만 인용했으면 그 답, 그 밖이면 '직접' 사례만으로 다시 답한다.
임베딩 중복 제거(`service._embed`)는 모든 설정에 함께 들어간다(같은 글 → 같은 벡터라 결과가 바뀌지 않음).

사전 기준 (돌리기 전에 정함. 결과를 보고 규칙·기준을 고치지 않는다). 품질 기준은 f20에서 판정(G)을 채택할 때 쓴 기준과 같다.
- P1 정상 불일치 0
- P2 정상 일치 ≥ 268
- P3 정답 제외 불일치 ≤ 24
- P4 정답 제외 불일치 + 부분일치 ≤ 83
- P5 정상 불일치 + 부분일치 ≤ 17
- P6 답변불가 20 모두 거절
- P7 응답시간(--latency, 같은 시각에 문항마다 순서를 돌려 순차로 번갈아, dev11 앞 16문항, 설정마다 임베딩 기억을 비우고 잼):
       P 중앙값 ≤ 9.0초, 그리고 P 중앙값 ≤ 판정만(use_gate=True, 검증 끔) 중앙값 − 2.0초
- P8 시스템 오류 0, 잘못된 사례번호 0, 판정 호출 실패 3건 이하
모두 통과하면 데모 기본값을 P(동시 판정, 검증 끔)로 바꾼다. 하나라도 못 넘으면 지금 설정을 두고 보고한다.
한계: 합성 단일 정답 질문, 생성·판정·심사가 같은 모델. dev7~10은 판정·검증을 고를 때 쓴 세트다. 시간은 API 속도가 시간대마다 달라 같은 시각 비교만 의미가 있다.
사용: python pipeline/f24_parallel_gate_eval.py [--smoke] [--latency]
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
ASK = service.ask
LAT_CONF = {"끔": dict(use_gate=False, use_verify=False), "판정만": dict(use_gate=True, use_verify=False), "동시 판정(P)": dict(use_gate="parallel", use_verify=False)}


def fingerprint():
    paths = sorted((ROOT / "app").glob("*.py")) + [ROOT / "pipeline" / n for n in ("rag.py", "f4_rag_answer_eval.py", "f19_a1_a2_eval.py", Path(__file__).name)]
    data = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    data["model"] = rag.CHAT_MODEL
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16], data


def latency(n=16):
    """같은 시각, 순차, 설정 순서를 문항마다 돌림. 설정마다 임베딩 기억을 비워 첫 설정만 계산하는 불공평을 없앤다."""
    names = list(LAT_CONF)
    qs = [json.loads(x)["question"] for x in (EVAL / "dev11.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()][:n]
    T = {k: [] for k in names}
    for i, q in enumerate(qs):
        for k in names[i % len(names):] + names[:i % len(names)]:
            service._embed_cached.cache_clear()
            t = time.time()
            ASK(q, **LAT_CONF[k])
            T[k].append(round(time.time() - t, 2))
        print(f"시간 {i + 1}/{len(qs)}", {k: T[k][-1] for k in names}, flush=True)
    return {k: {"median": statistics.median(v), "mean": round(statistics.mean(v), 2), "max": max(v), "n": len(v), "values": v} for k, v in T.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--latency", action="store_true")
    args = ap.parse_args()
    tag = "_f24s" if args.smoke else "_f24"
    ANS, JDG, RUN = (EVAL / f"{tag}_{x}" for x in ("answers.jsonl", "judge.jsonl", "run.json"))
    OUT = EVAL / ("f24_smoke.json" if args.smoke else "f24_parallel_gate_eval.json")
    fp, parts = fingerprint()
    if RUN.exists():
        assert json.loads(RUN.read_text(encoding="utf-8"))["fingerprint"] == fp, "설정이 바뀌었다: 캐시와 섞지 않는다"
    else:
        f19.save(RUN, {"started_at": f19.now(), "fingerprint": fp, "parts": parts})
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    if args.latency:
        rag.embed("준비")
        res["latency"] = latency(4 if args.smoke else 16)
        f19.save(OUT, res)
        print(json.dumps({k: {x: v[x] for x in ("median", "mean", "max")} for k, v in res["latency"].items()}, ensure_ascii=False))
        return
    f19.install()
    service.ask = functools.partial(ASK, use_gate="parallel", use_verify=False)
    data = {n: [json.loads(x) for x in (EVAL / f"{n}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] for n in SETS + [UNANS]}
    if args.smoke:
        data = {n: v[:2] for n, v in data.items()}
    rows = [r for n in SETS for r in data[n]]
    with rag.connect() as conn:
        groups = {r["case_key"]: r["group_id"] for r in conn.execute("select case_key, group_id from interp.cases union all select case_key, group_id from interp.exmnt_cases").fetchall()}
        refs = f19.gold_refs(conn, rows)
    rag.embed("준비")
    answers = {(x["arm"], x["set"], x["qid"]): x for x in f19.read(ANS)}
    for arm in ("parallel", "parallel_holdout"):
        es.MAX_N = f19.ES_MAX + 1 if arm.endswith("holdout") else f19.ES_MAX
        todo = [(n, r) for n in SETS + ([UNANS] if arm == "parallel" else []) for r in data[n] if (arm, n, r["qid"]) not in answers]
        t0 = time.time()
        with ThreadPoolExecutor(f19.WORKERS) as ex:
            for i, a in enumerate(ex.map(lambda x: f19.run_one(arm, x[0], x[1], groups), todo), 1):
                f19.append(ANS, a)
                answers[(arm, a["set"], a["qid"])] = a
                if i % 20 == 0 or i == len(todo):
                    print(f"[{arm}] 답변 {i}/{len(todo)} ({time.time() - t0:.0f}초)", flush=True)
    es.MAX_N = f19.ES_MAX
    judged = {(x["arm"], x["set"], x["qid"]): x for x in f19.read(JDG)}
    todo = [(r, answers[(arm, n, r["qid"])], refs[r["qid"]]) for arm in ("parallel", "parallel_holdout") for n in SETS for r in data[n] if (arm, n, r["qid"]) not in judged]
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
        g = [a.get("gate") or {} for a in A]
        return {"correctness": dict(C), "faithful_answered": round(G["충실"] / answered, 4) if answered else None,
                "by_corpus": {cp: dict(Counter(j["correctness"] for j, c in zip(J, corpus) if c == cp)) for cp in ("interp", "exmnt")},
                "gold_in_context": sum(a.get("gold_in_context", False) for a in A), "invalid": sum(len(a.get("invalid") or []) for a in A),
                "errors": sum(bool(a.get("error")) for a in A), "gate_fail": sum(bool(x.get("error")) for x in g),
                "kept_first": sum(x.get("kept_first", 0) for x in g), "reanswered": sum(x.get("reanswered", 0) for x in g)}
    norm, hold = col("parallel"), col("parallel_holdout")
    un = [answers[("parallel", UNANS, r["qid"])] for r in data[UNANS]]
    nc, hc = Counter(norm["correctness"]), Counter(hold["correctness"])
    refused = sum(a.get("not_found", False) for a in un)
    errs = norm["errors"] + hold["errors"] + sum(bool(a.get("error")) for a in un)
    crit = {"P1_norm_wrong0": {"value": nc["불일치"], "pass": nc["불일치"] == 0},
            "P2_norm_match": {"value": nc["일치"], "limit": 268, "pass": nc["일치"] >= 268},
            "P3_hold_wrong": {"value": hc["불일치"], "limit": 24, "pass": hc["불일치"] <= 24},
            "P4_hold_other": {"value": hc["불일치"] + hc["부분일치"], "limit": 83, "pass": hc["불일치"] + hc["부분일치"] <= 83},
            "P5_norm_other": {"value": nc["불일치"] + nc["부분일치"], "limit": 17, "pass": nc["불일치"] + nc["부분일치"] <= 17},
            "P6_unanswerable": {"value": refused, "limit": len(un), "pass": refused == len(un)},
            "P8_errors": {"errors": errs, "invalid": norm["invalid"] + hold["invalid"], "gate_fail": norm["gate_fail"] + hold["gate_fail"],
                          "pass": errs == 0 and norm["invalid"] + hold["invalid"] == 0 and norm["gate_fail"] + hold["gate_fail"] <= 3}}
    lat = res.get("latency")
    if lat:
        p, g = lat["동시 판정(P)"]["median"], lat["판정만"]["median"]
        crit["P7_latency"] = {"value": p, "gate_only": g, "pass": p <= 9.0 and p <= g - 2.0}
    res.update(fingerprint=fp, updated_at=f19.now(), criteria=crit, all_pass=all(c["pass"] for c in crit.values()) and "P7_latency" in crit,
               normal=norm, holdout=hold, unanswerable={"n": len(un), "refused": refused})
    f19.save(OUT, res)
    print("통과" if res["all_pass"] else "미달(또는 시간 측정 전)", json.dumps({k: (v.get("value", v.get("errors")), v["pass"]) for k, v in crit.items()}, ensure_ascii=False))
    for k in ("normal", "holdout"):
        print(" ", k, json.dumps(res[k], ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
