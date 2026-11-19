"""F22: dev11·test11로 지금 데모 설정 확인 (worklog 122번, 118번 A4, 사용자 "open ai로 dev11,test11 50건씩 만들어주고 검증해봐", 2026-11-17).
120·121번의 직접성 판정 + 답변 검증은 dev7~10으로 고르고 쟀다. 한 번도 보지 않은 문항에서도 같은 방향인지 확인한다.

갈래 (같은 코드·같은 시점, 프롬프트는 모두 V3)
- cur  = 지금 데모 기본값(`service.ask()`: 리랭킹 + 현장건의 + 키워드 완화 + 기간 조건 + 직접성 판정 + 답변 검증)
- base = 120번 이전 설정(같은 경로에서 use_gate=False, use_verify=False). f19 V3와 같다
- cur_holdout, base_holdout = 각각 정답 묶음을 검색 대상에서 뺀 것(f19와 같은 방식)
심사: f19와 같은 전문 심사. 문항: dev11 50, test11 50(각 회신 35 + 현장건의 15).

순서: dev11을 먼저 돌려 동작을 확인하고, 시스템을 바꾸지 않은 채 test11을 한 번 돌린다(설정 지문이 같아야 함). test11은 끝나면
data/eval/f22_test11_lock.json으로 잠그고 다시 돌리지 않는다. test11의 질문·답변은 콘솔에 출력하지 않는다.

사전 기준 (dev7~10에서 본 방향을 50문항 규모에 맞춘 것. 돌리기 전에 정했고 결과를 보고 고치지 않는다. 세트마다 판정)
- T1 cur 정상 불일치 0
- T2 cur 정상 일치 ≥ base 정상 일치 − 3 (dev7~10: −11/300)
- T3 cur 정답 제외 불일치 ≤ base 정답 제외 불일치 / 3 (dev7~10: 8 / 74)
- T4 cur 정답 제외 불일치 + 부분 ≤ base 정답 제외 불일치 + 부분 / 2 (dev7~10: 30 / 166)
- T5 시스템 오류·판정/검증 호출 실패 0, 잘못된 사례번호 0
test11에서 하나라도 못 넘으면 설정을 고치지 않고 보고한다(되돌릴지는 사용자가 정한다).
한계: 합성 단일 정답 질문, 생성·판정·검증·심사가 같은 모델, 50문항이라 1문항 = 2%p. 응답시간은 4개씩 동시에 돌려 참고값이다.
사용: python pipeline/f22_dev11_test11_eval.py [--dev-only]
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
EVAL = f19.EVAL
ANS, JDG, RUN, LOCK, OUT = (EVAL / x for x in ("_f22_answers.jsonl", "_f22_judge.jsonl", "_f22_run.json", "f22_test11_lock.json", "f22_dev11_test11_eval.json"))
ARMS = {"cur": {}, "base": {"use_gate": False, "use_verify": False}}
ASK = service.ask


def fingerprint():
    paths = sorted((ROOT / "app").glob("*.py")) + [ROOT / "pipeline" / n for n in ("rag.py", "f4_rag_answer_eval.py", "f19_a1_a2_eval.py", Path(__file__).name)]
    data = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    data.update(model=rag.CHAT_MODEL, sets={n: hashlib.sha256((EVAL / f"{n}.jsonl").read_bytes()).hexdigest() for n in ("dev11", "test11")})
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16], data


def run_set(name, rows, groups, refs, answers, judged):
    for arm in ("cur", "base", "cur_holdout", "base_holdout"):
        service.ask = functools.partial(ASK, **ARMS[arm.replace("_holdout", "")])
        es.MAX_N = f19.ES_MAX + 1 if arm.endswith("holdout") else f19.ES_MAX
        todo = [r for r in rows if (arm, name, r["qid"]) not in answers]
        t0 = time.time()
        with ThreadPoolExecutor(f19.WORKERS) as ex:
            for i, a in enumerate(ex.map(lambda r: f19.run_one(arm, name, r, groups), todo), 1):
                f19.append(ANS, a)
                answers[(arm, name, a["qid"])] = a
                if i % 25 == 0 or i == len(todo):
                    print(f"{name} [{arm}] 답변 {i}/{len(todo)} ({time.time() - t0:.0f}초)", flush=True)
    service.ask, es.MAX_N = ASK, f19.ES_MAX
    todo = [(r, answers[(arm, name, r["qid"])], refs[r["qid"]]) for arm in ("cur", "base", "cur_holdout", "base_holdout") for r in rows if (arm, name, r["qid"]) not in judged]
    with ThreadPoolExecutor(6) as ex:
        for i, j in enumerate(ex.map(f19.judge, todo), 1):
            f19.append(JDG, j)
            judged[(j["arm"], j["set"], j["qid"])] = j
    print(f"{name} 심사 {len(todo)}건", flush=True)


def report(name, rows, answers, judged):
    out = {}
    for arm in ("cur", "base", "cur_holdout", "base_holdout"):
        A = [answers[(arm, name, r["qid"])] for r in rows]
        C = Counter(judged[(arm, name, r["qid"])]["correctness"] for r in rows)
        G = Counter(judged[(arm, name, r["qid"])]["grounded"] for r in rows)
        answered = sum(v for k, v in C.items() if k not in ("답변없음", "시스템오류"))
        by = {cp: dict(Counter(judged[(arm, name, r["qid"])]["correctness"] for r in rows if r["corpus"] == cp)) for cp in ("interp", "exmnt")}
        out[arm] = {"correctness": dict(C), "faithful_answered": round(G["충실"] / answered, 3) if answered else None, "by_corpus": by,
                    "gold_in_context": sum(a.get("gold_in_context", False) for a in A), "gold_cited": sum(a.get("gold_cited", False) for a in A),
                    "invalid": sum(len(a.get("invalid") or []) for a in A), "errors": sum(bool(a.get("error")) for a in A),
                    "gate_fail": sum(bool((a.get("gate") or {}).get("error")) for a in A),
                    "seconds_p50": statistics.median(a["seconds"] for a in A if "seconds" in a)}
    c, b, ch, bh = (Counter(out[k]["correctness"]) for k in ("cur", "base", "cur_holdout", "base_holdout"))
    crit = {"T1_cur_norm_wrong0": {"value": c["불일치"], "pass": c["불일치"] == 0},
            "T2_cur_norm_match": {"value": c["일치"], "limit": b["일치"] - 3, "pass": c["일치"] >= b["일치"] - 3},
            "T3_cur_hold_wrong": {"value": ch["불일치"], "limit": round(bh["불일치"] / 3, 2), "pass": ch["불일치"] <= bh["불일치"] / 3},
            "T4_cur_hold_other": {"value": ch["불일치"] + ch["부분일치"], "limit": (bh["불일치"] + bh["부분일치"]) / 2, "pass": ch["불일치"] + ch["부분일치"] <= (bh["불일치"] + bh["부분일치"]) / 2},
            "T5_errors": {"value": sum(out[k]["errors"] + out[k]["gate_fail"] + out[k]["invalid"] for k in out), "pass": sum(out[k]["errors"] + out[k]["gate_fail"] + out[k]["invalid"] for k in out) == 0}}
    return {"arms": out, "criteria": crit, "all_pass": all(x["pass"] for x in crit.values())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev-only", action="store_true")
    args = ap.parse_args()
    if not args.dev_only and LOCK.exists() and json.loads(LOCK.read_text(encoding="utf-8")).get("completed_at"):
        raise SystemExit("test11은 이미 평가했고 잠겼다")
    fp, parts = fingerprint()
    if RUN.exists():
        assert json.loads(RUN.read_text(encoding="utf-8"))["fingerprint"] == fp, "설정이 바뀌었다: dev11과 같은 설정으로만 test11을 돈다"
    else:
        f19.save(RUN, {"started_at": f19.now(), "fingerprint": fp, "parts": parts})
    f19.install()
    with rag.connect() as conn:
        groups = {r["case_key"]: r["group_id"] for r in conn.execute("select case_key, group_id from interp.cases union all select case_key, group_id from interp.exmnt_cases").fetchall()}
    rag.embed("준비")
    answers = {(x["arm"], x["set"], x["qid"]): x for x in f19.read(ANS)}
    judged = {(x["arm"], x["set"], x["qid"]): x for x in f19.read(JDG)}
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"sets": {}}
    for name in ["dev11"] + ([] if args.dev_only else ["test11"]):
        rows = [json.loads(x) for x in (EVAL / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        assert len(rows) == 50
        if name == "test11" and not LOCK.exists():
            f19.save(LOCK, {"started_at": f19.now(), "fingerprint": fp})
        with rag.connect() as conn:
            refs = f19.gold_refs(conn, rows)
        run_set(name, rows, groups, refs, answers, judged)
        res["sets"][name] = report(name, rows, answers, judged)
        res.update(fingerprint=fp, updated_at=f19.now())
        f19.save(OUT, res)
        r = res["sets"][name]
        print(name, "통과" if r["all_pass"] else "미달", json.dumps({k: (v["value"], v["pass"]) for k, v in r["criteria"].items()}, ensure_ascii=False))
        for arm, v in r["arms"].items():
            print(f"  {arm}: {v['correctness']} 충실 {v['faithful_answered']} 정답문맥 {v['gold_in_context']} 중앙값 {v['seconds_p50']}초 회신·현장건의 {v['by_corpus']}")
        if name == "test11":
            f19.save(LOCK, {**json.loads(LOCK.read_text(encoding="utf-8")), "completed_at": f19.now()})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
