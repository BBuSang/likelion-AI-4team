"""F17 dev9·test9 평가: 지금 데모 설정(A) + 검증 구성 "직접 다루지 않으면 거절, 없는 줄은 표시만"(C) (사용자 결정 2026-11-12, worklog 99번).
A 지금 데모 설정: 벡터 20 → 리랭킹 5 → `APP_SYSTEM_V2`, 현장건의 88번 조건 (f13·f16과 같은 코드, 첫 답변 기준)
C `verify.check`: A의 답변을 검증자에게 보여 ① 인용 사례가 질문의 대상·조건을 직접 다루지 않으면("간접") "찾지 못했습니다"로 바꾸고,
  ② 인용 사례에 없다고 본 줄은 지우지 않고 표시만 한다. 검증자 프롬프트는 96번 그대로다.
배경: 줄을 지우는 방식은 test8 확인에서 일치 손실이 한도를 넘었다(98번). 이 구성은 그 결과를 본 뒤에 고른 것이라 dev8·test8 점수를 근거로 쓸 수 없고, 처음 보는 문항으로 확인한다.
C의 답변 글은 거절한 문항만 A와 다르다. 거절한 문항은 심사 없이 "답변없음"이 되고(지금까지와 같은 규칙), 나머지는 A의 심사 결과와 같다. 따라서 추가 심사 호출이 없다.
test9는 이번 한 번 평가하고 data/eval/f17_test9_lock.json으로 잠근다. 결과 파일에 test9 질문·답변 글은 넣지 않는다. 결과를 보고 설정·프롬프트·규칙을 고치지 않는다.

사전 등록한 확인 기준 (결과를 보기 전에 정함, test9 100문항 기준. 주 심사 = 지금까지 쓴 심사)
1) 정확도 "일치"가 2문항(2%p) 넘게 줄지 않는다
2) "불일치"가 늘지 않는다
3) "답변없음"이 3문항 넘게 늘지 않는다
4) 검증 호출 실패 0, 근거 번호 오류 0
모두 통과하면 데모에 토글(기본 켬)로 연결한다. 하나라도 미달이면 연결하지 않고 그대로 보고한다.
함께 보고 (기준 아님)
- 거절한 문항이 A에서 무엇이었는지(불일치·부분일치·일치), 정답이 문맥에 있었는지
- 표시: 줄이 하나 이상 표시된 답변의 비율, 표시된 줄의 비율. 전문 심사에서 "일부추정"인 답변 중 표시가 붙은 비율(잡아낸 비율)과 "충실"인 답변 중 표시가 붙은 비율(잘못 붙은 비율)
- 검색 Hit, A의 답변 수치(주 심사·전문 심사), dev9의 같은 수치
사용: python f17_dev9_test9_eval.py [--dev-only]     캐시: _f17_stage/answers/check/judge/judge_full.jsonl     결과: data/eval/f17_dev9_test9_eval.json
"""
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import f11_dev6_test6_eval as f11  # noqa: E402
import f13_dev7_test7_eval as f13  # noqa: E402
import f15_verify_eval as f15  # noqa: E402
import f2_test_eval as f2  # noqa: E402
import f4_rag_answer_eval as f4  # noqa: E402
import rag  # noqa: E402
import verify as vf  # noqa: E402
from common import DATA, manifest  # noqa: E402

EVAL = DATA / "eval"
f13.ST = EVAL / "_f17_stage.jsonl"                    # f13.stage가 이 세트의 검색·리랭킹·현장건의 결과를 여기에 쓴다
ANS, CHK, JDG, JFULL = EVAL / "_f17_answers.jsonl", EVAL / "_f17_check.jsonl", EVAL / "_f17_judge.jsonl", EVAL / "_f17_judge_full.jsonl"
LOCK, OUT = EVAL / "f17_test9_lock.json", EVAL / "f17_dev9_test9_eval.json"
key = f13.key
NONE = {"correctness": "답변없음", "grounded": "해당없음", "reason": "검증 단계가 거절함(인용 사례가 질문을 직접 다루지 않음)"}
rate = lambda n, d: round(n / d, 3) if d else None


def main():
    dev_only = "--dev-only" in sys.argv
    sets = ["dev9"] + ([] if dev_only else ["test9"])
    data = {n: f11.load(n) for n in sets}
    t0 = time.time()
    with rag.connect() as conn:
        fp = f2.fingerprint(conn)["id"]
        prev = {v["fingerprint"] for v in json.load(open(f2.LOCK, encoding="utf-8")).values()}
        if fp not in prev:
            sys.exit(f"설정 지문 {fp}가 이전 평가({sorted(prev)})와 다릅니다.")
        lock = json.load(open(LOCK, encoding="utf-8")) if LOCK.exists() else {}
        if not dev_only and lock.get("completed_at"):
            sys.exit(f"test9는 이미 평가했습니다 ({lock['completed_at']}). --dev-only로 dev9만 다시 돌릴 수 있습니다.")
        if not dev_only and not lock:
            LOCK.write_text(json.dumps({"started_at": datetime.now().isoformat(timespec="seconds"), "fingerprint": fp}, ensure_ascii=False), encoding="utf-8")
        grp_rep = {r["group_id"]: r["representative_case_key"] for r in conn.execute("select group_id, representative_case_key from interp.case_groups").fetchall()}
        ex_rep = {r["group_id"]: r["representative_case_key"] for r in conn.execute("select group_id, representative_case_key from interp.exmnt_groups").fetchall()}
        rag.embed("준비")
        f13.stage(conn, sets, data, grp_rep, ex_rep)
        recs = [r for r in f4.read_jsonl(f13.ST) if r["set"] in sets]
        gk = [r["gold_key"] for r in recs]
        golds = {x["case_key"]: x for x in conn.execute("select case_key, title, question, answer, reason from interp.cases where case_key = any(%s)", (gk,)).fetchall()}
        golds.update({x["case_key"]: x for x in conn.execute(
            "select case_key, title, question, '검토결과: ' || coalesce(answer_type, '') || E'\\n' || coalesce(answer, '') as answer, '' as reason from interp.exmnt_cases where case_key = any(%s)", (gk,)).fetchall()})
    print(f"[1 검색·리랭킹·현장건의] {time.time() - t0:.0f}초", flush=True)
    questions = {(n, q["qid"]): q["question"] for n in sets for q in data[n]}
    for r in recs:
        r["question"], r["context"], r["ids"] = questions[key(r)], r["ctx_b"], r["ids_a"] + r["f_ids"]
    A = {key(r): r for r in f4.read_jsonl(ANS)}
    todo = [r for r in recs if key(r) not in A]
    print(f"[2 답변] 호출 대상 {len(todo)}", flush=True)
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for x in ex.map(lambda r: f11.answer(r, "B" if r["f_ids"] else "A", r["question"]), todo):
            f4.append(ANS, x)
            A[key(x)] = x
    C = {key(r): r for r in f4.read_jsonl(CHK)}
    todo = [r for r in recs if key(r) not in C]
    print(f"[3 검증] 호출 대상 {len(todo)}", flush=True)

    def run(r):
        text, info = vf.check(r["question"], A[key(r)]["answer"], r["context"])
        return {"set": r["set"], "qid": r["qid"], "info": {**info, "flagged": len(info["flagged"])}}          # 표시된 줄의 글은 남기지 않는다 (test9 답변 글)
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for x in ex.map(run, todo):
            f4.append(CHK, x)
            C[key(x)] = x
    for r in recs:
        r["base_a"], r["info"] = A[key(r)], C[key(r)]["info"]
        r["abst"] = r["info"]["abstained"]
        r["new_a"] = {"set": r["set"], "qid": r["qid"], "answer": rag.NOT_FOUND, "check": f11.check(rag.NOT_FOUND, r["ids"])} if r["abst"] else r["base_a"]
    todo = [(r, r["base_a"], "base") for r in recs]
    J = f15.run_judge(JDG, todo, golds, f15.SHORT, "4 주 심사")
    JF = f15.run_judge(JFULL, todo, golds, f15.FULL, "5 전문 심사")
    j0, f0 = (lambda r: J[(r["set"], r["qid"], "base")]), (lambda r: JF[(r["set"], r["qid"], "base")])
    j1, f1 = (lambda r: NONE if r["abst"] else j0(r)), (lambda r: NONE if r["abst"] else f0(r))
    in_ctx = lambda r: bool(r["rank"]) if r["corpus"] == "interp" else r["gold_group_id"] in r["f_groups"]

    out = {"ran_at": datetime.now().isoformat(timespec="seconds"), "fingerprint": fp, "sets": {}}
    for n in sets:
        rs = [r for r in recs if r["set"] == n]
        JA = {key(r): j0(r) for r in rs}
        chk = [r for r in rs if r["info"]["checked"] and not r["abst"]]
        est, ok = [r for r in chk if f0(r)["grounded"] == "일부추정"], [r for r in chk if f0(r)["grounded"] == "충실"]
        s = {"검색·답변(A)": {"전체": f13.summarize(rs, A, JA), "기존 회신 정답": f13.summarize([r for r in rs if r["corpus"] == "interp"], A, JA),
                            "현장건의 정답": f13.summarize([r for r in rs if r["corpus"] == "exmnt"], A, JA)},
             "주 심사": {"A": f15.score(rs, lambda r: r["base_a"], j0), "C 검증 구성": f15.score(rs, lambda r: r["new_a"], j1)},
             "전문 심사": {"A": f15.score(rs, lambda r: r["base_a"], f0), "C 검증 구성": f15.score(rs, lambda r: r["new_a"], f1)},
             "거절": {"문항 수": sum(r["abst"] for r in rs), "A에서의 판정": dict(Counter(j0(r)["correctness"] for r in rs if r["abst"])),
                    "정답이 문맥에 있었음": sum(1 for r in rs if r["abst"] and in_ctx(r)), "정답 종류": dict(Counter(r["corpus"] for r in rs if r["abst"]))},
             "표시": {"검증한 답변": len(chk), "줄이 표시된 답변": sum(1 for r in chk if r["info"]["flagged"]), "비율": rate(sum(1 for r in chk if r["info"]["flagged"]), len(chk)),
                    "표시된 줄 / 전체 줄": f"{sum(r['info']['flagged'] for r in chk)}/{sum(r['info']['n_lines'] for r in chk)}",
                    "전문 심사 일부추정 답변 중 표시 붙음": f"{sum(1 for r in est if r['info']['flagged'])}/{len(est)}", "전문 심사 충실 답변 중 표시 붙음": f"{sum(1 for r in ok if r['info']['flagged'])}/{len(ok)}"},
             "검증 실패": sum(1 for r in rs if r["info"].get("error"))}
        a, c = s["주 심사"]["A"], s["주 심사"]["C 검증 구성"]
        s["확인 기준"] = {"1 일치 감소 ≤ 2문항": a["correctness"]["일치"] - c["correctness"]["일치"] <= 2, "2 불일치 늘지 않음": c["correctness"]["불일치"] <= a["correctness"]["불일치"],
                      "3 답변없음 증가 ≤ 3문항": c["correctness"]["답변없음"] - a["correctness"]["답변없음"] <= 3, "4 검증 실패 0·번호 오류 0": s["검증 실패"] == 0 and c["번호 오류"] == 0}
        if n != "test9":
            s["items"] = [{"qid": r["qid"], "corpus": r["corpus"], "tag": r["tag"], "gold_key": r["gold_key"], "vec_rank": r["vec_rank"], "rank": r["rank"], "ex_rank": r["ex_rank"], "f_ids": r["f_ids"],
                           "abstained": r["abst"], "flagged": r["info"]["flagged"], "n_lines": r["info"]["n_lines"], "A": [j0(r)["correctness"], j0(r)["grounded"]], "A 전문": [f0(r)["correctness"], f0(r)["grounded"]],
                           "reason_A": j0(r)["reason"]} for r in rs]
        out["sets"][n] = s
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not dev_only:
        LOCK.write_text(json.dumps({**json.load(open(LOCK, encoding="utf-8")), "completed_at": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False), encoding="utf-8")
    manifest("f17_dev9_test9_eval", sets=sets, fingerprint=fp)
    print(json.dumps({n: {k: v for k, v in s.items() if k not in ("items", "검색·답변(A)")} for n, s in out["sets"].items()}, ensure_ascii=False))
    print(f"{time.time() - t0:.0f}초")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
