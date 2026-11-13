"""F16 dev8·test8 평가: 지금 데모 설정(A) + 답변 검증 단계(B) (사용자 결정 2026-11-12 "평가 진행", worklog 98번).
A 지금 데모 설정: 벡터 20 → 리랭킹 5 → `APP_SYSTEM_V2`, 현장건의는 점수 ≥ 0.65이면서 회신 1위 점수보다 높을 때 최대 2건 (f13과 같은 코드, 첫 답변 기준)
B 검증 후: A의 답변을 `app/verify.py`에 그대로 넣어 고친 답변 (96번에서 개발셋 560문항으로 정한 것. 프롬프트·규칙을 바꾸지 않았다)
목적: 96번의 검증 단계 효과를 처음 보는 문항으로 확인한다. test8는 이번 한 번 평가하고 data/eval/f16_test8_lock.json으로 잠근다.
결과 파일에 test8 질문·답변 글은 넣지 않는다. 결과를 보고 설정·프롬프트·규칙을 고치지 않는다.

사전 등록한 확인 기준 (결과를 보기 전에 정함, test8 50문항 기준. 주 심사 = 지금까지 쓴 심사)
세트가 50문항이라 96번처럼 유의성을 요구할 수 없다(96번 효과 크기면 50문항에서 충실 됨 약 3건). 방향과 손실 한도만 본다.
1) 충실도: "충실"이 된 문항 수 ≥ "충실"이 아니게 된 문항 수 (방향이 뒤집히지 않음)
2) 정확도 "일치"가 2문항(4%p) 넘게 줄지 않는다
3) "답변없음"이 2문항 넘게 늘지 않는다
4) "불일치"가 늘지 않는다
5) 근거 번호 오류 0, 단정 표현이 늘지 않는다
함께 보고: 검색 Hit(벡터·리랭킹·현장건의 후보군), 인용 사례 전문으로 본 심사(96번에서 1,200자 제한이 충실도를 낮게 잡는다는 것을 확인), dev8의 같은 수치, 검증자가 고친 수
사용: python f16_dev8_test8_eval.py [--dev-only]     캐시: _f16_stage/answers/verify/judge/judge_full.jsonl     결과: data/eval/f16_dev8_test8_eval.json
"""
import json
import sys
import time
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
f13.ST = EVAL / "_f16_stage.jsonl"                    # f13.stage가 이 세트의 검색·리랭킹·현장건의 결과를 여기에 쓴다 (f13의 dev7·test7 캐시는 건드리지 않음)
ANS, VER, JDG, JFULL = EVAL / "_f16_answers.jsonl", EVAL / "_f16_verify.jsonl", EVAL / "_f16_judge.jsonl", EVAL / "_f16_judge_full.jsonl"
LOCK, OUT = EVAL / "f16_test8_lock.json", EVAL / "f16_dev8_test8_eval.json"
key = f13.key


def main():
    dev_only = "--dev-only" in sys.argv
    sets = ["dev8"] + ([] if dev_only else ["test8"])
    data = {n: f11.load(n) for n in sets}
    t0 = time.time()
    with rag.connect() as conn:
        fp = f2.fingerprint(conn)["id"]
        prev = {v["fingerprint"] for v in json.load(open(f2.LOCK, encoding="utf-8")).values()}
        if fp not in prev:
            sys.exit(f"설정 지문 {fp}가 이전 평가({sorted(prev)})와 다릅니다.")
        lock = json.load(open(LOCK, encoding="utf-8")) if LOCK.exists() else {}
        if not dev_only and lock.get("completed_at"):
            sys.exit(f"test8는 이미 평가했습니다 ({lock['completed_at']}). --dev-only로 dev8만 다시 돌릴 수 있습니다.")
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
    # 2) A 답변
    A = {key(r): r for r in f4.read_jsonl(ANS)}
    todo = [r for r in recs if key(r) not in A]
    print(f"[2 답변] 호출 대상 {len(todo)}", flush=True)
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for x in ex.map(lambda r: f11.answer(r, "B" if r["f_ids"] else "A", r["question"]), todo):
            f4.append(ANS, x)
            A[key(x)] = x
    # 3) 검증 단계
    V = {key(r): r for r in f4.read_jsonl(VER)}
    todo = [r for r in recs if key(r) not in V]
    print(f"[3 검증] 호출 대상 {len(todo)}", flush=True)

    def run(r):
        text, info = vf.verify(r["question"], A[key(r)]["answer"], r["context"])
        return {"set": r["set"], "qid": r["qid"], "text": text, "info": info}
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for x in ex.map(run, todo):
            f4.append(VER, x)
            V[key(x)] = x
    for r in recs:
        v = V[key(r)]
        r["base_a"], r["changed"] = A[key(r)], v["info"]["changed"]
        r["new_a"] = {"set": r["set"], "qid": r["qid"], "answer": f"{v['text']}\n\n{rag.CLOSING}", "check": f11.check(v["text"], r["ids"])} if r["changed"] else r["base_a"]
    ch = [r for r in recs if r["changed"]]
    # 4) 심사: 주 심사와 전문 심사, A는 전부 · B는 달라진 문항만
    todo = [(r, r["base_a"], "base") for r in recs] + [(r, r["new_a"], "new") for r in ch]
    J = f15.run_judge(JDG, todo, golds, f15.SHORT, "4 주 심사")
    JF = f15.run_judge(JFULL, todo, golds, f15.FULL, "5 전문 심사")
    pick = lambda D, which: (lambda r: D[(r["set"], r["qid"], which if which == "base" or r["changed"] else "base")])
    j0, j1, f0, f1 = pick(J, "base"), pick(J, "new"), pick(JF, "base"), pick(JF, "new")

    out = {"ran_at": datetime.now().isoformat(timespec="seconds"), "fingerprint": fp, "sets": {}}
    for n in sets:
        rs = [r for r in recs if r["set"] == n]
        JA = {key(r): j0(r) for r in rs}
        s = {"검색·답변(A, f13과 같은 집계)": {"전체": f13.summarize(rs, A, JA), "기존 회신 정답": f13.summarize([r for r in rs if r["corpus"] == "interp"], A, JA),
                                          "현장건의 정답": f13.summarize([r for r in rs if r["corpus"] == "exmnt"], A, JA)},
             "검증자": {"고침": sum(r["changed"] for r in rs), "간접(찾지 못함으로 바꿈)": sum(1 for r in rs if V[key(r)]["info"]["direct"] == "간접"),
                     "첫 줄 삭제": sum(1 for r in rs if V[key(r)]["info"]["changed"] and V[key(r)]["info"]["direct"] == "직접" and V[key(r)]["info"]["lead_removed"]),
                     "검증 실패": sum(1 for r in rs if V[key(r)]["info"].get("error"))},
             "주 심사": {"A": f15.score(rs, lambda r: r["base_a"], j0), "B 검증 후": f15.score(rs, lambda r: r["new_a"], j1), "같은 문항 비교": f15.paired(rs, j0, j1)},
             "전문 심사": {"A": f15.score(rs, lambda r: r["base_a"], f0), "B 검증 후": f15.score(rs, lambda r: r["new_a"], f1), "같은 문항 비교": f15.paired(rs, f0, f1)}}
        a, b, p = s["주 심사"]["A"], s["주 심사"]["B 검증 후"], s["주 심사"]["같은 문항 비교"]
        s["확인 기준"] = {"1 충실 방향": p["충실 됨/충실 아니게 됨"][0] >= p["충실 됨/충실 아니게 됨"][1], "2 일치 감소 ≤ 2문항": a["correctness"]["일치"] - b["correctness"]["일치"] <= 2,
                      "3 답변없음 증가 ≤ 2문항": b["correctness"]["답변없음"] - a["correctness"]["답변없음"] <= 2, "4 불일치 늘지 않음": b["correctness"]["불일치"] <= a["correctness"]["불일치"],
                      "5 번호 오류 0·단정 늘지 않음": b["번호 오류"] == 0 and b["단정"] <= a["단정"]}
        if n != "test8":
            s["items"] = [{"qid": r["qid"], "corpus": r["corpus"], "tag": r["tag"], "gold_key": r["gold_key"], "vec_rank": r["vec_rank"], "rank": r["rank"], "ex_rank": r["ex_rank"], "f_ids": r["f_ids"],
                           "changed": r["changed"], "A": [j0(r)["correctness"], j0(r)["grounded"]], "B": [j1(r)["correctness"], j1(r)["grounded"]], "A 전문": [f0(r)["correctness"], f0(r)["grounded"]],
                           "B 전문": [f1(r)["correctness"], f1(r)["grounded"]], "reason_A": j0(r)["reason"], "reason_B": j1(r)["reason"] if r["changed"] else None} for r in rs]
        out["sets"][n] = s
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not dev_only:
        LOCK.write_text(json.dumps({**json.load(open(LOCK, encoding="utf-8")), "completed_at": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False), encoding="utf-8")
    manifest("f16_dev8_test8_eval", sets=sets, fingerprint=fp)
    print(json.dumps({n: {k: v for k, v in s.items() if k != "items"} for n, s in out["sets"].items()}, ensure_ascii=False))
    print(f"{time.time() - t0:.0f}초")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
