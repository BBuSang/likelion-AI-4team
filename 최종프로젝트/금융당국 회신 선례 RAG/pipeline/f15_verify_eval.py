"""F15 답변 검증 단계(`app/verify.py`)의 개발셋 평가 (사용자 결정 2026-11-12 "진행", worklog 96번).
대상: 지금 데모 설정으로 답변·심사가 이미 있는 개발 데이터 560문항 = 합성 150 + dev1~4 260(f10) + dev6 50(f11 A) + dev7 100(f13). 시험 세트는 읽지 않는다(test7 포함).
방법: 이미 있는 답변을 그대로 검증 단계에 넣어 고친 답변을 만든다(답변을 새로 생성하지 않으므로 차이는 검증 단계뿐이다).
고친 답변이 원래와 같으면 기존 심사 결과를 쓰고, 달라진 문항만 다시 심사한다.

사전 등록한 합격 기준 (결과를 보기 전에 정함. 미달이어도 검증 프롬프트·규칙을 고쳐 다시 재지 않는다)
주 심사 = 지금까지 쓴 심사(f4.judge_one, 인용 사례를 사례당 1,200자·최대 3건까지 봄). 이전 수치와 비교할 수 있게 그대로 쓴다.
1) 충실도: 같은 문항 비교에서 "충실"이 된 문항이 "충실"이 아니게 된 문항보다 많고 부호 검정 p < 0.05
2) 정확도 "일치" 비율이 3%p 넘게 떨어지지 않는다
3) "답변없음"이 2%p(11문항) 넘게 늘지 않는다
4) "불일치" 수가 늘지 않는다
5) 근거 번호 오류 0, 단정 표현이 늘지 않는다
함께 보고: 검증자가 고친 비율(줄 삭제, 첫 줄 삭제, "간접"), 판정 변화표, 토큰, 검증 실패 수

덧붙인 측정 (기준 아님): 심사자가 인용 사례 글을 1,200자까지만 보는 것의 영향
- 개발셋 사례 글의 약 25%가 1,200자를 넘는다. 근거가 잘린 뒤쪽에 있으면 답변이 맞아도 "일부추정"이 될 수 있다
- 기존 답변 560개를 인용 사례 전문으로 한 번 더 심사한다(같은 심사 프롬프트). 충실도가 어느 쪽으로 얼마나 바뀌는지 본다
- 고친 답변도 전문 심사를 함께 보고한다
사용: python f15_verify_eval.py     캐시: _f15_verify.jsonl, _f15_judge.jsonl(주 심사), _f15_judge_full.jsonl(전문 심사)     결과: data/eval/f15_verify_eval.json
"""
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import f11_dev6_test6_eval as f11  # noqa: E402  (f4.cited_context를 현장건의 포함 판으로 바꿔 끼움, check 재사용)
import f13_dev7_test7_eval as f13  # noqa: E402
import f4_rag_answer_eval as f4  # noqa: E402
import f9_rerank_eval as f9  # noqa: E402
import rag  # noqa: E402
import verify as vf  # noqa: E402
from common import DATA, manifest  # noqa: E402
from f5_app_prompt_eval import RANK, sign_p  # noqa: E402

EVAL = DATA / "eval"
VER, JDG, JFULL, OUT = EVAL / "_f15_verify.jsonl", EVAL / "_f15_judge.jsonl", EVAL / "_f15_judge_full.jsonl", EVAL / "f15_verify_eval.json"
F10_SETS = ("rewrite_150", "dev1", "dev2", "dev3", "dev4")
key = lambda r: (r["set"], r["qid"])
SHORT = f11.cited_context                                              # 사례당 1,200자, 최대 3건 (지금까지의 심사)
FULL = lambda ctx, cited: f11.cited_context(ctx, cited, limit=10 ** 6, max_cases=99)


def load():
    """문항마다 {set, qid, gold_key, context, ids, question, base_a, base_j}."""
    recs, questions = [], {}
    for n in F10_SETS + ("dev6", "dev7"):
        for q in f11.load(n):
            questions[(n, q["qid"])] = q["question"]
    a10, j10 = {key(r): r for r in f4.read_jsonl(EVAL / "_f10_answers.jsonl")}, {key(r): r for r in f4.read_jsonl(EVAL / "_f10_judge.jsonl")}
    rer = {key(r): r for r in f4.read_jsonl(f9.R9)}
    with rag.connect() as conn:
        for b in f4.read_jsonl(f4.S1):                                 # f10과 같은 방식으로 문맥을 다시 만든다 (리랭킹 뒤 5건)
            if b["set"] in F10_SETS and b["gold_key"]:
                ids = rer[key(b)]["order"]
                cases = rag.load_cases(ids, conn)
                recs.append({"set": b["set"], "qid": b["qid"], "gold_key": b["gold_key"], "ids": ids, "context": "\n\n".join(rag.case_block(cases[i]) for i in ids), "base_a": a10[key(b)], "base_j": j10[key(b)]})
        a11 = {key(r): r for r in f4.read_jsonl(f11.ANS) if r["mode"] == "A"}
        j11 = {key(r): r for r in f4.read_jsonl(f11.JDG) if r["mode"] == "A"}
        for s in f4.read_jsonl(f11.ST):
            if s["set"] == "dev6":
                recs.append({"set": "dev6", "qid": s["qid"], "gold_key": s["gold_key"], "ids": s["ids_a"], "context": s["ctx_a"], "base_a": a11[key(s)], "base_j": j11[key(s)]})
        a13, j13 = {key(r): r for r in f4.read_jsonl(f13.ANS)}, {key(r): r for r in f4.read_jsonl(f13.JDG)}
        for s in f4.read_jsonl(f13.ST):
            if s["set"] == "dev7":
                recs.append({"set": "dev7", "qid": s["qid"], "gold_key": s["gold_key"], "ids": s["ids_a"] + s["f_ids"], "context": s["ctx_b"], "base_a": a13[key(s)], "base_j": j13[key(s)]})
        gk = [r["gold_key"] for r in recs]
        golds = {x["case_key"]: x for x in conn.execute("select case_key, title, question, answer, reason from interp.cases where case_key = any(%s)", (gk,)).fetchall()}
        golds.update({x["case_key"]: x for x in conn.execute(
            "select case_key, title, question, '검토결과: ' || coalesce(answer_type, '') || E'\\n' || coalesce(answer, '') as answer, '' as reason from interp.exmnt_cases where case_key = any(%s)", (gk,)).fetchall()})
    for r in recs:
        r["question"] = questions[key(r)]
    return recs, golds


def run_judge(path, todo, golds, ctx_fn, tag):
    """todo: [(rec, answer, which)]. which = 'base' | 'new'."""
    done = {(r["set"], r["qid"], r["which"]): r for r in f4.read_jsonl(path)}
    f4.cited_context = ctx_fn
    rest = [t for t in todo if (t[0]["set"], t[0]["qid"], t[2]) not in done]
    print(f"[{tag}] 호출 대상 {len(rest)}", flush=True)
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for x in ex.map(lambda t: {**f4.judge_one(t[0], t[1], t[0]["question"], golds[t[0]["gold_key"]]), "which": t[2]}, rest):
            f4.append(path, x)
            done[(x["set"], x["qid"], x["which"])] = x
    f4.cited_context = SHORT
    return done


def score(recs, a_of, j_of):
    n = len(recs)
    a, j = [a_of(r) for r in recs], [j_of(r) for r in recs]
    c = Counter(x["correctness"] for x in j)
    g = Counter(x["grounded"] for x in j if x["correctness"] != "답변없음")
    return {"n": n, "일치": round(c["일치"] / n, 4), "correctness": {k: c[k] for k in RANK}, "충실": round(g["충실"] / max(sum(g.values()), 1), 4), "grounded": {k: g[k] for k in ("충실", "일부추정", "근거없음")},
            "충실(전체 문항 대비)": round(g["충실"] / n, 4), "찾지 못함": sum(x["check"]["not_found"] for x in a), "번호 오류": sum(len(x["check"]["invalid"]) for x in a), "단정": sum(x["check"]["assertive"] for x in a)}


def paired(recs, j0, j1):
    b = sum(1 for r in recs if RANK[j1(r)["correctness"]] > RANK[j0(r)["correctness"]])
    w = sum(1 for r in recs if RANK[j1(r)["correctness"]] < RANK[j0(r)["correctness"]])
    gb = sum(1 for r in recs if j0(r)["grounded"] != "충실" and j1(r)["grounded"] == "충실")
    gw = sum(1 for r in recs if j0(r)["grounded"] == "충실" and j1(r)["grounded"] != "충실")
    return {"정확도 좋아짐/나빠짐": [b, w], "p": round(sign_p(b, w), 4), "충실 됨/충실 아니게 됨": [gb, gw], "p_충실": round(sign_p(gb, gw), 4),
            "정확도 변화": {f"{x}→{y}": n for (x, y), n in Counter((j0(r)["correctness"], j1(r)["correctness"]) for r in recs).most_common() if x != y},
            "충실도 변화": {f"{x}→{y}": n for (x, y), n in Counter((j0(r)["grounded"], j1(r)["grounded"]) for r in recs).most_common() if x != y}}


def main():
    t0 = time.time()
    recs, golds = load()
    print(f"[0] 문항 {len(recs)} {dict(Counter(r['set'] for r in recs))}", flush=True)
    # 1) 검증 단계
    V = {key(r): r for r in f4.read_jsonl(VER)}
    todo = [r for r in recs if key(r) not in V]
    print(f"[1 검증] 호출 대상 {len(todo)}", flush=True)

    def run(r):
        text, info = vf.verify(r["question"], r["base_a"]["answer"], r["context"])
        return {"set": r["set"], "qid": r["qid"], "text": text, "info": info}
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for x in ex.map(run, todo):
            f4.append(VER, x)
            V[key(x)] = x
    for r in recs:
        v = V[key(r)]
        r["changed"] = v["info"]["changed"]
        r["new_a"] = {"set": r["set"], "qid": r["qid"], "answer": f"{v['text']}\n\n{rag.CLOSING}", "check": f11.check(v["text"], r["ids"])} if r["changed"] else r["base_a"]
    ch = [r for r in recs if r["changed"]]
    print(f"[1 검증] 고친 문항 {len(ch)} / {len(recs)}  {time.time() - t0:.0f}초", flush=True)
    # 2) 주 심사: 달라진 문항만
    J = run_judge(JDG, [(r, r["new_a"], "new") for r in ch], golds, SHORT, "2 주 심사")
    j0 = lambda r: r["base_j"]
    j1 = lambda r: J[(r["set"], r["qid"], "new")] if r["changed"] else r["base_j"]
    # 3) 전문 심사: 기존 답변 전부 + 달라진 문항의 고친 답변
    JF = run_judge(JFULL, [(r, r["base_a"], "base") for r in recs] + [(r, r["new_a"], "new") for r in ch], golds, FULL, "3 전문 심사")
    f0 = lambda r: JF[(r["set"], r["qid"], "base")]
    f1 = lambda r: JF[(r["set"], r["qid"], "new")] if r["changed"] else f0(r)

    base, new = score(recs, lambda r: r["base_a"], j0), score(recs, lambda r: r["new_a"], j1)
    info = [V[key(r)]["info"] for r in recs]
    out = {"ran_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n": len(recs),
           "검증자": {"검증함": sum(i["checked"] for i in info), "고침": len(ch), "줄 삭제만": sum(1 for i in info if i["changed"] and i["direct"] == "직접" and not i["lead_removed"]),
                   "첫 줄 삭제": sum(1 for i in info if i["changed"] and i["direct"] == "직접" and i["lead_removed"]), "간접(찾지 못함으로 바꿈)": sum(1 for i in info if i["direct"] == "간접"),
                   "지운 줄 수": sum(i["removed"] for i in info if i["direct"] == "직접"), "검증 실패": sum(1 for i in info if i.get("error")),
                   "토큰": {"input": sum(i["usage"]["input"] for i in info), "output": sum(i["usage"]["output"] for i in info)}},
           "주 심사": {"기존": base, "검증 후": new, "같은 문항 비교": paired(recs, j0, j1)},
           "전문 심사": {"기존": score(recs, lambda r: r["base_a"], f0), "검증 후": score(recs, lambda r: r["new_a"], f1), "같은 문항 비교": paired(recs, f0, f1)},
           "심사 방식 비교(기존 답변, 1,200자 → 전문)": paired(recs, j0, f0),
           "세트별(주 심사, 일치 / 충실)": {n: {"기존": [score(rs, lambda r: r["base_a"], j0)[k] for k in ("일치", "충실")], "검증 후": [score(rs, lambda r: r["new_a"], j1)[k] for k in ("일치", "충실")]}
                                    for n in F10_SETS + ("dev6", "dev7") for rs in [[r for r in recs if r["set"] == n]]}}
    p = out["주 심사"]["같은 문항 비교"]
    gb, gw = p["충실 됨/충실 아니게 됨"]
    out["판정"] = {"1 충실도(좋아짐 > 나빠짐, p < 0.05)": gb > gw and p["p_충실"] < 0.05, "2 일치 하락 ≤ 3%p": new["일치"] >= base["일치"] - 0.03,
                 "3 답변없음 증가 ≤ 2%p": new["correctness"]["답변없음"] - base["correctness"]["답변없음"] <= 0.02 * len(recs),
                 "4 불일치 늘지 않음": new["correctness"]["불일치"] <= base["correctness"]["불일치"], "5 번호 오류 0·단정 늘지 않음": new["번호 오류"] == 0 and new["단정"] <= base["단정"]}
    out["items_changed"] = [{"set": r["set"], "qid": r["qid"], "info": {k: v for k, v in V[key(r)]["info"].items() if k != "usage"}, "주 심사": [j0(r)["correctness"], j0(r)["grounded"], j1(r)["correctness"], j1(r)["grounded"]],
                             "전문 심사": [f0(r)["correctness"], f0(r)["grounded"], f1(r)["correctness"], f1(r)["grounded"]], "reason_new": j1(r)["reason"]} for r in ch]
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    manifest("f15_verify_eval", questions=len(recs), changed=len(ch))
    print(json.dumps({k: v for k, v in out.items() if k != "items_changed"}, ensure_ascii=False, indent=1))
    print(f"{time.time() - t0:.0f}초")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
