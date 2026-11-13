"""F12 현장건의 붙이는 조건 변경 평가 (사용자 결정 "1번으로", worklog 88번). 개발 데이터만 쓴다. test6는 87번에서 잠겼으므로 읽지 않는다.
새 조건(B2): 현장건의 묶음 점수 ≥ 0.65 이면서 기존 회신 1위 벡터 점수보다 높은 것만 최대 2건. 예전 조건(B1, 87번): 0.65 이상이면 최대 2건.
사전 등록한 기준 (결과를 보기 전에 정함)
- dev6: B2의 일치·충실 비율이 A(현장건의 끔, 87번 답변·심사 재사용)보다 각각 3%p 넘게 떨어지지 않는다
- 답변불가 20: 전부 "찾지 못했습니다", 근거 번호 오류 0, 단정 0
- 함께 보고: 붙는 비율(B1 vs B2) — 회신 정답 질문(합성·dev1~6)에서는 낮을수록, 현장건의 정답 질문(devF 40)에서는 정답 묶음이 붙는 비율이 높을수록 좋다
- 결과를 보고 기준값을 다시 고치지 않는다 (devF·개발셋은 84번에서 이 조건을 떠올리는 데 쓴 데이터라 낙관적일 수 있다)
처리: 질문 임베딩은 한 번에 묶어서, LLM 호출은 동시에(87번에서 1단계를 차례로 돌려 느렸음)
캐시: _f12_answers.jsonl, _f12_judge.jsonl     결과: data/eval/f12_exmnt_gate_eval.json
"""
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import exmnt_search as es  # noqa: E402
import f11_dev6_test6_eval as f11  # noqa: E402  (cited_context 교체, check, answer 형식 재사용)
import f4_rag_answer_eval as f4  # noqa: E402
import rag  # noqa: E402
from common import DATA, manifest  # noqa: E402
from f5_app_prompt_eval import RANK, sign_p  # noqa: E402
from prompts import APP_SYSTEM_V2  # noqa: E402

EVAL = DATA / "eval"
ANS, JDG, OUT = EVAL / "_f12_answers.jsonl", EVAL / "_f12_judge.jsonl", EVAL / "f12_exmnt_gate_eval.json"
ANSWER_SETS = ("dev6", "unanswerable_20")
ATTACH_ONLY = ("devF", "rewrite_150", "dev1", "dev2", "dev3", "dev4", "dev5")
key = lambda r: (r["set"], r["qid"])


def ref_score(conn, v):
    """기존 회신(interp.chunks) 1위 벡터 점수 (rag.search와 같은 검색)."""
    conn.execute("SET hnsw.ef_search = 200")
    r = conn.execute("select 1 - (embedding <=> %(v)s::vector) s from interp.chunks order by embedding <=> %(v)s::vector limit 1", {"v": v}).fetchone()
    return float(r["s"]) if r else None


def main():
    sets = ANSWER_SETS + ATTACH_ONLY
    qs = [(n, json.loads(l)) for n in sets for l in open(EVAL / f"{n}.jsonl", encoding="utf-8") if l.strip()]
    t0 = time.time()
    rag.embed("준비")
    vecs = rag._local_model.encode([rag.mask_pii(q["question"]) for _, q in qs], normalize_embeddings=True, batch_size=32)
    print(f"[1 임베딩] {len(qs)}문항 {time.time() - t0:.0f}초", flush=True)
    rows = []
    with rag.connect() as conn:
        for (n, q), qv in zip(qs, vecs):
            v = "[" + ",".join(map(str, qv.tolist())) + "]"
            ref = ref_score(conn, v)
            old, top1 = es.search(conn, qv.tolist())
            new, _ = es.search(conn, qv.tolist(), ref)
            rows.append({"set": n, "qid": q["qid"], "gold_group_id": q.get("gold_group_id"), "ref": ref, "f_top1": top1,
                         "b1": [f["group_id"] for f in old], "b2": [f["group_id"] for f in new], "b2_rows": new})
    print(f"[2 검색] {time.time() - t0:.0f}초", flush=True)

    attach = {}
    for n in sets:
        rs = [r for r in rows if r["set"] == n]
        a = {"n": len(rs), "B1 붙는 비율": round(sum(1 for r in rs if r["b1"]) / len(rs), 3), "B2 붙는 비율": round(sum(1 for r in rs if r["b2"]) / len(rs), 3)}
        if n == "devF":
            a["B1 정답 붙음"] = round(sum(1 for r in rs if r["gold_group_id"] in r["b1"]) / len(rs), 3)
            a["B2 정답 붙음"] = round(sum(1 for r in rs if r["gold_group_id"] in r["b2"]) / len(rs), 3)
        attach[n] = a
    dev_all = [r for r in rows if r["set"] in ATTACH_ONLY[1:] + ("dev6",)]
    attach["회신 정답 개발셋 합계(합성·dev1~6)"] = {"n": len(dev_all), "B1 붙는 비율": round(sum(1 for r in dev_all if r["b1"]) / len(dev_all), 3),
                                         "B2 붙는 비율": round(sum(1 for r in dev_all if r["b2"]) / len(dev_all), 3)}

    # dev6·답변불가: 87번(f11) 검색·리랭킹 문맥과 A 답변·심사를 재사용, B2가 붙은 문항만 새로 답변·심사
    st = {key(r): r for r in f4.read_jsonl(f11.ST) if r["set"] in ANSWER_SETS}
    A11 = {(r["set"], r["qid"], r["mode"]): r for r in f4.read_jsonl(f11.ANS)}
    J11 = {(r["set"], r["qid"], r["mode"]): r for r in f4.read_jsonl(f11.JDG)}
    questions = {(n, q["qid"]): q["question"] for n, q in qs}
    recs = []
    for r in rows:
        if r["set"] not in ANSWER_SETS:
            continue
        s = st[key(r)]
        f_ids = [f["case_key"] for f in r["b2_rows"]]
        recs.append({**s, "f_ids": f_ids, "ctx_b": s["ctx_a"] + "".join("\n\n" + es.block(f) for f in r["b2_rows"])})
    A = {(r["set"], r["qid"]): r for r in f4.read_jsonl(ANS)}
    todo = [r for r in recs if r["f_ids"] and key(r) not in A]
    print(f"[3 답변] B2가 붙은 문항 {sum(1 for r in recs if r['f_ids'])}, 호출 대상 {len(todo)}", flush=True)
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for x in ex.map(lambda r: f11.answer(r, "B", questions[key(r)]), todo):
            f4.append(ANS, x)
            A[key(x)] = x
    golds = {}
    with rag.connect() as conn:
        golds = {g["case_key"]: g for g in conn.execute("select case_key, title, question, answer, reason from interp.cases where case_key = any(%s)", ([r["gold_key"] for r in recs if r["gold_key"]],)).fetchall()}
    J = {(r["set"], r["qid"]): r for r in f4.read_jsonl(JDG)}
    todo = [r for r in recs if r["f_ids"] and r["gold_key"] and key(r) not in J]

    def judge(r):
        return f4.judge_one({"set": r["set"], "qid": r["qid"], "gold_key": r["gold_key"], "context": r["ctx_b"]}, A[key(r)], questions[key(r)], golds[r["gold_key"]])
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for x in ex.map(judge, todo):
            f4.append(JDG, x)
            J[key(x)] = x

    def ans_of(r, mode):
        return A11[(r["set"], r["qid"], "A")] if mode == "A" or not r["f_ids"] else A[key(r)]

    def jdg_of(r, mode):
        return J11[(r["set"], r["qid"], "A")] if mode == "A" or not r["f_ids"] else J[key(r)]

    res = {}
    for n in ANSWER_SETS:
        rs = [r for r in recs if r["set"] == n]
        out = {}
        for mode in ("A", "B2"):
            a = [ans_of(r, mode) for r in rs]
            o = {"not_found": sum(x["check"]["not_found"] for x in a), "invalid": sum(len(x["check"]["invalid"]) for x in a), "assertive": sum(x["check"]["assertive"] for x in a),
                 "cited_field": sum(1 for x in a if any(c.startswith("F") for c in x["check"]["cited"]))}
            g = [r for r in rs if r["gold_key"]]
            if g:
                j = [jdg_of(r, mode) for r in g]
                c = Counter(x["correctness"] for x in j)
                gr = Counter(x["grounded"] for x in j if x["correctness"] != "답변없음")
                o.update({"correct_rate": round(c["일치"] / len(g), 3), "correctness": {k: c[k] for k in RANK}, "grounded_rate": round(gr["충실"] / max(sum(gr.values()), 1), 3)})
            out[mode] = o
        att = [r for r in rs if r["f_ids"] and r["gold_key"]]
        if att:
            b = sum(1 for r in att if RANK[jdg_of(r, "B2")["correctness"]] > RANK[jdg_of(r, "A")["correctness"]])
            w = sum(1 for r in att if RANK[jdg_of(r, "B2")["correctness"]] < RANK[jdg_of(r, "A")["correctness"]])
            out["attached_paired"] = {"n": len(att), "좋아짐/나빠짐": [b, w], "p": round(sign_p(b, w), 4),
                                      "items": [{"qid": r["qid"], "f_ids": r["f_ids"], "A": jdg_of(r, "A")["correctness"], "B2": jdg_of(r, "B2")["correctness"],
                                                 "A_grounded": jdg_of(r, "A")["grounded"], "B2_grounded": jdg_of(r, "B2")["grounded"], "B2_cited": ans_of(r, "B2")["check"]["cited"]} for r in att]}
        res[n] = out
    result = {"rule": {"threshold": es.THRESH, "max": es.MAX_N, "B2": "점수 ≥ 0.65 이면서 기존 회신 1위 벡터 점수보다 높을 때"}, "attach": attach, "answers": res}
    json.dump(result, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    manifest("f12_exmnt_gate_eval", questions=len(qs))
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
