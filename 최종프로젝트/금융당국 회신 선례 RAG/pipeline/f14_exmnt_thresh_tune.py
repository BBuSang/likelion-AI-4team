"""F14 현장건의 붙이는 기준값(지금 0.65)을 개발셋으로 다시 잡기 (사용자 요청 2026-11-11 "재구현 전에 개발셋으로 다시 잡아보자", worklog 91번).
배경(89번): dev7 현장건의 정답 2문항이 1위 점수 0.635·0.647로 기준에 못 미쳐 붙지 않았다. 규칙의 모양은 그대로 둔다
(현장건의 묶음 점수 ≥ T 이면서 기존 회신 1위 벡터 점수보다 높을 때 최대 2건, 88번). T만 0.50~0.70을 0.01 간격으로 훑는다.
쓰는 데이터: 개발 데이터만. test1~7·final_test는 읽지 않는다(test7는 90번에서 열어 봤지만 기준값을 고르는 데 쓰지 않는다).
- 현장건의 정답 70: devF 40 + dev7의 현장건의 30
- 회신 정답 580: 합성 150 + dev1~6 360 + dev7의 회신 70
- 답변불가 20
사전 등록한 선택 규칙 (결과를 보기 전에 정함. 결과를 보고 규칙이나 범위를 고치지 않는다)
1) 제약: 회신 정답 580문항에 현장건의가 붙는 비율 ≤ 3.0%, 답변불가 20문항 중 붙는 문항 수 ≤ 2 (둘 다 지금 0.65의 수준: 2.5%, 2문항)
2) 제약을 지키는 T 중 현장건의 정답 70문항에서 정답 묶음이 붙는 비율이 가장 높은 것. 같으면 가장 큰 T(보수적인 쪽)
3) 고른 T가 0.65와 같으면 그대로 두고 끝낸다
답변 검증 (--answers, 고른 T가 0.65와 다를 때만): dev7 100 + 답변불가 20을 고른 T로 다시 답한다(붙는 현장건의가 달라진 문항만 새로 답변·심사, 나머지는 89번·87번·88번 결과 재사용)
- dev7 전체의 일치·충실 비율이 0.65 때(89번: 93%, 86.9%)보다 각각 3%p 넘게 떨어지지 않는다
- 답변불가 20개 전부 "찾지 못했습니다", 근거 번호 오류 0, 단정 0
- 통과하면 app/exmnt_search.py의 THRESH를 바꾸고, 통과하지 못하면 0.65를 유지하고 근거와 함께 보고한다(값을 바꿔 가며 다시 재지 않는다)
사용: python f14_exmnt_thresh_tune.py [--answers]     캐시: _f14_search.jsonl, _f14_answers.jsonl, _f14_judge.jsonl     결과: data/eval/f14_exmnt_thresh_tune.json
"""
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import exmnt_search as es  # noqa: E402
import f11_dev6_test6_eval as f11  # noqa: E402
import f12_exmnt_gate_eval as f12  # noqa: E402  (ref_score)
import f13_dev7_test7_eval as f13  # noqa: E402
import f4_rag_answer_eval as f4  # noqa: E402
import rag  # noqa: E402
from common import DATA, manifest  # noqa: E402
from f5_app_prompt_eval import RANK  # noqa: E402

EVAL = DATA / "eval"
SEARCH, ANS, JDG, OUT = EVAL / "_f14_search.jsonl", EVAL / "_f14_answers.jsonl", EVAL / "_f14_judge.jsonl", EVAL / "f14_exmnt_thresh_tune.json"
SETS = ("devF", "dev7", "rewrite_150", "dev1", "dev2", "dev3", "dev4", "dev5", "dev6", "unanswerable_20")
GRID = [round(0.50 + 0.01 * i, 2) for i in range(21)]
NOW, MAX_INTERP, MAX_UNANS = 0.65, 0.03, 2
key = lambda r: (r["set"], r["qid"])


def corpus_of(name, q):
    return "none" if name == "unanswerable_20" else "exmnt" if name == "devF" else q.get("corpus", "interp")


def search_all():
    done = {key(r): r for r in f4.read_jsonl(SEARCH)}
    qs = [(n, json.loads(l)) for n in SETS for l in open(EVAL / f"{n}.jsonl", encoding="utf-8") if l.strip()]
    todo = [(n, q) for n, q in qs if (n, q["qid"]) not in done]
    if todo:
        t0 = time.time()
        rag.embed("준비")
        vecs = rag._local_model.encode([rag.mask_pii(q["question"]) for _, q in todo], normalize_embeddings=True, batch_size=32)
        with rag.connect() as conn:
            for (n, q), qv in zip(todo, vecs):
                v = "[" + ",".join(map(str, qv.tolist())) + "]"
                conn.execute("SET hnsw.ef_search = 200")
                rows = conn.execute("select group_id, 1 - (embedding <=> %(v)s::vector) s from interp.exmnt_chunks order by embedding <=> %(v)s::vector limit %(n)s", {"v": v, "n": es.FETCH}).fetchall()
                best = {}
                for r in rows:
                    best.setdefault(r["group_id"], float(r["s"]))
                row = {"set": n, "qid": q["qid"], "corpus": corpus_of(n, q), "gold_group_id": q.get("gold_group_id"), "ref": f12.ref_score(conn, v), "top": list(best.items())[:es.MAX_N]}
                f4.append(SEARCH, row)
                done[key(row)] = row
        print(f"[검색] {len(todo)}문항 {time.time() - t0:.0f}초", flush=True)
    return [done[(n, q["qid"])] for n, q in qs]


def attached(r, t):
    """es.search와 같은 조건: 상위 MAX_N 묶음 중 점수 ≥ max(T, 회신 1위 점수 초과)."""
    return [g for g, s in r["top"] if s >= max(t, r["ref"] + 1e-9)]


def sweep(rows):
    ex, it, na = ([r for r in rows if r["corpus"] == c] for c in ("exmnt", "interp", "none"))
    table = []
    for t in GRID:
        table.append({"T": t, "현장건의 정답 붙음": round(sum(1 for r in ex if r["gold_group_id"] in attached(r, t)) / len(ex), 4),
                      "현장건의 정답 1위로 붙음": round(sum(1 for r in ex if attached(r, t)[:1] == [r["gold_group_id"]]) / len(ex), 4),
                      "회신 질문에 붙음": round(sum(1 for r in it if attached(r, t)) / len(it), 4), "회신 질문에 붙음(문항)": sum(1 for r in it if attached(r, t)),
                      "답변불가에 붙음(문항)": sum(1 for r in na if attached(r, t))})
    ok = [x for x in table if x["회신 질문에 붙음"] <= MAX_INTERP and x["답변불가에 붙음(문항)"] <= MAX_UNANS]
    best = max(x["현장건의 정답 붙음"] for x in ok) if ok else None
    chosen = max(x["T"] for x in ok if x["현장건의 정답 붙음"] == best) if ok else NOW
    return table, chosen, {"현장건의 정답": len(ex), "회신 정답": len(it), "답변불가": len(na)}


def answers(rows, t):
    """dev7 + 답변불가를 T로 다시 답한다. 붙는 현장건의가 0.65 때와 같으면 기존 답변·심사를 쓴다."""
    tgt = [r for r in rows if r["set"] in ("dev7", "unanswerable_20")]
    qs = {(n, q["qid"]): q["question"] for n in ("dev7", "unanswerable_20") for q in f11.load(n)}
    st = {key(r): r for r in f4.read_jsonl(f13.ST) if r["set"] == "dev7"}
    st.update({key(r): r for r in f4.read_jsonl(f11.ST) if r["set"] == "unanswerable_20"})
    oldA = {key(r): r for r in f4.read_jsonl(f13.ANS) if r["set"] == "dev7"}
    oldJ = {key(r): r for r in f4.read_jsonl(f13.JDG) if r["set"] == "dev7"}
    oldA.update({(r["set"], r["qid"]): r for r in f4.read_jsonl(f11.ANS) if r["set"] == "unanswerable_20" and r["mode"] == "A"})
    oldA.update({key(r): r for r in f4.read_jsonl(f12.ANS) if r["set"] == "unanswerable_20"})          # 88번 조건(0.65)으로 붙은 문항의 답변
    changed = [r for r in tgt if attached(r, t) != attached(r, NOW)]
    recs = {}
    es.THRESH = t
    with rag.connect() as conn:
        ex_rep = {x["group_id"]: x["representative_case_key"] for x in conn.execute("select group_id, representative_case_key from interp.exmnt_groups").fetchall()}
        for r in changed:
            s = st[key(r)]
            fields, _ = es.search(conn, rag.embed(rag.mask_pii(qs[key(r)])), r["ref"])
            assert [f["group_id"] for f in fields] == attached(r, t), key(r)
            recs[key(r)] = {**s, "f_ids": [f["case_key"] for f in fields], "f_groups": [f["group_id"] for f in fields], "ctx_b": s["ctx_a"] + "".join("\n\n" + es.block(f) for f in fields)}
        gk = [st[key(r)].get("gold_key") or ex_rep.get(r["gold_group_id"]) for r in tgt if r["set"] == "dev7"]
        golds = {x["case_key"]: x for x in conn.execute("select case_key, title, question, answer, reason from interp.cases where case_key = any(%s)", (gk,)).fetchall()}
        golds.update({x["case_key"]: x for x in conn.execute(
            "select case_key, title, question, '검토결과: ' || coalesce(answer_type, '') || E'\\n' || coalesce(answer, '') as answer, '' as reason from interp.exmnt_cases where case_key = any(%s)", (gk,)).fetchall()})
    es.THRESH = NOW
    A = {key(r): r for r in f4.read_jsonl(ANS)}
    todo = [r for k, r in recs.items() if k not in A]
    print(f"[답변] 붙는 현장건의가 달라진 문항 {len(changed)}, 호출 대상 {len(todo)}", flush=True)
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for x in ex.map(lambda r: f11.answer(r, "B" if r["f_ids"] else "A", qs[key(r)]), todo):
            f4.append(ANS, x)
            A[key(x)] = x
    J = {key(r): r for r in f4.read_jsonl(JDG)}
    todo = [r for k, r in recs.items() if k[0] == "dev7" and k not in J]
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for x in ex.map(lambda r: f4.judge_one({"set": r["set"], "qid": r["qid"], "gold_key": r["gold_key"], "context": r["ctx_b"]}, A[key(r)], qs[key(r)], golds[r["gold_key"]]), todo):
            f4.append(JDG, x)
            J[key(x)] = x

    def score(rs, a_of, j_of):
        a, j = [a_of(r) for r in rs], [j_of(r) for r in rs]
        c = Counter(x["correctness"] for x in j)
        g = Counter(x["grounded"] for x in j if x["correctness"] != "답변없음")
        return {"n": len(rs), "일치": round(c["일치"] / len(rs), 3), "correctness": {k: c[k] for k in RANK}, "충실": round(g["충실"] / max(sum(g.values()), 1), 3),
                "찾지 못함": sum(x["check"]["not_found"] for x in a), "번호 오류": sum(len(x["check"]["invalid"]) for x in a), "단정": sum(x["check"]["assertive"] for x in a)}
    d7 = [r for r in tgt if r["set"] == "dev7"]
    na = [r for r in tgt if r["set"] == "unanswerable_20"]
    new_a = lambda r: A[key(r)] if key(r) in recs else oldA[key(r)]
    new_j = lambda r: J[key(r)] if key(r) in recs else oldJ[key(r)]
    res = {"T": t, "달라진 문항": len(changed), "dev7": {"0.65": score(d7, lambda r: oldA[key(r)], lambda r: oldJ[key(r)]), "새 T": score(d7, new_a, new_j)}}
    for name, sub in (("dev7 회신 70", [r for r in d7 if r["corpus"] == "interp"]), ("dev7 현장건의 30", [r for r in d7 if r["corpus"] == "exmnt"])):
        res[name] = {"0.65": score(sub, lambda r: oldA[key(r)], lambda r: oldJ[key(r)]), "새 T": score(sub, new_a, new_j)}
    ua = [new_a(r) for r in na]
    res["답변불가 20"] = {"거절": sum(x["check"]["not_found"] for x in ua), "붙은 문항": sum(1 for r in na if attached(r, t)), "번호 오류": sum(len(x["check"]["invalid"]) for x in ua), "단정": sum(x["check"]["assertive"] for x in ua)}
    res["items"] = [{"set": r["set"], "qid": r["qid"], "corpus": r["corpus"], "ref": round(r["ref"], 4), "top": [[g, round(s, 4)] for g, s in r["top"]], "gold": r["gold_group_id"],
                     "0.65": [oldJ[key(r)]["correctness"], oldJ[key(r)]["grounded"]] if key(r) in oldJ else None, "새 T": [J[key(r)]["correctness"], J[key(r)]["grounded"]] if key(r) in J else None,
                     "cited": A[key(r)]["check"]["cited"], "not_found": A[key(r)]["check"]["not_found"]} for r in changed]
    a0, a1 = res["dev7"]["0.65"], res["dev7"]["새 T"]
    res["판정"] = {"dev7 일치": a1["일치"] >= a0["일치"] - 0.03, "dev7 충실": a1["충실"] >= a0["충실"] - 0.03,
                 "답변불가": res["답변불가 20"]["거절"] == len(na) and res["답변불가 20"]["번호 오류"] == 0 and res["답변불가 20"]["단정"] == 0}
    return res


def main():
    rows = search_all()
    table, chosen, n = sweep(rows)
    out = {"rule": "점수 ≥ T 이면서 기존 회신 1위 벡터 점수보다 높을 때 최대 2건", "n": n, "constraints": {"회신 질문에 붙음 ≤": MAX_INTERP, "답변불가에 붙음 ≤": MAX_UNANS}, "table": table, "chosen": chosen, "now": NOW}
    for x in table:
        print(json.dumps(x, ensure_ascii=False))
    print("고른 T:", chosen, flush=True)
    if "--answers" in sys.argv and chosen != NOW:
        out["answers"] = answers(rows, chosen)
        print(json.dumps({k: v for k, v in out["answers"].items() if k != "items"}, ensure_ascii=False, indent=1))
        for x in out["answers"]["items"]:
            print(json.dumps(x, ensure_ascii=False))
    elif OUT.exists():
        out.update({k: v for k, v in json.load(open(OUT, encoding="utf-8")).items() if k == "answers"})
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    manifest("f14_exmnt_thresh_tune", questions=len(rows), chosen=chosen)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
