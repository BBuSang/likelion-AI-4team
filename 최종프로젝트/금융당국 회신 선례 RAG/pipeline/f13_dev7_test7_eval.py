"""F13 dev7·test7 평가: 기존 회신 70 + 현장건의 30씩 (사용자 요청 2026-11-11 "데모를 재구현하기 전에 dev7, test7을 돌리고 수치로 정리", worklog 89번).
지금 데모 설정 그대로 잰다(`app/demo.py`의 ask와 같은 흐름, 첫 답변 기준이라 '못 찾을 때만 키워드 완화'는 넣지 않음. f10·f11과 같음).
- 기존 회신: 벡터 후보 20 → LLM 리랭킹 5 (`app/rerank.py`, 실패 시 벡터 상위 5)
- 현장건의: 별도 후보군. 점수 ≥ 0.65 이면서 기존 회신 1위 벡터 점수보다 높은 묶음만 최대 2건을 뒤에 붙인다 (`app/exmnt_search.py`, 88번 조건)
- 답변: `APP_SYSTEM_V2`(현장건의가 붙으면 현장건의 규칙 추가) → LLM 심사(f4.judge_one, 현장건의 정답은 건의내용·검토결과·사유를 정답 사례로 넘김)
보고하는 수치 (합격 기준 없이 재기만 한다. 결과를 보고 설정·기준값을 고치지 않는다)
- 기존 회신 정답 문항: 벡터 Hit@1·3·5·10·20, MRR@10 / 리랭킹 Hit@1·3·5, MRR@5 / 현장건의가 잘못 붙은 비율
- 현장건의 정답 문항: 현장건의 후보군 안에서의 Hit@1·3·5·10, MRR@10 / 붙이는 조건을 통과해 정답 묶음이 문맥에 들어간 비율
- 전체: 정답이 최종 문맥(회신 5 + 현장건의 최대 2)에 들어간 비율, 정답이 맨 앞(회신은 리랭킹 1위, 현장건의는 붙은 것 중 1위)인 비율
- 답변: 정확도(일치/부분일치/불일치/답변없음), 충실도, 정답 사례 인용, "찾지 못했습니다", 근거 번호 오류, 단정 표현
- test7는 한 번만 평가하고 data/eval/f13_test7_lock.json으로 잠근다. 결과 파일에 test7 질문 글·답변은 넣지 않는다
사용: python f13_dev7_test7_eval.py [--dev-only]     캐시: _f13_stage.jsonl, _f13_answers.jsonl, _f13_judge.jsonl     결과: data/eval/f13_dev7_test7_eval.json
"""
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import exmnt_search as es  # noqa: E402
import f11_dev6_test6_eval as f11  # noqa: E402  (answer·check 재사용, 심사자에게 현장건의 사례 글도 넘기도록 f4.cited_context 교체)
import f2_test_eval as f2  # noqa: E402
import f4_rag_answer_eval as f4  # noqa: E402
import rag  # noqa: E402
import rerank as rr  # noqa: E402
from common import DATA, manifest  # noqa: E402
from f5_app_prompt_eval import RANK  # noqa: E402

EVAL = DATA / "eval"
ST, ANS, JDG = EVAL / "_f13_stage.jsonl", EVAL / "_f13_answers.jsonl", EVAL / "_f13_judge.jsonl"
LOCK, OUT = EVAL / "f13_test7_lock.json", EVAL / "f13_dev7_test7_eval.json"
key = lambda r: (r["set"], r["qid"])
rate = lambda n, d: round(n / d, 3) if d else None


def exmnt_ranked(conn, qvec):
    """현장건의 후보군 묶음 순위 (es.search와 같은 검색, 조건 없이 순위만)."""
    v = "[" + ",".join(map(str, qvec)) + "]"
    conn.execute("SET hnsw.ef_search = 200")
    rows = conn.execute("select group_id, 1 - (embedding <=> %(v)s::vector) s from interp.exmnt_chunks order by embedding <=> %(v)s::vector limit %(n)s", {"v": v, "n": es.FETCH}).fetchall()
    return list(dict.fromkeys(r["group_id"] for r in rows))


def stage(conn, sets, data, grp_rep, ex_rep):
    done = {key(r) for r in f4.read_jsonl(ST)}
    todo = [(n, q) for n in sets for q in data[n] if (n, q["qid"]) not in done]
    if not todo:
        return
    masked = [rag.mask_pii(q["question"]) for _, q in todo]
    vecs = {m: v.tolist() for m, v in zip(masked, rag._local_model.encode(masked, normalize_embeddings=True, batch_size=32))}
    one = rag.embed
    rag.embed = lambda t: vecs[t] if t in vecs else one(t)       # rr.candidates가 질문마다 다시 임베딩하지 않게 (값은 같은 모델·같은 정규화)
    pre = []
    for (n, q), m in zip(todo, masked):
        cands = rr.candidates(conn, m)
        ids = [c["doc_id"] for c in cands]
        pre.append((n, q, m, cands, ids, rr.load_brief(conn, ids)))

    def rerank(p):
        try:
            return f4.with_retry(lambda: rr.rerank(p[2], p[4], p[5]))[0], True
        except Exception:
            return p[4][:5], False
    with ThreadPoolExecutor(f4.THREADS) as ex:
        orders = list(ex.map(rerank, pre))
    for (n, q, m, cands, ids, _), (order, ok) in zip(pre, orders):
        gold, corpus = q["gold_group_id"], q["corpus"]
        groups = [c["group_id"] for c in cands]
        g_of = dict(zip(ids, groups))
        ref = max(c["score"] for c in cands)
        fields, top1 = es.search(conn, vecs[m], ref)
        ex_rank = exmnt_ranked(conn, vecs[m])
        cases = rag.load_cases(order, conn)
        ctx_a = "\n\n".join(rag.case_block(cases[i]) for i in order)
        f4.append(ST, {"set": n, "qid": q["qid"], "corpus": corpus, "tag": q.get("tag"), "gubun": q.get("gubun"), "gold_group_id": gold,
                       "gold_key": (grp_rep if corpus == "interp" else ex_rep).get(gold),
                       "vec_rank": groups.index(gold) + 1 if gold in groups else None,
                       "rank": next((i + 1 for i, d in enumerate(order) if g_of[d] == gold), None),
                       "ex_rank": ex_rank.index(gold) + 1 if gold in ex_rank else None,
                       "reranked_ok": ok, "ref": round(ref, 4), "f_top1": top1 and round(top1, 4),
                       "ids_a": order, "ctx_a": ctx_a, "f_ids": [f["case_key"] for f in fields], "f_groups": [f["group_id"] for f in fields],
                       "ctx_b": ctx_a + "".join("\n\n" + es.block(f) for f in fields)})
    rag.embed = one


def summarize(rs, A, J):
    n = len(rs)
    if not n:
        return {"n": 0}
    hit = lambda field, k: rate(sum(1 for r in rs if r[field] and r[field] <= k), n)
    mrr = lambda field, k: round(sum(1 / r[field] for r in rs if r[field] and r[field] <= k) / n, 3)
    a = [A[key(r)] for r in rs]
    j = [J[key(r)] for r in rs]
    c = Counter(x["correctness"] for x in j)
    g = Counter(x["grounded"] for x in j if x["correctness"] != "답변없음")
    in_ctx = lambda r: bool(r["rank"]) if r["corpus"] == "interp" else r["gold_group_id"] in r["f_groups"]
    first = lambda r: r["rank"] == 1 if r["corpus"] == "interp" else r["f_groups"][:1] == [r["gold_group_id"]]
    out = {"n": n}
    if any(r["corpus"] == "interp" for r in rs):
        out["회신 벡터"] = {"hit1": hit("vec_rank", 1), "hit3": hit("vec_rank", 3), "hit5": hit("vec_rank", 5), "hit10": hit("vec_rank", 10), "hit20": hit("vec_rank", 20), "mrr10": mrr("vec_rank", 10)}
        out["회신 리랭킹"] = {"hit1": hit("rank", 1), "hit3": hit("rank", 3), "hit5": hit("rank", 5), "mrr5": mrr("rank", 5), "리랭킹 실패": sum(1 for r in rs if not r["reranked_ok"])}
    if any(r["corpus"] == "exmnt" for r in rs):
        out["현장건의 후보군"] = {"hit1": hit("ex_rank", 1), "hit3": hit("ex_rank", 3), "hit5": hit("ex_rank", 5), "hit10": hit("ex_rank", 10), "mrr10": mrr("ex_rank", 10)}
    out.update({"현장건의 붙음": rate(sum(1 for r in rs if r["f_ids"]), n), "현장건의 인용": sum(1 for x in a if any(c_.startswith("F") for c_ in x["check"]["cited"])),
                "정답이 문맥에": rate(sum(map(in_ctx, rs)), n), "정답이 맨 앞": rate(sum(map(first, rs)), n),
                "정답 인용(문맥에 있을 때)": f"{sum(1 for r, x in zip(rs, a) if in_ctx(r) and r['gold_key'] in x['check']['cited'])}/{sum(map(in_ctx, rs))}",
                "일치": rate(c["일치"], n), "일치+부분": rate(c["일치"] + c["부분일치"], n), "correctness": {k: c[k] for k in RANK},
                "충실": rate(g["충실"], sum(g.values())), "grounded": {k: g[k] for k in ("충실", "일부추정", "근거없음")},
                "찾지 못함": sum(x["check"]["not_found"] for x in a), "번호 오류": sum(len(x["check"]["invalid"]) for x in a), "단정": sum(x["check"]["assertive"] for x in a)})
    return out


def main():
    dev_only = "--dev-only" in sys.argv
    sets = ["dev7"] + ([] if dev_only else ["test7"])
    data = {n: f11.load(n) for n in sets}
    with rag.connect() as conn:
        fp = f2.fingerprint(conn)["id"]
        prev = {v["fingerprint"] for v in json.load(open(f2.LOCK, encoding="utf-8")).values()}
        if fp not in prev:
            sys.exit(f"설정 지문 {fp}가 이전 평가({sorted(prev)})와 다릅니다.")
        lock = json.load(open(LOCK, encoding="utf-8")) if LOCK.exists() else {}
        if not dev_only and lock.get("completed_at"):
            sys.exit(f"test7는 이미 평가했습니다 ({lock['completed_at']}). --dev-only로 dev7만 다시 돌릴 수 있습니다.")
        if not dev_only and not lock:
            LOCK.write_text(json.dumps({"started_at": datetime.now().isoformat(timespec="seconds"), "fingerprint": fp}, ensure_ascii=False), encoding="utf-8")
        grp_rep = {r["group_id"]: r["representative_case_key"] for r in conn.execute("select group_id, representative_case_key from interp.case_groups").fetchall()}
        ex_rep = {r["group_id"]: r["representative_case_key"] for r in conn.execute("select group_id, representative_case_key from interp.exmnt_groups").fetchall()}
        rag.embed("준비")
        t0 = time.time()
        stage(conn, sets, data, grp_rep, ex_rep)
        print(f"[1 검색·리랭킹·현장건의] {time.time() - t0:.0f}초", flush=True)
        recs = [r for r in f4.read_jsonl(ST) if r["set"] in sets]
        gkeys = [r["gold_key"] for r in recs]
        golds = {r["case_key"]: r for r in conn.execute("select case_key, title, question, answer, reason from interp.cases where case_key = any(%s)", (gkeys,)).fetchall()}
        golds.update({r["case_key"]: r for r in conn.execute(
            "select case_key, title, question, '검토결과: ' || coalesce(answer_type, '') || E'\\n' || coalesce(answer, '') as answer, '' as reason from interp.exmnt_cases where case_key = any(%s)", (gkeys,)).fetchall()})
    questions = {(n, q["qid"]): q["question"] for n in sets for q in data[n]}
    mode = lambda r: "B" if r["f_ids"] else "A"                  # 데모와 같다: 현장건의가 붙은 문항만 문맥·규칙 추가

    A = {key(r): r for r in f4.read_jsonl(ANS)}
    todo = [r for r in recs if key(r) not in A]
    print(f"[2 답변] 호출 대상 {len(todo)}", flush=True)
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for x in ex.map(lambda r: f11.answer(r, mode(r), questions[key(r)]), todo):
            f4.append(ANS, x)
            A[key(x)] = x
    J = {key(r): r for r in f4.read_jsonl(JDG)}
    todo = [r for r in recs if key(r) not in J]
    print(f"[3 심사] 호출 대상 {len(todo)}", flush=True)

    def judge(r):
        return f4.judge_one({"set": r["set"], "qid": r["qid"], "gold_key": r["gold_key"], "context": r["ctx_b"]}, A[key(r)], questions[key(r)], golds[r["gold_key"]])
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for x in ex.map(judge, todo):
            f4.append(JDG, x)
            J[key(x)] = x

    out = {"ran_at": datetime.now().isoformat(timespec="seconds"), "fingerprint": fp, "config": "벡터 20 → 리랭킹 5 → APP_SYSTEM_V2, 현장건의는 점수 ≥ 0.65이면서 회신 1위 벡터 점수보다 높을 때 최대 2건",
           "exmnt_threshold": es.THRESH, "exmnt_max": es.MAX_N, "sets": {}}
    for n in sets:
        rs = [r for r in recs if r["set"] == n]
        s = {"전체": summarize(rs, A, J), "기존 회신 정답": summarize([r for r in rs if r["corpus"] == "interp"], A, J), "현장건의 정답": summarize([r for r in rs if r["corpus"] == "exmnt"], A, J)}
        s["유형별(일치 / 정답이 문맥에 / n)"] = {f"{c_}:{t}": [rate(sum(1 for r in g if J[key(r)]["correctness"] == "일치"), len(g)),
                                                    rate(sum(1 for r in g if (r["rank"] if r["corpus"] == "interp" else r["gold_group_id"] in r["f_groups"])), len(g)), len(g)]
                                    for (c_, t), g in sorted({(r["corpus"], r["tag"]): [x for x in rs if (x["corpus"], x["tag"]) == (r["corpus"], r["tag"])] for r in rs}.items())}
        if n != "test7":
            s["items"] = [{"qid": r["qid"], "corpus": r["corpus"], "tag": r["tag"], "gold_key": r["gold_key"], "vec_rank": r["vec_rank"], "rank": r["rank"], "ex_rank": r["ex_rank"],
                           "ref": r["ref"], "f_top1": r["f_top1"], "f_ids": r["f_ids"], "cited": A[key(r)]["check"]["cited"],
                           "correctness": J[key(r)]["correctness"], "grounded": J[key(r)]["grounded"], "reason": J[key(r)]["reason"]} for r in rs]
        out["sets"][n] = s
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not dev_only:
        LOCK.write_text(json.dumps({**json.load(open(LOCK, encoding="utf-8")), "completed_at": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False), encoding="utf-8")
    manifest("f13_dev7_test7_eval", sets=sets, fingerprint=fp)
    print(json.dumps({n: {k: v for k, v in s.items() if k != "items"} for n, s in out["sets"].items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
