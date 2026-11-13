"""F11 현장건의를 포함한 전체 데이터로 dev6·test6 검증 (사용자 요청 2026-11-11, worklog 87번).
데모 설정 그대로: 벡터 후보 20 → LLM 리랭킹 5(`app/rerank.py`, 실패 시 벡터 상위 5) → 프롬프트 `APP_SYSTEM_V2`. 첫 답변 기준('못 찾을 때만 키워드 완화'는 넣지 않음, f10과 같음).
- A 현장건의 끔: 위 그대로
- B 현장건의 켬: 현장건의 1위 점수 ≥ 0.65면 최대 2건을 회신 뒤에 붙이고 현장건의 규칙을 더한 프롬프트(`app/exmnt_search.py`). 붙지 않은 문항은 A와 입력이 같으므로 A 답변·심사를 그대로 쓴다
dev6·test6의 정답은 기존 회신(법령해석·비조치)이다. 현장건의가 정답을 돕는지가 아니라, 붙였을 때 기존 답이 나빠지지 않는지를 본다. 답변불가 20(개발용)도 B로 함께 돌린다.
사전 등록한 기준 (결과를 보기 전에 정함)
- B의 정확도 "일치"와 충실도 "충실" 비율이 A보다 각각 3%p 넘게 떨어지지 않는다 (세트마다)
- 답변불가 20개 전부 "찾지 못했습니다", 근거 번호 오류 0, 단정 표현 0
- 함께 보고: 검색 Hit@1·5(벡터·리랭킹), 현장건의가 붙은 비율·인용된 수, 붙은 문항만의 A→B 변화(부호 검정), "찾지 못했습니다" 수
- test6는 한 번만 평가하고 data/eval/f11_test6_lock.json으로 잠근다. 결과 파일에 test6 질문 글·답변은 넣지 않는다. 결과를 보고 설정을 고치지 않는다
사용: python f11_dev6_test6_eval.py [--dev-only]     캐시: _f11_stage.jsonl, _f11_answers.jsonl, _f11_judge.jsonl     결과: data/eval/f11_dev6_test6_eval.json
"""
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import exmnt_search as es  # noqa: E402
import f2_test_eval as f2  # noqa: E402
import f4_rag_answer_eval as f4  # noqa: E402
import rag  # noqa: E402
import rerank as rr  # noqa: E402
from common import DATA, manifest  # noqa: E402
from f5_app_prompt_eval import RANK, sign_p  # noqa: E402
from prompts import APP_SYSTEM_V2  # noqa: E402

EVAL = DATA / "eval"
ST, ANS, JDG = EVAL / "_f11_stage.jsonl", EVAL / "_f11_answers.jsonl", EVAL / "_f11_judge.jsonl"
LOCK, OUT = EVAL / "f11_test6_lock.json", EVAL / "f11_dev6_test6_eval.json"
CITE = re.compile(r"\[([LOF]\d+)\]")
key = lambda r: (r["set"], r["qid"])


def cited_context(context, cited, limit=1200, max_cases=3):
    """f4.cited_context와 같되 현장건의(F) 사례 블록도 심사자에게 넘긴다 (f4는 [LO]만 찾는다)."""
    blocks = {m.group(1): m.group(0) for m in re.finditer(r"\[사례 ([LOF]\d+)\].*?(?=\n\n\[사례 |\Z)", context, re.S)}
    return "\n\n".join(blocks[c][:limit] for c in cited[:max_cases] if c in blocks)


f4.cited_context = cited_context          # judge_one이 모듈 전역 이름으로 부르므로 바꿔 끼운다 (f4 파일은 고치지 않음)


def load(name):
    return [json.loads(l) for l in open(EVAL / f"{name}.jsonl", encoding="utf-8") if l.strip()]


def stage(conn, name, rows, grp_rep):
    done = {key(r) for r in f4.read_jsonl(ST)}
    for q in rows:
        if (name, q["qid"]) in done:
            continue
        masked = rag.mask_pii(q["question"])
        cands = rr.candidates(conn, masked)
        cand_ids, cand_groups = [c["doc_id"] for c in cands], [c["group_id"] for c in cands]
        try:
            order, _, _ = f4.with_retry(lambda: rr.rerank(masked, cand_ids, rr.load_brief(conn, cand_ids)))
            reranked_ok = True
        except Exception:
            order, reranked_ok = cand_ids[:5], False
        gold = q.get("gold_group_id")
        g_of = dict(zip(cand_ids, cand_groups))
        vec_rank = cand_groups.index(gold) + 1 if gold in cand_groups else None
        rer_rank = next((i + 1 for i, d in enumerate(order) if g_of[d] == gold), None) if gold else None
        cases = rag.load_cases(order, conn)
        fields, top1 = es.search(conn, rag.embed(masked))
        ctx_a = "\n\n".join(rag.case_block(cases[i]) for i in order)
        f4.append(ST, {"set": name, "qid": q["qid"], "tag": q.get("tag"), "gubun": q.get("gubun"), "gold_group_id": gold, "gold_key": grp_rep.get(gold) if gold else None,
                       "vec_rank": vec_rank, "rank": rer_rank, "reranked_ok": reranked_ok, "ids_a": order, "ctx_a": ctx_a,
                       "f_ids": [f["case_key"] for f in fields], "f_scores": [round(f["score"], 4) for f in fields], "f_top1": top1 and round(top1, 4),
                       "ctx_b": ctx_a + "".join("\n\n" + es.block(f) for f in fields)})


def check(text, ids):
    c = rag.check_output(text, ids)
    cited = sorted(set(CITE.findall(text)))
    return {**c, "cited": cited, "invalid": sorted(set(cited) - set(ids))}


def answer(rec, mode, question):
    ctx, ids = (rec["ctx_a"], rec["ids_a"]) if mode == "A" else (rec["ctx_b"], rec["ids_a"] + rec["f_ids"])
    system = APP_SYSTEM_V2 if mode == "A" else APP_SYSTEM_V2 + es.EXMNT_RULES
    text, usage = f4.with_retry(lambda: rag.llm([{"role": "system", "content": system}, {"role": "user", "content": f"[질문]\n{rag.mask_pii(question)}\n\n[사례]\n{ctx}"}]))
    text = text.strip()
    chk = check(text, ids)
    if not text.endswith(rag.CLOSING):
        text = f"{text}\n\n{rag.CLOSING}"
    return {"set": rec["set"], "qid": rec["qid"], "mode": mode, "answer": text, "check": chk, "usage": usage}


def summarize(rs, A, J, mode):
    pick = lambda r: "B" if mode == "B" and r["f_ids"] else "A"
    a = [A[(r["set"], r["qid"], pick(r))] for r in rs]
    gold = [r for r in rs if r["gold_key"]]
    out = {"n": len(rs), "not_found": sum(x["check"]["not_found"] for x in a), "invalid": sum(len(x["check"]["invalid"]) for x in a), "assertive": sum(x["check"]["assertive"] for x in a),
           "attached": sum(1 for r in rs if r["f_ids"]) if mode == "B" else 0,
           "cited_field": sum(1 for x in a if any(c.startswith("F") for c in x["check"]["cited"])) if mode == "B" else 0}
    if gold:
        n = len(gold)
        j = [J[(r["set"], r["qid"], pick(r))] for r in gold]
        c = Counter(x["correctness"] for x in j)
        g = Counter(x["grounded"] for x in j if x["correctness"] != "답변없음")
        out.update({"vec_hit1": round(sum(1 for r in gold if r["vec_rank"] == 1) / n, 3), "vec_hit5": round(sum(1 for r in gold if r["vec_rank"] and r["vec_rank"] <= 5) / n, 3),
                    "rerank_hit1": round(sum(1 for r in gold if r["rank"] == 1) / n, 3), "rerank_hit5": round(sum(1 for r in gold if r["rank"]) / n, 3),
                    "correct_rate": round(c["일치"] / n, 3), "correct_or_partial": round((c["일치"] + c["부분일치"]) / n, 3), "correctness": {k: c[k] for k in RANK},
                    "grounded_rate": round(g["충실"] / max(sum(g.values()), 1), 3), "grounded": {k: g[k] for k in ("충실", "일부추정", "근거없음")}})
    return out


def main():
    dev_only = "--dev-only" in sys.argv
    sets = ["dev6", "unanswerable_20"] + ([] if dev_only else ["test6"])
    data = {n: load(n) for n in sets}
    with rag.connect() as conn:
        fp = f2.fingerprint(conn)["id"]
        prev = {v["fingerprint"] for v in json.load(open(f2.LOCK, encoding="utf-8")).values()}
        if fp not in prev:
            sys.exit(f"설정 지문 {fp}가 이전 평가({sorted(prev)})와 다릅니다.")
        lock = json.load(open(LOCK, encoding="utf-8")) if LOCK.exists() else {}
        if not dev_only and lock.get("completed_at"):
            sys.exit(f"test6는 이미 평가했습니다 ({lock['completed_at']}). --dev-only로 dev6만 다시 돌릴 수 있습니다.")
        if not dev_only and not lock:
            LOCK.write_text(json.dumps({"started_at": datetime.now().isoformat(timespec="seconds"), "fingerprint": fp}, ensure_ascii=False), encoding="utf-8")
        rep = {r["representative_case_key"]: r["group_id"] for r in conn.execute("select group_id, representative_case_key from interp.case_groups").fetchall()}
        grp_rep = {v: k for k, v in rep.items()}
        n_ex = conn.execute("select count(*) n from interp.exmnt_chunks").fetchone()["n"]
        rag.embed("준비")
        t0 = time.time()
        for n in sets:
            stage(conn, n, data[n], grp_rep)
        print(f"[1 검색·리랭킹·현장건의] {time.time() - t0:.0f}초 (현장건의 조각 {n_ex})", flush=True)
        recs = [r for r in f4.read_jsonl(ST) if r["set"] in sets]
        golds = {r["case_key"]: r for r in conn.execute("select case_key, title, question, answer, reason from interp.cases where case_key = any(%s)", ([r["gold_key"] for r in recs if r["gold_key"]],)).fetchall()}
    questions = {(n, q["qid"]): q["question"] for n in sets for q in data[n]}

    A = {(r["set"], r["qid"], r["mode"]): r for r in f4.read_jsonl(ANS)}
    todo = [(r, "A") for r in recs if (r["set"], r["qid"], "A") not in A] + [(r, "B") for r in recs if r["f_ids"] and (r["set"], r["qid"], "B") not in A]
    print(f"[2 답변] 호출 대상 {len(todo)}", flush=True)
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for x in ex.map(lambda t: answer(t[0], t[1], questions[key(t[0])]), todo):
            f4.append(ANS, x)
            A[(x["set"], x["qid"], x["mode"])] = x
    J = {(r["set"], r["qid"], r["mode"]): r for r in f4.read_jsonl(JDG)}
    todo = [(r, m) for r in recs if r["gold_key"] for m in (("A", "B") if r["f_ids"] else ("A",)) if (r["set"], r["qid"], m) not in J]
    print(f"[3 심사] 호출 대상 {len(todo)}", flush=True)

    def judge(t):
        r, m = t
        rec = {"set": r["set"], "qid": r["qid"], "gold_key": r["gold_key"], "context": r["ctx_a"] if m == "A" else r["ctx_b"]}
        return {**f4.judge_one(rec, A[(r["set"], r["qid"], m)], questions[key(r)], golds[r["gold_key"]]), "mode": m}
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for x in ex.map(judge, todo):
            f4.append(JDG, x)
            J[(x["set"], x["qid"], x["mode"])] = x

    out = {"ran_at": datetime.now().isoformat(timespec="seconds"), "fingerprint": fp, "exmnt_threshold": es.THRESH, "exmnt_max": es.MAX_N, "sets": {}}
    for n in sets:
        rs = [r for r in recs if r["set"] == n]
        s = {"A": summarize(rs, A, J, "A"), "B": summarize(rs, A, J, "B")}
        att = [r for r in rs if r["f_ids"] and r["gold_key"]]
        if att:
            ja, jb = [J[(r["set"], r["qid"], "A")] for r in att], [J[(r["set"], r["qid"], "B")] for r in att]
            b = sum(1 for x, y in zip(ja, jb) if RANK[y["correctness"]] > RANK[x["correctness"]])
            w = sum(1 for x, y in zip(ja, jb) if RANK[y["correctness"]] < RANK[x["correctness"]])
            gb = sum(1 for x, y in zip(ja, jb) if x["grounded"] != "충실" and y["grounded"] == "충실")
            gw = sum(1 for x, y in zip(ja, jb) if x["grounded"] == "충실" and y["grounded"] != "충실")
            s["attached_paired"] = {"n": len(att), "정확도 좋아짐/나빠짐": [b, w], "p": round(sign_p(b, w), 4), "충실도 좋아짐/나빠짐": [gb, gw], "p_충실": round(sign_p(gb, gw), 4)}
        s["f_top1_quantiles"] = [round(x, 3) for x in sorted(r["f_top1"] for r in rs if r["f_top1"] is not None)[::max(1, len(rs) // 4)]]
        if n != "test6":
            s["items"] = [{"qid": r["qid"], "rank": r["rank"], "f_ids": r["f_ids"], "f_top1": r["f_top1"], "A": J.get((r["set"], r["qid"], "A"), {}).get("correctness"),
                           "B": J.get((r["set"], r["qid"], "B" if r["f_ids"] else "A"), {}).get("correctness"),
                           "B_cited": A[(r["set"], r["qid"], "B" if r["f_ids"] else "A")]["check"]["cited"],
                           "B_reason": J.get((r["set"], r["qid"], "B"), {}).get("reason") if r["f_ids"] else None} for r in rs]
        out["sets"][n] = s
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not dev_only:
        LOCK.write_text(json.dumps({**json.load(open(LOCK, encoding="utf-8")), "completed_at": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False), encoding="utf-8")
    manifest("f11_dev6_test6_eval", sets=sets, fingerprint=fp)
    print(json.dumps({n: {k: v for k, v in s.items() if k != "items"} for n, s in out["sets"].items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
