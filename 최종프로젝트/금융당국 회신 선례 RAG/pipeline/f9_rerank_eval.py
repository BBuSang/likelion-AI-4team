"""F9 LLM 리랭킹(`app/rerank.py`) 개발셋 평가: 벡터 상위 20개 묶음을 LLM이 다시 정렬해 상위 5개를 고른다. 문맥 크기(5개)는 그대로다.
대상은 개발셋(합성 150, dev1~4)과 답변불가 20뿐이다. 시험 세트(test*, final_test)는 읽지 않는다.

사전 등록한 기준 (결과를 보기 전에 정함)
[A 검색] `python f9_rerank_eval.py`
- 보고: Hit@1·3·5·MRR@5(벡터 순서 vs 리랭킹), 정답이 상위 20 안인 비율(리랭킹의 상한), 상위 5로 들어온 문항(회수)과 밀려난 문항(손실), 같은 문항 순위 비교(부호 검정)
- B로 넘어가는 조건: 개발셋 합계에서 회수 > 손실이고 Hit@1이 1%p 넘게 떨어지지 않는다. 아니면 여기서 멈추고 보류한다
[B 답변] `python f9_rerank_eval.py --answers` (A 조건을 통과했을 때만)
- 리랭킹 상위 5개로 답변(rag.SYSTEM, f4와 같은 프롬프트)과 심사(f4와 같은 방식)를 다시 한다. 상위 5개 사례와 순서가 f4와 같은 문항은 f4의 답변·심사를 그대로 쓴다
- 합격: 정확도 "일치"와 충실도 "충실" 비율이 f4 기존보다 각각 3%p 넘게 떨어지지 않는다. 답변불가 20개 전부 "찾지 못했습니다", 근거 번호 오류 0, 단정 표현 0
- 함께 본다: "불일치"(틀린 결론) 수, 정답이 상위 5 밖이던 문항의 변화, 질문당 추가 LLM 호출 1회와 입력 토큰
- 이 결과로 리랭킹 프롬프트·후보 수·발췌 길이를 여러 번 고쳐 가며 맞추지 않는다 (결과를 그대로 보고하고 사용자가 결정)
캐시: data/eval/_f9_rerank.jsonl, _f9_answers.jsonl, _f9_judge.jsonl     결과: data/eval/f9_rerank_eval.json
"""
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import f4_rag_answer_eval as f4  # noqa: E402
import rag  # noqa: E402
import rerank as rr  # noqa: E402
from common import DATA, manifest  # noqa: E402
from f5_app_prompt_eval import RANK, sign_p  # noqa: E402

EVAL = DATA / "eval"
R9, A9, J9, OUT = EVAL / "_f9_rerank.jsonl", EVAL / "_f9_answers.jsonl", EVAL / "_f9_judge.jsonl", EVAL / "f9_rerank_eval.json"
SETS = f4.DEV_SETS + (f4.UNANS,)
ANSWERS = "--answers" in sys.argv
key = lambda r: (r["set"], r["qid"])
sys.stdout.reconfigure(encoding="utf-8")


def stage_a(base, questions):
    done = {key(r): r for r in f4.read_jsonl(R9)}
    todo = [b for b in base if key(b) not in done]
    print(f"[A 후보 검색] 대상 {len(todo)} / {len(base)}", flush=True)
    cands, t0 = {}, time.time()
    with rag.connect() as conn:
        for i, b in enumerate(todo, 1):
            cs = rr.candidates(conn, questions[key(b)])
            cands[key(b)] = (cs, rr.load_brief(conn, [c["doc_id"] for c in cs]))
            if i % 100 == 0:
                print(f"[A 후보 검색] {i}/{len(todo)} {time.time() - t0:.0f}초", flush=True)

    def one(b):
        cs, brief = cands[key(b)]
        ids = [c["doc_id"] for c in cs]
        order, usage, llm_order = f4.with_retry(lambda: rr.rerank(questions[key(b)], ids, brief))
        return {"set": b["set"], "qid": b["qid"], "cand_ids": ids, "cand_groups": [c["group_id"] for c in cs], "cand_scores": [round(c["score"], 3) for c in cs],
                "order": order, "llm_order": llm_order, "usage": usage}
    t0 = time.time()
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for i, r in enumerate(ex.map(one, todo), 1):
            f4.append(R9, r)
            done[key(r)] = r
            if i % 50 == 0:
                print(f"[A 리랭킹] {i}/{len(todo)} {time.time() - t0:.0f}초", flush=True)
    return done


def ranks(b, r):
    """(벡터 순위, 리랭킹 순위). 리랭킹 순위는 고른 5개 안이면 그 자리, 아니면 None(상위 5 밖)."""
    gold = b["gold_group_id"]
    vec = r["cand_groups"].index(gold) + 1 if gold in r["cand_groups"] else None
    g_of = dict(zip(r["cand_ids"], r["cand_groups"]))
    new = next((i + 1 for i, d in enumerate(r["order"]) if g_of[d] == gold), None)
    return vec, new


def retrieval_summary(base, rer):
    out = {}
    for name in list(f4.DEV_SETS) + ["개발셋 합계"]:
        rs = [b for b in base if b["gold_group_id"] and (name == "개발셋 합계" or b["set"] == name)]
        pairs = [ranks(b, rer[key(b)]) for b in rs]
        n = len(rs)
        hit = lambda xs, k: round(sum(1 for x in xs if x and x <= k) / n, 4)
        mrr5 = lambda xs: round(sum(1 / x for x in xs if x and x <= 5) / n, 4)
        v, w = [p[0] for p in pairs], [p[1] for p in pairs]
        eff = lambda x: x if x and x <= 5 else 99                      # 상위 5 밖은 같은 것으로 본다 (리랭킹은 5개만 고른다)
        better, worse = sum(1 for a, b_ in pairs if eff(b_) < eff(a)), sum(1 for a, b_ in pairs if eff(b_) > eff(a))
        out[name] = {"n": n, "in_top20": sum(1 for x in v if x), "vector": {"hit1": hit(v, 1), "hit3": hit(v, 3), "hit5": hit(v, 5), "mrr5": mrr5(v)},
                     "rerank": {"hit1": hit(w, 1), "hit3": hit(w, 3), "hit5": hit(w, 5), "mrr5": mrr5(w)},
                     "recovered": sum(1 for a, b_ in pairs if not (a and a <= 5) and b_), "lost": sum(1 for a, b_ in pairs if a and a <= 5 and not b_),
                     "rank_better": better, "rank_worse": worse, "p_sign": round(sign_p(better, worse), 4)}
    return out


def stage_b(base, rer, questions):
    base_a = {key(r): r for r in f4.read_jsonl(f4.ANS)}
    base_j = {key(r): r for r in f4.read_jsonl(f4.JDG)}
    recs, reuse = [], set()
    with rag.connect() as conn:
        rep = {r["representative_case_key"]: r["group_id"] for r in conn.execute("select group_id, representative_case_key from interp.case_groups").fetchall()}
        for b in base:
            ids = rer[key(b)]["order"]
            if ids == b["ids"]:
                recs.append({**b})
                reuse.add(key(b))
                continue
            cases = rag.load_cases(ids, conn)
            _, new = ranks(b, rer[key(b)]) if b["gold_group_id"] else (None, None)
            recs.append({**b, "ids": ids, "context": "\n\n".join(rag.case_block(cases[i]) for i in ids), "rank": new, "rank_vector": b["rank"]})
        golds = {r["case_key"]: r for r in conn.execute("select case_key, title, question, answer, reason from interp.cases where case_key = any(%s)", ([r["gold_key"] for r in recs if r["gold_key"]],)).fetchall()}
    for b in recs:                                                    # 재사용 문항의 순위도 '상위 5 안/밖' 기준으로 맞춘다
        if key(b) in reuse and b["gold_group_id"]:
            b["rank"] = b["rank"] if b["rank"] and b["rank"] <= 5 else None
    new_a = {key(r): r for r in f4.read_jsonl(A9)}
    todo = [r for r in recs if key(r) not in reuse and key(r) not in new_a]
    print(f"[B 답변] 문맥이 바뀐 문항 {len(recs) - len(reuse)} / {len(recs)}, 호출 대상 {len(todo)}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for i, a in enumerate(ex.map(lambda r: f4.answer_one(r, questions[key(r)]), todo), 1):
            f4.append(A9, a)
            new_a[key(a)] = a
            if i % 50 == 0:
                print(f"[B 답변] {i}/{len(todo)} {time.time() - t0:.0f}초", flush=True)
    new_j = {key(r): r for r in f4.read_jsonl(J9)}
    todo = [r for r in recs if r["gold_key"] and key(r) not in reuse and key(r) not in new_j]
    print(f"[B 심사] 호출 대상 {len(todo)}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for i, j in enumerate(ex.map(lambda r: f4.judge_one(r, new_a[key(r)], questions[key(r)], golds[r["gold_key"]]), todo), 1):
            f4.append(J9, j)
            new_j[key(j)] = j
            if i % 50 == 0:
                print(f"[B 심사] {i}/{len(todo)} {time.time() - t0:.0f}초", flush=True)
    for k in reuse:
        new_a[k] = base_a[k]
        if k in base_j:
            new_j[k] = base_j[k]
    dev = [r for r in recs if r["gold_key"] and r["set"] in f4.DEV_SETS]
    b_of = {key(b): b for b in base}
    in5 = lambda r: bool(b_of[key(r)]["rank"] and b_of[key(r)]["rank"] <= 5)
    better = sum(1 for r in dev if RANK[new_j[key(r)]["correctness"]] > RANK[base_j[key(r)]["correctness"]])
    worse = sum(1 for r in dev if RANK[new_j[key(r)]["correctness"]] < RANK[base_j[key(r)]["correctness"]])
    gb = sum(1 for r in dev if base_j[key(r)]["grounded"] != "충실" and new_j[key(r)]["grounded"] == "충실")
    gw = sum(1 for r in dev if base_j[key(r)]["grounded"] == "충실" and new_j[key(r)]["grounded"] != "충실")
    bs, ns = f4.aggregate(base, base_a, base_j), f4.aggregate(recs, new_a, new_j)
    out = {"summary": {g: {"base": bs[g], "new": ns[g]} for g in bs}, "changed_context": len(recs) - len(reuse),
           "paired": {"n": len(dev), "better": better, "worse": worse, "p_sign": round(sign_p(better, worse), 4), "grounded_better": gb, "grounded_worse": gw, "p_sign_grounded": round(sign_p(gb, gw), 4),
                      "transitions": {f"{a}→{b}": n for (a, b), n in Counter((base_j[key(r)]["correctness"], new_j[key(r)]["correctness"]) for r in dev).most_common()}},
           "missed_before": [{"set": r["set"], "qid": r["qid"], "rank_vector": b_of[key(r)]["rank"], "rank_rerank": r["rank"] if key(r) not in reuse else None,
                              "base": base_j[key(r)]["correctness"], "new": new_j[key(r)]["correctness"]} for r in dev if not in5(r)],
           "items_worse": [{"set": r["set"], "qid": r["qid"], "base": base_j[key(r)]["correctness"], "new": new_j[key(r)]["correctness"], "gold_in_new_top5": bool(r["rank"]), "new_reason": new_j[key(r)]["reason"]}
                           for r in dev if RANK[new_j[key(r)]["correctness"]] < RANK[base_j[key(r)]["correctness"]]],
           "tokens": {"answer_in": sum(a["usage"]["input"] for k, a in new_a.items() if k not in reuse), "judge_in": sum(j["usage"]["input"] for k, j in new_j.items() if k not in reuse)}}
    return out


def main():
    base = [r for r in f4.read_jsonl(f4.S1) if r["set"] in SETS]
    questions = {}
    for name in SETS:
        for l in open(EVAL / f"{name}.jsonl", encoding="utf-8"):
            x = json.loads(l)
            questions[(name, x["qid"])] = x["question"]
    rer = stage_a(base, questions)
    out = {"retrieval": retrieval_summary(base, rer),
           "rerank_calls": {"n": len(rer), "tokens_in": sum(r["usage"]["input"] for r in rer.values()), "tokens_out": sum(r["usage"]["output"] for r in rer.values()),
                            "llm_returned_5": sum(1 for r in rer.values() if len(r["llm_order"]) >= 5), "top5_same_as_vector": sum(1 for r in rer.values() if r["order"] == r["cand_ids"][:5]),
                            "top5_same_set": sum(1 for r in rer.values() if set(r["order"]) == set(r["cand_ids"][:5]))}}
    # 벡터 순위가 f4 기록과 같은지 (같은 검색인지 확인)
    out["vector_rank_matches_f4"] = sum(1 for b in base if b["gold_group_id"] and (ranks(b, rer[key(b)])[0] if (ranks(b, rer[key(b)])[0] or 99) <= 10 else None) == b["rank"])
    out["n_with_gold"] = sum(1 for b in base if b["gold_group_id"])
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if ANSWERS:
        out["answers"] = stage_b(base, rer, questions)
        a = out["answers"]
        print(json.dumps({k: a[k] for k in ("changed_context", "paired", "missed_before", "tokens")}, ensure_ascii=False, indent=1))
        keys = ["n", "correct_rate", "correct_or_partial_rate", "grounded_rate", "correctness", "grounded", "not_found", "invalid_citations", "assertive"]
        for g, v in a["summary"].items():
            print(f"\n[{g}]")
            for k in keys:
                if k in v["base"]:
                    print(f"  {k:26s} 기존 {str(v['base'].get(k))} → 새 {str(v['new'].get(k))}")
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    manifest("f9_rerank_eval", n=len(rer), answers=ANSWERS)


if __name__ == "__main__":
    main()
