"""F6 키워드 순차 완화(`app/keyword_relax.py`) 개발셋 평가: 벡터 상위 5개(f4와 같은 검색 문맥)에 키워드로 넓혀 찾은 사례 최대 3개를 더해 답변을 다시 만들고 f4 기존 결과와 비교한다.
프롬프트는 f4와 같은 `rag.SYSTEM`을 쓴다(검색 문맥만 바뀌도록). 대상은 개발셋(합성 150, dev1~4)과 답변불가 20뿐이다. 시험 세트는 읽지 않는다.

사전 등록한 기준 (결과를 보기 전에 정함)
- 검색: 정답 묶음이 LLM에 넘어간 문맥 안에 있는 비율(기존 = Hit@5, 새 = 상위 5 + 추가분). 기존에 상위 5 밖이던 개발셋 8문항 중 몇 개가 추가분으로 들어오는지
- 답변: 정확도 "일치" 비율과 충실도 "충실" 비율이 기존보다 각각 3%p 넘게 떨어지지 않는다
- 정답이 상위 5 밖이던 문항의 "찾지 못했습니다"가 줄어드는지, 정답이 상위 5 안이던 문항의 정답 인용률이 유지되는지(사례가 늘어 흐려지지 않는지)
- 답변불가 20개는 전부 "찾지 못했습니다", 근거 번호 오류 0, 단정 표현 0
- 비용: 질문당 추가 LLM 호출 1회(키워드 추출), 문맥 길이 증가
- 이 결과로 키워드 추출·완화 규칙을 여러 번 고쳐 가며 맞추지 않는다 (결과를 그대로 보고하고 사용자가 결정)
사용: python f6_keyword_relax_eval.py     캐시: data/eval/_f6_keywords.jsonl, _f6_answers.jsonl, _f6_judge.jsonl     결과: data/eval/f6_keyword_relax_eval.json
대조 실행(결과를 본 뒤 추가): `--control`은 추가 사례 없이 f4와 똑같은 문맥·프롬프트로 답변과 심사만 다시 한다. 다시 생성·심사하는 것만으로 생기는 차이(노이즈, API 모델 변화)를 재기 위해서다.
    캐시: _f6c_answers.jsonl, _f6c_judge.jsonl     결과: f6_control_eval.json
"""
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import f4_rag_answer_eval as f4  # noqa: E402
import keyword_relax as kr  # noqa: E402
import rag  # noqa: E402
from common import DATA, manifest  # noqa: E402
from f5_app_prompt_eval import sign_p, RANK  # noqa: E402

EVAL = DATA / "eval"
CONTROL = "--control" in sys.argv
K6 = EVAL / "_f6_keywords.jsonl"
A6, J6 = (EVAL / "_f6c_answers.jsonl", EVAL / "_f6c_judge.jsonl") if CONTROL else (EVAL / "_f6_answers.jsonl", EVAL / "_f6_judge.jsonl")
OUT = EVAL / ("f6_control_eval.json" if CONTROL else "f6_keyword_relax_eval.json")
SETS = f4.DEV_SETS + (f4.UNANS,)
sys.stdout.reconfigure(encoding="utf-8")


def answer_one(rec, question):
    masked = rag.mask_pii(question)
    text, usage = f4.with_retry(lambda: rag.llm([{"role": "system", "content": rag.SYSTEM}, {"role": "user", "content": f"[질문]\n{masked}\n\n[사례]\n{rec['context']}"}]))
    text = text.strip()
    chk = rag.check_output(text, rec["ids"])
    if not text.endswith(rag.CLOSING):
        text = f"{text}\n\n{rag.CLOSING}"
    return {"set": rec["set"], "qid": rec["qid"], "answer": text, "check": chk, "usage": usage}


def main():
    base_recs = [r for r in f4.read_jsonl(f4.S1) if r["set"] in SETS]
    questions = {}
    for name in SETS:
        for l in open(EVAL / f"{name}.jsonl", encoding="utf-8"):
            x = json.loads(l)
            questions[(name, x["qid"])] = x["question"]

    # 1 키워드 추출 + 완화 검색 (결정적 부분은 매번 다시 계산, LLM 추출만 캐시)
    kw = {(r["set"], r["qid"]): r for r in f4.read_jsonl(K6)}
    todo = [r for r in base_recs if (r["set"], r["qid"]) not in kw]
    print(f"[1 키워드] 호출 대상 {len(todo)} / {len(base_recs)}", flush=True)

    def ext(r):
        c, u = f4.with_retry(lambda: kr.extract_concepts(questions[(r["set"], r["qid"])]))
        return {"set": r["set"], "qid": r["qid"], "concepts": c, "usage": u}
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for k in ex.map(ext, todo):
            f4.append(K6, k)
            kw[(k["set"], k["qid"])] = k

    recs = []
    with rag.connect() as conn:
        rep = {r["representative_case_key"]: r["group_id"] for r in conn.execute("select group_id, representative_case_key from interp.case_groups").fetchall()}
        rag.embed("준비")
        t0 = time.time()
        for i, b in enumerate(base_recs, 1):
            q = questions[(b["set"], b["qid"])]
            res = kr.relax_search(conn, kw[(b["set"], b["qid"])]["concepts"], rag.embed(rag.mask_pii(q)), {rep[x] for x in b["ids"]})
            extra = [] if CONTROL else res["extra"]
            cases = rag.load_cases([e["doc_id"] for e in extra], conn) if extra else {}
            ctx = b["context"] + "".join("\n\n" + kr.extra_block(cases[e["doc_id"]], e) for e in extra)
            gold_extra = next((j + 1 for j, e in enumerate(extra) if e["group_id"] == b["gold_group_id"]), None) if b["gold_group_id"] else None
            recs.append({**b, "context": ctx, "ids": b["ids"] + [e["doc_id"] for e in extra], "extra": extra, "trail": res["trail"], "gold_in_extra": gold_extra})
            if i % 100 == 0:
                print(f"[1 완화 검색] {i}/{len(base_recs)} {time.time() - t0:.0f}초", flush=True)
        golds = {r["case_key"]: r for r in conn.execute("select case_key, title, question, answer, reason from interp.cases where case_key = any(%s)", ([r["gold_key"] for r in recs if r["gold_key"]],)).fetchall()}

    # 2 답변
    new_a = {(r["set"], r["qid"]): r for r in f4.read_jsonl(A6)}
    todo = [r for r in recs if (r["set"], r["qid"]) not in new_a]
    t0 = time.time()
    print(f"[2 답변] 호출 대상 {len(todo)}", flush=True)
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for i, a in enumerate(ex.map(lambda r: answer_one(r, questions[(r["set"], r["qid"])]), todo), 1):
            f4.append(A6, a)
            new_a[(a["set"], a["qid"])] = a
            if i % 50 == 0:
                print(f"[2 답변] {i}/{len(todo)} {time.time() - t0:.0f}초", flush=True)

    # 3 심사
    new_j = {(r["set"], r["qid"]): r for r in f4.read_jsonl(J6)}
    todo = [r for r in recs if r["gold_key"] and (r["set"], r["qid"]) not in new_j]
    t0 = time.time()
    print(f"[3 심사] 호출 대상 {len(todo)}", flush=True)
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for i, j in enumerate(ex.map(lambda r: f4.judge_one(r, new_a[(r["set"], r["qid"])], questions[(r["set"], r["qid"])], golds[r["gold_key"]]), todo), 1):
            f4.append(J6, j)
            new_j[(j["set"], j["qid"])] = j
            if i % 50 == 0:
                print(f"[3 심사] {i}/{len(todo)} {time.time() - t0:.0f}초", flush=True)

    # 비교
    base_a = {(r["set"], r["qid"]): r for r in f4.read_jsonl(f4.ANS)}
    base_j = {(r["set"], r["qid"]): r for r in f4.read_jsonl(f4.JDG)}
    key = lambda r: (r["set"], r["qid"])
    dev = [r for r in recs if r["gold_key"] and r["set"] in f4.DEV_SETS]
    in5 = lambda r: bool(r["rank"] and r["rank"] <= 5)
    out = {"retrieval": {}, "summary": {}, "paired": {}}
    for name in list(f4.DEV_SETS) + ["개발셋 합계"]:
        rs = [r for r in dev if name == "개발셋 합계" or r["set"] == name]
        out["retrieval"][name] = {"n": len(rs), "hit5": sum(map(in5, rs)), "in_context_new": sum(1 for r in rs if in5(r) or r["gold_in_extra"]),
                                  "recovered": sum(1 for r in rs if not in5(r) and r["gold_in_extra"]), "missed_before": sum(1 for r in rs if not in5(r))}
    allr = [r for r in recs if r["set"] in SETS]
    out["extra_stats"] = {"avg_extra": round(sum(len(r["extra"]) for r in allr) / len(allr), 2),
                          "no_concepts": sum(1 for r in allr if not kw[key(r)]["concepts"]),
                          "n_concepts": dict(Counter(len(kw[key(r)]["concepts"]) for r in allr)),
                          "top_level_full_match": sum(1 for r in allr if r["trail"] and r["trail"][0]["groups"] > 0),
                          "extra_matched_of": dict(Counter(f"{e['matched']}/{e['of']}" for r in allr for e in r["extra"])),
                          "context_chars_base": round(sum(len(b["context"]) for b in base_recs) / len(base_recs)), "context_chars_new": round(sum(len(r["context"]) for r in allr) / len(allr))}
    bs, ns = f4.aggregate(recs, base_a, base_j), f4.aggregate(recs, new_a, new_j)
    for g in bs:
        out["summary"][g] = {"base": bs[g], "new": ns[g]}
    # 정답 위치별 비교
    for label, cond in (("상위 5 안", in5), ("상위 5 밖", lambda r: not in5(r))):
        rs = [r for r in dev if cond(r)]
        out["summary"][f"개발셋 정답 {label}"] = {
            "n": len(rs),
            "base": dict(Counter(base_j[key(r)]["correctness"] for r in rs)), "new": dict(Counter(new_j[key(r)]["correctness"] for r in rs)),
            "gold_cited_base": sum(1 for r in rs if r["gold_key"] in base_a[key(r)]["check"]["cited"]), "gold_cited_new": sum(1 for r in rs if r["gold_key"] in new_a[key(r)]["check"]["cited"]),
            "cited_extra_new": sum(1 for r in rs if set(new_a[key(r)]["check"]["cited"]) & {e["doc_id"] for e in r["extra"]})}
    better = sum(1 for r in dev if RANK[new_j[key(r)]["correctness"]] > RANK[base_j[key(r)]["correctness"]])
    worse = sum(1 for r in dev if RANK[new_j[key(r)]["correctness"]] < RANK[base_j[key(r)]["correctness"]])
    gb = sum(1 for r in dev if base_j[key(r)]["grounded"] != "충실" and new_j[key(r)]["grounded"] == "충실")
    gw = sum(1 for r in dev if base_j[key(r)]["grounded"] == "충실" and new_j[key(r)]["grounded"] != "충실")
    out["paired"] = {"n": len(dev), "better": better, "worse": worse, "p_sign": round(sign_p(better, worse), 4), "grounded_better": gb, "grounded_worse": gw, "p_sign_grounded": round(sign_p(gb, gw), 4),
                     "transitions": {f"{a}→{b}": n for (a, b), n in Counter((base_j[key(r)]["correctness"], new_j[key(r)]["correctness"]) for r in dev).most_common()}}
    out["missed_items"] = [{"set": r["set"], "qid": r["qid"], "rank": r["rank"], "gold": r["gold_key"], "concepts": kw[key(r)]["concepts"], "trail": r["trail"],
                            "extra": [(e["doc_id"], f"{e['matched']}/{e['of']}") for e in r["extra"]], "gold_in_extra": r["gold_in_extra"],
                            "base": base_j[key(r)]["correctness"], "new": new_j[key(r)]["correctness"], "new_reason": new_j[key(r)]["reason"]} for r in dev if not in5(r)]
    out["items_worse"] = [{"set": r["set"], "qid": r["qid"], "base": base_j[key(r)]["correctness"], "new": new_j[key(r)]["correctness"], "cited_extra": sorted(set(new_a[key(r)]["check"]["cited"]) & {e["doc_id"] for e in r["extra"]}),
                           "new_reason": new_j[key(r)]["reason"]} for r in dev if RANK[new_j[key(r)]["correctness"]] < RANK[base_j[key(r)]["correctness"]]]
    out["tokens"] = {"keyword_in": sum(k["usage"]["input"] for k in kw.values()), "keyword_out": sum(k["usage"]["output"] for k in kw.values()),
                     "answer_in": sum(a["usage"]["input"] for a in new_a.values()), "answer_out": sum(a["usage"]["output"] for a in new_a.values()),
                     "judge_in": sum(j["usage"]["input"] for j in new_j.values()), "judge_out": sum(j["usage"]["output"] for j in new_j.values())}
    json.dump({**out, "items": [{"set": r["set"], "qid": r["qid"], "concepts": kw[key(r)]["concepts"], "trail": r["trail"], "extra": r["extra"], "answer_new": new_a[key(r)]["answer"],
                                 "judge_new": {k: new_j[key(r)][k] for k in ("correctness", "grounded", "reason")} if key(r) in new_j else None} for r in recs]},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    manifest("f6_control_eval" if CONTROL else "f6_keyword_relax_eval", n=len(recs))
    print(json.dumps({k: out[k] for k in ("retrieval", "extra_stats", "paired", "tokens")}, ensure_ascii=False, indent=1))
    keys = ["n", "correct_rate", "correct_or_partial_rate", "grounded_rate", "gold_cited_when_in_top5", "not_found", "invalid_citations", "assertive", "answers_with_uncited_sentences"]
    for g, v in out["summary"].items():
        print(f"\n[{g}]")
        if "n" in v and "base" in v and "correct_rate" not in v["base"] and "invalid_citations" not in v["base"]:
            print(" ", {k: v[k] for k in v})
            continue
        for k in keys:
            if k in v["base"]:
                print(f"  {k:32s} 기존 {str(v['base'].get(k)):>10s} → 새 {str(v['new'].get(k)):>10s}")
    print("\n[놓쳤던 문항]")
    for m in out["missed_items"]:
        print(" ", m)


if __name__ == "__main__":
    main()
