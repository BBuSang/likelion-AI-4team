"""F10 데모 설정 그대로의 개발셋 평가: 리랭킹 상위 5개(f9의 정렬 결과) + 데모 프롬프트 `APP_SYSTEM_V2`. 화면이 실제로 쓰는 조합을 잰다.
f5 --v2(프롬프트만)와 f9(리랭킹만)는 따로 쟀고, 둘을 함께 쓴 조합은 재지 않았기 때문이다. 개발셋(합성 150, dev1~4)과 답변불가 20만 쓴다. 시험 세트는 읽지 않는다.
'못 찾을 때만 키워드 완화'(76번)는 포함하지 않는다(첫 답변 기준).

사전 등록한 합격 기준 (결과를 보기 전에 정함, f5·f9와 같다)
- f4 기존(벡터 상위 5 + rag.SYSTEM: 일치 93.4%, 충실 82.8%) 대비 각각 3%p 넘게 떨어지지 않는다 → 일치 ≥ 90.4%, 충실 ≥ 79.8%
- 답변불가 20개 전부 "찾지 못했습니다", 근거 번호 오류 0, 단정 표현 0
- 함께 본다: "불일치" 수, "찾지 못했습니다" 수, 같은 문항 변화(부호 검정)
- 미달이어도 프롬프트·리랭킹을 다시 고쳐 맞추지 않는다 (결과를 그대로 보고)
캐시: data/eval/_f10_answers.jsonl, _f10_judge.jsonl     결과: data/eval/f10_demo_config_eval.json
"""
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import f4_rag_answer_eval as f4  # noqa: E402
import f9_rerank_eval as f9  # noqa: E402
import rag  # noqa: E402
from common import DATA, manifest  # noqa: E402
from f5_app_prompt_eval import RANK, sign_p  # noqa: E402
from prompts import APP_SYSTEM_V2  # noqa: E402

EVAL = DATA / "eval"
A10, J10, OUT = EVAL / "_f10_answers.jsonl", EVAL / "_f10_judge.jsonl", EVAL / "f10_demo_config_eval.json"
key = f9.key


def answer_one(rec, question):
    masked = rag.mask_pii(question)
    text, usage = f4.with_retry(lambda: rag.llm([{"role": "system", "content": APP_SYSTEM_V2}, {"role": "user", "content": f"[질문]\n{masked}\n\n[사례]\n{rec['context']}"}]))
    text = text.strip()
    chk = rag.check_output(text, rec["ids"])
    if not text.endswith(rag.CLOSING):
        text = f"{text}\n\n{rag.CLOSING}"
    return {"set": rec["set"], "qid": rec["qid"], "answer": text, "check": chk, "usage": usage}


def main():
    base = [r for r in f4.read_jsonl(f4.S1) if r["set"] in f9.SETS]
    rer = {key(r): r for r in f4.read_jsonl(f9.R9)}
    assert all(key(b) in rer for b in base), "먼저 f9_rerank_eval.py를 실행해야 한다"
    questions = {}
    for name in f9.SETS:
        for l in open(EVAL / f"{name}.jsonl", encoding="utf-8"):
            x = json.loads(l)
            questions[(name, x["qid"])] = x["question"]
    recs = []
    with rag.connect() as conn:
        for b in base:
            ids = rer[key(b)]["order"]
            cases = rag.load_cases(ids, conn)
            new = f9.ranks(b, rer[key(b)])[1] if b["gold_group_id"] else None
            recs.append({**b, "ids": ids, "context": "\n\n".join(rag.case_block(cases[i]) for i in ids), "rank": new})
        golds = {r["case_key"]: r for r in conn.execute("select case_key, title, question, answer, reason from interp.cases where case_key = any(%s)", ([r["gold_key"] for r in recs if r["gold_key"]],)).fetchall()}
    new_a = {key(r): r for r in f4.read_jsonl(A10)}
    todo = [r for r in recs if key(r) not in new_a]
    print(f"[1 답변] 호출 대상 {len(todo)} / {len(recs)}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for i, a in enumerate(ex.map(lambda r: answer_one(r, questions[key(r)]), todo), 1):
            f4.append(A10, a)
            new_a[key(a)] = a
            if i % 100 == 0:
                print(f"[1 답변] {i}/{len(todo)} {time.time() - t0:.0f}초", flush=True)
    new_j = {key(r): r for r in f4.read_jsonl(J10)}
    todo = [r for r in recs if r["gold_key"] and key(r) not in new_j]
    print(f"[2 심사] 호출 대상 {len(todo)}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for i, j in enumerate(ex.map(lambda r: f4.judge_one(r, new_a[key(r)], questions[key(r)], golds[r["gold_key"]]), todo), 1):
            f4.append(J10, j)
            new_j[key(j)] = j
            if i % 100 == 0:
                print(f"[2 심사] {i}/{len(todo)} {time.time() - t0:.0f}초", flush=True)
    base_a = {key(r): r for r in f4.read_jsonl(f4.ANS)}
    base_j = {key(r): r for r in f4.read_jsonl(f4.JDG)}
    dev = [r for r in recs if r["gold_key"] and r["set"] in f4.DEV_SETS]
    better = sum(1 for r in dev if RANK[new_j[key(r)]["correctness"]] > RANK[base_j[key(r)]["correctness"]])
    worse = sum(1 for r in dev if RANK[new_j[key(r)]["correctness"]] < RANK[base_j[key(r)]["correctness"]])
    gb = sum(1 for r in dev if base_j[key(r)]["grounded"] != "충실" and new_j[key(r)]["grounded"] == "충실")
    gw = sum(1 for r in dev if base_j[key(r)]["grounded"] == "충실" and new_j[key(r)]["grounded"] != "충실")
    bs, ns = f4.aggregate(base, base_a, base_j), f4.aggregate(recs, new_a, new_j)
    out = {"summary": {g: {"base": bs[g], "new": ns[g]} for g in bs},
           "paired": {"n": len(dev), "better": better, "worse": worse, "p_sign": round(sign_p(better, worse), 4), "grounded_better": gb, "grounded_worse": gw, "p_sign_grounded": round(sign_p(gb, gw), 4),
                      "transitions": {f"{a}→{b}": n for (a, b), n in Counter((base_j[key(r)]["correctness"], new_j[key(r)]["correctness"]) for r in dev).most_common()}},
           "items_mismatch": [{"set": r["set"], "qid": r["qid"], "gold_in_top5": bool(r["rank"]), "reason": new_j[key(r)]["reason"]} for r in dev if new_j[key(r)]["correctness"] == "불일치"],
           "tokens": {"answer_in": sum(a["usage"]["input"] for a in new_a.values()), "answer_out": sum(a["usage"]["output"] for a in new_a.values()), "judge_in": sum(j["usage"]["input"] for j in new_j.values())}}
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    manifest("f10_demo_config_eval", n=len(recs))
    print(json.dumps({k: out[k] for k in ("paired", "items_mismatch", "tokens")}, ensure_ascii=False, indent=1))
    for g, v in out["summary"].items():
        b, n = v["base"], v["new"]
        print(g, "| 일치", b.get("correct_rate"), "->", n.get("correct_rate"), "| 충실", b.get("grounded_rate"), "->", n.get("grounded_rate"), "|", b.get("correctness"), "->", n.get("correctness"),
              "| 못찾음", b["not_found"], "->", n["not_found"], "| 번호오류", n["invalid_citations"], "단정", n["assertive"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
