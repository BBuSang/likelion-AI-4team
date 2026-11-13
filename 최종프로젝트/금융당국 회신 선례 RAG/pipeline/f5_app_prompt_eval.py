"""F5 데모 프롬프트(`app/prompts.py`의 APP_SYSTEM) 개발셋 비교: 기존 평가(f4)와 같은 문항·같은 검색 문맥·같은 심사 방식에서 프롬프트만 바꿔 답변을 다시 만들고 비교한다.
대상은 개발셋(합성 150, dev1~4)과 답변불가 20뿐이다. 시험 세트(test*, final_test)는 읽지 않는다.

사전 등록한 합격 기준 (결과를 보기 전에 정함)
- 정확도 "일치" 비율이 기존(rag.SYSTEM)보다 3%p 넘게 떨어지지 않는다
- 충실도 "충실" 비율과 정답 사례 인용률이 눈에 띄게(각각 3%p 넘게) 나빠지지 않는다
- 답변불가 20개는 전부 "찾지 못했습니다", 근거 번호 오류 0, 단정 표현 0
- 새로 보는 것: 답변 글에 회신 날짜 표기가 남는 비율, 답변 길이
- 같은 문항의 정확도 변화는 좋아짐/나빠짐 수와 부호 검정으로 함께 본다. LLM 답변은 호출마다 달라질 수 있어 작은 차이는 노이즈일 수 있다
- 이 결과로 프롬프트를 여러 번 고쳐 가며 맞추지 않는다 (통과하지 못하면 근거와 함께 보고하고 사용자가 결정)
사용: python f5_app_prompt_eval.py     캐시: data/eval/_f5_answers.jsonl, _f5_judge.jsonl     결과: data/eval/f5_app_prompt_eval.json

`--v2` (81번 검토 뒤 추가): 규칙 8을 더한 `APP_SYSTEM_V2`를 같은 방식으로 한 번 잰다. 캐시 _f5v2_*.jsonl, 결과 f5_app_prompt_v2_eval.json
- 합격 기준은 위와 같다(결과를 보기 전에 고정): f4 기존(일치 93.4%, 충실 82.8%) 대비 각각 3%p 넘게 떨어지지 않는다 → 일치 ≥ 90.4%, 충실 ≥ 79.8%
- 참고로 대조 재실행(f6 --control, 충실 81.2%)과 V1(충실 75.9%)도 함께 적는다. 미달이면 문구를 다시 고치지 않고 예전 프롬프트(rag.SYSTEM)로 돌아간다
"""
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import f4_rag_answer_eval as f4  # noqa: E402
import rag  # noqa: E402
from common import DATA, manifest  # noqa: E402
from prompts import APP_SYSTEM, APP_SYSTEM_V2  # noqa: E402

EVAL = DATA / "eval"
V2 = "--v2" in sys.argv
PROMPT = APP_SYSTEM_V2 if V2 else APP_SYSTEM
A5, J5 = (EVAL / "_f5v2_answers.jsonl", EVAL / "_f5v2_judge.jsonl") if V2 else (EVAL / "_f5_answers.jsonl", EVAL / "_f5_judge.jsonl")
OUT = EVAL / ("f5_app_prompt_v2_eval.json" if V2 else "f5_app_prompt_eval.json")
SETS = f4.DEV_SETS + (f4.UNANS,)
RANK = {"일치": 3, "부분일치": 2, "불일치": 1, "답변없음": 0}
DATE_RX = re.compile(r"\d{4}[-.]\d{1,2}[-.]\d{1,2}|\d{4}년\s*\d{1,2}월\s*\d{1,2}일")
sys.stdout.reconfigure(encoding="utf-8")


def app_answer(rec, question):
    masked = rag.mask_pii(question)
    text, usage = f4.with_retry(lambda: rag.llm([{"role": "system", "content": PROMPT}, {"role": "user", "content": f"[질문]\n{masked}\n\n[사례]\n{rec['context']}"}]))
    text = text.strip()
    chk = rag.check_output(text, rec["ids"])
    if not text.endswith(rag.CLOSING):
        text = f"{text}\n\n{rag.CLOSING}"
    return {"set": rec["set"], "qid": rec["qid"], "answer": text, "check": chk, "usage": usage}


def body(a):
    return a["answer"].replace(rag.CLOSING, "").strip()


def sign_p(better, worse):
    n = better + worse
    return 1.0 if not n else min(1.0, 2 * sum(comb(n, i) for i in range(min(better, worse) + 1)) / 2 ** n)


def extra(recs, anss):
    a = [anss[(r["set"], r["qid"])] for r in recs]
    bodies = [body(x) for x in a]
    return {"date_in_answer": sum(1 for b in bodies if DATE_RX.search(b)), "avg_chars": round(sum(len(b) for b in bodies) / len(bodies)), "tokens_out": sum(x["usage"]["output"] for x in a),
            "bullets_style": sum(1 for b in bodies if re.search(r"^- ", b, re.M)), "bold": sum(1 for b in bodies if "**" in b)}


def main():
    recs = [r for r in f4.read_jsonl(f4.S1) if r["set"] in SETS]
    questions = {}
    for name in SETS:
        for l in open(EVAL / f"{name}.jsonl", encoding="utf-8"):
            x = json.loads(l)
            questions[(name, x["qid"])] = x["question"]
    base_a = {(r["set"], r["qid"]): r for r in f4.read_jsonl(f4.ANS)}
    base_j = {(r["set"], r["qid"]): r for r in f4.read_jsonl(f4.JDG)}
    new_a = {(r["set"], r["qid"]): r for r in f4.read_jsonl(A5)}
    t0 = time.time()
    todo = [r for r in recs if (r["set"], r["qid"]) not in new_a]
    print(f"[1 답변] 호출 대상 {len(todo)} / {len(recs)}", flush=True)
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for i, a in enumerate(ex.map(lambda r: app_answer(r, questions[(r["set"], r["qid"])]), todo), 1):
            f4.append(A5, a)
            new_a[(a["set"], a["qid"])] = a
            if i % 50 == 0:
                print(f"[1 답변] {i}/{len(todo)} {time.time() - t0:.0f}초", flush=True)
    with rag.connect() as conn:
        golds = {r["case_key"]: r for r in conn.execute("select case_key, title, question, answer, reason from interp.cases where case_key = any(%s)", ([r["gold_key"] for r in recs if r["gold_key"]],)).fetchall()}
    new_j = {(r["set"], r["qid"]): r for r in f4.read_jsonl(J5)}
    todo = [r for r in recs if r["gold_key"] and (r["set"], r["qid"]) not in new_j]
    t0 = time.time()
    print(f"[2 심사] 호출 대상 {len(todo)}", flush=True)
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for i, j in enumerate(ex.map(lambda r: f4.judge_one(r, new_a[(r["set"], r["qid"])], questions[(r["set"], r["qid"])], golds[r["gold_key"]]), todo), 1):
            f4.append(J5, j)
            new_j[(j["set"], j["qid"])] = j
            if i % 50 == 0:
                print(f"[2 심사] {i}/{len(todo)} {time.time() - t0:.0f}초", flush=True)

    # 비교
    out = {"summary": {}, "paired": {}}
    groups = {name: [r for r in recs if r["set"] == name] for name in SETS}
    groups["개발셋 합계(합성·dev1~4)"] = [r for r in recs if r["set"] in f4.DEV_SETS]
    bs_all, ns_all = f4.aggregate(recs, base_a, base_j), f4.aggregate(recs, new_a, new_j)
    for g in bs_all:
        out["summary"][g] = {"base": {**bs_all[g], **extra(groups[g], base_a)}, "new": {**ns_all[g], **extra(groups[g], new_a)}}
    dev = [r for r in recs if r["gold_key"] and r["set"] in f4.DEV_SETS]
    trans = Counter((base_j[(r["set"], r["qid"])]["correctness"], new_j[(r["set"], r["qid"])]["correctness"]) for r in dev)
    better = sum(1 for r in dev if RANK[new_j[(r["set"], r["qid"])]["correctness"]] > RANK[base_j[(r["set"], r["qid"])]["correctness"]])
    worse = sum(1 for r in dev if RANK[new_j[(r["set"], r["qid"])]["correctness"]] < RANK[base_j[(r["set"], r["qid"])]["correctness"]])
    out["paired"] = {"n": len(dev), "better": better, "worse": worse, "same": len(dev) - better - worse, "p_sign": round(sign_p(better, worse), 4),
                     "transitions": {f"{a}→{b}": n for (a, b), n in sorted(trans.items(), key=lambda x: -x[1])}}
    gb = sum(1 for r in dev if base_j[(r["set"], r["qid"])]["grounded"] != "충실" and new_j[(r["set"], r["qid"])]["grounded"] == "충실")
    gw = sum(1 for r in dev if base_j[(r["set"], r["qid"])]["grounded"] == "충실" and new_j[(r["set"], r["qid"])]["grounded"] != "충실")
    out["paired"].update(grounded_better=gb, grounded_worse=gw, p_sign_grounded=round(sign_p(gb, gw), 4))
    out["items_worse"] = [{"set": r["set"], "qid": r["qid"], "base": base_j[(r["set"], r["qid"])]["correctness"], "new": new_j[(r["set"], r["qid"])]["correctness"], "new_reason": new_j[(r["set"], r["qid"])]["reason"]}
                          for r in dev if RANK[new_j[(r["set"], r["qid"])]["correctness"]] < RANK[base_j[(r["set"], r["qid"])]["correctness"]]]
    out["tokens"] = {"answer_in": sum(a["usage"]["input"] for a in new_a.values()), "answer_out": sum(a["usage"]["output"] for a in new_a.values()),
                     "judge_in": sum(j["usage"]["input"] for j in new_j.values()), "judge_out": sum(j["usage"]["output"] for j in new_j.values())}
    json.dump({**out, "items": [{"set": r["set"], "qid": r["qid"], "answer_new": new_a[(r["set"], r["qid"])]["answer"], "judge_new": {k: new_j[(r["set"], r["qid"])][k] for k in ("correctness", "grounded", "reason")} if (r["set"], r["qid"]) in new_j else None} for r in recs]},
              open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    manifest("f5_app_prompt_v2_eval" if V2 else "f5_app_prompt_eval", n=len(recs))
    keys = ["n", "correct_rate", "correct_or_partial_rate", "grounded_rate", "gold_cited_when_in_top5", "not_found", "invalid_citations", "assertive", "answers_with_uncited_sentences", "date_in_answer", "avg_chars"]
    for g, v in out["summary"].items():
        print(f"\n[{g}]")
        for k in keys:
            if k in v["base"]:
                print(f"  {k:32s} 기존 {str(v['base'].get(k)):>12s} → 새 {str(v['new'].get(k)):>12s}")
    print("\n같은 문항 정확도 변화(개발셋, 정답 있는 문항):", {k: v for k, v in out["paired"].items() if k != "transitions"}, "\n전이:", out["paired"]["transitions"])
    print("토큰", out["tokens"])


if __name__ == "__main__":
    main()
