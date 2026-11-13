"""F4 LLM 포함 RAG 전체 평가(답변 수준): 합성 150, dev1~4, test1~3, final_test, 답변불가 20을 `rag.answer`와 같은 흐름(KURE-v1 검색 → 상위 5개 묶음 → gpt-6-luna 답변)으로 돌리고, 답변을 LLM 심사로 채점한다.
test4는 이미 f3_test4_rag.py로 한 번 실행했으므로 저장된 답변을 재사용한다(다시 호출하지 않음).

사전 등록한 지표 (결과를 보기 전에 정함)
- 검색: 사례 묶음 단위 Hit@1·3·5·10, MRR@10 (이전 평가와 같은 값이어야 한다)
- 답변 형식·안전: 근거 번호 오류, 단정 표현 수, 근거 없는 문장이 있는 답변 수, "찾지 못했습니다" 비율 (답변불가 질문은 이것이 정답)
- 정답 사례 인용률: 정답 묶음이 상위 5에 있던 질문 중 답변이 정답 사례를 인용한 비율
- 답변 정확도(LLM 심사, `gpt-6-luna` 추론 medium): 정답 사례의 회답과 비교해 일치 / 부분일치 / 불일치 / 답변없음. 정답은 사례 하나라 같은 주제의 다른 회신으로 정당하게 답해도 결론이 다르면 불일치가 될 수 있다
- 답변 충실도(LLM 심사): 답변의 주장이 인용한 사례에 있는지 충실 / 일부추정 / 근거없음
- 세트마다 따로, 개발셋 합계와 시험 세트 합계를 따로 보고한다. 심사자가 생성 모델과 같은 계열이라 점수가 후할 수 있어, 표본 직접 읽기로 신뢰도를 따로 점검한다
- 이 결과로 검색·프롬프트·코드를 바꾸지 않는다

안전장치: 설정 지문이 f2 평가(test1~3·final_test)와 같아야 한다. 시험 세트(test1~3, final_test)는 한 번만 평가하고 `data/eval/f4_test_lock.json`으로 잠근다. 결과 파일에 시험 세트 질문 글은 넣지 않는다.
단계별 캐시(`_f4_stage1.jsonl` 검색·문맥, `_f4_answers.jsonl` 답변, `_f4_judge.jsonl` 심사)로 끊겨도 이어서 한다.
사용: python f4_rag_answer_eval.py [--dev-only]
"""
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from statistics import median

import f2_test_eval as f2
import rag
from common import DATA, manifest

EVAL = DATA / "eval"
DEV_SETS = ("rewrite_150", "dev1", "dev2", "dev3", "dev4")
TEST_SETS = ("test1", "test2", "test3", "final_test")
UNANS = "unanswerable_20"
LOCK = EVAL / "f4_test_lock.json"
S1, ANS, JDG = EVAL / "_f4_stage1.jsonl", EVAL / "_f4_answers.jsonl", EVAL / "_f4_judge2.jsonl"      # judge2: 고정 안내 문구를 뺀 심사 (1차 심사 _f4_judge.jsonl은 기록으로만 남김)
OUT_FILE = EVAL / "f4_answer_eval.json"
THREADS = 6
sys.stdout.reconfigure(encoding="utf-8")

JUDGE_SYSTEM = """당신은 금융당국 회신 검색 시스템의 답변을 채점하는 심사자입니다. 아래 정보를 보고 JSON으로만 답하세요.
정답 사례는 이 질문의 정답으로 정해 둔 회신 하나입니다. 같은 주제의 다른 회신이 있을 수 있으므로 정답 사례와 답변을 비교하되, 두 항목을 따로 판정하세요.
1) correctness: 답변의 핵심 결론이 정답 사례의 회답과 같은 방향인가.
   - "일치": 질문이 묻는 쟁점에 대해 정답 사례와 같은 결론이다 (표현이 달라도 됨)
   - "부분일치": 일부 쟁점만 맞거나 중요한 단서·조건을 빠뜨렸다
   - "불일치": 정답 사례와 반대이거나 다른 쟁점에 답했다
   - "답변없음": 관련 회신을 찾지 못했다고만 답했다
2) grounded: 답변의 주장이 '답변이 인용한 사례' 안에 있는가 (정답 사례와 무관하게 인용 사례 기준)
   - "충실": 모든 주장이 인용한 사례에 있다
   - "일부추정": 일부 문장이 인용한 사례에 없거나 사례의 범위를 넘어선 추론이다
   - "근거없음": 대부분 인용한 사례에 없다
   - 답변이 "찾지 못했다"뿐이면 "해당없음"
답변 끝의 "회신은 개별 사안에 대한 당국 의견이며, 최종 판단은 준법감시부서 확인이 필요합니다." 문구는 시스템이 모든 답변에 고정으로 붙이는 안내이므로 판정에서 제외합니다(이미 제거해서 전달합니다).
출력: {"correctness": "...", "grounded": "...", "reason": "한 문장"}"""


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []


def append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def with_retry(fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:                       # 일시적 API 오류는 간격을 두고 다시 시도 (키·주소가 담길 수 있어 메시지는 출력하지 않는다)
            if i == tries - 1:
                raise RuntimeError(type(e).__name__)
            time.sleep(3 * (i + 1))


def stage1(conn, sets, rep, grp_rep, limit=None):
    done = {(r["set"], r["qid"]) for r in read_jsonl(S1)}
    t0, n = time.time(), 0
    for name in sets:
        rows = f2.load(name) if name != UNANS else [json.loads(l) for l in open(EVAL / f"{UNANS}.jsonl", encoding="utf-8")]
        for r in (rows[:limit] if limit else rows):
            if (name, r["qid"]) in done:
                continue
            hits = rag.search(r["question"], 10, conn)
            groups = [rep[h["doc_id"]] for h in hits]
            gold = r.get("gold_group_id")
            rank = groups.index(gold) + 1 if gold in groups else None
            ids = [h["doc_id"] for h in hits[:5]]
            cases = rag.load_cases(ids, conn)
            append(S1, {"set": name, "qid": r["qid"], "tag": r.get("tag"), "gubun": r.get("gubun"), "gold_group_id": gold, "gold_key": grp_rep.get(gold) if gold else None,
                        "rank": rank, "ids": ids, "scores": [round(h["score"], 3) for h in hits[:5]], "context": "\n\n".join(rag.case_block(cases[i]) for i in ids),
                        "marks": {i: cases[i]["time_mark"] for i in ids}})
            n += 1
            if n % 50 == 0:
                print(f"[1 검색] {n}문항 {time.time() - t0:.0f}초", flush=True)


def answer_one(rec, question):
    masked = rag.mask_pii(question)
    text, usage = with_retry(lambda: rag.llm([{"role": "system", "content": rag.SYSTEM}, {"role": "user", "content": f"[질문]\n{masked}\n\n[사례]\n{rec['context']}"}]))
    text = text.strip()
    chk = rag.check_output(text, rec["ids"])
    if not text.endswith(rag.CLOSING):
        text = f"{text}\n\n{rag.CLOSING}"
    return {"set": rec["set"], "qid": rec["qid"], "answer": text, "check": chk, "usage": usage}


def cited_context(context, cited, limit=1200, max_cases=3):
    blocks = {m.group(1): m.group(0) for m in re.finditer(r"\[사례 ([LO]\d+)\].*?(?=\n\n\[사례 |\Z)", context, re.S)}
    return "\n\n".join(blocks[c][:limit] for c in cited[:max_cases] if c in blocks)


def judge_one(rec, ans, question, gold_case):
    if ans["check"]["not_found"] or not ans["check"]["cited"]:
        return {"set": rec["set"], "qid": rec["qid"], "correctness": "답변없음", "grounded": "해당없음", "reason": "관련 회신을 찾지 못했다고 답함", "usage": {"input": 0, "output": 0}}
    ref = f"제목: {gold_case['title']}\n질의요지: {(gold_case['question'] or '')[:600]}\n회답: {(gold_case['answer'] or '')[:800]}\n이유: {(gold_case['reason'] or '')[:600]}"
    body = ans["answer"].replace(rag.CLOSING, "").strip()          # 시스템이 붙이는 고정 안내 문구는 심사에서 뺀다
    user = f"[질문]\n{question}\n\n[시스템 답변]\n{body}\n\n[정답 사례({rec['gold_key']})]\n{ref}\n\n[답변이 인용한 사례]\n{cited_context(rec['context'], ans['check']['cited'])}"

    def call():
        r = rag.client.chat.completions.create(model=rag.CHAT_MODEL, reasoning_effort="medium", response_format={"type": "json_object"},
                                               messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}])
        return json.loads(r.choices[0].message.content), r.usage
    j, u = with_retry(call)
    return {"set": rec["set"], "qid": rec["qid"], "correctness": j.get("correctness"), "grounded": j.get("grounded"), "reason": j.get("reason"),
            "usage": {"input": u.prompt_tokens, "output": u.completion_tokens}}


def aggregate(recs, anss, jdgs):
    out = {}
    groups = {name: [r for r in recs if r["set"] == name] for name in sorted({r["set"] for r in recs})}
    groups["개발셋 합계(합성·dev1~4)"] = [r for r in recs if r["set"] in DEV_SETS]
    groups["시험 세트 합계(test1~4·final_test)"] = [r for r in recs if r["set"] in TEST_SETS + ("test4",)]
    for g, rs in groups.items():
        if not rs:
            continue
        n = len(rs)
        a = [anss[(r["set"], r["qid"])] for r in rs]
        js = [jdgs.get((r["set"], r["qid"])) for r in rs]
        has_gold = [r for r in rs if r["gold_key"]]
        hit = lambda k: sum(1 for r in has_gold if r["rank"] and r["rank"] <= k)
        m = {"n": n}
        if has_gold:
            m.update(hit1=round(hit(1) / len(has_gold), 4), hit3=round(hit(3) / len(has_gold), 4), hit5=round(hit(5) / len(has_gold), 4), hit10=round(hit(10) / len(has_gold), 4),
                     mrr10=round(sum(1 / r["rank"] for r in has_gold if r["rank"]) / len(has_gold), 4))
            top5 = [r for r in has_gold if r["rank"] and r["rank"] <= 5]
            m["gold_cited_when_in_top5"] = f"{sum(1 for r in top5 if r['gold_key'] in anss[(r['set'], r['qid'])]['check']['cited'])}/{len(top5)}"
        m.update(not_found=sum(x["check"]["not_found"] for x in a), invalid_citations=sum(len(x["check"]["invalid"]) for x in a), assertive=sum(x["check"]["assertive"] for x in a),
                 answers_with_uncited_sentences=sum(1 for x in a if x["check"]["no_citation_sentences"] > 0))
        if has_gold:
            jj = [j for r, j in zip(rs, js) if r["gold_key"] and j]
            c = Counter(j["correctness"] for j in jj)
            m["correctness"] = {k: c[k] for k in ("일치", "부분일치", "불일치", "답변없음")}
            m["correct_rate"] = round(c["일치"] / len(jj), 4) if jj else None
            m["correct_or_partial_rate"] = round((c["일치"] + c["부분일치"]) / len(jj), 4) if jj else None
            gr = Counter(j["grounded"] for j in jj if j["correctness"] != "답변없음")
            m["grounded"] = {k: gr[k] for k in ("충실", "일부추정", "근거없음")}
            m["grounded_rate"] = round(gr["충실"] / max(sum(gr.values()), 1), 4)
            # 정답이 상위 5 안에 있는지에 따라
            for label, cond in (("정답이 상위 5 안", lambda r: r["rank"] and r["rank"] <= 5), ("정답이 상위 5 밖", lambda r: not (r["rank"] and r["rank"] <= 5))):
                sub = [j for r, j in zip(rs, js) if r["gold_key"] and j and cond(r)]
                if sub:
                    cc = Counter(j["correctness"] for j in sub)
                    m[f"correctness_{label}"] = {"n": len(sub), **{k: cc[k] for k in ("일치", "부분일치", "불일치", "답변없음")}}
        out[g] = m
    return out


def main():
    dev_only = "--dev-only" in sys.argv
    rejudge = "--rejudge" in sys.argv          # 저장된 답변을 다시 생성하지 않고 심사만 다시 한다 (심사 방법을 고친 뒤)
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None      # 시험 삼아 세트마다 앞 N문항만 (개발셋 전용)
    if limit and not dev_only:
        sys.exit("--limit은 --dev-only와 함께만 씁니다 (시험 세트는 일부만 돌리지 않는다).")
    sets = list(DEV_SETS) + [UNANS] + ([] if dev_only else list(TEST_SETS))
    with rag.connect() as conn:
        fp = f2.fingerprint(conn)
        prev = {v["fingerprint"] for v in json.load(open(f2.LOCK, encoding="utf-8")).values()}
        if fp["id"] not in prev:
            sys.exit(f"설정 지문 {fp['id']}가 f2 평가({sorted(prev)})와 다릅니다. 설정이 바뀌었습니다.")
        lock = json.load(open(LOCK, encoding="utf-8")) if LOCK.exists() else {}
        if not dev_only and not rejudge and lock.get("completed_at"):
            sys.exit(f"시험 세트는 이미 이 평가에 쓰였습니다 ({lock['completed_at']}). --dev-only로 개발셋만 다시 돌릴 수 있습니다.")
        if not dev_only and lock and lock.get("fingerprint") != fp["id"]:
            sys.exit("중단된 이전 실행과 설정 지문이 다릅니다. 시험 세트는 설정이 바뀐 뒤 이어서 평가하지 않습니다.")
        if not dev_only and not lock:
            LOCK.write_text(json.dumps({"started_at": datetime.now().isoformat(timespec="seconds"), "fingerprint": fp["id"]}, ensure_ascii=False), encoding="utf-8")   # 시작하면 바로 잠근다 (끊기면 같은 설정에서만 이어서)
        rep = {r["representative_case_key"]: r["group_id"] for r in conn.execute("select group_id, representative_case_key from interp.case_groups").fetchall()}
        grp_rep = {v: k for k, v in rep.items()}
        rag.embed("준비")
        print(f"설정 지문 {fp['id']} (f2와 같음) | 평가 세트 {sets}", flush=True)
        stage1(conn, sets, rep, grp_rep, limit)

        recs = read_jsonl(S1)
        if limit:
            want = {(name, r["qid"]) for name in sets for r in (f2.load(name) if name != UNANS else [json.loads(l) for l in open(EVAL / f"{UNANS}.jsonl", encoding="utf-8")])[:limit]}
            recs = [r for r in recs if (r["set"], r["qid"]) in want]
        # test4: f3에서 저장한 답변을 재사용, 검색 맥락은 다시 만든다(검색은 결정적)
        t4 = json.load(open(EVAL / "f3_test4_result.json", encoding="utf-8")) if (EVAL / "f3_test4_result.json").exists() and not dev_only else None
        questions = {}
        for name in sets + (["test4"] if t4 else []):
            path = EVAL / f"{name}.jsonl"
            for l in open(path, encoding="utf-8"):
                x = json.loads(l)
                questions[(name, x["qid"])] = x["question"]
        anss = {(r["set"], r["qid"]): r for r in read_jsonl(ANS)}
        if t4:
            for it in t4["items"]:
                ids = [h[0] for h in it["top5"]]
                cases = rag.load_cases(ids, conn)
                rec = {"set": "test4", "qid": it["qid"], "tag": it["tag"], "gubun": it["gubun"], "gold_key": it["gold_case"], "rank": it["rank"], "ids": ids,
                       "gold_group_id": rep.get(it["gold_case"]) and rep[it["gold_case"]], "context": "\n\n".join(rag.case_block(cases[i]) for i in ids), "marks": {}}
                if not any(r["set"] == "test4" and r["qid"] == it["qid"] for r in recs):
                    recs.append(rec)
                anss[("test4", it["qid"])] = {"set": "test4", "qid": it["qid"], "answer": it["answer"], "check": it["check"], "usage": it["usage"]}

        # 2 답변
        todo = [r for r in recs if (r["set"], r["qid"]) not in anss]
        if rejudge and todo:
            sys.exit(f"--rejudge는 저장된 답변만 심사합니다. 답변이 없는 문항이 {len(todo)}개 있습니다.")
        t0 = time.time()
        print(f"[2 답변] 호출 대상 {len(todo)}", flush=True)
        with ThreadPoolExecutor(THREADS) as ex:
            for i, a in enumerate(ex.map(lambda r: answer_one(r, questions[(r["set"], r["qid"])]), todo), 1):
                append(ANS, a)
                anss[(a["set"], a["qid"])] = a
                if i % 50 == 0:
                    print(f"[2 답변] {i}/{len(todo)} {time.time() - t0:.0f}초", flush=True)

        # 3 심사 (정답 사례가 있는 질문만)
        golds = {r["case_key"]: r for r in conn.execute("select case_key, title, question, answer, reason from interp.cases where case_key = any(%s)", ([r["gold_key"] for r in recs if r["gold_key"]],)).fetchall()}
        jdgs = {(r["set"], r["qid"]): r for r in read_jsonl(JDG)}
        todo = [r for r in recs if r["gold_key"] and (r["set"], r["qid"]) not in jdgs]
        t0 = time.time()
        print(f"[3 심사] 호출 대상 {len(todo)}", flush=True)
        with ThreadPoolExecutor(THREADS) as ex:
            for i, j in enumerate(ex.map(lambda r: judge_one(r, anss[(r["set"], r["qid"])], questions[(r["set"], r["qid"])], golds[r["gold_key"]]), todo), 1):
                append(JDG, j)
                jdgs[(j["set"], j["qid"])] = j
                if i % 50 == 0:
                    print(f"[3 심사] {i}/{len(todo)} {time.time() - t0:.0f}초", flush=True)

    result = {"ran_at": datetime.now().isoformat(timespec="seconds"), "fingerprint": fp["id"], "dev_only": dev_only, "summary": aggregate(recs, anss, jdgs),
              "tokens": {"answer_in": sum(a["usage"]["input"] for a in anss.values()), "answer_out": sum(a["usage"]["output"] for a in anss.values()),
                         "judge_in": sum(j["usage"]["input"] for j in jdgs.values()), "judge_out": sum(j["usage"]["output"] for j in jdgs.values())},
              "items": [{"set": r["set"], "qid": r["qid"], "tag": r["tag"], "gubun": r["gubun"], "rank": r["rank"], "gold_key": r["gold_key"], "ids": r["ids"],
                         "answer": anss[(r["set"], r["qid"])]["answer"], "check": anss[(r["set"], r["qid"])]["check"],
                         "judge": ({k: jdgs[(r["set"], r["qid"])][k] for k in ("correctness", "grounded", "reason")} if (r["set"], r["qid"]) in jdgs else None)} for r in recs]}
    json.dump(result, open(OUT_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not dev_only and not rejudge:
        LOCK.write_text(json.dumps({**json.load(open(LOCK, encoding="utf-8")), "completed_at": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False), encoding="utf-8")
    manifest("f4_rag_answer_eval", n=len(recs), dev_only=dev_only, fingerprint=fp["id"])
    print(json.dumps({"fingerprint": fp["id"], "tokens": result["tokens"], "summary": result["summary"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
