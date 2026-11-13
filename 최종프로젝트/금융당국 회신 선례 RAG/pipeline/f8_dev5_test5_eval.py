"""F8 새 세트(dev5 50, test5 30)에서 키워드 완화 적용 규칙 비교: 검색 + 답변 + LLM 심사.
규칙 (73·74번과 같은 정의)
- A 끔: 벡터 상위 5개 묶음 → `rag.SYSTEM` 답변 (f4와 같은 흐름)
- B 항상 켬: 상위 5 + 키워드 순차 완화로 찾은 사례 최대 3개(`app/keyword_relax.py`)
- C 못 찾을 때만: A 답변이 "찾지 못했습니다"(또는 인용 없음)면 B 답변을 쓴다. B의 입력이 C의 두 번째 호출 입력과 같으므로 B 답변을 그대로 쓴다(실제 두 번째 호출)
사전에 정한 보고 항목 (결과를 보기 전에 정함)
- 검색: Hit@1·5·10, MRR@10, 정답이 LLM 문맥 안(A·C 첫 호출 = Hit@5, B = 상위 5 + 추가분)
- 답변: 정확도(일치/부분일치/불일치/답변없음), 충실도(충실/일부추정/근거없음), "찾지 못했습니다" 수, 근거 번호 오류, 단정 표현, 추가 LLM 호출 수
- A 대비 B·C의 같은 문항 변화(부호 검정). 구분(법령해석/비조치)별·유형별 분포. 세트별로 따로 보고하고 합치지 않는다
- 이 결과로 검색·키워드 규칙·프롬프트를 고치지 않는다. test5는 한 번만 평가하고 `data/eval/f8_test5_lock.json`으로 잠근다. 결과 파일에 test5 질문 글은 넣지 않는다
사용: python f8_dev5_test5_eval.py [--dev-only]     캐시: _f8_stage.jsonl, _f8_answers.jsonl, _f8_judge.jsonl     결과: data/eval/f8_dev5_test5_eval.json
"""
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import f2_test_eval as f2  # noqa: E402
import f4_rag_answer_eval as f4  # noqa: E402
import keyword_relax as kr  # noqa: E402
import rag  # noqa: E402
from common import DATA, manifest  # noqa: E402

EVAL = DATA / "eval"
ST, ANS, JDG = EVAL / "_f8_stage.jsonl", EVAL / "_f8_answers.jsonl", EVAL / "_f8_judge.jsonl"
LOCK, OUT = EVAL / "f8_test5_lock.json", EVAL / "f8_dev5_test5_eval.json"
R = {"일치": 3, "부분일치": 2, "불일치": 1, "답변없음": 0}
sys.stdout.reconfigure(encoding="utf-8")


def sign_p(b, w):
    n = b + w
    return 1.0 if not n else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, w) + 1)) / 2 ** n)


def nf(a):
    return a["check"]["not_found"] or not a["check"]["cited"]


def stage(conn, name, rows, rep, grp_rep):
    done = {(r["set"], r["qid"]) for r in f4.read_jsonl(ST)}
    for r in rows:
        if (name, r["qid"]) in done:
            continue
        q = rag.mask_pii(r["question"])
        hits = rag.search(q, 10, conn)
        groups = [rep[h["doc_id"]] for h in hits]
        rank = groups.index(r["gold_group_id"]) + 1 if r["gold_group_id"] in groups else None
        ids = [h["doc_id"] for h in hits[:5]]
        concepts, usage = f4.with_retry(lambda: kr.extract_concepts(q))
        rr = kr.relax_search(conn, concepts, rag.embed(q), set(groups[:5]))
        cases = rag.load_cases(ids + [e["doc_id"] for e in rr["extra"]], conn)
        ctx_a = "\n\n".join(rag.case_block(cases[i]) for i in ids)
        ctx_b = ctx_a + "".join("\n\n" + kr.extra_block(cases[e["doc_id"]], e) for e in rr["extra"])
        f4.append(ST, {"set": name, "qid": r["qid"], "tag": r["tag"], "gubun": r["gubun"], "gold_group_id": r["gold_group_id"], "gold_key": grp_rep[r["gold_group_id"]], "rank": rank,
                       "ids_a": ids, "ids_b": ids + [e["doc_id"] for e in rr["extra"]], "ctx_a": ctx_a, "ctx_b": ctx_b, "concepts": concepts, "trail": rr["trail"],
                       "extra": rr["extra"], "gold_in_extra": next((j + 1 for j, e in enumerate(rr["extra"]) if e["group_id"] == r["gold_group_id"]), None), "kw_usage": usage})


def answer(rec, mode, question):
    ctx, ids = (rec["ctx_a"], rec["ids_a"]) if mode == "A" else (rec["ctx_b"], rec["ids_b"])
    masked = rag.mask_pii(question)
    text, usage = f4.with_retry(lambda: rag.llm([{"role": "system", "content": rag.SYSTEM}, {"role": "user", "content": f"[질문]\n{masked}\n\n[사례]\n{ctx}"}]))
    text = text.strip()
    chk = rag.check_output(text, ids)
    if not text.endswith(rag.CLOSING):
        text = f"{text}\n\n{rag.CLOSING}"
    return {"set": rec["set"], "qid": rec["qid"], "mode": mode, "answer": text, "check": chk, "usage": usage}


def metrics(recs, ans, jdg, rule):
    pick = lambda r: ("B" if rule == "B" else "A") if rule != "C" else ("B" if nf(ans[(r["set"], r["qid"], "A")]) else "A")
    n = len(recs)
    a = [ans[(r["set"], r["qid"], pick(r))] for r in recs]
    j = [jdg[(r["set"], r["qid"], pick(r))] for r in recs]
    c = Counter(x["correctness"] for x in j)
    g = Counter(x["grounded"] for x in j if x["correctness"] != "답변없음")
    hit = lambda k: sum(1 for r in recs if r["rank"] and r["rank"] <= k)
    ctx_in = sum(1 for r in recs if (r["rank"] and r["rank"] <= 5) or (pick(r) == "B" and r["gold_in_extra"]))
    extra_calls = 0 if rule == "A" else (n if rule == "B" else 2 * sum(1 for r in recs if pick(r) == "B"))
    return {"n": n, "hit1": round(hit(1) / n, 3), "hit5": round(hit(5) / n, 3), "hit10": round(hit(10) / n, 3), "mrr10": round(sum(1 / r["rank"] for r in recs if r["rank"]) / n, 3),
            "정답이 문맥 안": ctx_in, "정확도 일치": round(c["일치"] / n, 3), "일치+부분": round((c["일치"] + c["부분일치"]) / n, 3), "정확도": {k: c[k] for k in R},
            "충실도 충실": round(g["충실"] / max(sum(g.values()), 1), 3), "충실도": {k: g[k] for k in ("충실", "일부추정", "근거없음")},
            "찾지 못했습니다": sum(1 for x in a if x["check"]["not_found"]), "근거 번호 오류": sum(len(x["check"]["invalid"]) for x in a), "단정 표현": sum(x["check"]["assertive"] for x in a),
            "넓혀 찾은 사례 인용": sum(1 for r, x in zip(recs, a) if set(x["check"]["cited"]) & {e["doc_id"] for e in r["extra"]} and pick(r) == "B"),
            "추가 LLM 호출": extra_calls, "재검색한 문항": sum(1 for r in recs if pick(r) == "B") if rule == "C" else None}


def main():
    dev_only = "--dev-only" in sys.argv
    sets = ["dev5"] + ([] if dev_only else ["test5"])
    data = {name: f2.load(name) for name in sets}
    with rag.connect() as conn:
        fp = f2.fingerprint(conn)
        prev = {v["fingerprint"] for v in json.load(open(f2.LOCK, encoding="utf-8")).values()}
        if fp["id"] not in prev:
            sys.exit(f"설정 지문 {fp['id']}가 이전 평가({sorted(prev)})와 다릅니다.")
        lock = json.load(open(LOCK, encoding="utf-8")) if LOCK.exists() else {}
        if not dev_only and lock.get("completed_at"):
            sys.exit(f"test5는 이미 평가했습니다 ({lock['completed_at']}). --dev-only로 dev5만 다시 돌릴 수 있습니다.")
        if not dev_only and not lock:
            LOCK.write_text(json.dumps({"started_at": datetime.now().isoformat(timespec="seconds"), "fingerprint": fp["id"]}, ensure_ascii=False), encoding="utf-8")
        rep = {r["representative_case_key"]: r["group_id"] for r in conn.execute("select group_id, representative_case_key from interp.case_groups").fetchall()}
        grp_rep = {v: k for k, v in rep.items()}
        rag.embed("준비")
        t0 = time.time()
        for name in sets:
            stage(conn, name, data[name], rep, grp_rep)
        print(f"[1 검색·키워드] {time.time() - t0:.0f}초", flush=True)
        recs = [r for r in f4.read_jsonl(ST) if r["set"] in sets]
        golds = {r["case_key"]: r for r in conn.execute("select case_key, title, question, answer, reason from interp.cases where case_key = any(%s)", ([r["gold_key"] for r in recs],)).fetchall()}
    questions = {(name, r["qid"]): r["question"] for name in sets for r in data[name]}

    ans = {(r["set"], r["qid"], r["mode"]): r for r in f4.read_jsonl(ANS)}
    todo = [(r, m) for r in recs for m in ("A", "B") if (r["set"], r["qid"], m) not in ans]
    print(f"[2 답변] 호출 대상 {len(todo)}", flush=True)
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for x in ex.map(lambda t: answer(t[0], t[1], questions[(t[0]["set"], t[0]["qid"])]), todo):
            f4.append(ANS, x)
            ans[(x["set"], x["qid"], x["mode"])] = x

    jdg = {(r["set"], r["qid"], r["mode"]): r for r in f4.read_jsonl(JDG)}
    todo = [(r, m) for r in recs for m in ("A", "B") if (r["set"], r["qid"], m) not in jdg]
    print(f"[3 심사] 호출 대상 {len(todo)}", flush=True)

    def judge(t):
        r, m = t
        rec = {"set": r["set"], "qid": r["qid"], "gold_key": r["gold_key"], "context": r["ctx_a"] if m == "A" else r["ctx_b"]}
        out = f4.judge_one(rec, ans[(r["set"], r["qid"], m)], questions[(r["set"], r["qid"])], golds[r["gold_key"]])
        return {**out, "mode": m}
    with ThreadPoolExecutor(f4.THREADS) as ex:
        for x in ex.map(judge, todo):
            f4.append(JDG, x)
            jdg[(x["set"], x["qid"], x["mode"])] = x

    out = {"ran_at": datetime.now().isoformat(timespec="seconds"), "fingerprint": fp["id"], "sets": {}}
    for name in sets:
        rs = [r for r in recs if r["set"] == name]
        s = {"rules": {rule: metrics(rs, ans, jdg, rule) for rule in ("A", "B", "C")}, "paired_vs_A": {}, "by_gubun": {}, "by_tag": {}}
        for rule in ("B", "C"):
            pick = (lambda r: "B") if rule == "B" else (lambda r: "B" if nf(ans[(r["set"], r["qid"], "A")]) else "A")
            ja = [jdg[(r["set"], r["qid"], "A")] for r in rs]
            jr = [jdg[(r["set"], r["qid"], pick(r))] for r in rs]
            b = sum(1 for x, y in zip(ja, jr) if R[y["correctness"]] > R[x["correctness"]])
            w = sum(1 for x, y in zip(ja, jr) if R[y["correctness"]] < R[x["correctness"]])
            gb = sum(1 for x, y in zip(ja, jr) if x["grounded"] != "충실" and y["grounded"] == "충실")
            gw = sum(1 for x, y in zip(ja, jr) if x["grounded"] == "충실" and y["grounded"] != "충실")
            s["paired_vs_A"][rule] = {"정확도 좋아짐/나빠짐": [b, w], "p": round(sign_p(b, w), 4), "충실도 좋아짐/나빠짐": [gb, gw], "p_충실": round(sign_p(gb, gw), 4)}
        for key, field in (("by_gubun", "gubun"), ("by_tag", "tag")):
            for v in sorted({r[field] for r in rs}):
                sub = [r for r in rs if r[field] == v]
                s[key][v] = {rule: {k: metrics(sub, ans, jdg, rule)[k] for k in ("n", "hit1", "hit5", "정확도 일치", "충실도 충실", "찾지 못했습니다")} for rule in ("A", "B", "C")}
        s["kw"] = {"no_concepts": sum(1 for r in rs if not r["concepts"]), "avg_extra": round(sum(len(r["extra"]) for r in rs) / len(rs), 2),
                   "gold_in_extra": sum(1 for r in rs if r["gold_in_extra"]), "gold_in_extra_when_missed": sum(1 for r in rs if r["gold_in_extra"] and not (r["rank"] and r["rank"] <= 5)),
                   "missed": sum(1 for r in rs if not (r["rank"] and r["rank"] <= 5))}
        if name == "dev5":
            s["missed_items"] = [{"qid": r["qid"], "tag": r["tag"], "gubun": r["gubun"], "rank": r["rank"], "gold": r["gold_key"], "concepts": r["concepts"], "gold_in_extra": r["gold_in_extra"],
                                  "A": jdg[(r["set"], r["qid"], "A")]["correctness"], "B": jdg[(r["set"], r["qid"], "B")]["correctness"]} for r in rs if not (r["rank"] and r["rank"] <= 5)]
        out["sets"][name] = s
    out["tokens"] = {"keyword_in": sum(r["kw_usage"]["input"] for r in recs), "answer_in": sum(a["usage"]["input"] for a in ans.values() if a["set"] in sets),
                     "judge_in": sum(j["usage"]["input"] for j in jdg.values() if j["set"] in sets)}
    out["items"] = [{"set": r["set"], "qid": r["qid"], "tag": r["tag"], "gubun": r["gubun"], "rank": r["rank"], "gold_key": r["gold_key"], "gold_in_extra": r["gold_in_extra"],
                     "concepts": r["concepts"] if r["set"] == "dev5" else None,
                     **{f"judge_{m}": {k: jdg[(r["set"], r["qid"], m)][k] for k in ("correctness", "grounded", "reason")} for m in ("A", "B")},
                     **({f"answer_{m}": ans[(r["set"], r["qid"], m)]["answer"] for m in ("A", "B")} if r["set"] == "dev5" else {})} for r in recs]
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if not dev_only:
        LOCK.write_text(json.dumps({**json.load(open(LOCK, encoding="utf-8")), "completed_at": datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False), encoding="utf-8")
    manifest("f8_dev5_test5_eval", sets=sets, fingerprint=fp["id"])
    print(json.dumps({k: v for k, v in out.items() if k != "items"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
