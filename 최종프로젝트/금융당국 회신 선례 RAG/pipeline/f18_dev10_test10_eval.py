"""현재 service.ask 기본 설정의 dev10/test10 실제 전체 경로 평가.
사전 고정: 전체/회신/현장건의별 정확도, 충실도(응답분모/전체분모), 근거포함,
인용오류, 무응답, 재검색, 실제 직렬 요청 p50/p95. 별도 LLM 전문 심사.
골드 없는 질문이 없으므로 거절의 안전성을 입증하는 시험은 아니다.
dev를 먼저 평가하되 시스템 조정 없이 test 1회. 캐시와 전체 설정 지문으로 재개.
test 문항/답변은 캐시에만 보관하고 콘솔/요약에 출력하지 않는다.
사용: python f18_dev10_test10_eval.py [--dev-only]
"""
import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import service
import rag
import f4_rag_answer_eval as f4
from common import DATA

EVAL = DATA / "eval"
ANS = EVAL / "_f18_answers.jsonl"
JDG = EVAL / "_f18_judge_full.jsonl"
LOCK = EVAL / "f18_test10_lock.json"
RUN = EVAL / "f18_run.json"
OUT = EVAL / "f18_dev10_test10_eval.json"
CITE = re.compile(r"\[([LOF]\d+)\]")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def append(path, row):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def fingerprint():
    paths = list((ROOT / "app").glob("*.py")) + [ROOT / "pipeline" / n for n in
             ("rag.py", "common.py", "time_marks.py", "f4_rag_answer_eval.py", Path(__file__).name)]
    data = {"files": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "sets": {n: hashlib.sha256((EVAL / f"{n}.jsonl").read_bytes()).hexdigest() for n in ("dev10", "test10")},
            "model": rag.CHAT_MODEL, "embedding": rag.QUERY_MODEL, "reasoning": rag.REASONING,
            "settings": {"relax": True, "use_rerank": True, "use_exmnt": True, "use_verify": False}}
    with rag.connect() as conn:
        data["db"] = {}
        for table in ("cases", "chunks", "case_groups", "exmnt_cases", "exmnt_chunks", "exmnt_groups", "case_article_status"):
            result = conn.execute(f"select count(*) n, md5(string_agg(h, '' order by h)) hash from (select md5(row_to_json(t)::text) h from interp.{table} t) s").fetchone()
            data["db"][table] = dict(result)
    return {"id": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(), "parts": data}


def run_question(row, name, groups):
    captures = []
    original = service._answer

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captures.append({"context": args[1], "ids": args[2]})
        return result

    start = time.perf_counter()
    try:
        with patch.object(service, "_answer", side_effect=capture):
            res = service.ask(row["question"])
        wall = time.perf_counter() - start
        displayed = rag.NOT_FOUND if res["check"]["not_found"] else res["raw"]
        ids = res["order"]
        cited = list(dict.fromkeys(CITE.findall(displayed)))
        ctx = captures[-1]["context"] if captures else ""
        blocks = {m.group(1): m.group(0) for m in re.finditer(r"\[사례 ([LOF]\d+)\].*?(?=\n\n\[사례 |\Z)", ctx, re.S)}
        return {"set": name, "qid": row["qid"], "corpus": row["corpus"], "seconds": wall, "result": res,
                "displayed_answer": displayed, "context": ctx,
                "cited_context": "\n\n".join(blocks[c] for c in cited if c in blocks),
                "invalid": sorted(set(cited) - set(ids)), "cited": cited,
                "gold_in_context": row["gold_group_id"] in {groups.get(k) for k in ids},
                "gold_cited": row["gold_group_id"] in {groups.get(k) for k in cited}}
    except Exception as error:
        return {"set": name, "qid": row["qid"], "corpus": row["corpus"], "seconds": time.perf_counter()-start,
                "error": type(error).__name__}


def judge(item):
    row, ans = item
    base = {"set": ans["set"], "qid": ans["qid"]}
    if ans.get("error"):
        return {**base, "correctness": "시스템오류", "grounded": "해당없음"}
    if ans["displayed_answer"] == rag.NOT_FOUND:
        return {**base, "correctness": "답변없음", "grounded": "해당없음"}
    ref = row["gold_reference"]
    prompt = f"[질문]\n{row['question']}\n\n[시스템 답변]\n{ans['displayed_answer']}\n\n[정답 사례]\n" + json.dumps(ref, ensure_ascii=False) + f"\n\n[답변이 인용한 사례 전문]\n{ans['cited_context']}"
    for attempt in range(3):
        try:
            response = rag.client.chat.completions.create(model=rag.CHAT_MODEL, reasoning_effort="medium", response_format={"type": "json_object"},
                       messages=[{"role": "system", "content": f4.JUDGE_SYSTEM}, {"role": "user", "content": prompt}])
            j = json.loads(response.choices[0].message.content)
            assert j["correctness"] in ("일치", "부분일치", "불일치", "답변없음")
            assert j["grounded"] in ("충실", "일부추정", "근거없음", "해당없음")
            return {**base, **j}
        except Exception:
            if attempt == 2:
                raise RuntimeError("judge failed; resume from caches") from None
            time.sleep(2 ** attempt)


def pct(values, p):
    values = sorted(values)
    idx = (len(values)-1)*p
    lo = int(idx)
    return round(values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(idx-lo), 3)


def summarize(rows, answers, judges):
    n = len(rows)
    a = [answers[r["qid"]] for r in rows]
    j = [judges[r["qid"]] for r in rows]
    correct = Counter(x["correctness"] for x in j)
    grounded = Counter(x["grounded"] for x in j)
    answered = sum(x["correctness"] not in ("답변없음", "시스템오류") for x in j)
    return {"n": n, "correctness": dict(correct), "accuracy": correct["일치"]/n,
            "grounded": dict(grounded), "faithfulness_answered": grounded["충실"]/answered if answered else None,
            "faithfulness_all": grounded["충실"]/n, "gold_in_context": sum(x.get("gold_in_context", False) for x in a)/n,
            "gold_cited": sum(x.get("gold_cited", False) for x in a)/n,
            "invalid_citations": sum(len(x.get("invalid", [])) for x in a),
            "keyword_retries": sum(bool((x.get("result", {}).get("relax") or {}).get("retried")) for x in a),
            "errors": sum(bool(x.get("error")) for x in a),
            "seconds_p50": pct([x["seconds"] for x in a], .5), "seconds_p95": pct([x["seconds"] for x in a], .95)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-only", action="store_true")
    args = parser.parse_args()
    if not args.dev_only and LOCK.exists() and json.loads(LOCK.read_text(encoding="utf-8")).get("completed_at"):
        raise SystemExit("test10 already evaluated; locked")
    fp = fingerprint()
    if RUN.exists():
        assert json.loads(RUN.read_text(encoding="utf-8"))["fingerprint"]["id"] == fp["id"], "configuration changed"
    else:
        save(RUN, {"started_at": now(), "fingerprint": fp})
    with rag.connect() as conn:
        groups = {r["case_key"]: r["group_id"] for r in conn.execute("select case_key, group_id from interp.cases union all select case_key, group_id from interp.exmnt_cases").fetchall()}
    started = time.perf_counter()
    rag.embed("준비")
    warmup = time.perf_counter()-started
    result = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"sets": {}}
    for name in ["dev10"] + ([] if args.dev_only else ["test10"]):
        if name == "test10" and not LOCK.exists():
            save(LOCK, {"started_at": now(), "fingerprint": fp["id"]})
        rows = read(EVAL / f"{name}.jsonl")
        assert len(rows) == 50
        answers = {x["qid"]: x for x in read(ANS) if x["set"] == name}
        for row in rows:
            if row["qid"] in answers:
                continue
            ans = run_question(row, name, groups)
            append(ANS, ans)
            answers[row["qid"]] = ans
            print(f"{name} answers {len(answers)}/50", flush=True)
        judges = {x["qid"]: x for x in read(JDG) if x["set"] == name}
        pending = [(r, answers[r["qid"]]) for r in rows if r["qid"] not in judges]
        with ThreadPoolExecutor(4) as executor:
            for j in executor.map(judge, pending):
                append(JDG, j)
                judges[j["qid"]] = j
                print(f"{name} judged {len(judges)}/50", flush=True)
        result["sets"][name] = {label: summarize(sub, answers, judges) for label, sub in
                                  [("all", rows), ("interp", [r for r in rows if r["corpus"] == "interp"]),
                                   ("exmnt", [r for r in rows if r["corpus"] == "exmnt"])]}
        result.update(fingerprint=fp["id"], warmup_seconds=round(warmup, 2), updated_at=now(),
                      limitations=["Synthetic single-gold questions; same model as judge", "No unanswerable test cases", "Full app path/full-source grading differs from f17"])
        save(OUT, result)
        print(json.dumps({name: result["sets"][name]}, ensure_ascii=False), flush=True)
        if name == "test10":
            save(LOCK, {**json.loads(LOCK.read_text(encoding="utf-8")), "completed_at": now()})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
