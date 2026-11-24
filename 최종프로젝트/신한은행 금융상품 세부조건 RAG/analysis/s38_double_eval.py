"""16-1 '질문 두 개' 채점 — 답을 한 번 만들고 하위 질문마다 따로 채점

사용: python analysis/s38_double_eval.py subs dev            → retrieval_doublesub_dev.jsonl (하위 질문 한 줄씩, 이어서 s17로 보완 정답)
      python analysis/s38_double_eval.py eval dev [--version=v9]
      python analysis/s38_double_eval.py eval test --final [--version=...]   (다 고친 뒤 한 번만)
- 답변: s27과 같은 흐름 (근거 고르기 · 판매 상태 · 답변 원칙 · 추정 표시). 결과는 data/eval/double_eval_{split}.jsonl (이미 한 것은 다시 하지 않음)
- 채점: 하위 질문마다 s15 채점 지시문 + "답변 중 이 하위 질문 부분만 판단"
- 검색: 하위 질문의 정답(gold + 보완 gold_alt)이 답변에 넣은 근거 안에 있는지
"""
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import s12_retrieval as R  # noqa: E402
import s15_answer_eval as A  # noqa: E402
from s8_extract import load_env  # noqa: E402

EVAL = ROOT / "data" / "eval"
SUB_JUDGE = A.JUDGE_PROMPT + "\n\n**[답변]은 여러 질문을 한 번에 받은 답이다. 그중 [질문]에 해당하는 부분만 보고 판정한다.** 다른 질문에 대한 내용은 무시한다."


def subs(split):
    rows = [json.loads(l) for l in open(EVAL / f"double_{split}.jsonl")]
    out = []
    for r in rows:
        for k, s in enumerate(r["subs"]):
            out.append({"qid": f"{r['qid']}{'ab'[k]}", "type": "A", "question": s["question"], "product_id": r["product_id"],
                        "category": r["category"], "gold": s["gold"], "answer": s["answer"], "evidence": s["evidence"], "split": f"doublesub_{split}"})
    p = EVAL / f"retrieval_doublesub_{split}.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    py = sys.executable
    subprocess.run([py, str(ROOT / "analysis" / "s10_chunks.py"), "embed-questions-local", str(p), str(EVAL / f"double_{split}.jsonl")], check=True)
    subprocess.run([py, str(ROOT / "analysis" / "s17_multigold.py"), f"doublesub_{split}"], check=True)


class Judge(BaseModel):
    verdict: str
    reason: str


def evaluate(split):
    if split != "dev" and "--final" not in sys.argv:
        sys.exit("test는 잠근 시험 세트 — 방식을 다 정한 뒤 --final을 붙여 한 번만 실행")
    version = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--version=")), "v9")
    qs = [json.loads(l) for l in open(EVAL / f"double_{split}.jsonl")]
    sub = {r["qid"]: r for r in map(json.loads, open(EVAL / f"retrieval_doublesub_{split}.jsonl"))}
    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
    idx = R.Index()
    idx._name_index()
    idx._client = client
    qv = R.embed_questions(qs, idx)
    docs = {r["doc_id"]: r for r in csv.DictReader(open(ROOT / "data" / "meta" / "documents.csv"))}
    pn_all = {r["product_id"]: r["name"] for r in csv.DictReader(open(ROOT / "data" / "meta" / "products.csv"))}
    prompt = A.prompt_of(version)
    ph = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    out_path = EVAL / f"double_eval_{split}.jsonl"
    done = {}
    if out_path.exists():
        for l in open(out_path):
            r = json.loads(l)
            done[r["key"]] = r

    def job(arg):
        q, v = arg
        ctx_ids, groups = A.context_ids(idx, q["question"], v, version, k=5)
        extra = A.extra_context(q["question"], idx)
        key = hashlib.sha256(f"{env['CHAT_MODEL']}|{q['qid']}|{q['question']}|{ctx_ids}|{version}|{ph}|{extra}".encode()).hexdigest()
        if key in done:
            return done[key], False
        pids = set().union(*groups) if len(groups) >= 2 else idx.detect_products(q["question"])
        linked = set().union(*(idx.links[p] for p in pids)) if pids else set()
        lcl = set().union(*(idx.doc_clause[d] for d in linked)) if linked else set()
        pn = ", ".join(sorted(pn_all[p] for p in pids))[:120]
        if len(groups) >= 2:
            ctx = "\n\n".join(f"[{n}] {A.multi_header(idx, i, docs, groups, pn_all)}\n{idx.items[i]['text'][:2000]}" for n, i in enumerate(ctx_ids, 1))
        else:
            ctx = "\n\n".join(f"[{n}] {A.context_header(idx, i, docs, linked, lcl, pn)}\n{idx.items[i]['text'][:2000]}" for n, i in enumerate(ctx_ids, 1))
        inp = extra + "\n\n" + A.answer_input(version, ctx, q["question"], pn, inferred=bool(pids) and idx.detect_how(q["question"]) != "exact")
        ans = client.responses.create(model=env["CHAT_MODEL"], instructions=prompt, input=inp).output_text
        if len(groups) < 2:
            ans = A.finalize_answer(version, ans, idx.detect_how(q["question"]), pn)
        res = []
        for k, s in enumerate(q["subs"]):
            sq = sub[f"{q['qid']}{'ab'[k]}"]
            j = client.responses.parse(model=env["CHAT_MODEL"], instructions=SUB_JUDGE, text_format=A.Judge,
                                       input=f"[질문] {s['question']}\n[정답] {s['answer']}\n[정답 근거 원문] {s['evidence']}\n\n[답변] {ans}").output_parsed
            in_ctx = any(idx.is_gold_any(i, sq) for i in ctx_ids)
            res.append({"verdict": j.verdict, "reason": j.reason, "gold_in_context": in_ctx})
        rec = {"key": key, "version": version, "qid": q["qid"], "style": q["style"], "question": q["question"], "answer": ans, "subs": res,
               "context": [idx.items[i]["text"].split("\n", 1)[0] for i in ctx_ids]}
        return rec, True

    with ThreadPoolExecutor(6) as ex, open(out_path, "a") as f:
        recs = []
        for rec, new in ex.map(job, zip(qs, qv)):
            if new:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            recs.append(rec)
    summary(recs, split, version)


def summary(recs, split, version):
    n = len(recs)
    both = sum(all(s["verdict"] == "correct" for s in r["subs"]) for r in recs)
    first = sum(r["subs"][0]["verdict"] == "correct" for r in recs)
    second = sum(r["subs"][1]["verdict"] == "correct" for r in recs)
    ctx_both = sum(all(s["gold_in_context"] for s in r["subs"]) for r in recs)
    print(f"[{split} {version}] 질문 {n} | 둘 다 맞음 {both}/{n} ({both / n:.2f}) · 첫 질문 {first}/{n} · 두 번째 질문 {second}/{n} · 근거에 두 정답 다 {ctx_both}/{n}")
    print("   하위 질문 판정", dict(Counter(s["verdict"] for r in recs for s in r["subs"])),
          "| 근거에 정답 없음인데 정답 아님", sum(1 for r in recs for s in r["subs"] if not s["gold_in_context"] and s["verdict"] != "correct"),
          "| 근거에 정답 있는데 정답 아님", sum(1 for r in recs for s in r["subs"] if s["gold_in_context"] and s["verdict"] != "correct"))
    for st in ("formal", "real"):
        rs = [r for r in recs if r["style"] == st]
        if rs:
            print(f"   {st}: 둘 다 맞음 {sum(all(s['verdict'] == 'correct' for s in r['subs']) for r in rs)}/{len(rs)}")


if __name__ == "__main__":
    cmd, split = sys.argv[1], sys.argv[2]
    subs(split) if cmd == "subs" else evaluate(split)
