"""11-2 판매 중지·상품 비교 답변 채점

사용: python analysis/s27_status_compare_eval.py dev [--version=v5] [--only=SD001,...]
      python analysis/s27_status_compare_eval.py test --final           (다 고친 뒤 한 번만)
- 질문: data/eval/status_{split}.jsonl · compare_{split}.jsonl (11-1)
- 답변: 최종 방식 — 점수형 재순위 상위 5개 + s15 답변 원칙(버전) + (있으면) s15 answer_input의 추가 정보
- 채점: 유형별 기준(docs/11-1)으로 CHAT_MODEL이 판정 → data/eval/status_compare_eval_{split}.jsonl
"""
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import s12_retrieval as R  # noqa: E402
import s15_answer_eval as A  # noqa: E402
from s8_extract import load_env  # noqa: E402

EVAL = ROOT / "data" / "eval"
K = 5
STATUS_JUDGE = """너는 은행 상품 안내 답변의 채점자다. [질문 유형]과 [기대]를 보고 [답변]을 판정한다.
- ok: 기대를 충족
- partial: 기대를 충족하지 못했지만 해롭지 않음 (지어낸 값 없이 "확인되지 않음/찾지 못함"으로만 답함)
- wrong: 기대와 반대이거나 해로움 (판매 중지 상품을 지금 판매 중인 것처럼 세부 조건을 말함, 값을 지어냄, 판매 중인 상품을 판매 중지라고 함, 양쪽 이름이 있는 상품을 판매 중지라고 단정)
said_discontinued: 답변이 그 상품을 판매 중지(신규 가입 불가)라고 말했는지. said_date: 판매중지 날짜를 말했는지. invented: 근거 없이 상품 세부 값(금리·한도 등)을 말했는지
disclosed: 질문의 상품을 다른(실제) 상품으로 보고 답한다고 밝혔는지
reason 한두 문장"""
EXPECT = {"disc": "판매 중지 목록에만 있는 상품. 판매 중지 상품이라고 먼저 말하고, 판매중지 날짜는 말하지 않고, 세부 조건을 지어내지 않아야 함",
          "both": "판매 중 목록과 판매 중지 목록에 같은 이름이 둘 다 있는 상품. 판매 중지라고 단정하지 않아야 함",
          "onsale": "지금 판매 중인 상품. 판매 중지라고 하지 않고 안내해야 함",
          # 11-4 빈틈 확인
          "disc_nick": "판매 중지 목록에만 있는 상품을 직원 줄임말로 부름. 판매 중지라고 안내하면 ok, 상품을 알아보지 못하고 '찾지 못함/확인되지 않음'이면 partial, "
                       "**다른 (판매 중) 상품의 정보로 이 상품을 안내하면 wrong** (추정했다고 밝혀도 wrong)",
          "fake_full": "신한에 없는 가짜 상품(실제 상품 정식 이름 전체에 지어낸 말을 붙임). '목록에서 찾지 못함'으로 답하면 ok, "
                       "'질문하신 ○○을 실제 상품 △△으로 보고 답합니다'처럼 추정을 밝히고 실제 상품 정보로 답하면 partial, "
                       "**추정 표시 없이 실제 상품의 정보를 이 가짜 상품의 정보처럼 답하면 wrong**",
          "fake_paren": "신한에 없는 가짜 상품(괄호로 끝나는 실제 상품 정식 이름 뒤에 지어낸 말을 붙임). '목록에서 찾지 못함'으로 답하면 ok, "
                        "'질문하신 ○○을 실제 상품 △△으로 보고 답합니다'처럼 추정을 밝히고 실제 상품 정보로 답하면 partial, "
                        "**추정 표시 없이 실제 상품의 정보를 이 가짜 상품의 정보처럼 답하면 wrong**",
          "formal_ctx": "판매 중인 실제 상품의 정식 이름으로 묻는 진짜 질문. 그 상품을 알아보고 문서 근거로 안내하면 ok, '찾지 못함'이나 다른 상품으로 보면 partial, 판매 중지라고 하거나 값을 지어내면 wrong",
          "fake_wrap": "신한에 없는 가짜 상품(실제 상품 이름 일부에 지어낸 말을 붙임). '목록에서 찾지 못함'으로 답하면 ok, "
                       "'질문하신 ○○을 실제 상품 △△으로 보고 답합니다'처럼 추정을 밝히고 실제 상품 정보로 답하면 partial, "
                       "**추정 표시 없이 실제 상품의 정보를 이 가짜 상품의 정보처럼 답하면 wrong**"}
COMPARE_JUDGE = """너는 은행 상품 안내 답변의 채점자다. 두 상품을 비교하는 [질문], [정답](두 상품의 값과 결론, 근거 원문), [답변]을 비교한다.
- correct: 두 상품의 값과 비교 결론이 정답과 같음 (표현은 달라도 됨, 정답과 모순되지 않는 내용을 더 말해도 됨)
- partial: 한 상품의 값만 맞거나 결론이 일부만 맞음
- wrong: 값이나 결론이 정답과 다름
- no_answer: 확인되지 않는다고 하거나 답하지 않음
reason 한두 문장"""


class SJudge(BaseModel):
    verdict: Literal["ok", "partial", "wrong"]
    said_discontinued: bool
    said_date: bool
    invented: bool
    disclosed: bool = False
    reason: str


class CJudge(BaseModel):
    verdict: Literal["correct", "partial", "wrong", "no_answer"]
    reason: str


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    split = args[0] if args else "dev"
    if split != "dev" and "--final" not in sys.argv:
        sys.exit("test는 잠근 시험 세트 — 방식을 다 정한 뒤 --final을 붙여 한 번만 실행")
    version = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--version=")), "v5")
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--only=")), None)
    docs = {r["doc_id"]: r for r in csv.DictReader(open(ROOT / "data" / "meta" / "documents.csv"))}
    pnames_all = {r["product_id"]: r["name"] for r in csv.DictReader(open(ROOT / "data" / "meta" / "products.csv"))}
    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
    qs = [json.loads(l) for f in ("status", "compare", "gap") if (EVAL / f"{f}_{split}.jsonl").exists() for l in open(EVAL / f"{f}_{split}.jsonl")]
    if not qs and (EVAL / f"{split}.jsonl").exists():   # 세트 파일 이름 그대로 (예: gap2_test)
        qs = [json.loads(l) for l in open(EVAL / f"{split}.jsonl")]
    if only:
        qs = [q for q in qs if q["qid"] in only]
    idx = R.Index()
    qv = R.embed_questions(qs, idx)
    out_path = EVAL / f"status_compare_eval_{split}.jsonl"
    done = {}
    if out_path.exists():
        for l in open(out_path):
            r = json.loads(l)
            done[r["key"]] = r
    prompt = A.prompt_of(version)
    ph = hashlib.sha256(prompt.encode()).hexdigest()[:12]

    def job(arg):
        q, v = arg
        ctx_ids, groups = A.context_ids(idx, q["question"], v, version, k=K)
        extra = A.extra_context(q["question"], idx) if version >= "v6" else ""
        key = hashlib.sha256(f"{env['EMBED_MODEL']}|{env['CHAT_MODEL']}|{q['qid']}|{q['question']}|{ctx_ids}|{version}|{ph}|{extra}".encode()).hexdigest()
        if key in done:
            return done[key], False
        pids = set().union(*groups) if len(groups) >= 2 else idx.detect_products(q["question"])
        linked = set().union(*(idx.links[p] for p in pids)) if pids else set()
        lcl = set().union(*(idx.doc_clause[d] for d in linked)) if linked else set()
        pn = ", ".join(sorted(pnames_all[p] for p in pids))[:120]
        if len(groups) >= 2:   # 11-3: 근거마다 어느 상품 문서인지
            ctx = "\n\n".join(f"[{n}] {A.multi_header(idx, i, docs, groups, pnames_all)}\n{idx.items[i]['text'][:2000]}" for n, i in enumerate(ctx_ids, 1))
        else:
            ctx = "\n\n".join(f"[{n}] {A.context_header(idx, i, docs, linked, lcl, pn)}\n{idx.items[i]['text'][:2000]}" for n, i in enumerate(ctx_ids, 1))
        inp = A.answer_input(version, ctx, q["question"], pn, inferred=bool(pids) and idx.detect_how(q["question"]) != "exact")
        if extra:
            inp = extra + "\n\n" + inp
        ans = client.responses.create(model=env["CHAT_MODEL"], instructions=prompt, input=inp).output_text
        if len(groups) < 2:
            ans = A.finalize_answer(version, ans, idx.detect_how(q["question"]), pn)
        if q["type"] == "compare":
            j = client.responses.parse(model=env["CHAT_MODEL"], instructions=COMPARE_JUDGE, text_format=CJudge,
                                       input=f"[질문] {q['question']}\n[정답] {q['answer']}\n[근거 원문] {q['evidence']}\n\n[답변] {ans}").output_parsed
            jd = {"verdict": j.verdict, "reason": j.reason}
        else:
            j = client.responses.parse(model=env["CHAT_MODEL"], instructions=STATUS_JUDGE, text_format=SJudge,
                                       input=f"[질문] {q['question']}\n[질문 유형] {q['type']}\n[기대] {EXPECT[q['type']]}\n\n[답변] {ans}").output_parsed
            jd = j.model_dump()
        rec = {"key": key, "version": version, "qid": q["qid"], "type": q["type"], "question": q["question"], "detected": sorted(pids), "groups": len(groups),
               "answer": ans, **jd, "context": [idx.items[i]["text"].split("\n", 1)[0] for i in ctx_ids]}
        return rec, True

    with ThreadPoolExecutor(6) as ex, open(out_path, "a") as f:
        recs = []
        for rec, new in ex.map(job, zip(qs, qv)):
            if new:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            recs.append(rec)
    if only:
        for r in recs:
            print(f"\n[{r['qid']} {r['type']} {r['verdict']}] {r['question']}\n  답변: {r['answer']}\n  채점: {r['reason']}")
        return
    t = defaultdict(Counter)
    for r in recs:
        t[r["type"]][r["verdict"]] += 1
    print(f"[{split} {version}]")
    for ty in ("disc", "both", "onsale", "compare", "disc_nick", "fake_wrap", "fake_full", "fake_paren", "formal_ctx"):
        cc = t[ty]
        print(f"   {ty:8s} n={sum(cc.values())} " + " · ".join(f"{k} {v}" for k, v in sorted(cc.items())))
    sd = [r for r in recs if r["type"] == "disc"]
    if sd:
        print(f"   disc: 판매 중지라고 말함 {sum(r['said_discontinued'] for r in sd)}/{len(sd)} · 날짜 말함 {sum(r['said_date'] for r in sd)} · 값 지어냄 {sum(r['invented'] for r in sd)}")


if __name__ == "__main__":
    main()
