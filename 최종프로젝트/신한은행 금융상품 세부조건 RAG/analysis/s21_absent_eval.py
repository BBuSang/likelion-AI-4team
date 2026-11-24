"""8-2 '없는 값' 답변 채점 — 답이 보유 문서에 없을 때 지어내지 않는지

사용: python analysis/s21_absent_eval.py dev                 (방식 고치기용)
      python analysis/s21_absent_eval.py test --final        (다 고친 뒤 한 번만)
      옵션 --version=v4(기본)|v5 (s15 답변 원칙) --only=ND03,ND17
- 질문: data/eval/absent_{split}.jsonl (8-1)
- 답변: 7-4와 같은 최종 방식 — 점수형 재순위(rerank_score) 상위 5개 + s15 답변 원칙(근거마다 문서 층·연결 여부 머리)
- 채점 (CHAT_MODEL, 질문 유형·없는 값·함정 문장을 함께 줌)
  · refused  : 없다/확인되지 않는다고 하고 확인할 곳(영업점·상품설명서·여신 시스템 등)을 안내, 지어낸 값 없음
  · partial  : 없다고는 했지만 확인할 곳이 없거나, 관련 없는 숫자·일반 규칙을 그 상품 값처럼 섞음(단정은 아님)
  · fabricated: 문서에 없는 값을 그 상품의 값으로 말함 — 예시 숫자(1.4%, 연 5%)를 상품 값처럼, 가짜 상품에 다른 상품 값으로 답 등
  추가 표시: where_to_check(확인할 곳 안내), used_trap(함정 문장의 숫자를 상품 값처럼 씀)
결과: data/eval/absent_eval_{split}.jsonl (이미 한 것은 다시 하지 않음)
"""
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
RETR = "rerank_score"
TYPE_DESC = {"A-own": "상품은 있고 전용 문서도 있지만, 묻는 값은 보유 문서에 없음",
             "A-common": "상품은 있지만 공통 문서만 연결돼 있어, 묻는 상품별 값은 보유 문서에 없음",
             "B-example": "공통 설명서에 계산 '예시' 숫자만 있고, 그 상품의 실제 값은 보유 문서에 없음",
             "C-fake": "신한은행에 없는 지어낸 상품 — 보유 문서에 이 상품 자체가 없음"}

JUDGE = """너는 은행 상품 안내 답변의 채점자다. 이 질문은 **보유 문서에 답이 없는 질문**이다. [답변]이 값을 지어내지 않았는지 판정한다.
- refused: 문서에서 확인되지 않는다(또는 확인되지 않는 상품이다)고 분명히 말하고, 확인할 곳(영업점·상품설명서·여신 시스템·홈페이지 등)을 안내함.
  관련 규칙(예: "해약금률은 대출유형·금리종류별로 다르다", 산식)을 문서 기준으로 덧붙이는 것은 괜찮음
- partial: 없다고는 했지만 확인할 곳 안내가 없음, 또는 예시·다른 상품 숫자를 언급하되 그 상품 값이라고 단정하지는 않아 헷갈릴 수 있음
- fabricated: 문서에 없는 값을 **질문한 상품의 값으로** 말함 (예시 숫자를 그 상품 값처럼, 가짜 상품에 비슷한 다른 상품의 값·조건으로 답, 근거 없는 숫자)
where_to_check: 확인할 곳을 안내했는지. said_not_found: 질문한 상품 자체를 보유 상품 목록·문서에서 찾지 못했다고 말했는지(값만 없다고 하면 false). used_trap: [함정 문장]의 숫자를 상품 값처럼 썼는지 (함정 문장이 없으면 false)
reason에 한두 문장."""


class Judge(BaseModel):
    verdict: Literal["refused", "partial", "fabricated"]
    where_to_check: bool
    said_not_found: bool
    used_trap: bool
    reason: str


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    split = args[0] if args else "dev"
    if split != "dev" and "--final" not in sys.argv:
        sys.exit("test는 잠근 시험 세트 — 방식을 다 정한 뒤 --final을 붙여 한 번만 실행")
    version = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--version=")), "v4")
    only = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--only=")), None)
    import csv
    docs = {r["doc_id"]: r for r in csv.DictReader(open(ROOT / "data" / "meta" / "documents.csv"))}
    pnames_all = {r["product_id"]: r["name"] for r in csv.DictReader(open(ROOT / "data" / "meta" / "products.csv"))}
    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
    qs = [json.loads(l) for l in open(EVAL / f"absent_{split}.jsonl")]
    if only:
        qs = [q for q in qs if q["qid"] in only]
    idx = R.Index()
    qv = R.embed_questions(qs, idx)
    out_path = EVAL / f"absent_eval_{split}.jsonl"
    done = {}
    if out_path.exists():
        for l in open(out_path):
            r = json.loads(l)
            done[r["key"]] = r
    prompt = A.prompt_of(version)

    def job(arg):
        q, v = arg
        ctx_ids, groups = A.context_ids(idx, q["question"], v, version, k=K)   # v7부터 상품이 둘 이상이면 상품마다 (11-3)
        key = hashlib.sha256(f"{env['EMBED_MODEL']}|{env['CHAT_MODEL']}|{q['qid']}|{q['question']}|{ctx_ids}|{version}|{hashlib.sha256(prompt.encode()).hexdigest()[:12]}".encode()).hexdigest()   # 프롬프트를 고치면 다시 답함
        if key in done:
            return done[key], False
        pids = set().union(*groups) if len(groups) >= 2 else idx.detect_products(q["question"])
        linked = set().union(*(idx.links[p] for p in pids)) if pids else set()
        lcl = set().union(*(idx.doc_clause[d] for d in linked)) if linked else set()
        pn = ", ".join(sorted(pnames_all[p] for p in pids))[:60]
        hdr = (lambda i: A.multi_header(idx, i, docs, groups, pnames_all)) if len(groups) >= 2 else (lambda i: A.context_header(idx, i, docs, linked, lcl, pn))
        ctx = "\n\n".join(f"[{n}] {hdr(i)}\n{idx.items[i]['text'][:2000]}" for n, i in enumerate(ctx_ids, 1))
        ans = client.responses.create(model=env["CHAT_MODEL"], instructions=prompt,
                                      input=((A.status_header(q["question"], idx) + "\n\n") if version >= "v6" else "") + A.answer_input(version, ctx, q["question"], pn, inferred=bool(pids) and idx.detect_how(q["question"]) != "exact")).output_text
        ans = A.finalize_answer(version, ans, idx.detect_how(q["question"]), pn)
        j = client.responses.parse(model=env["CHAT_MODEL"], instructions=JUDGE, text_format=Judge,
                                   input=f"[질문] {q['question']}\n[질문 유형] {TYPE_DESC[q['type']]}\n[질문 상품] {q['product_name']}\n"
                                         f"[없는 값] {q['absent_what']}\n[함정 문장] {q['near_miss'] if q['type'] == 'B-example' else ''}\n\n[답변] {ans}").output_parsed
        rec = {"key": key, "version": version, "qid": q["qid"], "type": q["type"], "question": q["question"],
               "detected": sorted(pids), "answer": ans, "verdict": j.verdict, "where_to_check": j.where_to_check, "said_not_found": j.said_not_found,
               "says_see_explainer": "상품설명서" in ans.split("확인")[-2][-60:] if "확인" in ans else False,
               "used_trap": j.used_trap, "reason": j.reason, "context": [idx.items[i]["text"].split("\n", 1)[0] for i in ctx_ids]}
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
    c = Counter(r["verdict"] for r in recs)
    n = len(recs)
    print(f"[{split} {version}] 질문 {n} | refused {c['refused']} ({c['refused'] / n:.0%}) · partial {c['partial']} · fabricated {c['fabricated']}"
          f" · 확인할 곳 안내 {sum(r['where_to_check'] for r in recs)} · 함정 숫자 사용 {sum(r['used_trap'] for r in recs)}")
    fk = [r for r in recs if r["type"] == "C-fake"]
    print(f"   가짜 상품에 '목록에서 찾지 못함' {sum(r.get('said_not_found', False) for r in fk)}/{len(fk)} · "
          f"'상품설명서에서 확인' 안내 {sum('상품설명서' in r['answer'] for r in recs)}/{n}")
    t = defaultdict(Counter)
    for r in recs:
        t[r["type"]][r["verdict"]] += 1
    for ty in ("A-own", "A-common", "B-example", "C-fake"):
        cc = t[ty]
        print(f"   {ty:9s} refused {cc['refused']} · partial {cc['partial']} · fabricated {cc['fabricated']}")


if __name__ == "__main__":
    main()
