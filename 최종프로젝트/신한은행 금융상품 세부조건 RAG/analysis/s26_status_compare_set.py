"""11-1 판매 중지·상품 비교 평가 세트 (구현 전에 만들고 test는 잠금)

사용: python analysis/s26_status_compare_set.py   → data/eval/status_{dev,test}.jsonl · compare_{dev,test}.jsonl (한 번만, 있으면 멈춤)
왜: 판매 중지 상품 질문(답변원칙 1번)과 여러 상품 비교 질문은 한 번도 재지 않음
구성 (seed 20261119, OpenAI CHAT_MODEL, 상품 단위로 dev/test 반씩)
  status  disc 30   판매 중지 목록에만 있는 상품(예금·대출, 판매 중에 같은 이름 없음) — 기대: 판매 중지라고 먼저 말함·날짜 안 말함·세부 조건 안 지어냄
          both 10   판매 중·판매 중지 양쪽에 같은 이름이 있는 상품 — 기대: 판매 중지라고 단정하지 않음
          onsale 10 판매 중 상품의 "지금 가입돼요?"류 — 기대: 판매 중지라고 하지 않고 정상 안내
  compare 30        같은 대분류(저축·대출)의 판매 중 상품 두 개, 각자 전용 상품설명서의 LLM 추출값(근거 문장 포함)으로 비교 질문
                    — 근거 문장이 각 상품 설명서 원문에 실제로 있는지 코드로 확인 (10-1 교훈: 값이 그 상품에 해당하는지)
"""
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from s8_extract import lines_in_order, load_env, ns  # noqa: E402

META, EVAL = ROOT / "data" / "meta", ROOT / "data" / "eval"
SEED = 20261119
OUT = {f"{k}_{s}": EVAL / f"{k}_{s}.jsonl" for k in ("status", "compare") for s in ("dev", "test")}

STATUS_PROMPT = """너는 신한은행 영업점 직원이다. 고객 응대 중 동료나 상품 안내 챗봇에게 주어진 상품에 대해 물어볼 짧은 질문 하나를 만든다.
- 상품 이름은 주어진 그대로 넣는다
- {focus}
- 말하듯 자연스럽게, 한 문장"""
FOCUS = {"disc": "금리·가입 조건·한도·우대 조건 중 하나를 묻는다 (판매 여부는 묻지 않음)",
         "both": "지금 가입할 수 있는지 또는 금리를 묻는다",
         "onsale": "지금 신규 가입이 되는지(판매 중인지)를 묻는다"}
COMPARE_PROMPT = """너는 신한은행 영업점 직원이다. 아래 [상품 A]와 [상품 B]의 추출 자료를 보고, 고객 응대 중 물어볼 **두 상품을 비교하는 질문** 하나를 만든다.
- 두 상품 모두 자료에 값이 있는 항목 하나만 비교 (예: 기본금리, 최고 우대금리, 가입기간, 가입금액 한도, 가입 대상, 이자 지급 방식)
- 질문에 두 상품 이름을 그대로 넣는다. answer: 두 상품의 값과 비교 결론을 한두 문장으로
- evidence_a / evidence_b: 각 상품 자료의 source_text 또는 값이 적힌 문장을 **그대로** 복사
- 비교할 만한 항목이 없으면 usable=false"""


class SQ(BaseModel):
    question: str


class CQ(BaseModel):
    usable: bool
    question: str
    attribute: str
    answer: str
    evidence_a: str
    evidence_b: str
    note: Optional[str] = None


def main():
    only_compare = "--only=compare" in sys.argv   # 비교 세트만 다시 (짝을 대분류로 넓힘, 판매 상태 세트는 그대로)
    targets = [k for k in OUT if k.startswith("compare")] if only_compare else list(OUT)
    if any(OUT[k].exists() for k in targets):
        sys.exit("이미 있음 — 잠근 세트라 다시 만들지 않음")
    rnd = random.Random(SEED)
    prods = list(csv.DictReader(open(META / "products.csv")))
    by_name = {}
    for p in prods:
        by_name.setdefault(p["name"], p)
    docs = {r["doc_id"]: r for r in csv.DictReader(open(META / "documents.csv"))}
    links, plinks = defaultdict(set), defaultdict(set)
    for r in csv.DictReader(open(META / "product_docs.csv")):
        links[r["product_id"]].add(r["doc_id"])
        plinks[r["doc_id"]].add(r["product_id"])
    disc = list(csv.DictReader(open(META / "discontinued.csv")))

    def split_of(name):
        return "test" if int(hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest(), 16) % 2 else "dev"

    def pick(names, n):
        out = {"dev": [], "test": []}
        for nm in names:
            sp = split_of(nm)
            if len(out[sp]) < n // 2:
                out[sp].append(nm)
        return out

    d_only = sorted({r["name"] for r in disc if r["category"] in ("저축상품", "대출상품") and r["also_on_sale"] == "0" and r["name"] not in by_name})
    d_both = sorted({r["name"] for r in disc if r["also_on_sale"] == "1"})
    onsale = sorted(by_name)
    for lst in (d_only, d_both, onsale):
        rnd.shuffle(lst)
    status_jobs = [("disc", sp, n) for sp, ns_ in pick(d_only, 30).items() for n in ns_]
    status_jobs += [("both", sp, n) for sp, ns_ in pick(d_both, 10).items() for n in ns_]
    status_jobs += [("onsale", sp, n) for sp, ns_ in pick(onsale, 10).items() for n in ns_]

    # 비교: 전용 상품설명서가 하나 있고 추출이 ok인 판매 중 상품, 같은 소분류끼리 짝
    ext = {}
    for f in (ROOT / "data" / "extract" / "full").glob("*.json"):
        x = json.load(open(f))
        if x.get("status") == "ok":
            ext[x["doc_id"]] = x["extraction"]
    own = {}
    for nm, p in by_name.items():
        ds = [d for d in links[p["product_id"]] if len(plinks[d]) == 1 and d in ext and docs[d]["doc_type"] == "상품설명서"]
        if len(ds) == 1:
            own[nm] = ds[0]
    groups = defaultdict(list)
    for nm in own:
        groups[by_name[nm]["category"]].append(nm)   # 같은 대분류(저축·대출)끼리 — 소분류로만 짝지으면 test 짝이 8개뿐이라 넓힘
    pairs = {"dev": [], "test": []}
    for g, nms in sorted(groups.items()):
        nms = sorted(nms)
        rnd.shuffle(nms)
        for sp in ("dev", "test"):
            side = [n for n in nms if split_of(n) == sp]
            for i in range(0, len(side) - 1, 2):
                pairs[sp].append((side[i], side[i + 1]))
    for sp in pairs:
        rnd.shuffle(pairs[sp])
    print("비교 후보 쌍", {k: len(v) for k, v in pairs.items()}, "| 판매중지 이름", len(d_only), "양쪽", len(d_both))

    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)

    def s_one(job):
        kind, sp, name = job
        r = client.responses.parse(model=env["CHAT_MODEL"], instructions=STATUS_PROMPT.format(focus=FOCUS[kind]), text_format=SQ, input=f"상품 이름: {name}")
        return job, r.output_parsed

    def compact(nm):
        x = ext[own[nm]]
        return json.dumps({"상품": nm, "summary": x.get("summary"), "target_customer": x.get("target_customer"), "detail": x.get("detail"),
                           "rates": [{k: r.get(k) for k in ("rate_kind", "rate_min", "rate_max", "term_min_m", "term_max_m", "condition", "source_text")} for r in x.get("rates", [])][:12]},
                          ensure_ascii=False)

    def c_one(arg):
        sp, a, b = arg
        r = client.responses.parse(model=env["CHAT_MODEL"], instructions=COMPARE_PROMPT, text_format=CQ,
                                   input=f"[상품 A]\n{compact(a)}\n\n[상품 B]\n{compact(b)}")
        return arg, r.output_parsed

    with ThreadPoolExecutor(6) as ex:
        s_res = [] if only_compare else list(ex.map(s_one, status_jobs))
        c_res = list(ex.map(c_one, [(sp, a, b) for sp in ("dev", "test") for a, b in pairs[sp][:22]]))
    out = defaultdict(list)
    for n, ((kind, sp, name), q) in enumerate(s_res):
        if q and name in q.question:
            pid = by_name[name]["product_id"] if name in by_name else None
            out[f"status_{sp}"].append({"qid": f"S{sp[0].upper()}{n:03d}", "type": kind, "question": q.question, "name": name, "product_id": pid, "split": sp})
    sec_text = {}
    for d in set(own.values()):
        sec_text[d] = ns("\n".join(json.loads(l)["text"] for l in open(ROOT / "data" / "sections" / f"{d}.jsonl")))
    dropped = Counter()
    for n, ((sp, a, b), q) in enumerate(c_res):
        if len([x for x in out[f"compare_{sp}"]]) >= 15:
            continue
        if not q or not q.usable:
            dropped["usable=false"] += 1
            continue
        if a not in q.question or b not in q.question:
            dropped["상품 이름 없음"] += 1
            continue
        if not (lines_in_order(q.evidence_a, sec_text[own[a]]) and lines_in_order(q.evidence_b, sec_text[own[b]])):
            dropped["근거가 각 상품 설명서에 없음"] += 1
            continue
        out[f"compare_{sp}"].append({"qid": f"P{sp[0].upper()}{n:03d}", "type": "compare", "question": q.question, "attribute": q.attribute,
                                     "products": [a, b], "product_ids": [by_name[a]["product_id"], by_name[b]["product_id"]],
                                     "docs": [own[a], own[b]], "answer": q.answer, "evidence": [q.evidence_a, q.evidence_b], "split": sp})
    for k in targets:
        p = OUT[k]
        with open(p, "w") as f:
            for r in out[k]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(p.name, Counter(r["type"] for r in out[k]))
    print("비교 버림", dict(dropped))
    for k in [x for x in ("status_test", "compare_test") if x in targets]:
        print(k, "sha256", hashlib.sha256(OUT[k].read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
