"""9-1 상품명 찾기 확인 세트 2 (잠금) — 줄임말 개선을 공정하게 재려고 **구현 전에** 만듦

사용: python analysis/s22_alias_check2.py gen                 → data/eval/alias_check2.jsonl (한 번만, 해시로 잠금)
      python analysis/s22_alias_check2.py score --final [--method=exact|fallback]   (다 고친 뒤 한 번만)
      python analysis/s22_alias_check2.py score-dev [--method=...]  → 고치기용 세트(alias_check 60 + 가짜 상품 + 공통 질문)
구성 (seed 20261115)
  nick    60: 기존 alias_check·test2 질문에 쓰지 않은 상품 이름에서 무작위 — s18과 **같은 지시문**으로 직원 줄임말 질문
  fake    30: 신한에 없는 지어낸 상품 이름 질문 (실제 이름과 겹치지 않는 것만) → 어떤 상품도 고르면 안 됨
  general 20: 특정 상품이 아닌 공통 규정·용어 질문 ("예금자보호 한도", "기한의 이익 상실") → 상품을 붙이면 안 됨
판정: nick 맞음(찾은 상품 이름에 정답) / 틀림(다른 상품만) / 못 찾음, fake·general은 상품을 골랐는지
"""
import csv
import hashlib
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from s8_extract import load_env  # noqa: E402
from s18_alias_check import PROMPT as NICK_PROMPT, Q  # noqa: E402

EVAL, META = ROOT / "data" / "eval", ROOT / "data" / "meta"
OUT = EVAL / "alias_check2.jsonl"
SEED = 20261115
FAKE_PROMPT = """너는 은행 평가 질문을 만드는 사람이다. **신한은행에 실제로는 없는, 그럴듯한 가짜 상품 이름**을 넣은 영업점 직원 질문을 만든다.
- 이름은 실제 은행 상품처럼 (적금·예금·통장·대출·외화·카드 연계 등 다양하게), 정식 이름이 아니라 직원이 부르듯 짧게 불러도 됨
- 아래 [실제 상품 이름]과 핵심 단어가 겹치면 안 됨
- 질문 내용은 금리·가입대상·한도·해지 등 짧게. name에는 질문 속 상품 호칭만"""
GEN_PROMPT = """너는 신한은행 영업점 직원이다. **특정 상품 이름 없이** 은행 거래 공통 규정이나 용어를 묻는 짧은 질문을 만든다.
- 예: 예금자보호 한도, 휴면계좌·거래중지계좌, 기한의 이익 상실, 중도상환해약금 뜻, 연체이자 계산, 통장 분실 신고, 대출 청약철회, 금리인하요구권, 이자 지급 시기 등 다양하게
- 상품 이름(○○적금, ○○대출 같은 고유한 이름)은 넣지 않는다. '적금', '대출' 같은 일반어는 괜찮음"""


class Fake(BaseModel):
    name: str
    question: str


class FakeList(BaseModel):
    items: list[Fake]


class GenList(BaseModel):
    questions: list[str]


def norm(t):
    return re.sub(r"[\s·\-()（）\[\]_/&]", "", t).replace("신한은행", "").replace("신한", "").lower()


def gen():
    if OUT.exists():
        sys.exit("이미 있음 — 잠근 확인 세트라 다시 만들지 않음")
    prods = list(csv.DictReader(open(META / "products.csv")))
    uniq = {}
    for p in prods:
        uniq.setdefault(p["name"], p)
    used = {json.loads(l)["name"] for l in open(EVAL / "alias_check.jsonl")}
    pid2name = {p["product_id"]: p["name"] for p in prods}
    used |= {pid2name.get(json.loads(l)["product_id"]) for l in open(EVAL / "retrieval_test2.jsonl")}
    rnd = random.Random(SEED)
    pick = rnd.sample(sorted(n for n in uniq if n not in used), 60)
    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)

    def one(name):
        r = client.responses.parse(model=env["CHAT_MODEL"], instructions=NICK_PROMPT, text_format=Q, input=f"상품: {name}")
        return {"kind": "nick", "name": name, "product_id": uniq[name]["product_id"], **r.output_parsed.model_dump()}

    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(one, pick))
    real = set(uniq) | {r["name"] for r in csv.DictReader(open(META / "discontinued.csv"))}
    real_n = {norm(n) for n in real}
    r = client.responses.parse(model=env["CHAT_MODEL"], instructions=FAKE_PROMPT, text_format=FakeList,
                               input="40개 만들어라.\n\n[실제 상품 이름]\n" + "\n".join(sorted(real)))
    nf = 0
    for f in r.output_parsed.items:
        fn = norm(f.name)
        if nf >= 30 or not fn or fn in real_n or any(fn in x or (len(x) >= 4 and x in fn) for x in real_n) or f.name not in f.question:
            continue
        rows.append({"kind": "fake", "name": f.name, "product_id": None, "nickname": f.name, "question": f.question})
        nf += 1
    g = client.responses.parse(model=env["CHAT_MODEL"], instructions=GEN_PROMPT, text_format=GenList, input="20개 만들어라.")
    for qq in g.output_parsed.questions[:20]:
        rows.append({"kind": "general", "name": None, "product_id": None, "nickname": None, "question": qq})
    with open(OUT, "w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    from collections import Counter
    print(OUT.name, Counter(x["kind"] for x in rows), "sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())


def judge(rows, detect, names):
    from collections import Counter
    c = Counter()
    bad = []
    for r in rows:
        found = {names[p] for p in detect(r["question"])}
        if r["kind"] == "nick":
            k = "못 찾음" if not found else "맞음" if r["name"] in found else "틀림"
        else:
            k = "상품 붙임" if found else "안 붙임"
        c[(r["kind"], k)] += 1
        if k in ("틀림", "못 찾음", "상품 붙임"):
            bad.append((r["kind"], k, r.get("nickname") or r["question"][:30], r["name"], sorted(found)[:3]))
    return c, bad


def dev_rows():
    """고치기용: alias_check 60(7-2에서 한 번 채점해 확인용 역할 끝) + 가짜 상품(8-1 후보 중 absent test에 안 들어간 것) + 검색 dev의 공통 조항·용어 질문"""
    rows = [{"kind": "nick", **json.loads(l)} for l in open(EVAL / "alias_check.jsonl")]
    test_q = {json.loads(l)["question"] for l in open(EVAL / "absent_test.jsonl")}
    for l in open(EVAL / "absent_candidates.jsonl"):
        c = json.loads(l)
        if c["type"] == "C-fake" and c["question"] not in test_q:
            rows.append({"kind": "fake", "name": c["product_name"], "product_id": None, "nickname": c["product_name"], "question": c["question"]})
    for l in open(EVAL / "retrieval_dev.jsonl"):
        q = json.loads(l)
        if q["type"] in ("B-clause", "B-term") and not q.get("product_id"):
            rows.append({"kind": "general", "name": None, "product_id": None, "nickname": None, "question": q["question"]})
    return rows


def score(final):
    import s12_retrieval as R
    method = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--method=")), "fallback")
    if final:
        if "--final" not in sys.argv:
            sys.exit("잠근 확인 세트 — 다 고친 뒤 --final을 붙여 한 번만")
        rows = [json.loads(l) for l in open(OUT)]
    else:
        rows = dev_rows()
    names = {p["product_id"]: p["name"] for p in csv.DictReader(open(META / "products.csv"))}
    idx = R.Index()
    detect = {"fallback": idx.detect_products, "products": idx.detect_products}.get(method) or getattr(idx, f"detect_{method}")   # exact · v91 (9-2)
    c, bad = judge(rows, detect, names)
    print(f"[{'잠금 세트' if final else 'dev'} · {method}]")
    for kind, ks in (("nick", ("맞음", "틀림", "못 찾음")), ("fake", ("상품 붙임", "안 붙임")), ("general", ("상품 붙임", "안 붙임"))):
        print(f"   {kind:8s} " + " · ".join(f"{k} {c[(kind, k)]}" for k in ks))
    for b in bad:
        print("   ", b)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "gen":
        gen()
    else:
        score(final=(cmd == "score"))
