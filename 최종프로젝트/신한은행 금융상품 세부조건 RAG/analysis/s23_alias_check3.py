"""9-2 상품명 찾기 확인 세트 3 (잠금) — 1단계 별칭 오탐 개선을 공정하게 재려고 **구현 전에** 만듦

사용: python analysis/s23_alias_check3.py gen                         → data/eval/alias_check3.jsonl (한 번만, 해시로 잠금)
      python analysis/s23_alias_check3.py score --final [--method=...]   (다 고친 뒤 한 번만)
왜: 9-1 잠금 세트(alias_check2)에서 1단계 별칭의 오탐(틀림 4 · 가짜 1)을 찾음 → 그 세트로는 개선 효과를 잴 수 없음
구성 (seed 20261116, OpenAI CHAT_MODEL, 기존 alias_check·alias_check2에 쓴 상품·질문은 제외 (test2는 상품 사실 세트라 상품은 겹칠 수 있음, 질문은 새로))
  nick_sim   ≤40: 이름이 비슷한 다른 상품이 있는 상품(한 상품 이름의 별칭이 다른 상품 이름 안에 들어 있음) — s18과 같은 줄임말 지시문
  nick_other 20: 그 밖의 상품 — 같은 지시문
  formal     20: 정식 이름 그대로 넣은 질문
  fake       30: 실제 별칭 단어(4글자 이상)를 품은 가짜 상품 이름 — 실제 이름과 겹치지 않는 것만
  general    20: 상품 이름 없는 공통 규정·용어 질문 (alias_check2의 공통 질문과 겹치지 않게)
판정: nick·formal 맞음(찾은 상품 이름에 정답) / 틀림 / 못 찾음, fake·general은 상품을 골랐는지
"""
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from s8_extract import load_env  # noqa: E402
from s18_alias_check import PROMPT as NICK_PROMPT, Q  # noqa: E402
from s22_alias_check2 import GEN_PROMPT, GenList, norm as fnorm  # noqa: E402

EVAL, META = ROOT / "data" / "eval", ROOT / "data" / "meta"
OUT = EVAL / "alias_check3.jsonl"
SEED = 20261116
FORMAL_PROMPT = """너는 신한은행 영업점 직원이다. 주어진 상품의 **정식 이름을 그대로** 넣어 동료에게 물을 짧은 질문 하나를 만든다.
- 질문 내용은 금리·가입대상·해지·한도 등 아무거나. nickname에는 질문 속 상품 이름을 그대로 적는다"""
FAKE_PROMPT = """너는 은행 평가 질문을 만드는 사람이다. **신한은행에 실제로는 없는 가짜 상품 이름**을 넣은 영업점 직원 질문을 만든다.
- 각 가짜 이름은 아래 [쓸 단어] 중 하나를 **그대로 포함**하되, 앞뒤에 다른 말(브랜드·대상·용도)을 붙여 실제 상품과 다른 상품이 되게 한다 (예: 단어 '소상공인' → '하모니 소상공인통장')
- [실제 상품 이름]과 같거나 거의 같으면 안 됨. 질문 내용은 금리·가입대상·한도·해지 등 짧게
- name에는 질문 속 가짜 상품 이름만"""


class Fake(BaseModel):
    name: str
    word: str
    question: str


class FakeList(BaseModel):
    items: list[Fake]


def gen():
    if OUT.exists():
        sys.exit("이미 있음 — 잠근 확인 세트라 다시 만들지 않음")
    import s12_retrieval as R
    prods = list(csv.DictReader(open(META / "products.csv")))
    uniq = {}
    for p in prods:
        uniq.setdefault(p["name"], p)
    pid2name = {p["product_id"]: p["name"] for p in prods}
    used = {json.loads(l)["name"] for l in open(EVAL / "alias_check.jsonl")}
    used |= {json.loads(l)["name"] for l in open(EVAL / "alias_check2.jsonl") if json.loads(l)["kind"] == "nick"}
    old_q = {json.loads(l)["question"] for f in ("alias_check.jsonl", "alias_check2.jsonl") for l in open(EVAL / f)}
    # 이름이 비슷한 다른 상품이 있는 상품: 이 상품의 별칭(4글자 이상)이 다른 상품 이름 안에 들어 있거나, 다른 상품의 별칭이 이 이름 안에 들어 있음
    idx = R.Index()
    alias_of = {}
    for a, ps in idx.alias.items():
        for x in ps:
            alias_of.setdefault(pid2name[x], set()).add(a)
    nn = {n: R.norm(n) for n in uniq}
    sim = set()
    for n, als in alias_of.items():
        for a in als:
            owners = {pid2name[x] for x in idx.alias[a]}
            if len(a) >= 4 and any(a in nn[m] for m in uniq if m not in owners):   # 이 별칭을 이름에 품은 '다른' 상품이 있음
                sim.add(n)
    rnd = random.Random(SEED)
    free = sorted(n for n in uniq if n not in used)
    pool_sim = sorted(n for n in free if n in sim)
    pick_sim = rnd.sample(pool_sim, min(40, len(pool_sim)))   # 상품명 확인 세트에 쓴 상품만 빼면 남는 수만큼 (9-2)
    rest = [n for n in free if n not in pick_sim]
    pick_other = rnd.sample(sorted(n for n in rest if n not in sim), 20)
    rest = [n for n in rest if n not in pick_other]
    pick_formal = rnd.sample(sorted(rest), 20)

    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)

    def one(arg):
        kind, name = arg
        instr = FORMAL_PROMPT if kind == "formal" else NICK_PROMPT
        r = client.responses.parse(model=env["CHAT_MODEL"], instructions=instr, text_format=Q, input=f"상품: {name}")
        return {"kind": kind, "name": name, "product_id": uniq[name]["product_id"], **r.output_parsed.model_dump()}

    jobs = [("nick_sim", n) for n in pick_sim] + [("nick_other", n) for n in pick_other] + [("formal", n) for n in pick_formal]
    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(one, jobs))
    # 가짜: 실제 별칭 단어(4글자 이상, 상품 종류 일반어만으로 된 것 제외)를 품은 이름
    words = sorted({a for a in idx.alias if 4 <= len(a) <= 7 and a not in {R.norm(w) for w in R.GENERIC_WORDS}})
    rnd.shuffle(words)
    real = set(uniq) | {r["name"] for r in csv.DictReader(open(META / "discontinued.csv"))}
    real_n = {fnorm(n) for n in real}
    r = client.responses.parse(model=env["CHAT_MODEL"], instructions=FAKE_PROMPT, text_format=FakeList,
                               input="45개 만들어라. 단어는 서로 다르게.\n\n[쓸 단어]\n" + "\n".join(words[:80])
                               + "\n\n[실제 상품 이름]\n" + "\n".join(sorted(real)))
    nf = 0
    for f in r.output_parsed.items:
        fn = fnorm(f.name)
        if nf >= 30 or not fn or fn in real_n or f.name not in f.question or R.norm(f.word) not in R.norm(f.name):
            continue
        if any(fn == x or (len(fn) >= 4 and fn in x) for x in real_n):   # 실제 이름의 일부이면 가짜가 아님
            continue
        rows.append({"kind": "fake", "name": f.name, "product_id": None, "nickname": f.name, "question": f.question, "word": f.word})
        nf += 1
    g = client.responses.parse(model=env["CHAT_MODEL"], instructions=GEN_PROMPT + "\n- 아래 [이미 쓴 질문]과 다른 주제·표현으로",
                               text_format=GenList, input="25개 만들어라.\n\n[이미 쓴 질문]\n" + "\n".join(sorted(old_q)))
    ng = 0
    for qq in g.output_parsed.questions:
        if ng < 20 and qq not in old_q:
            rows.append({"kind": "general", "name": None, "product_id": None, "nickname": None, "question": qq})
            ng += 1
    rows = [x for x in rows if x["question"] not in old_q]
    with open(OUT, "w") as fo:
        for x in rows:
            fo.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(OUT.name, Counter(x["kind"] for x in rows), "sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())


def score():
    if "--final" not in sys.argv:
        sys.exit("잠근 확인 세트 — 다 고친 뒤 --final을 붙여 한 번만")
    import s12_retrieval as R
    from s22_alias_check2 import judge
    method = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--method=")), "current")
    rows = [json.loads(l) for l in open(OUT)]
    for r in rows:
        if r["kind"] in ("nick_sim", "nick_other", "formal"):
            r["_k"], r["kind"] = r["kind"], "nick"
        else:
            r["_k"] = r["kind"]
    names = {p["product_id"]: p["name"] for p in csv.DictReader(open(META / "products.csv"))}
    idx = R.Index()
    detect = {"current": idx.detect_products, "exact": idx.detect_exact}.get(method) or getattr(idx, f"detect_{method}")
    by = {}
    for r in rows:
        by.setdefault(r["_k"], []).append(r)
    print(f"[alias_check3 잠금 · {method}]")
    for k in ("nick_sim", "nick_other", "formal", "fake", "general"):
        c, bad = judge(by.get(k, []), detect, names)
        print(f"   {k:10s} " + " · ".join(f"{kk[1]} {v}" for kk, v in sorted(c.items())))
        for b in bad:
            print("      ", b)


if __name__ == "__main__":
    {"gen": gen, "score": score}[sys.argv[1]]()
