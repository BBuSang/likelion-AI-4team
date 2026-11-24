"""9-3 상품명 찾기 확인 세트 4 (잠금) — 가짜 상품 오탐 개선을 공정하게 재려고 **구현 전에** 만듦

사용: python analysis/s24_alias_check4.py gen                         → data/eval/alias_check4.jsonl (한 번만, 해시로 잠금)
      python analysis/s24_alias_check4.py score --final [--method=...]   (다 고친 뒤 한 번만)
왜: 9-2 잠금 세트(alias_check3)에서 '실제 상품 고유 이름 전체 + 지어낸 말' 가짜 상품에 실제 상품을 붙이는 문제(17/30)를 찾음
구성 (seed 20261117, OpenAI CHAT_MODEL, 상품은 alias_check 1~3에 쓰지 않은 것만)
  fake_wrap  40: 실제 상품의 고유한 이름 전체 + 지어낸 말 (예: '청춘 청년처음적금플랜') — 실제 이름과 같으면 버림
  fake_word  20: 실제 별칭 단어(4글자 이상)를 품은 가짜 이름 (9-2와 같은 방식)
  nick       20: s18과 같은 줄임말 지시문
  nick_abbr  20: 이름 속 말을 줄이거나 다른 표기로 (기관 줄임말·영어↔한글) — 새 원칙이 진짜 줄임말까지 막지 않는지
  formal     20: 정식 이름 그대로
  general    10: 상품 이름 없는 공통 질문 (기존 세트와 겹치지 않게)
"""
import csv
import hashlib
import json
import random
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
from s23_alias_check3 import FAKE_PROMPT, FORMAL_PROMPT, FakeList  # noqa: E402

EVAL, META = ROOT / "data" / "eval", ROOT / "data" / "meta"
OUT = EVAL / "alias_check4.jsonl"
SEED = 20261117
ABBR_PROMPT = """너는 신한은행 영업점 직원이다. 주어진 상품을 동료에게 물을 때, 상품 이름 속 말을 **줄이거나 다른 표기로 바꿔** 부르면서 질문 하나를 만든다.
- 예: '주택금융공사' → '주금공', '서울보증보험' → '서울보증', 'MY CAR' → '마이카', 'Value-up' → '밸류업', 'SOL' → '쏠'
- 이름에 없는 새 말을 덧붙이지 않는다. 직원들이 실제로 쓰는 수준으로만
- 질문 내용은 금리·가입대상·해지 등 짧게. nickname에는 질문 속 상품 호칭만"""
WRAP_PROMPT = """너는 은행 평가 질문을 만드는 사람이다. **신한은행에 실제로는 없는 가짜 상품 이름**을 넣은 영업점 직원 질문을 만든다.
- 각 가짜 이름은 아래 [실제 상품 이름] 중 하나의 고유한 부분('신한'·'쏠편한' 같은 머리말과 상품 종류 말을 뺀 핵심 이름)을 **그대로 품고**, 앞이나 뒤에 실제 이름에 없는 말(브랜드·대상·용도: 예 '청춘', '새집', '플랜', '케어', '플러스')을 붙여 다른 상품처럼 만든다
  (예: 실제 '청년 처음적금' → 가짜 '청춘 청년처음적금플랜')
- 핵심 이름은 **끊지 않고 통째로** 품는다 (사이에 말을 끼워 넣지 않음). 덧붙이는 말은 앞이나 뒤에만
- 붙이는 말은 실제 신한 상품 이름에 쓰이지 않는 말로. 질문 내용은 금리·가입대상·한도·해지 등 짧게
- **질문 문장에 가짜 상품 이름을 글자 그대로** 넣는다
- name에는 질문 속 가짜 상품 이름만, word에는 품은 핵심 이름(가짜 이름 안에 그대로 들어 있는 부분)만"""


def gen():
    if OUT.exists():
        sys.exit("이미 있음 — 잠근 확인 세트라 다시 만들지 않음")
    import s12_retrieval as R
    prods = list(csv.DictReader(open(META / "products.csv")))
    uniq = {}
    for p in prods:
        uniq.setdefault(p["name"], p)
    used, old_q = set(), set()
    for f in ("alias_check.jsonl", "alias_check2.jsonl", "alias_check3.jsonl"):
        for l in open(EVAL / f):
            r = json.loads(l)
            old_q.add(r["question"])
            if r.get("name") and r.get("product_id"):
                used.add(r["name"])
    rnd = random.Random(SEED)
    free = sorted(n for n in uniq if n not in used)
    rnd.shuffle(free)
    pick = {"nick": free[:20], "nick_abbr": free[20:40], "formal": free[40:60]}
    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
    instr = {"nick": NICK_PROMPT, "nick_abbr": ABBR_PROMPT, "formal": FORMAL_PROMPT}

    def one(arg):
        kind, name = arg
        r = client.responses.parse(model=env["CHAT_MODEL"], instructions=instr[kind], text_format=Q, input=f"상품: {name}")
        return {"kind": kind, "name": name, "product_id": uniq[name]["product_id"], **r.output_parsed.model_dump()}

    with ThreadPoolExecutor(6) as ex:
        rows = list(ex.map(one, [(k, n) for k, ns in pick.items() for n in ns]))
    real = set(uniq) | {r["name"] for r in csv.DictReader(open(META / "discontinued.csv"))}
    real_n = {fnorm(n) for n in real}

    def fakes(prompt, inp, n_keep, kind, need_word):
        """정한 개수가 찰 때까지 최대 3번 (LLM 출력이 매번 달라 한 번에 개수가 들쑥날쑥함)"""
        items = []
        for _ in range(3):
            r = client.responses.parse(model=env["CHAT_MODEL"], instructions=prompt, text_format=FakeList, input=inp)
            items += r.output_parsed.items
            if len(keep(items, n_keep, kind, need_word)) >= n_keep:
                break
        return keep(items, n_keep, kind, need_word)

    def keep(items, n_keep, kind, need_word):
        out, seen = [], set()
        for f in items:
            if fnorm(f.name) in seen:
                continue
            seen.add(fnorm(f.name))
            fn = fnorm(f.name)
            if len(out) >= n_keep or not fn or fn in real_n or f.name not in f.question:
                continue
            if any(len(fn) >= 4 and fn in x for x in real_n):          # 실제 이름의 일부이면 가짜가 아님
                continue
            if need_word and R.norm(f.word) not in R.norm(f.name):
                continue
            out.append({"kind": kind, "name": f.name, "product_id": None, "nickname": f.name, "question": f.question, "word": f.word})
        return out

    wrap_src = sorted(uniq)
    rnd.shuffle(wrap_src)
    rows += fakes(WRAP_PROMPT, "60개 만들어라. 품는 상품은 서로 다르게.\n\n[실제 상품 이름]\n" + "\n".join(wrap_src), 40, "fake_wrap", True)
    words = sorted({a for a in R.Index().alias if 4 <= len(a) <= 7 and a not in {R.norm(w) for w in R.GENERIC_WORDS}})
    rnd.shuffle(words)
    rows += fakes(FAKE_PROMPT, "30개 만들어라. 단어는 서로 다르게.\n\n[쓸 단어]\n" + "\n".join(words[:60]) + "\n\n[실제 상품 이름]\n" + "\n".join(sorted(real)),
                  20, "fake_word", True)
    g = client.responses.parse(model=env["CHAT_MODEL"], instructions=GEN_PROMPT + "\n- 아래 [이미 쓴 질문]과 다른 주제·표현으로",
                               text_format=GenList, input="15개 만들어라.\n\n[이미 쓴 질문]\n" + "\n".join(sorted(old_q)))
    rows += [{"kind": "general", "name": None, "product_id": None, "nickname": None, "question": q}
             for q in g.output_parsed.questions if q not in old_q][:10]
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
    method = next((a.split("=", 1)[1] for a in sys.argv[2:] if a.startswith("--method=")), "products")
    rows = [json.loads(l) for l in open(OUT)]
    for r in rows:
        r["_k"] = r["kind"]
        r["kind"] = "nick" if r["kind"] in ("nick", "nick_abbr", "formal") else "fake" if r["kind"].startswith("fake") else "general"
    names = {p["product_id"]: p["name"] for p in csv.DictReader(open(META / "products.csv"))}
    idx = R.Index()
    detect = getattr(idx, f"detect_{method}")
    by = {}
    for r in rows:
        by.setdefault(r["_k"], []).append(r)
    print(f"[alias_check4 잠금 · {method}]")
    for k in ("nick", "nick_abbr", "formal", "fake_wrap", "fake_word", "general"):
        c, bad = judge(by.get(k, []), detect, names)
        print(f"   {k:10s} " + " · ".join(f"{kk[1]} {v}" for kk, v in sorted(c.items())))
        for b in bad:
            print("      ", b)


if __name__ == "__main__":
    {"gen": gen, "score": score}[sys.argv[1]]()
