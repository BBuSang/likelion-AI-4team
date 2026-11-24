"""11-6 정식 이름 전체를 품은 가짜 상품 확인 세트 (잠금, 구현 전에 만듦)

사용: python analysis/s29_fullname_fake_set.py   → data/eval/gap2_test.jsonl (한 번만, 있으면 멈춤)
      채점: python analysis/s27_status_compare_eval.py gap2 --final --version=... (s27이 gap2 파일을 읽음)
왜: 11-4에서 실제 **정식 이름 전체**를 품은 가짜 상품('꽃온 Npay biz 신한통장')은 1단계가 바로 확정해 표시 없이 실제 상품 정보로 답함(13/40)
구성 (seed 20261121, OpenAI CHAT_MODEL, 기존 상품명·빈틈 세트에 쓴 이름 제외)
  fake_full 30: 판매 중 상품의 정식 이름 **전체**(신한 머리말 포함 가능) 앞에 지어낸 말을 붙이거나, 뒤에 띄어쓰기 없이 지어낸 말을 붙인 가짜 상품
  formal_ctx 20: (예전 세트에 쓴 상품도 됨) 실제 상품 정식 이름 앞에 평범한 말이 오는 진짜 질문 ("고객이 묻는데 신한 생계비계좌는…") — 새 신호가 진짜 질문을 막지 않는지
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
from s22_alias_check2 import norm as fnorm  # noqa: E402

META, EVAL = ROOT / "data" / "meta", ROOT / "data" / "eval"
OUT = EVAL / "gap2_test.jsonl"
SEED = 20261121
FAKE_PROMPT = """너는 은행 평가 질문을 만드는 사람이다. 주어진 [실제 상품 이름]을 **글자 하나 바꾸지 않고 통째로** 품은 **가짜 상품 이름**을 만들고, 그 가짜 상품에 대한 영업점 직원의 짧은 질문을 만든다.
- 방식 {mode}
- 붙이는 말은 실제 신한 상품 이름에 쓰이지 않는 지어낸 말(브랜드·대상·용도)로
- 질문 문장에 가짜 상품 이름을 글자 그대로 넣는다. 질문 내용은 금리·가입대상·한도·해지 등"""
MODES = {"pre": "앞: 실제 이름 앞에 띄어 쓴 지어낸 말 하나 (예: '꽃온 Npay biz 신한통장')",
         "suf": "뒤: 실제 이름 바로 뒤에 띄어쓰기 없이 지어낸 말 (예: '신한 청년 처음적금플랜')"}
CTX_PROMPT = """너는 신한은행 영업점 직원이다. 주어진 상품의 **정식 이름을 그대로** 넣되, 이름 **앞에** 평범한 말(예: '고객이 묻는데', '혹시', '어제 상담한', '우리 지점에서 많이 파는')을 넣은 짧은 질문 하나를 만든다.
- 상품 이름은 글자 그대로. 질문 내용은 금리·가입대상·해지 등"""


class FQ(BaseModel):
    fake_name: str
    question: str


class CQ(BaseModel):
    question: str


def main():
    if OUT.exists():
        sys.exit("이미 있음 — 잠근 세트라 다시 만들지 않음")
    rnd = random.Random(SEED)
    prods = sorted({p["name"] for p in csv.DictReader(open(META / "products.csv"))})
    real = set(prods) | {r["name"] for r in csv.DictReader(open(META / "discontinued.csv"))}
    real_n = {fnorm(n) for n in real}
    used = set()
    for f in EVAL.glob("*.jsonl"):
        if f.name.startswith(("alias_check", "gap", "status_", "compare_")):
            for l in open(f):
                r = json.loads(l)
                for k in ("name", "word"):
                    used.add(r.get(k))
    pool = [n for n in prods if n not in used and len(fnorm(n)) >= 5]
    rnd.shuffle(pool)
    fake_src = pool[:45]
    # 대조군은 질문 형태(이름 앞에 평범한 말)를 보는 것이라 예전 세트에 쓴 상품도 됨 — 가짜에 쓴 상품만 뺌 (안 쓴 상품이 49개뿐)
    ctx_pool = [n for n in prods if n not in fake_src and len(fnorm(n)) >= 5]
    rnd.shuffle(ctx_pool)
    ctx_src = ctx_pool[:60]
    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)

    def f_one(arg):
        i, name = arg
        mode = "pre" if i % 2 == 0 else "suf"
        r = client.responses.parse(model=env["CHAT_MODEL"], instructions=FAKE_PROMPT.format(mode=MODES[mode]), text_format=FQ,
                                   input=f"[실제 상품 이름] {name}")
        return name, mode, r.output_parsed

    def c_one(name):
        r = client.responses.parse(model=env["CHAT_MODEL"], instructions=CTX_PROMPT, text_format=CQ, input=f"상품: {name}")
        return name, r.output_parsed

    with ThreadPoolExecutor(6) as ex:
        fres = list(ex.map(f_one, enumerate(fake_src)))
        cres = list(ex.map(c_one, ctx_src))
    rows, nf, nc = [], 0, 0
    for name, mode, q in fres:
        if nf >= 30 or not q:
            continue
        fn = fnorm(q.fake_name)
        if name not in q.fake_name or q.fake_name not in q.question or fn in real_n or fn == fnorm(name):
            continue
        rows.append({"type": "fake_full", "question": q.question, "name": q.fake_name, "word": name, "mode": mode, "product_id": None})
        nf += 1
    for name, q in cres:
        if nc >= 20 or not q or name not in q.question or q.question.startswith(name):
            continue
        rows.append({"type": "formal_ctx", "question": q.question, "name": name, "product_id": None})
        nc += 1
    with open(OUT, "w") as fo:
        for n, r in enumerate(rows):
            fo.write(json.dumps({"qid": f"F{n:03d}", **r, "split": "gap2"}, ensure_ascii=False) + "\n")
    print(OUT.name, Counter(r["type"] for r in rows), Counter(r.get("mode") for r in rows if r["type"] == "fake_full"),
          "sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
