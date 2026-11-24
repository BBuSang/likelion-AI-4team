"""11-7 남은 빈틈 확인 세트 3 (잠금, 구현 전에 만듦)

사용: python analysis/s30_gap3_set.py   → data/eval/gap3_test.jsonl (한 번만, 있으면 멈춤)
      채점: python analysis/s27_status_compare_eval.py gap3_test --final --version=...
왜: 11-6 잠금 세트(gap2)에서 남은 빈틈 — ① 괄호로 끝나는 정식 이름 뒤에 붙은 말('…(주택도시보증)누리온')을 신호 ③이 놓침
    ② 이름이 두 군데서 맞으면 연결어가 없어도 '두 상품 비교'로 나뉘어 확인·추정 표시를 건너뜀 → gap2로는 효과를 잴 수 없어 새 세트
구성 (seed 20261122, OpenAI CHAT_MODEL)
  fake_paren 15: 괄호로 끝나는 판매 중 상품 정식 이름 바로 뒤에 지어낸 말을 붙인 가짜 (s29 지시문의 '뒤' 방식)
  fake_full  15: 정식 이름 앞에 지어낸 말을 띄어 붙인 가짜 (s29 지시문의 '앞' 방식)
  compare    10: 진짜 두 상품 비교 질문 'A와 B 중…' (s26 비교 세트와 같은 정답 형식은 아님 — 두 상품을 모두 찾는지만 봄)
  formal_ctx 10: 정식 이름 앞에 평범한 말이 오는 진짜 질문 (s29와 같은 지시문)
  가짜는 기존 상품명·빈틈 세트에 쓴 이름 제외, 대조군·비교는 가짜에 쓴 상품만 제외
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
from s29_fullname_fake_set import CTX_PROMPT, FAKE_PROMPT, MODES, CQ, FQ  # noqa: E402

META, EVAL = ROOT / "data" / "meta", ROOT / "data" / "eval"
OUT = EVAL / "gap3_test.jsonl"
SEED = 20261122
CMP_PROMPT = """너는 신한은행 영업점 직원이다. 주어진 **두 상품을 비교하는** 짧은 질문 하나를 만든다.
- 두 상품 이름을 글자 그대로 넣고, 그 사이를 '와/과', '하고', '이랑', '및', 쉼표 중 하나로 잇는다
- 비교 내용은 금리·가입대상·가입기간 등"""


def main():
    if OUT.exists():
        sys.exit("이미 있음 — 잠근 세트라 다시 만들지 않음")
    rnd = random.Random(SEED)
    prods = list(csv.DictReader(open(META / "products.csv")))
    names = sorted({p["name"] for p in prods})
    cat = {p["name"]: p["category"] for p in prods}
    real = set(names) | {r["name"] for r in csv.DictReader(open(META / "discontinued.csv"))}
    real_n = {fnorm(n) for n in real}
    used = set()
    for f in EVAL.glob("*.jsonl"):
        if f.name.startswith(("alias_check", "gap", "status_", "compare_")):
            for l in open(f):
                r = json.loads(l)
                for k in ("name", "word"):
                    used.add(r.get(k))
    paren = [n for n in names if n.rstrip().endswith(")") and n not in used]
    plain = [n for n in names if not n.rstrip().endswith(")") and n not in used and len(fnorm(n)) >= 5]
    rnd.shuffle(paren)
    rnd.shuffle(plain)
    if len(paren) < 20:   # 괄호 이름이 모자라면 예전 세트에 쓴 이름도 (가짜는 새로 만드는 이름이라 겹쳐도 됨)
        extra = [n for n in names if n.rstrip().endswith(")") and n not in paren]
        rnd.shuffle(extra)
        paren += extra
    f_src = [("suf", n) for n in paren[:24]] + [("pre", n) for n in plain[:24]]
    taken = {n for _, n in f_src}
    rest = [n for n in names if n not in taken and len(fnorm(n)) >= 5]
    rnd.shuffle(rest)
    ctx_src = rest[:30]
    by_cat = {}
    for n in rest[30:]:
        by_cat.setdefault(cat[n], []).append(n)
    pairs = []
    for c, ns in by_cat.items():
        for i in range(0, len(ns) - 1, 2):
            pairs.append((ns[i], ns[i + 1]))
    rnd.shuffle(pairs)
    pairs = pairs[:20]
    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)

    def f_one(arg):
        mode, name = arg
        r = client.responses.parse(model=env["CHAT_MODEL"], instructions=FAKE_PROMPT.format(mode=MODES[mode]), text_format=FQ, input=f"[실제 상품 이름] {name}")
        return mode, name, r.output_parsed

    def c_one(name):
        return name, client.responses.parse(model=env["CHAT_MODEL"], instructions=CTX_PROMPT, text_format=CQ, input=f"상품: {name}").output_parsed

    def p_one(pair):
        return pair, client.responses.parse(model=env["CHAT_MODEL"], instructions=CMP_PROMPT, text_format=CQ, input=f"상품 A: {pair[0]}\n상품 B: {pair[1]}").output_parsed

    with ThreadPoolExecutor(6) as ex:
        fres = list(ex.map(f_one, f_src))
        cres = list(ex.map(c_one, ctx_src))
        pres = list(ex.map(p_one, pairs))
    rows, cnt = [], Counter()
    for mode, name, q in fres:
        typ = "fake_paren" if mode == "suf" else "fake_full"
        if cnt[typ] >= 15 or not q:
            continue
        fn = fnorm(q.fake_name)
        if name not in q.fake_name or q.fake_name not in q.question or fn in real_n or fn == fnorm(name):
            continue
        if typ == "fake_paren" and not q.fake_name.startswith(name):
            continue
        rows.append({"type": typ, "question": q.question, "name": q.fake_name, "word": name, "product_id": None})
        cnt[typ] += 1
    for name, q in cres:
        if cnt["formal_ctx"] >= 10 or not q or name not in q.question or q.question.startswith(name):
            continue
        rows.append({"type": "formal_ctx", "question": q.question, "name": name, "product_id": None})
        cnt["formal_ctx"] += 1
    for (a, b), q in pres:
        if cnt["compare"] >= 10 or not q or a not in q.question or b not in q.question:
            continue
        rows.append({"type": "compare2", "question": q.question, "name": f"{a} / {b}", "products": [a, b], "product_id": None})
        cnt["compare"] += 1
    with open(OUT, "w") as fo:
        for n, r in enumerate(rows):
            fo.write(json.dumps({"qid": f"H{n:03d}", **r, "split": "gap3"}, ensure_ascii=False) + "\n")
    print(OUT.name, Counter(r["type"] for r in rows), "sha256", hashlib.sha256(OUT.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
