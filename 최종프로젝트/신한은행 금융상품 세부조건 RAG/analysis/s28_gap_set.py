"""11-4 빈틈 확인 세트 — 판매 중지 상품 줄임말 · 실제 이름을 통째로 품은 가짜 상품 (dev/test, test 잠금)

사용: python analysis/s28_gap_set.py   → data/eval/gap_{dev,test}.jsonl (한 번만, 있으면 멈춤)
      채점: python analysis/s27_status_compare_eval.py dev --version=v7 --only=<gap qid>   (s27이 gap 파일도 읽음)
왜: 11-2·11-3 이후 남은 빈틈 두 가지는 아직 잰 적이 없음
  disc_nick 30: 판매 중지 목록에만 있는 상품을 직원 줄임말로 부른 질문 (s18과 같은 줄임말 지시문, 정식 이름 그대로면 버림)
               — 위험: 목록 대조는 이름 그대로일 때만 맞추므로, 상품명 찾기가 비슷한 **판매 중** 상품으로 추정해 그 정보로 답함
  fake_wrap 30: 판매 중 상품 이름 전체 + 지어낸 말 (s24와 같은 지시문) — 위험: 1단계가 실제 상품으로 바로 확정해 표시 없이 그 상품 정보로 답함
  (seed 20261120, 기존 판매 상태·상품명 확인 세트에 쓴 이름 제외, 이름 단위로 dev/test 반씩)
"""
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from s8_extract import load_env  # noqa: E402
from s18_alias_check import PROMPT as NICK_PROMPT, Q  # noqa: E402
from s22_alias_check2 import norm as fnorm  # noqa: E402
from s23_alias_check3 import FakeList  # noqa: E402
from s24_alias_check4 import WRAP_PROMPT  # noqa: E402

META, EVAL = ROOT / "data" / "meta", ROOT / "data" / "eval"
SEED = 20261120
OUT = {s: EVAL / f"gap_{s}.jsonl" for s in ("dev", "test")}


def main():
    if any(p.exists() for p in OUT.values()):
        sys.exit("이미 있음 — 잠근 세트라 다시 만들지 않음")
    rnd = random.Random(SEED)
    onsale = sorted({p["name"] for p in csv.DictReader(open(META / "products.csv"))})
    disc = list(csv.DictReader(open(META / "discontinued.csv")))
    used = set()
    for f in ("status_dev", "status_test", "alias_check", "alias_check2", "alias_check3", "alias_check4"):
        for l in open(EVAL / f"{f}.jsonl"):
            r = json.loads(l)
            used.add(r.get("name"))
            used.add(r.get("word"))
    d_only = sorted({r["name"] for r in disc if r["category"] in ("저축상품", "대출상품") and r["also_on_sale"] == "0"
                     and r["name"] not in onsale and r["name"] not in used})
    rnd.shuffle(d_only)

    def split_of(name):
        return "test" if int(hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest(), 16) % 2 else "dev"

    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)

    def one(name):
        r = client.responses.parse(model=env["CHAT_MODEL"], instructions=NICK_PROMPT, text_format=Q, input=f"상품: {name}")
        return name, r.output_parsed

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, d_only[:60]))
    out, have = {"dev": [], "test": []}, Counter()
    for name, q in res:
        sp = split_of(name)
        if have[("disc_nick", sp)] >= 15 or not q:
            continue
        if fnorm(q.nickname) == fnorm(name) or fnorm(name) in fnorm(q.question) or q.nickname not in q.question:
            continue
        out[sp].append({"type": "disc_nick", "question": q.question, "name": name, "nickname": q.nickname, "product_id": None})
        have[("disc_nick", sp)] += 1
    # 가짜: 판매 중 상품 이름 전체 + 지어낸 말
    real = set(onsale) | {r["name"] for r in disc}
    real_n = {fnorm(n) for n in real}
    src = [n for n in onsale if n not in used]
    rnd.shuffle(src)
    items = []
    for _ in range(3):
        r = client.responses.parse(model=env["CHAT_MODEL"], instructions=WRAP_PROMPT, text_format=FakeList,
                                   input="50개 만들어라. 품는 상품은 서로 다르게.\n\n[실제 상품 이름]\n" + "\n".join(src))
        items += r.output_parsed.items
        if len(items) >= 70:
            break
    seen = set()
    import s12_retrieval as R
    for f in items:
        fn = fnorm(f.name)
        sp = split_of(f.name)
        if fn in seen or have[("fake_wrap", sp)] >= 15 or not fn or fn in real_n or f.name not in f.question:
            continue
        if any(len(fn) >= 4 and fn in x for x in real_n) or R.norm(f.word) not in R.norm(f.name):
            continue
        seen.add(fn)
        out[sp].append({"type": "fake_wrap", "question": f.question, "name": f.name, "word": f.word, "product_id": None})
        have[("fake_wrap", sp)] += 1
    for sp, p in OUT.items():
        with open(p, "w") as fo:
            for n, r in enumerate(out[sp]):
                fo.write(json.dumps({"qid": f"G{sp[0].upper()}{n:03d}", **r, "split": sp}, ensure_ascii=False) + "\n")
        print(p.name, Counter(r["type"] for r in out[sp]))
    print("test sha256", hashlib.sha256(OUT["test"].read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
