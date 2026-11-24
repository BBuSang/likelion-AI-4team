"""8-1 '없는 값' 답변 평가 세트 — 답이 보유 문서에 없을 때 지어내지 않는지 (답변원칙 5번)

사용:
  python analysis/s20_absent_set.py gen       → data/eval/absent_candidates.jsonl (후보, 유형마다 넉넉히)
  python analysis/s10_chunks.py embed-questions-local data/eval/absent_candidates.jsonl
  python analysis/s20_absent_set.py verify    → 정말 없는지 LLM 확인 → data/eval/absent_{dev,test}.jsonl (한 번만, 있으면 멈춤)
왜: 지금까지 답변 평가는 '답이 문서에 있는 질문'만 — 없을 때 숫자를 지어내면 직원이 고객에게 틀린 값을 안내하게 됨
유형 (seed 20261114)
  A-own    상품 전용 문서(설명서·특약)가 있는 상품: 그 문서 전체를 보여 주고 '이 문서에 없는 구체 값'을 묻는 질문
  A-common 공통 문서(대출 공통 설명서·여신 약관)만 연결된 대출 상품: 상품별 한도·기간·대상·서류처럼 공통 문서로는 정할 수 없는 값
           (고시 금리가 있는 상품(rate_product_map)은 금리를 묻지 않음 — 금리는 고시 금리(DB) 경로로 답할 수 있음)
  B-example 공통 대출 설명서의 '예시 숫자·빈칸' 항목(중도상환해약금률 1.4% 예시 등)을 그 상품의 실제 값으로 묻는 질문 = 함정
  C-fake   신한에 없는 지어낸 상품 이름 (실제 판매 중·판매 중지 이름과 겹치지 않고, 상품명 찾기에도 안 걸리는 것만)
'정말 없는지' 확인: 질문 상품에 연결된 문서·공통 조항 안에서 키워드+벡터 결합 상위 20개를 CHAT_MODEL이 읽고
  '이 상품의 구체 값으로 답할 수 있는 조각'이 하나라도 있으면 버림 (예시·빈칸·다른 상품 값은 답이 아님). 이후 사람이 전부 읽음
"""
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from s8_extract import load_env  # noqa: E402

META, EVAL = ROOT / "data" / "meta", ROOT / "data" / "eval"
CAND = EVAL / "absent_candidates.jsonl"
LOG = EVAL / "absent_verify.jsonl"
SEED = 20261114
N_GEN = {"A-own": 22, "A-common": 22, "B-example": 40, "C-fake": 24}   # 후보 (버릴 것 대비)
N_KEEP = {"A-own": 15, "A-common": 15, "B-example": 15, "C-fake": 15}
TRAP = {"D038": [8, 29, 32, 38], "D059": [9, 31, 34, 40]}      # 계산 '예시' 숫자가 있는 항목 (8-1 검수: 빈칸만 있는 항목은 뺌)
# 8-1 사람 검수로 뺀 후보 (verify에서 제외)
REVIEW_DROP = {"C007": "비과세종합저축 한도는 법정 상식(문서 없이도 맞힐 수 있음)",
               "C001": "고시 금리가 있는 상품의 금리 질문 — 고시 금리(DB) 경로로 답할 수 있음",
               "C012": "교부 권종(100·300·500만원)이 문서에 있어 사실상 답이 있음",
               **{c: "고시 금리가 있는 상품의 금리 질문 (B)" for c in ("C121", "C123", "C127", "C133")}}


def clean_q(t):
    """B 질문의 '실제로'·'실제' — 함정이 있다는 힌트가 되고 직원 말투도 아님 (8-1 검수)"""
    return re.sub(r"\s+", " ", re.sub(r"실제로\s*|실제\s+", "", t)).strip()

GEN_A = """너는 신한은행 영업점 직원이다. 아래는 상품 하나에 연결된 문서 전체(또는 목차)다.
고객 응대 중에 동료나 상품 안내 챗봇에게 물어볼 법한 질문을 하나 만들되, **이 문서들에는 답이 없는 구체적인 값**을 묻는다.
- 직원이 실제로 물을 만한 자연스러운 것 (예: 상품별 대출 한도, 최대 대출 기간, 특정 수수료 금액, 판매 한도, 가입 가능 나이, 우대금리 최대치 등)
- 문서에 그 값이 **적혀 있지 않아야** 한다. 비슷한 항목이 있지만 값이 빈칸·'별도 고시'·예시뿐인 경우는 좋다
- 은행 상품과 상관없는 질문, 누구나 아는 일반 상식 질문은 안 됨
- 질문에 상품 이름을 넣는다 (직원이 부르듯 자연스럽게, 다만 알아볼 수 있게)
- absent_what: 질문이 묻는 값이 무엇인지 짧게. near_miss: 문서에 있는 비슷하지만 답이 아닌 내용(있으면, 입력에서 그대로 짧게 복사), 없으면 빈 문자열
JSON으로만 답한다."""
NO_RATE = "\n- **금리(이자율)는 묻지 않는다** (이 상품은 고시 금리가 따로 있음)."
GEN_B = """너는 신한은행 영업점 직원이다. 아래는 대출 공통 상품설명서의 한 항목이다. 여기에는 계산을 보여 주려고 넣은 **예시 숫자**(예: '중도상환해약금률 1.4%', '1억원을 5년, 연 5%로')가 있다.
그 예시 숫자가 가리키는 항목에 대해 [상품 이름]의 **실제 값**을 묻는 자연스러운 질문을 하나 만든다 (예: "○○대출 중도상환해약금률은 몇 %예요?").
- 예시 숫자를 실제 값으로 착각하기 쉬운 질문이어야 한다. 설명서가 모든 상품에 공통으로 정한 실제 규칙(예: 연체가산이자율)을 묻는 질문은 안 됨
- absent_what: 질문이 묻는 값. near_miss: 설명서의 예시 숫자 문장을 입력에서 그대로 짧게 복사 (답변이 이걸 실제 값처럼 말하면 함정에 걸린 것)
- 이 항목으로 그런 질문을 만들 수 없으면 usable=false
JSON으로만 답한다."""
GEN_C = """너는 은행 상품 평가 질문을 만드는 사람이다. **신한은행에 실제로는 없는, 그럴듯한 가짜 상품 이름**과 그 상품에 대해 영업점 직원이 물을 법한 질문을 만든다.
- 이름은 실제 신한은행 상품 이름처럼 보이게 (예: '신한 ○○ 적금', '쏠편한 ○○ 대출', '○○ 외화 정기예금') — 단 아래 [실제 상품 이름]과 핵심 단어가 겹치면 안 됨
- 분류를 섞는다: 적금·예금·입출금·대출(가계·기업)·외화
- 질문은 금리·한도·가입 대상·기간·우대 조건 같은 구체적인 값. **질문 문장에 그 가짜 상품 이름을 그대로 넣는다** (예: "신한 ○○ 적금 12개월 금리가 몇 %예요?"), 한 문장으로 짧게
JSON으로만 답한다."""


class GenA(BaseModel):
    usable: bool
    question: str
    absent_what: str
    near_miss: str


class Fake(BaseModel):
    name: str
    category: str
    question: str
    absent_what: str


class FakeList(BaseModel):
    items: list[Fake]


class Check(BaseModel):
    answerable: bool
    ids: list[int]
    note: str


CHECK = """너는 평가 질문 검수자다. [질문]은 '보유 문서에 답이 없는 질문'으로 만들어졌다. 아래 [후보] 문서 조각들을 읽고,
**이 조각들만으로 질문한 상품의 구체적인 값(질문이 묻는 것)을 답할 수 있는지** 판단한다.
- answerable=true: 어느 조각에 질문 상품에 적용되는 그 값이 실제로 적혀 있음 (공통 약관이라도 상품에 그대로 적용되는 값이면 true)
- answerable=false: 값이 없음 / 빈칸 / '별도 고시·영업점 문의' / 계산 예시의 숫자뿐 / 다른 상품의 값뿐
- answerable=true면 ids에 그 조각 번호. note에 한두 문장 이유"""


def norm(t):
    return re.sub(r"[\s·\-()（）\[\]_/]", "", t).replace("신한은행", "").replace("신한", "").lower()


def load_meta():
    prods = {r["product_id"]: r for r in csv.DictReader(open(META / "products.csv"))}
    docs = {r["doc_id"]: r for r in csv.DictReader(open(META / "documents.csv"))}
    links = defaultdict(list)
    for r in csv.DictReader(open(META / "product_docs.csv")):
        links[r["product_id"]].append(r["doc_id"])
    n_link = Counter(d for v in links.values() for d in v)
    common = {d for d, n in n_link.items() if n >= 10}
    secs = defaultdict(list)
    for p in sorted((ROOT / "data" / "sections").glob("*.jsonl")):
        for l in open(p):
            s = json.loads(l)
            secs[s["doc_id"]].append(s)
    rated = set()
    for r in csv.DictReader(open(ROOT / "data" / "rates" / "rate_product_map.csv")):
        rated |= set(filter(None, re.split(r"[;|, ]+", r["product_ids"])))
    return prods, docs, links, common, secs, rated


def gen(only=None):
    """only: 다시 만들 유형 목록 — 후보 파일에서 그 유형만 바꿔 넣음 (8-1 검수: B·C 다시)"""
    old = [json.loads(l) for l in open(CAND)] if CAND.exists() else []
    if old and not only:
        sys.exit("후보가 이미 있음 — 유형만 다시 만들려면 gen --only=B-example,C-fake")
    rnd = random.Random(SEED)
    prods, docs, links, common, secs, rated = load_meta()
    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
    by_name = {}
    for pid, p in prods.items():
        by_name.setdefault(p["name"], pid)
    own = [pid for pid in by_name.values() if any(d not in common for d in links[pid])]
    com = [pid for pid in by_name.values() if links[pid] and all(d in common for d in links[pid])]
    rnd.shuffle(own)
    rnd.shuffle(com)
    jobs = []
    want = set(only or N_GEN)
    for pid in own[:N_GEN["A-own"] if "A-own" in want else 0]:
        txt = "\n\n".join(f"### {docs[d]['title']}\n" + "\n".join(s["text"] for s in secs[d] if not s["is_boilerplate"])
                          for d in links[pid] if d not in common)[:16000]
        jobs.append(("A-own", pid, GEN_A, f"상품 이름: {prods[pid]['name']}\n\n{txt}", ""))
    for pid in com[:N_GEN["A-common"] if "A-common" in want else 0]:
        toc = "\n".join(f"### {docs[d]['title']}\n" + "\n".join(s["section_path"] for s in secs[d]) for d in links[pid])[:12000]
        instr = GEN_A + (NO_RATE if pid in rated else "")
        jobs.append(("A-common", pid, instr, f"상품 이름: {prods[pid]['name']} ({prods[pid]['subcategory']})\n"
                     "이 상품에는 아래 공통 문서만 연결돼 있다 (상품 전용 문서 없음). 목차:\n\n" + toc, ""))
    pool = [pid for pid in com[N_GEN["A-common"]:]]
    k = 0
    for pid in pool:
        if k >= (N_GEN["B-example"] if "B-example" in want else 0):
            break
        d = next((d for d in ("D038", "D059") if d in links[pid]), None)
        if not d:
            continue
        seq = TRAP[d][k % len(TRAP[d])]
        s = next(s for s in secs[d] if s["seq"] == seq)
        jobs.append(("B-example", pid, GEN_B, f"상품 이름: {prods[pid]['name']}\n문서: {docs[d]['title']}\n\n{s['text']}", f"{d}:{seq}"))
        k += 1

    def one(job):
        typ, pid, instr, inp, src = job
        r = client.responses.parse(model=env["CHAT_MODEL"], instructions=instr, input=inp, text_format=GenA)
        return job, r.output_parsed

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, jobs))
    out = []
    for (typ, pid, _, _, src), q in res:
        if q and q.usable:
            out.append({"type": typ, "question": clean_q(q.question) if typ == "B-example" else q.question, "product_id": pid, "product_name": prods[pid]["name"],
                        "category": prods[pid]["category"], "absent_what": q.absent_what, "near_miss": q.near_miss, "trap_src": src})
    # C: 가짜 상품 — 실제 이름(판매 중 + 판매 중지)과 겹치지 않는 것만
    real = {p["name"] for p in prods.values()} | {r["name"] for r in csv.DictReader(open(META / "discontinued.csv"))}
    real_n = {norm(n) for n in real}
    r = None if "C-fake" not in want else client.responses.parse(model=env["CHAT_MODEL"], instructions=GEN_C, text_format=FakeList,
                               input=f"{N_GEN['C-fake'] + 8}개 만들어라.\n\n[실제 상품 이름]\n" + "\n".join(sorted(real)))
    import s12_retrieval as R
    idx = R.Index()
    nc = 0
    for f in (r.output_parsed.items if r else []):
        fn = norm(f.name)
        if norm(f.name) not in norm(f.question):
            continue
        if nc >= N_GEN["C-fake"] or fn in real_n or any(fn in x or (len(x) >= 4 and x in fn) for x in real_n):
            continue
        if idx.detect_exact(f.question):
            continue
        out.append({"type": "C-fake", "question": f.question, "product_id": None, "product_name": f.name,
                    "category": f.category, "absent_what": f.absent_what, "near_miss": "", "trap_src": ""})
        nc += 1
    start = max((int(q["cid"][1:]) for q in old), default=-1) + 1
    allq = [q for q in old if q["type"] not in want] + [{"cid": f"C{start + n:03d}", **q} for n, q in enumerate(out)]
    with open(CAND, "w") as fo:
        for q in allq:
            fo.write(json.dumps(q, ensure_ascii=False) + "\n")
    print("새 후보", Counter(q["type"] for q in out), "· 전체", Counter(q["type"] for q in allq))


def pool_for(idx, q, qv, pid, k=20):
    """질문 상품에 연결된 문서·공통 조항·용어 안에서 벡터+키워드 결합 상위 k (상품명 찾기와 상관없이 알려진 상품으로)"""
    vec, kw = idx.E @ qv, idx.bm25(q)
    score = np.zeros(idx.N)
    for r, i in enumerate(np.argsort(-vec)):
        score[i] += 1 / (60 + r)
    for r, i in enumerate(np.argsort(-kw)):
        if kw[i] > 0:
            score[i] += 1 / (60 + r)
    if pid:
        ds = set(idx.links[pid])
        cl = set().union(*(idx.doc_clause[d] for d in ds))
        ok = np.array([(i["kind"] == "chunk" and i["doc_id"] in ds) or (i["kind"] == "clause" and i["clause_id"] in cl)
                       or i["kind"] == "term" for i in idx.items])
        score = np.where(ok, score, -1)
    return [int(i) for i in np.argsort(-score)[:k]]


def verify():
    outs = [EVAL / "absent_dev.jsonl", EVAL / "absent_test.jsonl"]
    if any(p.exists() for p in outs):
        sys.exit("absent 세트가 이미 있음 — 잠근 세트라 다시 만들지 않음")
    import s12_retrieval as R
    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
    idx = R.Index()
    cs = [json.loads(l) for l in open(CAND)]
    qv = R.embed_questions(cs, idx)

    prev = {}
    if LOG.exists():
        for l in open(LOG):
            r = json.loads(l)
            prev[(r["cid"], r.get("question"))] = r

    def job(arg):
        c, v = arg
        if (c["cid"], c["question"]) in prev:      # 같은 질문은 다시 확인하지 않음
            return c, prev[(c["cid"], c["question"])]
        pool = pool_for(idx, c["question"], v, c["product_id"])
        body = "\n\n".join(f"<{n}>\n{idx.items[i]['text'][:1500]}" for n, i in enumerate(pool))
        r = client.responses.parse(model=env["CHAT_MODEL"], instructions=CHECK, text_format=Check,
                                   input=f"[질문] {c['question']}\n[질문 상품] {c['product_name']}\n\n[후보]\n{body}").output_parsed
        return c, {"cid": c["cid"], "question": c["question"], "answerable": r.answerable, "hits": [idx.items[pool[n]]["text"].split("\n", 1)[0] for n in r.ids if 0 <= n < len(pool)],
                   "note": r.note, "detected": sorted(idx.detect_exact(c["question"]))}

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(job, zip(cs, qv)))
    with open(LOG, "w") as f:
        for _, rec in res:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    rnd = random.Random(SEED)
    keep = defaultdict(list)
    for c, rec in res:
        if not rec["answerable"] and c["cid"] not in REVIEW_DROP:
            keep[c["type"]].append({**c, "detected": rec["detected"]})
    print("확인 후 남음", {t: len(v) for t, v in keep.items()}, "버림", sum(1 for _, r in res if r["answerable"]))
    split = {"dev": [], "test": []}
    for t, n in N_KEEP.items():
        xs = keep[t][:n]
        rnd.shuffle(xs)
        for j, x in enumerate(xs):
            split["dev" if j % 2 == 0 else "test"].append(x)
    for sp, xs in split.items():
        with open(EVAL / f"absent_{sp}.jsonl", "w") as f:
            for n, x in enumerate(xs):
                f.write(json.dumps({"qid": f"N{sp[0].upper()}{n:02d}", **x, "split": sp}, ensure_ascii=False) + "\n")
        print(sp, len(xs), Counter(x["type"] for x in xs))
    print("test sha256", hashlib.sha256((EVAL / "absent_test.jsonl").read_bytes()).hexdigest())


if __name__ == "__main__":
    if sys.argv[1] == "gen":
        o = next((a.split("=", 1)[1].split(",") for a in sys.argv[2:] if a.startswith("--only=")), None)
        gen(o)
    else:
        verify()
