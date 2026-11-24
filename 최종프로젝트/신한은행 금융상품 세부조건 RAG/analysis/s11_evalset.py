"""5-1 검색 평가셋 만들기

사용: python analysis/s11_evalset.py     # data/eval/retrieval_{dev,test}.jsonl
정답 형식 (질문 1개 = 한 줄)
  {"qid", "type", "question", "product_id", "gold": {"doc_id", "section_seq"} | {"clause_id"} | {"term"}, "answer", "evidence", "split"}
질문 유형
  A 상품 지정 사실 질문 — 정형 문구가 아닌 섹션을 문서 종류별로 층화 표본 → LLM이 은행 직원 말투로 질문(상품명 포함)·짧은 답·근거 원문
  B 공통 조항·용어 질문 — 정형 문구 공통 조항(절반은 상품명 포함)과 용어집
  (C 비교·SQL, D 판매 중지·미보유 값 질문은 답변 단계 평가에서 따로 만듦 — 검색 지표에 넣지 않음)
과적합 방지
  - dev/test는 **상품 단위**로 나눔(같은 상품이 양쪽에 없음), 공통 조항·용어는 항목 단위로 나눔. test는 검색 방식을 고를 때 보지 않고 마지막에 한 번만
  - 근거(evidence)가 정답 섹션에 원문 그대로 없으면 그 질문은 버림
"""
import csv
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from s8_extract import lines_in_order, load_env, ns  # noqa: E402

META, EVAL = ROOT / "data" / "meta", ROOT / "data" / "eval"
SEED = 20261111
N_A = {"상품설명서": 80, "특약": 30, "투자설명서": 12, "간이투자설명서": 8, "약관": 12, "약정서": 8}
N_CLAUSE, N_TERM = 20, 10

GEN_PROMPT = """너는 신한은행 영업점 직원이다. 아래 상품 문서의 한 부분을 읽고, 고객 응대 중에 동료나 상품 안내 챗봇에게 물어볼 법한 질문을 하나 만든다.
규칙
- 이 부분만 읽으면 답할 수 있는 **구체적인 사실 하나**를 묻는다 (금리·조건·기간·금액·절차·제한 등).
- 질문에 상품 이름을 넣되 직원이 실제로 부르는 대로 자연스럽게 (예: '신한 운동화 적금' → '운동화 적금'). 상품 이름이 '없음'이면 넣지 않는다.
- 문서 문장을 그대로 베끼지 말고 말하듯이 바꿔 쓴다. 섹션 제목 단어를 그대로 쓰지 않아도 된다.
- answer: 짧은 정답 (1문장). evidence: 정답 근거 원문을 입력에서 **그대로** 복사 (1~3줄).
- 이 부분으로 의미 있는 질문을 만들 수 없으면 usable=false.
JSON으로만 답한다."""


class GenQ(BaseModel):
    usable: bool
    question: str
    answer: str
    evidence: str
    note: Optional[str] = None


def split_of(key: str) -> str:
    return "test" if int(hashlib.sha256(f"{SEED}:{key}".encode()).hexdigest(), 16) % 10 < 4 else "dev"


def short_name(name: str) -> str:
    return name


def gen():
    random.seed(SEED)
    docs = {r["doc_id"]: r for r in csv.DictReader(open(META / "documents.csv"))}
    prods = {r["product_id"]: r for r in csv.DictReader(open(META / "products.csv"))}
    links = defaultdict(list)
    for r in csv.DictReader(open(META / "product_docs.csv")):
        links[r["doc_id"]].append(r["product_id"])
    secs = [json.loads(l) for p in sorted((ROOT / "data" / "sections").glob("*.jsonl")) for l in open(p)]
    cand = defaultdict(list)
    for s in secs:
        body = s["text"].split("\n", 1)[1] if "\n" in s["text"] else ""
        if not s["is_boilerplate"] and s["section_path"] != "(머리말)" and len(body) >= 40 and links[s["doc_id"]]:
            cand[docs[s["doc_id"]]["doc_type"]].append(s)
    jobs = []
    for dt, n in N_A.items():
        pool = cand[dt]
        # 한 문서에서 너무 많이 뽑히지 않게: 문서를 섞고 돌아가며 하나씩
        by_doc = defaultdict(list)
        for s in pool:
            by_doc[s["doc_id"]].append(s)
        order = list(by_doc)
        random.shuffle(order)
        for d in order:
            random.shuffle(by_doc[d])
        picked, i = [], 0
        while len(picked) < n and any(by_doc.values()):
            d = order[i % len(order)]
            if by_doc[d]:
                picked.append(by_doc[d].pop())
            i += 1
        for s in picked:
            pid = random.choice(links[s["doc_id"]])
            jobs.append(("A", s, pid, prods[pid]["name"], docs[s["doc_id"]]["title"]))

    clauses = [r for r in csv.DictReader(open(META / "common_clause.csv")) if r["is_boilerplate"] == "1"]
    random.shuffle(clauses)
    dclause = defaultdict(list)
    for r in csv.DictReader(open(META / "document_clause.csv")):
        dclause[r["clause_id"]].append(r["doc_id"])
    for k, c in enumerate(clauses[:N_CLAUSE]):
        d = random.choice(dclause[c["clause_id"]])
        pid = random.choice(links[d]) if links[d] and k % 2 == 0 else None
        jobs.append(("B-clause", {"clause_id": int(c["clause_id"]), "text": c["content"]}, pid,
                     prods[pid]["name"] if pid else "없음", "공통 조항"))
    gloss = list(csv.DictReader(open(META / "glossary.csv")))
    random.shuffle(gloss)
    for g in gloss[:N_TERM]:
        jobs.append(("B-term", {"term": g["term"], "text": f"{g['term']}\n{g['definition']}"}, None, "없음", "용어 안내"))

    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)

    def one(job):
        typ, s, pid, pname, title = job
        inp = f"상품 이름: {pname}\n문서: {title}\n\n{s['text']}"
        r = client.responses.parse(model=env["CHAT_MODEL"], instructions=GEN_PROMPT, input=inp, text_format=GenQ)
        return job, r.output_parsed

    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(one, jobs))

    out, dropped = [], defaultdict(int)
    for n, ((typ, s, pid, pname, title), q) in enumerate(results):
        if not q or not q.usable:
            dropped["usable=false"] += 1
            continue
        if not lines_in_order(q.evidence, ns(s["text"])):
            dropped["근거가 원문에 없음"] += 1
            continue
        if typ == "A":
            gold, key = {"doc_id": s["doc_id"], "section_seq": s["seq"]}, f"p:{pid}"
        elif typ == "B-clause":
            gold, key = {"clause_id": s["clause_id"]}, f"c:{s['clause_id']}"
        else:
            gold, key = {"term": s["term"]}, f"t:{s['term']}"
        out.append({"qid": f"{typ[0]}{n:03d}", "type": typ, "question": q.question, "product_id": pid, "gold": gold,
                    "answer": q.answer, "evidence": q.evidence, "split": split_of(key)})
    EVAL.mkdir(exist_ok=True)
    for sp in ("dev", "test"):
        with open(EVAL / f"retrieval_{sp}.jsonl", "w") as f:
            for r in out:
                if r["split"] == sp:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"만든 질문 {len(out)}개 (버림 {dict(dropped)}) — dev {sum(r['split'] == 'dev' for r in out)} / "
          f"test {sum(r['split'] == 'test' for r in out)}")
    for t in ("A", "B-clause", "B-term"):
        print(f"  {t}: {sum(r['type'] == t for r in out)}")


if __name__ == "__main__":
    gen()
