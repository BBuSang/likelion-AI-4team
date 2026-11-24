"""표 질문 세트: 재추출(data/runs/table_v2)의 md 표에서 '행 × 열' 칸 값을 묻는 질문을 OpenAI(CHAT_MODEL)로 만든다
- 표 고르기: 3열 이상·데이터 2행 이상 md 표, 문서당 최대 2개, 총 TARGET개
- 질문 1개 = 표 1개. 행 이름과 열 이름을 둘 다 알아야 답할 수 있는 칸을 고르게 함, 질문에는 상품 이름(공용 문서는 문서 이름)을 넣음
- 검사: 값·행 이름·열 이름이 표에 그대로 있고, 값이 그 행 줄에 있어야 통과
- dev/test 는 문서 단위로 나눔 (같은 문서가 양쪽에 없게)
출력: data/eval/retrieval_table_dev.jsonl, retrieval_table_test.jsonl (gold.doc_id + table: row·col·value)
사용: python analysis/s42_table_qset.py
"""
import collections
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
V2 = ROOT / "data" / "runs" / "table_v2" / "data" / "chunks" / "chunks.jsonl"
META, EVAL = ROOT / "data" / "meta", ROOT / "data" / "eval"
SEP = re.compile(r"^\|(\s*:?-{3,}:?\s*\|)+\s*$")
TARGET, PER_DOC, SEED = 60, 2, 42

INSTR = """은행 상품 문서의 표 하나를 보고, 고객이 챗봇에 물을 법한 질문 하나를 만드세요.
- 표의 칸 하나를 고르세요: 행 이름과 열 이름을 둘 다 알아야 값이 정해지는 칸 (숫자·기간·금리·조건 등). 빈칸·'-' 칸은 고르지 마세요.
- 질문에는 주어진 상품(문서) 이름을 그대로 넣고, 행·열의 뜻을 자연스러운 말로 담으세요 (표 머리글을 그대로 복사할 필요는 없음).
- row_label, col_label, value 는 표에 적힌 글자 그대로 (row_label 은 그 행의 첫 칸, col_label 은 그 열의 머리글 칸).
- answer 는 value 를 담은 한 문장."""


class Q(BaseModel):
    question: str
    answer: str
    row_label: str
    col_label: str
    value: str


def norm(t: str) -> str:
    return re.sub(r"[\s|·ㆍ•\-<>]|<br>", "", t.replace("<br>", ""))


def tables(chunk):
    L = chunk["content"].split("\n")
    for i in range(1, len(L)):
        if SEP.match(L[i].strip()) and L[i - 1].startswith("|"):
            j = i + 1
            while j < len(L) and L[j].startswith("|"):
                j += 1
            if L[i].count("|") - 1 >= 3 and j - i - 1 >= 2:
                yield "\n".join(L[i - 1:j])


def check(q: Q, table: str) -> bool:
    rows = [r for r in table.split("\n")[2:]]
    head = table.split("\n")[0]
    if not (norm(q.value) and norm(q.row_label) and norm(q.col_label)):
        return False
    if norm(q.col_label) not in norm(head) and norm(q.col_label) not in norm(table):
        return False
    return any(norm(q.row_label) in norm(r) and norm(q.value) in norm(r) for r in rows)


def main():
    from openai import OpenAI
    from s8_extract import load_env
    env = load_env()
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)

    docs = {r["doc_id"]: r for r in csv.DictReader(open(META / "documents.csv"))}
    pname = {r["product_id"]: r["name"] for r in csv.DictReader(open(META / "products.csv"))}
    links = collections.defaultdict(list)
    for r in csv.DictReader(open(META / "product_docs.csv")):
        links[r["doc_id"]].append(r["product_id"])

    by_doc = collections.defaultdict(list)
    for line in open(V2):
        c = json.loads(line)
        for t in tables(c):
            by_doc[c["doc_id"]].append((c, t))
    rng = random.Random(SEED)
    picks = []
    for d in sorted(by_doc):
        cand = by_doc[d][:]
        rng.shuffle(cand)
        picks += [(d, c, t) for c, t in cand[:PER_DOC]]
    rng.shuffle(picks)

    out, used = [], collections.Counter()
    for d, c, t in picks:
        if len(out) >= TARGET:
            break
        pids = links[d]
        pid = sorted(pids)[0] if 1 <= len(pids) <= 3 else ""
        name = pname[pid] if pid else re.sub(r"\s*(상품설명서|설명서)$", "", docs[d]["title"])
        r = client.responses.parse(model=env["CHAT_MODEL"], instructions=INSTR, text_format=Q,
                                   input=f"상품(문서) 이름: {name}\n섹션: {c['section']}\n\n표:\n{t[:3500]}")
        q = r.output_parsed
        ok = check(q, t)
        print(f"{'통과' if ok else '불통과'} {d} {name[:20]} | {q.question[:50]} | {q.row_label} × {q.col_label} = {q.value}")
        if not ok:
            continue
        used[d] += 1
        out.append({"type": "T", "question": q.question, "answer": q.answer, "product_id": pid,
                    "gold": {"doc_id": d}, "table": {"row": q.row_label, "col": q.col_label, "value": q.value},
                    "evidence": f"{q.row_label} {q.col_label} {q.value}", "source_chunk": c["chunk_key"], "gold_alt": []})

    # 문서 단위로 dev/test
    dlist = sorted({x["gold"]["doc_id"] for x in out}, key=lambda x: hashlib.md5(x.encode()).hexdigest())
    dev_docs = set(dlist[::2])
    for sp in ("dev", "test"):
        rows = [x for x in out if (x["gold"]["doc_id"] in dev_docs) == (sp == "dev")]
        with open(EVAL / f"retrieval_table_{sp}.jsonl", "w") as f:
            for n, x in enumerate(rows):
                f.write(json.dumps({"qid": f"T{sp[0].upper()}{n:03d}", "split": f"table_{sp}", **x}, ensure_ascii=False) + "\n")
        print(f"{sp}: {len(rows)}문항 · 문서 {len({x['gold']['doc_id'] for x in rows})}개")


if __name__ == "__main__":
    main()
