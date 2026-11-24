"""7-3 새 test 세트 (test2) 만들기 — 기존 dev·test·별칭 확인 질문에 쓰지 않은 상품·섹션·공통 조항·용어로

사용: python analysis/s19_testset2.py      → data/eval/retrieval_test2.jsonl (한 번만 만듦, 있으면 멈춤)
왜: 기존 test는 7-2에서 상품명 별칭 오류를 test에서 찾아 고치면서 그 부분이 오염됨 → 처음 보는 상품으로 다시 잠근 시험 세트
구성 (seed 20261113)
  A 상품 사실 90개: 안 쓴 상품(이름 기준 161개 — 대출 139·저축 19·기타 2·복합 1)에서 상품을 고르고, 그 상품에 연결된 문서의
    섹션(정형 문구 아님·머리말 아님·본문 40자 이상·기존 정답 섹션 아님, 같은 섹션은 한 번만)을 하나씩. 분류별: 저축 25 · 대출 60 · 기타/복합 5
    → **절반은 직원이 쓰는 줄임말로 상품을 부르게**(name_style=nickname), 절반은 정식 이름(name_style=formal) — 상품명 찾기까지 공정하게
  B 공통 조항 10개(정형 문구, 절반은 상품명 포함) · 용어 6개: 기존에 쓰지 않은 것
  질문 만들기는 s11과 같은 OpenAI(CHAT_MODEL) 지시문 + 줄임말 지시. 근거 원문이 정답 섹션에 그대로 없으면 버림
  같은 답을 담은 다른 문서(gold_alt)는 s17과 같은 방식으로 표시 (s17_multigold.py test2)
"""
import csv
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from s11_evalset import GEN_PROMPT, GenQ  # noqa: E402
from s8_extract import lines_in_order, load_env, ns  # noqa: E402

META, EVAL = ROOT / "data" / "meta", ROOT / "data" / "eval"
OUT = EVAL / "retrieval_test2.jsonl"
SEED = 20261113
N_A = {"저축상품": 25, "대출상품": 60, "기타": 5}
N_CLAUSE, N_TERM = 10, 6
NICK = ("\n- **상품 이름은 정식 이름 대신 영업점 직원이 실제로 줄여 부르는 말**로 쓴다 "
        "(예: '신한 장병내일준비 적금' → '장병적금', 'Hey Young(헤이영) 머니박스' → '헤이영 머니박스'). 아무도 못 알아들을 줄임말은 안 됨.")


def main():
    if OUT.exists():
        sys.exit("test2가 이미 있음 — 잠근 시험 세트라 다시 만들지 않음")
    rnd = random.Random(SEED)
    prods = list(csv.DictReader(open(META / "products.csv")))
    docs = {r["doc_id"]: r for r in csv.DictReader(open(META / "documents.csv"))}
    links = defaultdict(list)
    for r in csv.DictReader(open(META / "product_docs.csv")):
        links[r["product_id"]].append(r["doc_id"])
    used_p, used_sec, used_cl, used_t = set(), set(), set(), set()
    for sp in ("dev", "test"):
        for q in map(json.loads, open(EVAL / f"retrieval_{sp}.jsonl")):
            used_p.add(q["product_id"])
            g = q["gold"]
            used_sec.add((g.get("doc_id"), g.get("section_seq")))
            used_cl.add(g.get("clause_id"))
            used_t.add(g.get("term"))
    used_p |= {json.loads(l)["product_id"] for l in open(EVAL / "alias_check.jsonl")}
    names_used = {p["name"] for p in prods if p["product_id"] in used_p}
    free = {}
    for p in prods:
        if p["name"] not in names_used:
            free.setdefault(p["name"], p)
    secs = defaultdict(list)
    for p in (ROOT / "data" / "sections").glob("*.jsonl"):
        for l in open(p):
            s = json.loads(l)
            body = s["text"].split("\n", 1)[1] if "\n" in s["text"] else ""
            if not s["is_boilerplate"] and s["section_path"] != "(머리말)" and len(body) >= 40 and (s["doc_id"], s["seq"]) not in used_sec:
                secs[s["doc_id"]].append(s)

    jobs, taken = [], set()
    for cat, n in N_A.items():
        pool = [p for p in free.values() if (p["category"] == cat) or (cat == "기타" and p["category"] in ("기타상품", "복합금융상품"))]
        rnd.shuffle(pool)
        i = 0
        while len([j for j in jobs if j[0] == "A" and j[5] == cat]) < n and pool:
            p = pool[i % len(pool)]
            cand = [s for d in links[p["product_id"]] for s in secs[d] if (s["doc_id"], s["seq"]) not in taken]
            if cand:
                s = rnd.choice(cand)
                taken.add((s["doc_id"], s["seq"]))
                style = "nickname" if len(jobs) % 2 == 0 else "formal"
                jobs.append(("A", s, p["product_id"], p["name"], docs[s["doc_id"]]["title"], cat, style))
            i += 1
            if i > len(pool) * 6:
                break
    clauses = [r for r in csv.DictReader(open(META / "common_clause.csv")) if r["is_boilerplate"] == "1" and int(r["clause_id"]) not in used_cl]
    rnd.shuffle(clauses)
    dclause = defaultdict(list)
    for r in csv.DictReader(open(META / "document_clause.csv")):
        dclause[r["clause_id"]].append(r["doc_id"])
    plinks = defaultdict(list)
    for pid, ds in links.items():
        for d in ds:
            plinks[d].append(pid)
    for k, c in enumerate(clauses[:N_CLAUSE]):
        ps = [pid for d in dclause[c["clause_id"]] for pid in plinks[d] if prods_name(prods, pid) not in names_used]
        pid = rnd.choice(ps) if ps and k % 2 == 0 else None
        jobs.append(("B-clause", {"clause_id": int(c["clause_id"]), "text": c["content"]}, pid,
                     prods_name(prods, pid) if pid else "없음", "공통 조항", "공통", "formal"))
    gl = [g for g in csv.DictReader(open(META / "glossary.csv")) if g["term"] not in used_t]
    rnd.shuffle(gl)
    for g in gl[:N_TERM]:
        jobs.append(("B-term", {"term": g["term"], "text": f"{g['term']}\n{g['definition']}"}, None, "없음", "용어 안내", "용어", "formal"))

    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)

    def one(job):
        typ, s, pid, pname, title, cat, style = job
        instr = GEN_PROMPT + (NICK if style == "nickname" and pid else "")
        r = client.responses.parse(model=env["CHAT_MODEL"], instructions=instr, text_format=GenQ,
                                   input=f"상품 이름: {pname}\n문서: {title}\n\n{s['text']}")
        return job, r.output_parsed

    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(one, jobs))
    out, dropped = [], defaultdict(int)
    for n, ((typ, s, pid, pname, title, cat, style), q) in enumerate(results):
        if not q or not q.usable:
            dropped["usable=false"] += 1
            continue
        if not lines_in_order(q.evidence, ns(s["text"])):
            dropped["근거가 원문에 없음"] += 1
            continue
        gold = ({"doc_id": s["doc_id"], "section_seq": s["seq"]} if typ == "A" else
                {"clause_id": s["clause_id"]} if typ == "B-clause" else {"term": s["term"]})
        out.append({"qid": f"T{typ[0]}{n:03d}", "type": typ, "question": q.question, "product_id": pid, "category": cat,
                    "name_style": style if pid else None, "gold": gold, "answer": q.answer, "evidence": q.evidence, "split": "test2"})
    with open(OUT, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = defaultdict(int)
    for r in out:
        c[(r["type"], r["category"], r["name_style"])] += 1
    print(f"test2 {len(out)}개 (버림 {dict(dropped)})")
    for k, v in sorted(c.items(), key=str):
        print("  ", k, v)


def prods_name(prods, pid):
    return next((p["name"] for p in prods if p["product_id"] == pid), None)


if __name__ == "__main__":
    main()
