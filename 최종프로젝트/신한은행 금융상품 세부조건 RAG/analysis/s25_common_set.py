"""10-1 공통 약관 질문 세트 — 상품 이름으로 묻지만 답은 공통 약관·공통 조항에만 있는 질문 (구현 전에 만들고 test는 잠금)

사용: python analysis/s25_common_set.py      → data/eval/retrieval_common_dev.jsonl · retrieval_common_test.jsonl (한 번만, 있으면 멈춤)
      이어서 python analysis/s10_chunks.py embed-questions-local data/eval/retrieval_common_*.jsonl
            python analysis/s17_multigold.py common_dev common_test      (같은 답을 담은 다른 조각 = 보완 정답)
왜: test2 실패 2건(7-4)은 답이 공통 약관에 있는데, 상품 이름으로 좁힌 뒤 그 상품 문서 조각들(머리에 상품 이름이 붙음)이 위를 차지해
    공통 약관 조각이 재순위 후보(상위 10개)에 못 들어옴 (13위·16위, 다른 1건은 20위 밖)
구성 (seed 20261118, OpenAI CHAT_MODEL, s11과 같은 질문 지시문 + "문서 이름 없이 상품 이름으로만")
  common  60: 공통 약관 문서(예금거래기본약관·입출금/적립식/거치식 예금약관·외화예금거래 기본약관·여신거래 기본약관 가계/기업)의 섹션
              — 정형 문구 섹션은 공통 조항(clause_id)이 정답. 그 문서에 연결된 상품 하나를 골라 이름을 넣음
  control 40: 같은 방식으로 고른 상품의 **전용 문서**(설명서·특약) 섹션 — 상품 질문이 나빠지지 않는지
  상품 단위로 dev/test 반씩 (dev: common 30 · control 20 / test: common 30 · control 20). 기존 평가 세트의 정답 섹션·조항은 제외
"""
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from s11_evalset import GEN_PROMPT, GenQ  # noqa: E402
from s8_extract import lines_in_order, load_env, ns  # noqa: E402

META, EVAL = ROOT / "data" / "meta", ROOT / "data" / "eval"
OUT = {s: EVAL / f"retrieval_common_{s}.jsonl" for s in ("dev", "test")}
SEED = 20261118
COMMON_DOCS = ["D207", "D226", "D228", "D041", "D214", "D221", "D222"]
NO_DOC = "\n- 질문에 약관·설명서 같은 **문서 이름을 넣지 않는다** (직원은 상품 이름으로만 묻는다). 상품 이름은 정식 이름 그대로."


def main():
    if any(p.exists() for p in OUT.values()):
        sys.exit("이미 있음 — 잠근 세트라 다시 만들지 않음")
    rnd = random.Random(SEED)
    prods = {r["product_id"]: r for r in csv.DictReader(open(META / "products.csv"))}
    docs = {r["doc_id"]: r for r in csv.DictReader(open(META / "documents.csv"))}
    links, plinks = defaultdict(set), defaultdict(set)
    for r in csv.DictReader(open(META / "product_docs.csv")):
        links[r["product_id"]].add(r["doc_id"])
        plinks[r["doc_id"]].add(r["product_id"])
    n_link = Counter({d: len(ps) for d, ps in plinks.items()})
    shared = {d for d, n in n_link.items() if n >= 10}
    used_sec, used_cl = set(), set()
    for f in EVAL.glob("retrieval_*.jsonl"):
        for q in map(json.loads, open(f)):
            g = q["gold"]
            used_sec.add((g.get("doc_id"), g.get("section_seq")))
            used_cl.add(g.get("clause_id"))
    secs = defaultdict(list)
    for p in (ROOT / "data" / "sections").glob("*.jsonl"):
        for l in open(p):
            s = json.loads(l)
            body = s["text"].split("\n", 1)[1] if "\n" in s["text"] else ""
            if s["section_path"] != "(머리말)" and len(body) >= 40 and (s["doc_id"], s["seq"]) not in used_sec:
                if s["is_boilerplate"] and (not s.get("clause_id") or int(s["clause_id"]) in used_cl):
                    continue
                secs[s["doc_id"]].append(s)
    by_name = {}
    for pid, p in prods.items():
        by_name.setdefault(p["name"], pid)

    def split_of(pid):
        return "test" if int(hashlib.sha256(f"{SEED}:{prods[pid]['name']}".encode()).hexdigest(), 16) % 2 else "dev"

    jobs, taken_sec, taken_cl = [], set(), set()
    want = {("common", "dev"): 30, ("common", "test"): 30, ("control", "dev"): 20, ("control", "test"): 20}
    have = Counter()
    pool = [s for d in COMMON_DOCS for s in secs[d]]
    rnd.shuffle(pool)
    for s in pool:
        cid = int(s["clause_id"]) if s["is_boilerplate"] else None
        if (s["doc_id"], s["seq"]) in taken_sec or (cid and cid in taken_cl):
            continue
        cands = sorted(pid for pid in plinks[s["doc_id"]] if by_name.get(prods[pid]["name"]) == pid)
        rnd.shuffle(cands)
        for pid in cands:
            sp = split_of(pid)
            if have[("common", sp)] < want[("common", sp)]:
                jobs.append(("common", sp, s, pid, cid))
                have[("common", sp)] += 1
                taken_sec.add((s["doc_id"], s["seq"]))
                if cid:
                    taken_cl.add(cid)
                break
        if have[("common", "dev")] >= 30 and have[("common", "test")] >= 30:
            break
    # control: 상품 전용 문서(공유 문서가 아님)의 정형 문구가 아닌 섹션
    own_pids = sorted(by_name.values())
    rnd.shuffle(own_pids)
    for pid in own_pids:
        sp = split_of(pid)
        if have[("control", sp)] >= want[("control", sp)]:
            continue
        cand = [s for d in links[pid] if d not in shared for s in secs[d] if not s["is_boilerplate"] and (s["doc_id"], s["seq"]) not in taken_sec]
        if cand:
            s = rnd.choice(cand)
            jobs.append(("control", sp, s, pid, None))
            have[("control", sp)] += 1
            taken_sec.add((s["doc_id"], s["seq"]))
    print("만들 질문", dict(have))

    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)

    def one(job):
        typ, sp, s, pid, cid = job
        r = client.responses.parse(model=env["CHAT_MODEL"], instructions=GEN_PROMPT + NO_DOC, text_format=GenQ,
                                   input=f"상품 이름: {prods[pid]['name']}\n문서: {docs[s['doc_id']]['title']}\n\n{s['text']}")
        return job, r.output_parsed

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, jobs))
    out, dropped = defaultdict(list), Counter()
    for n, ((typ, sp, s, pid, cid), q) in enumerate(res):
        if not q or not q.usable:
            dropped["usable=false"] += 1
            continue
        if not lines_in_order(q.evidence, ns(s["text"])):
            dropped["근거가 원문에 없음"] += 1
            continue
        if prods[pid]["name"] not in q.question:
            dropped["상품 이름 없음"] += 1
            continue
        gold = {"clause_id": cid} if cid else {"doc_id": s["doc_id"], "section_seq": s["seq"]}
        out[sp].append({"qid": f"C{sp[0].upper()}{typ[0]}{n:03d}", "type": typ, "question": q.question, "product_id": pid,
                        "category": prods[pid]["category"], "gold": gold, "answer": q.answer, "evidence": q.evidence,
                        "gold_doc_title": docs[s["doc_id"]]["title"], "split": f"common_{sp}"})
    for sp, p in OUT.items():
        with open(p, "w") as f:
            for r in out[sp]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(p.name, Counter(r["type"] for r in out[sp]))
    print("버림", dict(dropped))


if __name__ == "__main__":
    main()
