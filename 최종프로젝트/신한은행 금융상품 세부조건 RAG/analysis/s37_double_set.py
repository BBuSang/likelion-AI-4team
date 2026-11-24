"""16-1 '질문 두 개' 세트 — 한 메시지에 같은 상품에 대해 두 가지를 묻는 질문 (dev / test 잠금, 고치기 전에 만듦)

사용: python analysis/s37_double_set.py   → data/eval/double_{dev,test}.jsonl (한 번만, 있으면 멈춤)
왜: 최종 검증(13-1)의 실제 말투 실패 7개 중 3개가 '질문 두 개' — 첫 질문만 답하고 두 번째는 "확인되지 않음"
구성 (seed 20261203, OpenAI CHAT_MODEL, 상품 단위로 dev/test)
  - 상품 하나에서 서로 다른 새 섹션 2개(기존 모든 평가 세트의 정답 섹션 제외, 상품 전용 문서, 정형 문구 아님, 제목이 다른 섹션)를 골라
    섹션마다 s11 지시문으로 질문·답·근거를 만들고(근거가 원문에 없으면 버림), 두 질문을 한 메시지로 합침
  - 합치는 말투: 절반은 단정한 두 문장('…인가요? 그리고 …는요?'), 절반은 실제 말투(짧은 반말·붙여 쓰기 등)
  - 상품 이름은 정식 이름 그대로 한 번 (상품명 찾기가 아니라 '두 번째 질문을 놓치는지'만 보려고)
  - 분류별 split마다: 저축 15 · 대출 10 · 기타·ELD 5 = 30
행: {qid, type: "double", style, question, product_id, category, subs: [{question, gold, answer, evidence, section}, {…}]}
"""
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from s8_extract import lines_in_order, load_env, ns  # noqa: E402
from s11_evalset import GEN_PROMPT, GenQ  # noqa: E402
from s25_common_set import NO_DOC  # noqa: E402

META, EVAL = ROOT / "data" / "meta", ROOT / "data" / "eval"
SEED = 20261203
OUT = {s: EVAL / f"double_{s}.jsonl" for s in ("dev", "test")}
N = {"저축상품": 15, "대출상품": 10, "기타": 5}
JOIN_PROMPT = """너는 신한은행 영업점 직원이다. 같은 상품에 대한 [질문 1]과 [질문 2]를 **한 메시지**로 합쳐, 동료나 사내 챗봇에 묻는 질문 하나를 만든다.
- 말투: {style}
- 두 질문의 내용(뜻)을 **모두** 그대로 묻는다. 하나라도 빠지거나 뜻이 바뀌면 안 된다
- 상품 이름은 정식 이름 그대로 **한 번만** 쓴다
- 정답을 질문에 넣지 않는다"""
STYLE = {"formal": "단정한 존댓말 두 문장 (예: '○○의 가입 대상은 어떻게 되나요? 그리고 중도해지하면 이자는 어떻게 되나요?')",
         "real": "바쁜 창구에서 치는 짧은 반말·축약, 띄어쓰기를 일부 생략해도 됨 (예: '○○ 가입대상 누구야? 중도해지하면 이자는?')"}


class JQ(BaseModel):
    question: str


def main():
    if any(p.exists() for p in OUT.values()):
        sys.exit("이미 있음 — 잠근 세트라 다시 만들지 않음")
    rnd = random.Random(SEED)
    prods = list(csv.DictReader(open(META / "products.csv")))
    by_name = {}
    for p in prods:
        by_name.setdefault(p["name"], p)
    docs = {r["doc_id"]: r for r in csv.DictReader(open(META / "documents.csv"))}
    links, plinks = defaultdict(set), defaultdict(set)
    for r in csv.DictReader(open(META / "product_docs.csv")):
        links[r["product_id"]].add(r["doc_id"])
        plinks[r["doc_id"]].add(r["product_id"])
    shared = {d for d, ps in plinks.items() if len(ps) >= 10}
    used = set()
    for f in EVAL.glob("*.jsonl"):
        for l in open(f):
            q = json.loads(l)
            for g in [q.get("gold")] + [s.get("gold") for s in q.get("subs", [])]:
                if isinstance(g, dict):
                    used.add((g.get("doc_id"), g.get("section_seq")))
    secs = defaultdict(list)
    for p in (ROOT / "data" / "sections").glob("*.jsonl"):
        for l in open(p):
            s = json.loads(l)
            body = s["text"].split("\n", 1)[1] if "\n" in s["text"] else ""
            if s["section_path"] != "(머리말)" and len(body) >= 40 and not s["is_boilerplate"] and (s["doc_id"], s["seq"]) not in used:
                secs[s["doc_id"]].append(s)

    def split_of(name):
        return "test" if int(hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest(), 16) % 2 else "dev"

    def cat_of(p):
        return "기타" if p["category"] in ("기타상품", "복합금융상품") else p["category"]

    names = sorted(by_name)
    rnd.shuffle(names)
    jobs, have = [], Counter()
    for n in names:
        p = by_name[n]
        c, sp = cat_of(p), split_of(n)
        if c not in N or have[(c, sp)] >= N[c] + 6:
            continue
        cand = [s for d in sorted(links[p["product_id"]]) if d not in shared for s in secs[d]]
        rnd.shuffle(cand)
        pair = None
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                if cand[i]["section_path"] != cand[j]["section_path"]:
                    pair = (cand[i], cand[j])
                    break
            if pair:
                break
        if pair:
            jobs.append((sp, c, p["product_id"], n, pair))
            have[(c, sp)] += 1

    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)

    def call(instr, inp, fmt):
        return client.responses.parse(model=env["CHAT_MODEL"], instructions=instr, input=inp, text_format=fmt).output_parsed

    def one(job):
        sp, c, pid, n, pair = job
        subs = []
        for s in pair:
            g = call(GEN_PROMPT + NO_DOC, f"상품 이름: {n}\n문서: {docs[s['doc_id']]['title']}\n\n{s['text']}", GenQ)
            if not g or not g.usable or not lines_in_order(g.evidence, ns(s["text"])) or n not in g.question:
                return job, None, None
            subs.append({"question": g.question, "gold": {"doc_id": s["doc_id"], "section_seq": s["seq"]}, "answer": g.answer,
                         "evidence": g.evidence, "section": f"{docs[s['doc_id']]['title']} > {s['section_path']}"})
        return job, subs, None

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(one, jobs))
    dropped = Counter()
    ok = []
    for (sp, c, pid, n, pair), subs, _ in res:
        if not subs:
            dropped["하위 질문 불량"] += 1
            continue
        ok.append((sp, c, pid, n, subs))
    # 말투: 분류·split 안에서 반씩
    cnt = Counter()
    joins = []
    for sp, c, pid, n, subs in ok:
        if cnt[(c, sp)] >= N[c]:
            continue
        style = "formal" if cnt[(c, sp)] % 2 == 0 else "real"
        cnt[(c, sp)] += 1
        joins.append((sp, c, pid, n, subs, style))

    def join(arg):
        sp, c, pid, n, subs, style = arg
        return arg, call(JOIN_PROMPT.format(style=STYLE[style]), f"[질문 1] {subs[0]['question']}\n[질문 2] {subs[1]['question']}", JQ)

    with ThreadPoolExecutor(6) as ex:
        jres = list(ex.map(join, joins))
    out = defaultdict(list)
    for k, ((sp, c, pid, n, subs, style), j) in enumerate(jres):
        if not j or n not in j.question:
            dropped["합친 질문에 상품 이름 없음"] += 1
            continue
        out[sp].append({"qid": f"W{sp[0].upper()}{k:03d}", "type": "double", "style": style, "question": j.question, "product_id": pid,
                        "category": by_name[n]["category"], "subs": subs, "split": f"double_{sp}"})
    for sp, p in OUT.items():
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out[sp]))
        print(p.name, len(out[sp]), dict(Counter((r["category"], r["style"]) for r in out[sp])))
    print("버림", dict(dropped))
    print("test sha256", hashlib.sha256(OUT["test"].read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
