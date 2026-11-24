"""12-1 실제 말투 평가 세트 — 같은 사실을 '단정한 말투'와 '실제 말투' 두 가지로 묻는 짝 비교 (dev / test 잠금)

사용: python analysis/s31_realistic_set.py
      → retrieval_real_{dev,test}.jsonl (상품 사실·공통 약관) · absent_real_{split}.jsonl (없는 값)
        status_real_{split}.jsonl (판매 중지) · compare_real_{split}.jsonl (비교)   — 기존 채점 스크립트(s12·s15·s21·s27) 형식 그대로
      이어서 질문 임베딩(s10 embed-questions-local) · 보완 정답(s17 real_dev real_test)
왜: 지금까지 평가 질문은 "○○의 가입 대상은 어떻게 되나요?"처럼 단정한 문장이 대부분 — 실제 영업점 말투(짧은 반말, 줄임말, 오타, 띄어쓰기 없음,
    고객 상황 섞기, 질문 두 개 한 번에, 구어체 숫자)에서도 버티는지 잰 적이 없음
구성 (seed 20261123, OpenAI CHAT_MODEL, 상품 단위로 dev/test)
  사실 30개씩: 상품 사실 12 · 공통 약관 6 · 없는 값 4 · 판매 중지 4 · 비교 4 → 사실마다 formal(단정) + real(실제 말투) 질문 2개 = 60개씩
  - 상품 사실: 상품 전용 문서 섹션 (s11과 같은 질문 지시문), 공통 약관: 공통 약관 섹션 + **그 조항이 그 상품 종류에 적용되는지 LLM 확인**(10-1 교훈)
  - 없는 값: s20 지시문(전용 문서에 없는 값) + 정말 없는지 LLM 확인(s20 CHECK), 판매 중지: 판매 중지 목록에만 있는 이름, 비교: 전용 설명서 추출값 + 근거 확인(s26)
  - 실제 말투: 단정한 질문을 뜻은 그대로 두고 아래 말투 중 1~2개로 바꿔 씀 (앞 대화가 있어야 하는 말·정답은 넣지 않음)
  - 기존 평가 세트의 정답 섹션은 뺌. 상품 이름은 판매 중지·비교에서만 뺌 (상품 사실·공통 약관·없는 값은 안 쓴 상품만으로는 거의 안 남아 섹션 기준으로만)
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
from s20_absent_set import CHECK, GEN_A, Check, GenA  # noqa: E402
from s26_status_compare_set import COMPARE_PROMPT, CQ  # noqa: E402

META, EVAL = ROOT / "data" / "meta", ROOT / "data" / "eval"
SEED = 20261123
SPLITS = ("dev", "test")
N = {"fact": 12, "common": 6, "absent": 4, "disc": 4, "compare": 4}
COMMON_DOCS = ["D207", "D226", "D228", "D041", "D214", "D221", "D222"]
STYLES = {
    "short": "짧은 반말·축약 (예: '쏠편한 직장인 금리 얼마?')",
    "nick": "상품 이름을 직원들이 쓰는 줄임말·별명으로 (예: '장병적금', '헤이영 머니박스')",
    "typo": "오타 하나나 띄어쓰기 없이 붙여 쓰기 (예: '운동화적금중도해지하면이자어케돼요')",
    "customer": "고객 상황을 섞어서 (예: '고객이 만기 전에 깬다는데 이자 어떻게 줘요?')",
    "double": "관련된 질문 두 개를 한 번에 (원래 질문 + 같은 상품의 가벼운 질문 하나)",
    "number": "구어체 숫자·단위 (예: '3천 넣으면', '1년 반')",
}
REAL_PROMPT = """너는 신한은행 영업점 직원이다. 아래 [질문]을 바쁜 창구에서 동료나 사내 챗봇에 실제로 칠 법한 말투로 바꿔 쓴다. **묻는 상품과 내용(뜻)은 그대로** 둔다.
- 사용할 말투: {styles}
- 상품 이름은 줄여 불러도 되지만 직원이 알아들을 수준으로. 두 상품을 비교하는 질문이면 두 상품 모두 언급
- 정답을 질문에 넣지 않는다. '그거', '아까 그 상품'처럼 앞 대화가 있어야 알 수 있는 말은 쓰지 않는다
- question에 바꾼 질문만"""
APPLY_PROMPT = """너는 은행 약관 검수자다. [약관 조항]이 [상품]에 **적용되는 조항인지** 판단한다.
- 조항이 특정 예금 종류(당좌예금·가계당좌예금·정액적립식·외화당좌 등)나 특정 대출 유형에만 해당하는데 상품이 그 종류가 아니면 applies=false
- 상품 종류를 알 수 없거나 확실하지 않으면 applies=false"""


class RQ(BaseModel):
    question: str


class Apply(BaseModel):
    applies: bool
    note: str


def main():
    targets = [EVAL / f"{k}_real_{s}.jsonl" for k in ("retrieval", "absent", "status", "compare") for s in SPLITS]
    if any(p.exists() for p in targets):
        sys.exit("이미 있음 — 잠근 세트라 다시 만들지 않음")
    import s12_retrieval as R
    rnd = random.Random(SEED)
    prods = list(csv.DictReader(open(META / "products.csv")))
    by_name = {}
    for p in prods:
        by_name.setdefault(p["name"], p)
    pid_name = {p["product_id"]: p["name"] for p in prods}
    docs = {r["doc_id"]: r for r in csv.DictReader(open(META / "documents.csv"))}
    links, plinks = defaultdict(set), defaultdict(set)
    for r in csv.DictReader(open(META / "product_docs.csv")):
        links[r["product_id"]].add(r["doc_id"])
        plinks[r["doc_id"]].add(r["product_id"])
    shared = {d for d, ps in plinks.items() if len(ps) >= 10}
    used_sec, used_cl, used_names = set(), set(), set()
    for f in EVAL.glob("*.jsonl"):
        for l in open(f):
            q = json.loads(l)
            g = q.get("gold") or {}
            used_sec.add((g.get("doc_id"), g.get("section_seq")))
            used_cl.add(g.get("clause_id"))
            for k in ("name", "product_name", "word"):
                used_names.add(q.get(k))
            for n in q.get("products") or []:
                used_names.add(n)
            if q.get("product_id"):
                used_names.add(pid_name.get(q["product_id"]))
    secs = defaultdict(list)
    for p in (ROOT / "data" / "sections").glob("*.jsonl"):
        for l in open(p):
            s = json.loads(l)
            body = s["text"].split("\n", 1)[1] if "\n" in s["text"] else ""
            if s["section_path"] != "(머리말)" and len(body) >= 40 and (s["doc_id"], s["seq"]) not in used_sec:
                if s["is_boilerplate"] and (not s.get("clause_id") or int(s["clause_id"]) in used_cl):
                    continue
                secs[s["doc_id"]].append(s)

    def split_of(name):
        return "test" if int(hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest(), 16) % 2 else "dev"

    free = sorted(n for n in by_name if n not in used_names)   # 판매 중지·비교용 (이름이 새것이어야 함)
    rnd.shuffle(free)
    any_names = sorted(by_name)   # 상품 사실·공통 약관·없는 값: 섹션만 겹치지 않으면 상품은 다시 써도 됨 (안 쓴 상품만으로는 거의 안 남음)
    rnd.shuffle(any_names)
    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)

    def call(instr, inp, fmt):
        return client.responses.parse(model=env["CHAT_MODEL"], instructions=instr, input=inp, text_format=fmt).output_parsed

    facts, have = [], Counter()
    # 상품 사실: 전용 문서 섹션
    fact_jobs = []
    for n in any_names:
        sp = split_of(n)
        if have[("fact", sp)] >= N["fact"] + 4:
            continue
        pid = by_name[n]["product_id"]
        cand = [s for d in links[pid] if d not in shared for s in secs[d] if not s["is_boilerplate"]]
        if cand:
            fact_jobs.append((sp, pid, rnd.choice(cand)))
            have[("fact", sp)] += 1
    # 공통 약관: 공통 문서 섹션 + 연결된 (안 쓴) 상품
    common_jobs = []
    pool = [s for d in COMMON_DOCS for s in secs[d]]
    rnd.shuffle(pool)
    for s in pool:
        cands = [pid for pid in sorted(plinks[s["doc_id"]]) if pid_name[pid] in by_name and by_name[pid_name[pid]]["product_id"] == pid]
        if not cands:
            continue
        pid = rnd.choice(cands)
        sp = split_of(pid_name[pid])
        if have[("common", sp)] >= N["common"] + 6:
            continue
        common_jobs.append((sp, pid, s))
        have[("common", sp)] += 1
        if all(have[("common", x)] >= N["common"] + 6 for x in SPLITS):
            break

    def gen_fact(job):
        sp, pid, s = job
        q = call(GEN_PROMPT + "\n- 질문에 약관·설명서 같은 문서 이름을 넣지 않는다. 상품 이름은 정식 이름 그대로.", f"상품 이름: {pid_name[pid]}\n문서: {docs[s['doc_id']]['title']}\n\n{s['text']}", GenQ)
        app = None
        if s["doc_id"] in COMMON_DOCS:
            app = call(APPLY_PROMPT, f"[상품] {pid_name[pid]} ({by_name[pid_name[pid]]['subcategory']})\n\n[약관 조항]\n{s['text'][:1500]}", Apply)
        return job, q, app

    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(gen_fact, fact_jobs + common_jobs))
    dropped = Counter()
    for (sp, pid, s), q, app in res:
        typ = "common" if s["doc_id"] in COMMON_DOCS else "fact"
        if sum(1 for f in facts if f["kind"] == typ and f["split"] == sp) >= N[typ]:
            continue
        if not q or not q.usable or not lines_in_order(q.evidence, ns(s["text"])) or pid_name[pid] not in q.question:
            dropped[f"{typ}: 질문 불량"] += 1
            continue
        if app is not None and not app.applies:
            dropped["common: 상품에 적용 안 되는 조항"] += 1
            continue
        cid = int(s["clause_id"]) if s["is_boilerplate"] else None
        facts.append({"kind": typ, "split": sp, "product_id": pid, "name": pid_name[pid], "formal": q.question,
                      "gold": {"clause_id": cid} if cid else {"doc_id": s["doc_id"], "section_seq": s["seq"]},
                      "answer": q.answer, "evidence": q.evidence, "category": by_name[pid_name[pid]]["category"]})

    # 없는 값: 전용 문서가 있는 상품 → 문서에 없는 값 질문 → 정말 없는지 확인
    idx = R.Index()
    abs_src = [n for n in any_names if any(d not in shared for d in links[by_name[n]["product_id"]]) and n not in {f["name"] for f in facts}][:30]

    def gen_abs(n):
        pid = by_name[n]["product_id"]
        txt = "\n\n".join(f"### {docs[d]['title']}\n" + "\n".join(s["text"] for s in secs[d] if not s["is_boilerplate"]) for d in links[pid] if d not in shared)[:16000]
        return n, call(GEN_A, f"상품 이름: {n}\n\n{txt}", GenA)

    with ThreadPoolExecutor(6) as ex:
        ares = list(ex.map(gen_abs, abs_src))
    abs_cand = [(n, q) for n, q in ares if q and q.usable and n in q.question]
    qv = [idx.text_vec(q.question) for _, q in abs_cand]   # 질문 벡터는 로컬 KURE로 바로
    from s20_absent_set import pool_for

    def chk(arg):
        (n, q), v = arg
        pool = pool_for(idx, q.question, v, by_name[n]["product_id"])
        body = "\n\n".join(f"<{k}>\n{idx.items[i]['text'][:1500]}" for k, i in enumerate(pool))
        return n, q, call(CHECK, f"[질문] {q.question}\n[질문 상품] {n}\n\n[후보]\n{body}", Check)

    with ThreadPoolExecutor(6) as ex:
        cres = list(ex.map(chk, zip(abs_cand, qv)))
    for n, q, c in cres:
        sp = split_of(n)
        if c.answerable:
            dropped["absent: 문서에 답이 있음"] += 1
            continue
        if sum(1 for f in facts if f["kind"] == "absent" and f["split"] == sp) < N["absent"]:
            facts.append({"kind": "absent", "split": sp, "name": n, "product_id": by_name[n]["product_id"], "formal": q.question,
                          "absent_what": q.absent_what, "near_miss": q.near_miss, "category": by_name[n]["category"]})

    # 판매 중지: 판매 중지 목록에만 있는 이름 (예금·대출)
    disc = [r["name"] for r in csv.DictReader(open(META / "discontinued.csv")) if r["category"] in ("저축상품", "대출상품") and r["also_on_sale"] == "0"]
    disc = sorted({n for n in disc if n not in by_name and n not in used_names})
    rnd.shuffle(disc)
    for n in disc:
        sp = split_of(n)
        if sum(1 for f in facts if f["kind"] == "disc" and f["split"] == sp) < N["disc"]:
            facts.append({"kind": "disc", "split": sp, "name": n, "product_id": None, "formal": f"{n} 금리와 가입 조건이 어떻게 되나요?"})

    # 비교: 전용 설명서 추출이 ok인 판매 중 상품 쌍 (같은 대분류)
    ext = {}
    for f in (ROOT / "data" / "extract" / "full").glob("*.json"):
        x = json.load(open(f))
        if x.get("status") == "ok":
            ext[x["doc_id"]] = x["extraction"]
    own = {}
    for n, p in by_name.items():
        ds = [d for d in links[p["product_id"]] if len(plinks[d]) == 1 and d in ext and docs[d]["doc_type"] == "상품설명서"]
        if len(ds) == 1:
            own[n] = ds[0]
    groups = defaultdict(list)
    for n in own:
        groups[(by_name[n]["category"], split_of(n))].append(n)
    pairs = []
    for (c, sp), ns_ in sorted(groups.items()):
        ns_ = sorted(ns_)
        rnd.shuffle(ns_)
        pairs += [(sp, ns_[i], ns_[i + 1]) for i in range(0, len(ns_) - 1, 2)]
    rnd.shuffle(pairs)

    def compact(n):
        x = ext[own[n]]
        return json.dumps({"상품": n, "summary": x.get("summary"), "target_customer": x.get("target_customer"), "detail": x.get("detail"),
                           "rates": [{k: r.get(k) for k in ("rate_kind", "rate_min", "rate_max", "term_min_m", "term_max_m", "condition", "source_text")} for r in x.get("rates", [])][:12]}, ensure_ascii=False)

    with ThreadPoolExecutor(6) as ex:
        pres = list(ex.map(lambda t: (t, call(COMPARE_PROMPT, f"[상품 A]\n{compact(t[1])}\n\n[상품 B]\n{compact(t[2])}", CQ)), pairs[:24]))
    sec_text = {d: ns("\n".join(json.loads(l)["text"] for l in open(ROOT / "data" / "sections" / f"{d}.jsonl"))) for d in set(own.values())}
    for (sp, a, b), q in pres:
        if sum(1 for f in facts if f["kind"] == "compare" and f["split"] == sp) >= N["compare"]:
            continue
        if not q or not q.usable or a not in q.question or b not in q.question:
            continue
        if not (lines_in_order(q.evidence_a, sec_text[own[a]]) and lines_in_order(q.evidence_b, sec_text[own[b]])):
            dropped["compare: 근거가 설명서에 없음"] += 1
            continue
        facts.append({"kind": "compare", "split": sp, "name": f"{a} / {b}", "products": [a, b], "product_ids": [by_name[a]["product_id"], by_name[b]["product_id"]],
                      "docs": [own[a], own[b]], "formal": q.question, "attribute": q.attribute, "answer": q.answer, "evidence": [q.evidence_a, q.evidence_b]})

    # 실제 말투로 바꿔 쓰기 (사실마다 말투 1~2개)
    keys = list(STYLES)
    for f in facts:
        f["styles"] = rnd.sample(keys, rnd.choice([1, 2]))

    def rewrite(f):
        return f, call(REAL_PROMPT.format(styles=" / ".join(STYLES[s] for s in f["styles"])), f"[질문] {f['formal']}", RQ)

    with ThreadPoolExecutor(6) as ex:
        rres = list(ex.map(rewrite, facts))
    out = defaultdict(list)
    for n, (f, r) in enumerate(rres):
        if not r or not r.question.strip():
            continue
        fid = f"R{f['split'][0].upper()}{n:03d}"
        for style, qtext in (("formal", f["formal"]), ("real", r.question.strip())):
            base = {"qid": f"{fid}{'F' if style == 'formal' else 'R'}", "fact_id": fid, "style": style, "styles": f["styles"] if style == "real" else [],
                    "question": qtext, "split": f"real_{f['split']}"}
            k = f["kind"]
            if k in ("fact", "common"):
                out[f"retrieval_real_{f['split']}"].append({**base, "type": "A" if k == "fact" else "B-common", "product_id": f["product_id"], "category": f["category"],
                                                            "gold": f["gold"], "answer": f["answer"], "evidence": f["evidence"]})
            elif k == "absent":
                out[f"absent_real_{f['split']}"].append({**base, "type": "A-own", "product_id": f["product_id"], "product_name": f["name"], "category": f["category"],
                                                         "absent_what": f["absent_what"], "near_miss": f["near_miss"]})
            elif k == "disc":
                out[f"status_real_{f['split']}"].append({**base, "type": "disc", "name": f["name"], "product_id": None})
            else:
                out[f"compare_real_{f['split']}"].append({**base, "type": "compare", **{x: f[x] for x in ("attribute", "products", "product_ids", "docs", "answer", "evidence")}})
    for p in targets:
        with open(p, "w") as fo:
            for r in out[p.stem]:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(p.name, len(out[p.stem]))
    print("사실", Counter((f["kind"], f["split"]) for f in facts), "| 버림", dict(dropped))
    print("말투", Counter(s for f in facts for s in f["styles"]))


if __name__ == "__main__":
    main()
