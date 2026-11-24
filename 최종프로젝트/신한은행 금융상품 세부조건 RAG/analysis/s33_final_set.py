"""13-1 최종 검증 세트 — 지금까지 잰 모든 질문 유형을 한 세트에 모아 크게, 지금 코드를 고정한 채 한 번만 채점

사용: python analysis/s33_final_set.py gen            → data/eval/{retrieval,absent,status,compare,gap}_final.jsonl (한 번만, 있으면 멈춤)
      python analysis/s33_final_set.py lock --drop=QID,...   공통 약관·공용 설명서 후보 중 사람이 뺀 것을 지우고 유형별 20개로 자른 뒤 sha256
      이어서 질문 임베딩(s10 embed-questions-local) · 보완 정답(s17 final)
      채점: s12 final · s15 final 5 --final · s21 final --final · s27 final --final (모두 지금 기본 버전)
구성 (seed 20261130, OpenAI CHAT_MODEL) — 유형마다 절반은 단정한 말투, 절반은 실제 말투(s31 지시문으로 바꿔 씀)
  상품 사실 120 (저축 50 · 대출 50 · 기타·ELD 20): 상품 전용 문서(10개 미만 상품이 공유)의 새 섹션 (s11 지시문, 대출·기타는 상품당 최대 3섹션)
  공통 약관 20: 공통 약관 7개의 새 섹션이 3개뿐이라 **이미 정답으로 쓴 섹션을 다른 상품·새 질문으로** — 조항이 그 상품 종류에 적용되는지 사람이 확인
  공용 대출 설명서 20: 가계·기업·전세자금·주택도시기금 설명서와 대출거래약정서의 새 섹션 (대출 270개 중 240개는 이 문서만 가짐) — 적용 여부 사람 확인
  없는 값 40: s20 지시문 + 정말 없는지 확인(s20 CHECK)
  판매 상태 40: 판매 중지 20 · 판매 중지 줄임말 10 · 판매 중 10 (s26·s28 지시문, 판매 중지 이름은 예전 세트에 안 쓴 것)
  두 상품 비교 20: 전용 설명서 추출값 + 근거 확인 (s26), 예전 세트에 없던 짝
  가짜 상품 20: 실제 이름 앞에 지어낸 말 7 · 뒤에 붙인 말 7 · 괄호로 끝나는 이름 뒤 6 (s29 지시문)
  상품 이름은 거의 다 예전 세트에 한 번은 나와서, 상품 사실·없는 값은 정답 섹션만 새것으로
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
from s8_extract import lines_in_order, load_env, ns  # noqa: E402
from s11_evalset import GEN_PROMPT, GenQ  # noqa: E402
from s18_alias_check import PROMPT as NICK_PROMPT, Q as NQ  # noqa: E402
from s20_absent_set import CHECK, GEN_A, Check, GenA, pool_for  # noqa: E402
from s22_alias_check2 import norm as fnorm  # noqa: E402
from s25_common_set import NO_DOC  # noqa: E402
from s26_status_compare_set import COMPARE_PROMPT, CQ, FOCUS, SQ, STATUS_PROMPT  # noqa: E402
from s29_fullname_fake_set import FAKE_PROMPT, FQ, MODES  # noqa: E402
from s31_realistic_set import APPLY_PROMPT, COMMON_DOCS, REAL_PROMPT, RQ, STYLES, Apply  # noqa: E402

META, EVAL = ROOT / "data" / "meta", ROOT / "data" / "eval"
SEED = 20261130
OUT = {k: EVAL / f"{k}_final.jsonl" for k in ("retrieval", "absent", "status", "compare", "gap")}
N_FACT = {"저축상품": 50, "대출상품": 50, "기타": 20}
SHARED_LOAN = ["D038", "D059", "D066", "D067", "D229", "D234"]
N_COMMON, N_SHARED, N_ABSENT, N_DISC, N_NICK, N_ONSALE, N_CMP = 20, 20, 40, 20, 10, 10, 20
N_FAKE = {"pre": 7, "suf": 7, "paren": 6}
PROFILE_KEYS = ("상품유형", "예금과목", "저축방법", "가입대상", "상품개요", "예금종류", "상품의 개요", "적립방법", "상품특징", "대출대상", "자금용도", "대출용도")
NAME_KEEP = ("disc", "onsale", "fake", "compare")   # 이름이 그대로 남아야 하는 유형 — 줄임말 말투 안 씀


def gen():
    if any(p.exists() for p in OUT.values()):
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
    used_sec, used_cl, used_names, sec_prods, used_pairs = set(), set(), set(), defaultdict(set), set()
    for f in EVAL.glob("*.jsonl"):
        for l in open(f):
            q = json.loads(l)
            g = q.get("gold") if isinstance(q.get("gold"), dict) else {}
            used_sec.add((g.get("doc_id"), g.get("section_seq")))
            used_cl.add(g.get("clause_id"))
            if g.get("doc_id") and q.get("product_id"):
                sec_prods[(g["doc_id"], g.get("section_seq"))].add(q["product_id"])
            for k in ("name", "product_name", "word", "nickname"):
                used_names.add(q.get(k))
            for n in q.get("products") or []:
                used_names.add(n)
            if q.get("products"):
                used_pairs.add(frozenset(q["products"]))
            if q.get("product_id"):
                used_names.add(pid_name.get(q["product_id"]))
    allsec, secs = {}, defaultdict(list)
    for p in (ROOT / "data" / "sections").glob("*.jsonl"):
        for l in open(p):
            s = json.loads(l)
            allsec[(s["doc_id"], s["seq"])] = s
            body = s["text"].split("\n", 1)[1] if "\n" in s["text"] else ""
            if s["section_path"] != "(머리말)" and len(body) >= 40 and (s["doc_id"], s["seq"]) not in used_sec:
                if s["is_boilerplate"] and (not s.get("clause_id") or int(s["clause_id"]) in used_cl):
                    continue
                secs[s["doc_id"]].append(s)

    def profile(pid):
        own = [s for (d, _), s in sorted(allsec.items()) if d in links[pid] and d not in shared and any(k in s["section_path"] for k in PROFILE_KEYS)]
        txt = "\n".join(s["text"] for s in own)[:1500]
        p = by_name[pid_name[pid]]
        return f"분류: {p['category']} / {p['subcategory']}\n" + (txt or "(상품 전용 문서 없음)")

    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)

    def call(instr, inp, fmt):
        return client.responses.parse(model=env["CHAT_MODEL"], instructions=instr, input=inp, text_format=fmt).output_parsed

    def pmap(fn, xs):
        with ThreadPoolExecutor(6) as ex:
            return list(ex.map(fn, xs))

    names = sorted(by_name)
    rnd.shuffle(names)
    dropped = Counter()

    # 1. 상품 사실 (전용 문서 새 섹션) — 여유 있게 후보
    fact_jobs, taken = [], set()
    for cat, n in N_FACT.items():
        per = 1 if cat == "저축상품" else 3
        pool = [x for x in names if by_name[x]["category"] == cat or (cat == "기타" and by_name[x]["category"] in ("기타상품", "복합금융상품"))]
        cnt = 0
        for rnd_i in range(per):
            for x in pool:
                if cnt >= n + 12:
                    break
                pid = by_name[x]["product_id"]
                cand = [s for d in sorted(links[pid]) if d not in shared for s in secs[d] if not s["is_boilerplate"] and (s["doc_id"], s["seq"]) not in taken]
                if cand:
                    s = rnd.choice(cand)
                    taken.add((s["doc_id"], s["seq"]))
                    fact_jobs.append((cat, pid, s))
                    cnt += 1

    def gen_q(job):
        kind, pid, s = job[0], job[1], job[2]
        q = call(GEN_PROMPT + NO_DOC, f"상품 이름: {pid_name[pid]}\n문서: {docs[s['doc_id']]['title']}\n\n{s['text']}", GenQ)
        app = None
        if kind in ("common", "shared"):
            app = call(APPLY_PROMPT, f"[상품] {pid_name[pid]}\n{profile(pid)}\n\n[약관 조항] ({docs[s['doc_id']]['title']})\n{s['text'][:1500]}", Apply)
        return job, q, app

    # 2. 공통 약관: 예전 정답 섹션을 다른 상품으로
    common_jobs = []
    pool = sorted(k for k in used_sec if k[0] in COMMON_DOCS and k in allsec and not allsec[k]["is_boilerplate"])
    rnd.shuffle(pool)
    for k in pool:
        cands = sorted(pid for pid in plinks[k[0]] if pid not in sec_prods[k] and by_name[pid_name[pid]]["product_id"] == pid)
        if cands and len(common_jobs) < N_COMMON * 2:
            common_jobs.append(("common", rnd.choice(cands), allsec[k]))
    # 3. 공용 대출 설명서·약정서 새 섹션
    shared_jobs = []
    pool = [s for d in SHARED_LOAN for s in secs[d]]
    rnd.shuffle(pool)
    for s in pool[:N_SHARED * 2]:
        cands = sorted(pid for pid in plinks[s["doc_id"]] if by_name[pid_name[pid]]["product_id"] == pid)
        shared_jobs.append(("shared", rnd.choice(cands), s))

    rows = defaultdict(list)   # 종류 → 사실 dict (말투 바꾸기 전)
    for (kind, pid, s), q, app in pmap(gen_q, fact_jobs + common_jobs + shared_jobs):
        key = "fact" if kind in N_FACT else kind
        if not q or not q.usable or not lines_in_order(q.evidence, ns(s["text"])) or pid_name[pid] not in q.question:
            dropped[f"{key}: 질문 불량"] += 1
            continue
        if app is not None and not app.applies:
            dropped[f"{key}: LLM이 적용 안 된다고 봄"] += 1
            continue
        if kind in N_FACT and sum(1 for r in rows["fact"] if r["cat"] == kind) >= N_FACT[kind]:
            continue
        cid = int(s["clause_id"]) if s["is_boilerplate"] and s.get("clause_id") else None
        rows[key].append({"kind": key, "cat": kind if kind in N_FACT else "공통", "product_id": pid, "name": pid_name[pid], "formal": q.question,
                          "gold": {"clause_id": cid} if cid else {"doc_id": s["doc_id"], "section_seq": s["seq"]},
                          "answer": q.answer, "evidence": q.evidence, "category": by_name[pid_name[pid]]["category"],
                          "apply_note": app.note if app else None, "section": f"{docs[s['doc_id']]['title']} > {s['section_path']}"})

    # 4. 없는 값
    idx = R.Index()
    fact_names = {r["name"] for r in rows["fact"]}
    abs_src = [x for x in names if any(d not in shared for d in links[by_name[x]["product_id"]]) and x not in fact_names][:N_ABSENT + 30]

    def gen_abs(x):
        pid = by_name[x]["product_id"]
        txt = "\n\n".join(f"### {docs[d]['title']}\n" + "\n".join(s["text"] for (dd, _), s in sorted(allsec.items()) if dd == d and not s["is_boilerplate"])
                          for d in sorted(links[pid]) if d not in shared)[:16000]
        return x, call(GEN_A, f"상품 이름: {x}\n\n{txt}", GenA)

    abs_cand = [(x, q) for x, q in pmap(gen_abs, abs_src) if q and q.usable and x in q.question]
    qv = [idx.text_vec(q.question) for _, q in abs_cand]

    def chk(arg):
        (x, q), v = arg
        pool_ = pool_for(idx, q.question, v, by_name[x]["product_id"])
        body = "\n\n".join(f"<{k}>\n{idx.items[i]['text'][:1500]}" for k, i in enumerate(pool_))
        return x, q, call(CHECK, f"[질문] {q.question}\n[질문 상품] {x}\n\n[후보]\n{body}", Check)

    for x, q, c in pmap(chk, list(zip(abs_cand, qv))):
        if c.answerable:
            dropped["absent: 문서에 답이 있음"] += 1
        elif len(rows["absent"]) < N_ABSENT:
            rows["absent"].append({"kind": "absent", "name": x, "product_id": by_name[x]["product_id"], "formal": q.question,
                                   "absent_what": q.absent_what, "near_miss": q.near_miss, "category": by_name[x]["category"]})

    # 5. 판매 상태: 판매 중지(예전 세트에 안 쓴 이름) · 판매 중지 줄임말 · 판매 중
    disc = sorted({r["name"] for r in csv.DictReader(open(META / "discontinued.csv"))
                   if r["category"] in ("저축상품", "대출상품") and r["also_on_sale"] == "0" and r["name"] not in by_name and r["name"] not in used_names})
    rnd.shuffle(disc)
    onsale = [x for x in names if len(fnorm(x)) >= 5][:N_ONSALE + 5]
    st_jobs = [("disc", x) for x in disc[:N_DISC + 5]] + [("onsale", x) for x in onsale]
    for (kind, x), q in pmap(lambda j: (j, call(STATUS_PROMPT.format(focus=FOCUS[j[0]]), f"상품 이름: {j[1]}", SQ)), st_jobs):
        n_max = N_DISC if kind == "disc" else N_ONSALE
        if q and x in q.question and len(rows[kind]) < n_max:
            rows[kind].append({"kind": kind, "name": x, "formal": q.question, "product_id": by_name[x]["product_id"] if x in by_name else None})
    for x, q in pmap(lambda x: (x, call(NICK_PROMPT, f"상품: {x}", NQ)), disc[N_DISC + 5:N_DISC + 5 + N_NICK * 2]):
        if len(rows["disc_nick"]) >= N_NICK or not q:
            continue
        if fnorm(q.nickname) == fnorm(x) or fnorm(x) in fnorm(q.question) or q.nickname not in q.question:
            dropped["disc_nick: 줄임말 아님"] += 1
            continue
        rows["disc_nick"].append({"kind": "disc_nick", "name": x, "nickname": q.nickname, "formal": q.question, "product_id": None})

    # 6. 가짜 상품
    real_n = {fnorm(x) for x in by_name} | {fnorm(r["name"]) for r in csv.DictReader(open(META / "discontinued.csv"))}
    paren = [x for x in names if x.rstrip().endswith(")")]
    plain = [x for x in names if not x.rstrip().endswith(")") and len(fnorm(x)) >= 5]
    f_jobs = [("pre", x) for x in plain[:N_FAKE["pre"] + 5]] + [("suf", x) for x in plain[N_FAKE["pre"] + 5:N_FAKE["pre"] + N_FAKE["suf"] + 10]] + \
             [("paren", x) for x in paren[:N_FAKE["paren"] + 5]]

    def f_one(j):
        mode, x = j
        return j, call(FAKE_PROMPT.format(mode=MODES["pre" if mode == "pre" else "suf"]), f"[실제 상품 이름] {x}", FQ)

    fcnt = Counter()
    for (mode, x), q in pmap(f_one, f_jobs):
        if not q or fcnt[mode] >= N_FAKE[mode]:
            continue
        fn = fnorm(q.fake_name)
        if x not in q.fake_name or q.fake_name not in q.question or fn in real_n or fn == fnorm(x) or (mode != "pre" and not q.fake_name.startswith(x)):
            dropped["fake: 형식 안 맞음"] += 1
            continue
        typ = "fake_paren" if mode == "paren" else "fake_full"
        rows["fake"].append({"kind": "fake", "type": typ, "mode": mode, "name": q.fake_name, "word": x, "formal": q.question, "product_id": None})
        fcnt[mode] += 1

    # 7. 두 상품 비교 (예전에 없던 짝)
    ext = {}
    for f in (ROOT / "data" / "extract" / "full").glob("*.json"):
        x = json.load(open(f))
        if x.get("status") == "ok":
            ext[x["doc_id"]] = x["extraction"]
    own = {}
    for x, p in by_name.items():
        ds = [d for d in links[p["product_id"]] if len(plinks[d]) == 1 and d in ext and docs[d]["doc_type"] == "상품설명서"]
        if len(ds) == 1:
            own[x] = ds[0]
    groups = defaultdict(list)
    for x in sorted(own):
        groups[by_name[x]["category"]].append(x)
    pairs = []
    for c, xs in sorted(groups.items()):
        rnd.shuffle(xs)
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                if frozenset((xs[i], xs[j])) not in used_pairs:
                    pairs.append((xs[i], xs[j]))
    rnd.shuffle(pairs)
    chosen, cnt_p = [], Counter()
    for a, b in pairs:   # 한 상품이 너무 여러 번 나오지 않게 (최대 2번)
        if cnt_p[a] < 2 and cnt_p[b] < 2:
            chosen.append((a, b))
            cnt_p[a] += 1
            cnt_p[b] += 1
        if len(chosen) >= N_CMP * 2:
            break

    def compact(x):
        e = ext[own[x]]
        return json.dumps({"상품": x, "summary": e.get("summary"), "target_customer": e.get("target_customer"), "detail": e.get("detail"),
                           "rates": [{k: r.get(k) for k in ("rate_kind", "rate_min", "rate_max", "term_min_m", "term_max_m", "condition", "source_text")} for r in e.get("rates", [])][:12]}, ensure_ascii=False)

    sec_text = {d: ns("\n".join(json.loads(l)["text"] for l in open(ROOT / "data" / "sections" / f"{d}.jsonl"))) for d in set(own.values())}
    for (a, b), q in pmap(lambda t: (t, call(COMPARE_PROMPT, f"[상품 A]\n{compact(t[0])}\n\n[상품 B]\n{compact(t[1])}", CQ)), chosen):
        if len(rows["compare"]) >= N_CMP:
            continue
        if not q or not q.usable or a not in q.question or b not in q.question:
            dropped["compare: 질문 불량"] += 1
            continue
        if not (lines_in_order(q.evidence_a, sec_text[own[a]]) and lines_in_order(q.evidence_b, sec_text[own[b]])):
            dropped["compare: 근거가 설명서에 없음"] += 1
            continue
        rows["compare"].append({"kind": "compare", "name": f"{a} / {b}", "products": [a, b], "product_ids": [by_name[a]["product_id"], by_name[b]["product_id"]],
                                "docs": [own[a], own[b]], "formal": q.question, "attribute": q.attribute, "answer": q.answer, "evidence": [q.evidence_a, q.evidence_b]})

    # 8. 말투: 유형마다 절반을 실제 말투로 (이름이 중요한 유형은 줄임말 말투 빼고, 바꾼 뒤에도 이름이 그대로 있어야 함)
    items = [r for k in rows for r in rows[k]]
    for k in rows:
        flags = [i % 2 == 1 for i in range(len(rows[k]))]
        rnd.shuffle(flags)
        for r, fl in zip(rows[k], flags):
            keys = [s for s in STYLES if not (s == "nick" and r["kind"] in NAME_KEEP)]
            r["styles"] = rnd.sample(keys, rnd.choice([1, 2])) if fl else []

    def rewrite(r):
        if not r["styles"]:
            return r, None
        return r, call(REAL_PROMPT.format(styles=" / ".join(STYLES[s] for s in r["styles"])), f"[질문] {r['formal']}", RQ)

    out = defaultdict(list)
    for n, (r, rq) in enumerate(pmap(rewrite, items)):
        q = r["formal"]
        style = "formal"
        if rq and rq.question.strip():
            cand = rq.question.strip()
            must = r.get("nickname") if r["kind"] == "disc_nick" else None
            names_ok = all(x in cand for x in r["products"]) if r["kind"] == "compare" else (r["name"] in cand if r["kind"] in NAME_KEEP else True)
            if names_ok and (must is None or must in cand):
                q, style = cand, "real"
            else:
                dropped["실제 말투: 이름이 바뀌어 단정 말투로 둠"] += 1
        base = {"style": style, "styles": r["styles"] if style == "real" else [], "formal_question": r["formal"], "question": q, "split": "final"}
        k = r["kind"]
        if k in ("fact", "common", "shared"):
            typ = {"fact": "A", "common": "B-common", "shared": "B-shared"}[k]
            out["retrieval"].append({"qid": f"F{ {'fact': 'A', 'common': 'C', 'shared': 'S'}[k] }{n:03d}", "type": typ, **base, "product_id": r["product_id"], "category": r["category"],
                                     "fact_cat": r["cat"], "gold": r["gold"], "answer": r["answer"], "evidence": r["evidence"],
                                     "apply_note": r["apply_note"], "section": r["section"]})
        elif k == "absent":
            out["absent"].append({"qid": f"FX{n:03d}", "type": "A-own", **base, "product_id": r["product_id"], "product_name": r["name"],
                                  "category": r["category"], "absent_what": r["absent_what"], "near_miss": r["near_miss"]})
        elif k in ("disc", "onsale"):
            out["status"].append({"qid": f"FD{n:03d}", "type": k, **base, "name": r["name"], "product_id": r["product_id"]})
        elif k == "disc_nick":
            out["gap"].append({"qid": f"FN{n:03d}", "type": "disc_nick", **base, "name": r["name"], "nickname": r["nickname"], "product_id": None})
        elif k == "fake":
            out["gap"].append({"qid": f"FF{n:03d}", "type": r["type"], "mode": r["mode"], **base, "name": r["name"], "word": r["word"], "product_id": None})
        else:
            out["compare"].append({"qid": f"FP{n:03d}", "type": "compare", **base, **{x: r[x] for x in ("attribute", "products", "product_ids", "docs", "answer", "evidence")}})
    for k, p in OUT.items():
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out[k]))
        print(p.name, len(out[k]), dict(Counter(r["type"] for r in out[k])), "| 실제 말투", sum(r["style"] == "real" for r in out[k]))
    print("상품 사실 분류", dict(Counter(r["fact_cat"] for r in out["retrieval"] if r["type"] == "A")))
    print("버림", dict(dropped))


def lock():
    drop = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--drop=")), [])
    p = OUT["retrieval"]
    rows, cnt = [], Counter()
    for q in map(json.loads, open(p)):
        if q["qid"] in drop:
            continue
        if q["type"] in ("B-common", "B-shared"):
            if cnt[q["type"]] >= (N_COMMON if q["type"] == "B-common" else N_SHARED):
                continue
            cnt[q["type"]] += 1
        rows.append(q)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(p.name, len(rows), dict(Counter(r["type"] for r in rows)))
    for k, f in OUT.items():
        print(f.name, hashlib.sha256(f.read_bytes()).hexdigest())


if __name__ == "__main__":
    {"gen": gen, "lock": lock}[sys.argv[1] if len(sys.argv) > 1 else "gen"]()
