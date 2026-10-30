"""새 ERD(plan_v2) 정보로 v1 검색 실패가 고쳐지는지 미리 재 보는 모의 실험.

적재 파이프라인을 다 만들기 전에, v2 테이블이 담을 정보를 v1 결과물에서 만들어 검색 순위를 다시 매긴다.
- v1 DB(rag.chunks)의 가명처리 본문·토큰·임베딩, v1 가명처리 결과(탐지기 위치·토큰)를 사용
- 역할·부서·지역은 적재 단계처럼 로컬에서 원문 주변 단어로 뽑음 (외부 전송 없음)
- 새로 임베딩하는 것은 가명처리된 텍스트뿐: 질의, 머리글 붙인 문서, 표 줄 청크
- 정답 라벨·인물 ID는 채점에만 사용

변형 (누적)
  S0 v1 재현 (문서 청크, 토큰 개수 → 의미 점수)
  S1 + 머리글 임베딩 (documents.doc_type/title → chunks.header)
  S2 + 표 줄 청크 (chunks.kind=row)
  S3 + 토큰 희귀도 (tokens.doc_freq)
  S4 + 문서 종류 의도 (documents.doc_type)
  S5 + 지역 (address_regions)
  S6 + 처리 상태 (doc_fields)
  S7 + 언급 역할 (mentions.role)
  S8 + 개체 묶기 (entities: 역할을 같은 개체의 다른 문서로 전파)
  L  = S0 임베딩 그대로 + S3~S8 (새 임베딩 없이 얻는 몫)

실행 (실전프로젝트2 폴더, v1 DB 실행 상태): python -X utf8 v2/eval/simulate_erd.py
"""
import collections
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

V2 = Path(__file__).resolve().parents[1]
ROOT = V2.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from starter import db  # noqa: E402
from starter.embed import embed  # noqa: E402
from eval_holdout_retrieval import score  # noqa: E402
sys.path.insert(0, str(V2))
from pseudo import roles  # noqa: E402

OUT = V2 / "outputs" / "sim"
CACHE = V2 / "outputs" / "sim_emb_cache.json"
TOKEN_RE = re.compile(r"<([A-Z_]+)_[0-9a-f]{8}>")
SEP_RE = re.compile(r"^\s*\|?\s*:?-{3}")

# 문서 종류 이름에서 나온 단어만 사용 (질의 문구에 맞춘 동의어는 넣지 않음, '부고'만 예외)
INTENT = {
    "공지": ["공지"], "녹취요약": ["녹취"], "출장보고": ["출장"], "한도심사결과": ["한도"],
    "분쟁조정": ["분쟁"], "감사지적": ["감사 지적", "감사지적"], "VOC요약": ["VOC"],
    "법인카드소명": ["법인카드", "소명"], "경조사공지": ["경조사", "부고"], "카드발급심사": ["카드 발급", "카드발급"],
    "가맹점변경": ["가맹점"], "고객편지": ["편지"], "채용메일": ["채용"], "권한신청": ["권한"],
    "피해접수": ["피해"], "영문혼용메일": ["영문"], "대리인위임확인": ["위임"], "자동이체신청서": ["자동이체"],
    "우편반송처리": ["반송"], "해외송금접수": ["해외송금"],
}
SIDO = {
    "서울특별시": ["서울"], "부산광역시": ["부산"], "대구광역시": ["대구"], "인천광역시": ["인천"],
    "광주광역시": ["광주"], "대전광역시": ["대전"], "울산광역시": ["울산"], "세종특별자치시": ["세종"],
    "경기도": ["경기"], "강원특별자치도": ["강원", "강원도"], "충청북도": ["충북"], "충청남도": ["충남"],
    "전북특별자치도": ["전북", "전라북도"], "전라남도": ["전남"], "경상북도": ["경북"], "경상남도": ["경남"],
    "제주특별자치도": ["제주", "제주도"],
}
SIDO_ALIAS = {a: full for full, al in SIDO.items() for a in [full, *al]}
STATUS_OPEN = ["처리 중", "진행 중", "아직", "안 끝난", "미처리"]
STATUS_DONE = ["완료", "마무리", "해결", "끝난"]
STAFF = ("직원", "대리", "과장", "팀장", "매니저", "사원", "부장", "차장", "주임", "상담사", "상담원",
         "본부장", "지점장", "선임", "책임", "수석", "실장", "센터장", "심사역")
ORG_RE = re.compile(r"([가-힣A-Za-z]+(?:팀|센터|본부|지점|부|실))\s*$")
ID_LABELS = ("PHONE", "RRN", "RESIDENT", "EMAIL", "ACCOUNT", "CARD")


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ───────────────────────── 임베딩 (가명처리 텍스트만, 캐시) ─────────────────────────
def _hk(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embed_all(texts):
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [t for t in dict.fromkeys(texts) if _hk(t) not in cache]
    for i, t in enumerate(todo, 1):
        cache[_hk(t)] = [float(x) for x in embed([t])[0]]
        if i % 20 == 0 or i == len(todo):
            CACHE.write_text(json.dumps(cache), encoding="utf-8")
            print(f"  임베딩 {i}/{len(todo)}", flush=True)
    return {t: unit(np.asarray(cache[_hk(t)], dtype=np.float32)) for t in texts}


def unit(v):
    return v / (np.linalg.norm(v) or 1.0)


# ───────────────────────── 청크 (v2 chunks) ─────────────────────────
def split_rows(text):
    """표·목록 줄과 나머지 본문으로 나눔. 표 머리 줄(성명|소속|…)은 각 줄 앞에 붙임."""
    lines, rows, body, cols = text.split("\n"), [], [], None
    for i, line in enumerate(lines):
        if line.count("|") >= 2:
            if SEP_RE.match(line):
                continue
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if nxt.count("|") >= 2 and SEP_RE.match(nxt):
                cols = line.strip()
                continue
            rows.append((i, (cols + "\n" if cols else "") + line.strip()))
        else:
            body.append(line)
    return rows, "\n".join(body)


def toks(text):
    return {m.group(0) for m in TOKEN_RE.finditer(text)}


def build_chunks(docs):
    """변형별 청크: plain(v1), header(문서+머리글), rows(표 문서는 본문+줄 청크)."""
    out = {"plain": [], "header": [], "rows": [], "rowsplain": []}
    for d in docs.values():
        title = next((l for l in d["masked"].split("\n") if l.strip()), "")
        head = f"[{d['doc_type']}] {title}"
        full = {"doc": d["id"], "kind": "doc", "text": d["masked"], "tokens": set(d["tokens"])}
        out["plain"].append({**full, "emb_key": None})
        out["header"].append({**full, "emb_key": head + "\n" + d["masked"]})
        rows, body = split_rows(d["masked"])
        if not rows:
            out["rows"].append({**full, "emb_key": head + "\n" + d["masked"]})
            out["rowsplain"].append({**full, "emb_key": None})
            continue
        # rows: 머리글 붙임 / rowsplain: 머리글 없이 (표 문서만 새로 임베딩, 나머지는 v1 임베딩)
        for v, pre in (("rows", head + "\n"), ("rowsplain", "")):
            out[v].append({"doc": d["id"], "kind": "doc", "text": body, "tokens": toks(body),
                           "emb_key": pre + body})
            for _, r in rows:
                out[v].append({"doc": d["id"], "kind": "row", "text": r, "tokens": toks(r),
                               "emb_key": pre + r})
    return out


# ───────────────────────── 언급 역할·부서 (v2 mentions) ─────────────────────────
def mention_role(text, s, e):
    """언급 앞뒤 단어로 역할·부서. 원문은 로컬에서만 보고 결과는 비개인정보 값."""
    line_start = text.rfind("\n", 0, s) + 1
    line_end = text.find("\n", e)
    line_end = len(text) if line_end < 0 else line_end
    after, before = text[e:line_end].lstrip(" "), text[line_start:s].rstrip(" ")
    role, org = None, None
    if "|" in text[line_start:line_end] and after.startswith("|"):
        cell = after[1:].split("|")[0].strip()
        if ORG_RE.search(cell):
            role, org = "staff", cell
        elif cell and not TOKEN_RE.search(cell) and not cell[0].isdigit():
            role, org = "external", cell
    if after.startswith(("고객", "대표")):
        role = "customer"
    elif after.startswith(STAFF):
        role = "staff"
    elif before.endswith(("담당", "담당자", "상담사", "상담원", "작성자", "처리자", "직원")):
        role = "staff"
    elif before.endswith("고객"):
        role = "customer"
    m = ORG_RE.search(before)
    if m and org is None:
        org = m.group(1)
        role = role or "staff"
    return role, org, text.count("\n", 0, s)


def build_mentions(raw, pseudo, extractor="v2"):
    """doc → token → [(role, org, line_no)]. extractor: v2 = pseudo/roles.py, v1 = 위 mention_role (첫 실험)."""
    ment = collections.defaultdict(lambda: collections.defaultdict(list))
    for doc_id, ents in pseudo.items():
        text = raw[doc_id]["text"]
        persons = [e for e in ents if e["label"] == "person"]
        if extractor == "v1":
            for e in persons:
                ment[doc_id][e["token"]].append(mention_role(text, e["start"], e["end"]))
            continue
        res = roles.extract(text, [{"start": e["start"], "end": e["end"], "key": e["token"]} for e in persons])
        for e, r in zip(persons, res):
            ment[doc_id][e["token"]].append((r["role"], r["org"], text.count("\n", 0, e["start"])))
    return ment


def line_tokens(masked):
    return {i: toks(l) for i, l in enumerate(masked.split("\n"))}


# ───────────────────────── 개체 묶기 (v2 entities) ─────────────────────────
def build_entities(docs, ment):
    """이름 토큰별로 문서를 개체로 묶음. 증거: 같은 줄 식별 토큰, 역할, 부서. 정답 인물 ID 사용 안 함."""
    prof = collections.defaultdict(dict)          # token → doc → {roles, orgs, ids}
    for doc_id, by_tok in ment.items():
        lt = line_tokens(docs[doc_id]["masked"])
        for t, ms in by_tok.items():
            ids = set()
            for _, _, ln in ms:
                persons = {x for x in lt.get(ln, ()) if x.startswith("<PERSON_")}
                if persons == {t}:   # 다른 사람이 같은 줄에 있으면 번호 주인이 불분명
                    ids |= {x for x in lt[ln] if x[1:].startswith(ID_LABELS)}
            prof[t][doc_id] = {"roles": {r for r, _, _ in ms if r}, "orgs": {o for _, o, _ in ms if o}, "ids": ids}

    def conflict(a, b):
        ra, rb = a["roles"], b["roles"]
        if ra and rb and not (ra & rb):
            return True
        if a["orgs"] and b["orgs"] and not (a["orgs"] & b["orgs"]):
            return True
        for lab in ID_LABELS:
            ia = {x for x in a["ids"] if x[1:].startswith(lab)}
            ib = {x for x in b["ids"] if x[1:].startswith(lab)}
            if ia and ib and not (ia & ib):
                return True
        return False

    def merge(a, b):
        return {k: a[k] | b[k] for k in a}

    entity_of = {}                                 # (token, doc) → entity 번호
    for t, by_doc in prof.items():
        clusters = [{"docs": {d}, **p} for d, p in by_doc.items()]
        changed = True
        while changed:                             # 1) 식별 토큰 공유 → 2) 역할이 확실히 같고 충돌 없음
            changed = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    a, b = clusters[i], clusters[j]
                    share = a["ids"] & b["ids"]
                    same_role = a["roles"] and a["roles"] == b["roles"]
                    if not conflict(a, b) and (share or same_role):
                        clusters[i] = {**merge(a, b), "docs": a["docs"] | b["docs"]}
                        clusters.pop(j)
                        changed = True
                        break
                if changed:
                    break
        known = [c for c in clusters if c["roles"] or c["ids"] or c["orgs"]]
        if len(known) == 1:                        # 3) 단서 없는 문서는 개체가 하나뿐일 때만 붙임
            k = known[0]
            for c in [c for c in clusters if c is not k]:
                k["docs"] |= c["docs"]
            clusters = [k]
        for n, c in enumerate(clusters):
            for d in c["docs"]:
                entity_of[(t, d)] = (n, frozenset(c["roles"]))
    return entity_of


# ───────────────────────── 질의 분석 (ops.query_log.parsed) ─────────────────────────
def parse_query(mq):
    intent = {dt for dt, words in INTENT.items() if any(w in mq for w in words)}
    status = "처리 중" if any(w in mq for w in STATUS_OPEN) else "처리 완료" if any(w in mq for w in STATUS_DONE) else None
    region = None
    words = re.findall(r"[가-힣]+", mq)
    for i, w in enumerate(words):
        if w in SIDO_ALIAS:
            sgg = [re.sub(r"(에|쪽|으로|로)$", "", x) for x in words[i + 1:i + 3]]
            region = (SIDO_ALIAS[w], [x for x in sgg if re.fullmatch(r"[가-힣]+[시군구]", x)])
            break
    roles = {}
    for m in TOKEN_RE.finditer(mq):
        if m.group(1) != "PERSON":
            continue
        after, before = mq[m.end():].lstrip(" "), mq[:m.start()].rstrip(" ")
        if after.startswith(("고객", "대표")) or before.endswith("고객"):
            roles[m.group(0)] = "customer"
        elif after.startswith(STAFF) or before.endswith(("직원", "담당", "상담사")):
            roles[m.group(0)] = "staff"
    return {"intent": intent, "status": status, "region": region, "roles": roles}


def doc_regions(raw_text, ents):
    out = []
    for e in ents:
        if e["label"] == "address":
            w = re.findall(r"[가-힣]+", re.sub(r"^\(?\d{5}\)?\s*", "", raw_text[e["start"]:e["end"]]))
            if w and w[0] in SIDO_ALIAS:
                out.append((SIDO_ALIAS[w[0]], set(w[1:4])))
    return out


# ───────────────────────── 순위 ─────────────────────────
def rank(q, chunks, emb, docs, opt, idf, ment, entity_of, regions):
    qv, qt, p = q["vec"], set(q["tokens"]), q["parsed"]
    best = {}
    for ch in chunks:
        d = ch["doc"]
        hits = qt & ch["tokens"]
        tok = sum(idf[t] for t in hits) if opt.get("idf") else len(hits)
        role = 0
        if opt.get("role") and p["roles"]:
            for t, want in p["roles"].items():
                if opt.get("entity") and (t, d) in entity_of:
                    have = set(entity_of[(t, d)][1])
                else:
                    have = {r for r, _, _ in ment.get(d, {}).get(t, []) if r}
                if want in have:
                    role = max(role, 1)
                elif have:
                    role = -1 if role == 0 else role
            if role < 0 and opt.get("role_filter"):
                tok = 0                            # 첫 실험: 역할이 다르면 토큰 일치에서 뺌 (너무 셈)
            # 기본: 감점만 — 역할이 다르다고 확인된 문서만 같은 토큰 일치 안에서 아래로.
            # 맞는 문서를 올리면 역할 단서가 없는 정답 문서가 밀림 (2차 실험 S6 가산 0.841)
            if not opt.get("role_bonus"):
                role = min(role, 0)
        status = int(bool(opt.get("status") and p["status"] and p["status"] in ch["text"] and (hits or not qt)))
        intent = int(bool(opt.get("intent") and docs[d]["doc_type"] in p["intent"]))
        region = 0
        if opt.get("region") and p["region"]:
            sido, sgg = p["region"]
            for rs, rw in regions.get(d, []):
                if rs == sido:
                    region = max(region, 2 if sgg and all(x in rw for x in sgg) else 1 if not sgg else 0)
        sem = float(qv @ (emb[ch["emb_key"]] if ch["emb_key"] else docs[d]["v1emb"]))
        key = (tok, role, status, intent, region, sem)
        if d not in best or key > best[d]:
            best[d] = key
    return [d for d, _ in sorted(best.items(), key=lambda x: x[1], reverse=True)][:20]


def confuse_split(gold, runs, q_person):
    """Confuse@10을 같은 사람의 다른 문서 몫과 동명이인 몫으로 나눔 (정답 인물 ID로 채점만)."""
    same, hom = [], []
    for g in gold:
        conf = set(g.get("confusable_doc_ids", []))
        if not conf:
            continue
        top = runs[g["qid"]][:10]
        pid = {e["person_id"] for e in g.get("query_entities", []) if e.get("person_id")}
        s = sum(1 for d in top if d in conf and pid & q_person.get(d, set()))
        same.append(s / len(top))
        hom.append((sum(1 for d in top if d in conf) - s) / len(top))
    return sum(same) / len(same), sum(hom) / len(hom)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {d["id"]: d for d in jl(ROOT / "data/holdout_labeled.jsonl")}
    pseudo = {d["id"]: d["entities"] for d in jl(ROOT / "outputs/pseudo_holdout_E2_Proposed.jsonl")}
    base = {r["qid"]: r for r in jl(ROOT / "outputs/run_E2_Proposed.jsonl")}
    gold = jl(ROOT / "data/holdout_retrieval_queries_gold.jsonl")
    pgold = {d["id"]: d["persons"] for d in jl(ROOT / "data/holdout_persons_gold.jsonl")}

    with db.rag_conn() as conn:
        rows = conn.execute("SELECT doc_id, masked_text, tokens, embedding FROM rag.chunks").fetchall()
    docs = {d: {"id": d, "masked": t, "tokens": tk, "v1emb": unit(np.asarray(e, dtype=np.float32)),
                "doc_type": raw[d]["doc_type"]} for d, t, tk, e in rows}
    n_docs = len(docs)
    df = collections.Counter(t for d in docs.values() for t in set(d["tokens"]))
    idf = {t: math.log(n_docs / c) + 1 for t, c in df.items()}

    chunks = build_chunks(docs)
    queries = [{"qid": q, "masked": r["masked_query"], "tokens": r["query_tokens"]} for q, r in base.items()]
    texts = [q["masked"] for q in queries] + [c["emb_key"] for v in ("header", "rows", "rowsplain")
                                              for c in chunks[v] if c["emb_key"]]
    print(f"임베딩 대상 {len(set(texts))}개 (캐시 제외 후 호출)")
    emb = embed_all(texts)
    for q in queries:
        q["vec"], q["parsed"] = emb[q["masked"]], parse_query(q["masked"])

    ment_v1 = build_mentions(raw, pseudo, "v1")
    ment = build_mentions(raw, pseudo, "v2")
    entity_v1 = build_entities(docs, ment_v1)
    entity_of = build_entities(docs, ment)
    regions = {d: doc_regions(raw[d]["text"], pseudo[d]) for d in docs}

    # 2차: 머리글 빼고(rowsplain), 희귀도 빼고, 역할은 새 추출기 + 가산점. 첫 실험 변형은 비교용으로 남김
    base_opt = {"intent": 1, "region": 1, "status": 1}
    variants = [
        ("S0 v1 재현", "plain", {}),
        ("S1 +머리글", "header", {}),
        ("A 첫실험 역할필터", "rows", {"idf": 1, **base_opt, "role": 1, "role_filter": 1, "role_bonus": 1, "old": 1}),
        ("S2 +줄 청크", "rowsplain", {}),
        ("S3 +문서종류", "rowsplain", {"intent": 1}),
        ("S4 +지역", "rowsplain", {"intent": 1, "region": 1}),
        ("S5 +상태", "rowsplain", base_opt),
        ("B 역할 가산(비교)", "rowsplain", {**base_opt, "role": 1, "role_bonus": 1}),
        ("S6 +역할 감점", "rowsplain", {**base_opt, "role": 1}),
        ("S7 +개체", "rowsplain", {**base_opt, "role": 1, "entity": 1}),
        ("C 새역할 필터", "rowsplain", {**base_opt, "role": 1, "role_filter": 1}),
        ("D 새역할+개체 필터", "rowsplain", {**base_opt, "role": 1, "entity": 1, "role_filter": 1}),
    ]
    # 문서별 정답 인물 (채점용)
    q_person = collections.defaultdict(set)
    for d, ps in pgold.items():
        q_person[d] |= {p["person_id"] for p in ps}

    result = {}
    types = sorted({g["type"] for g in gold})
    for name, ck, opt in variants:
        m_, e_ = (ment_v1, entity_v1) if opt.get("old") else (ment, entity_of)
        runs = {q["qid"]: rank(q, chunks[ck], emb, docs, opt, idf, m_, e_, regions) for q in queries}
        tag = name.split()[0]
        with open(OUT / f"run_{tag}.jsonl", "w", encoding="utf-8") as f:
            for qid, r in runs.items():
                f.write(json.dumps({"qid": qid, "ranked_doc_ids": r}, ensure_ascii=False) + "\n")
        m = score(gold, [{"qid": k, "ranked_doc_ids": v} for k, v in runs.items()], ks=(1, 3, 5, 10))
        avg = lambda g, c: sum(m[g][c]) / len(m[g][c]) if m[g][c] else None  # noqa: E731
        cs, ch_ = confuse_split(gold, runs, q_person)
        result[name] = {"ALL": {c: avg("ALL", c) for c in ("Hit@1", "Hit@3", "Hit@5", "Recall@10", "MRR", "Confuse@10")},
                        "confuse_same": cs, "confuse_homonym": ch_,
                        "types": {t: {"Hit@1": avg(t, "Hit@1"), "MRR": avg(t, "MRR")} for t in types}}
        if tag == "S0":
            same = sum(runs[q] == base[q]["ranked_doc_ids"] for q in runs)
            top10 = sum(runs[q][:10] == base[q]["ranked_doc_ids"][:10] for q in runs)
            result[name]["repro"] = {"top20_same": same, "top10_same": top10, "n": len(runs)}

    # 개체 묶기 정확도 (이름 토큰별 문서 묶음 vs 정답 인물, 채점만)
    pairs = collections.Counter()
    tok_gold = collections.defaultdict(lambda: collections.defaultdict(set))
    for d, ents in pseudo.items():
        gp = {(p["start"], p["end"]): p["person_id"] for p in pgold.get(d, [])}
        for e in ents:
            if e["label"] == "person" and (e["start"], e["end"]) in gp:
                tok_gold[e["token"]][d].add(gp[(e["start"], e["end"])])
    homonym = {t for t, by in tok_gold.items() if len(set().union(*by.values())) > 1}

    def entity_score(ent):
        pairs, hom_sep, wrong_split = collections.Counter(), [], 0
        for t, by in tok_gold.items():
            ds = [d for d in by if (t, d) in ent]
            for i in range(len(ds)):
                for j in range(i + 1, len(ds)):
                    pairs[(bool(by[ds[i]] & by[ds[j]]), ent[(t, ds[i])][0] == ent[(t, ds[j])][0])] += 1
            n_pred = len({ent[(t, d)][0] for d in ds})
            if t in homonym:
                hom_sep.append({"token": t, "gold_persons": len(set().union(*by.values())),
                                "pred_entities": n_pred, "docs": len(ds)})
            elif n_pred > 1:
                wrong_split += 1
        tp, fp, fn = pairs[(True, True)], pairs[(False, True)], pairs[(True, False)]
        return {"pair_precision": tp / (tp + fp) if tp + fp else None, "pair_recall": tp / (tp + fn) if tp + fn else None,
                "homonym_tokens": hom_sep, "single_person_tokens_split": wrong_split,
                "single_person_tokens": len(tok_gold) - len(homonym)}

    result["entities_v1"] = entity_score(entity_v1)
    result["entities"] = entity_score(entity_of)
    # holdout 역할 분포 (정답 없음, 참고)
    result["holdout_roles"] = dict(collections.Counter(r for by in ment.values() for ms in by.values() for r, _, _ in ms))
    result["holdout_roles_v1"] = dict(collections.Counter(r for by in ment_v1.values() for ms in by.values() for r, _, _ in ms))

    # LLM에 넘길 문맥에 섞이는 다른 사람 (문서 통째 vs 줄 청크)
    whole, row_only, n_case = 0, 0, 0
    for q in queries:
        qt = set(q["tokens"])
        for d in result_doc_ids(OUT / "run_S0.jsonl", q["qid"])[:3]:
            rows, body = split_rows(docs[d]["masked"])
            hit_rows = [r for _, r in rows if qt & toks(r)]
            if not hit_rows:
                continue
            n_case += 1
            others = lambda s: {x for x in toks(s) if x.startswith("<PERSON_")} - qt  # noqa: E731
            whole += len(others(docs[d]["masked"]))
            row_only += len(others("\n".join(hit_rows)))
    result["context_other_persons"] = {"cases": n_case, "whole_doc_avg": whole / max(n_case, 1),
                                       "rows_only_avg": row_only / max(n_case, 1)}

    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{'변형':<16}{'Hit@1':>7}{'Hit@3':>7}{'Hit@5':>7}{'R@10':>7}{'MRR':>7}{'Conf@10':>8}{'같은사람':>8}{'동명이인':>8}")
    for name, _, _ in variants:
        a = result[name]["ALL"]
        print(f"{name:<16}{a['Hit@1']:>7.3f}{a['Hit@3']:>7.3f}{a['Hit@5']:>7.3f}{a['Recall@10']:>7.3f}{a['MRR']:>7.3f}"
              f"{a['Confuse@10']:>8.3f}{result[name]['confuse_same']:>8.3f}{result[name]['confuse_homonym']:>8.3f}")
    print("\nS0 재현:", result["S0 v1 재현"]["repro"])
    print("\n유형별 Hit@1 (S0 → S5 → S6 → S7)")
    for t in types:
        v = [result[n]["types"][t]["Hit@1"] for n in ("S0 v1 재현", "S5 +상태", "S6 +역할 감점", "S7 +개체")]
        print(f"  {t:<26}" + " → ".join(f"{x:.3f}" for x in v))
    print("\n개체 묶기 (첫 실험):", json.dumps(result["entities_v1"], ensure_ascii=False))
    print("개체 묶기 (새 역할):", json.dumps(result["entities"], ensure_ascii=False))
    print("holdout 언급 역할 분포 첫 실험:", result["holdout_roles_v1"], "→ 새:", result["holdout_roles"])
    print("문맥의 다른 사람:", result["context_other_persons"])


def result_doc_ids(path, qid):
    for r in jl(path):
        if r["qid"] == qid:
            return r["ranked_doc_ids"]
    return []


if __name__ == "__main__":
    main()
