"""개체 묶기: 같은 이름 토큰을 쓰는 문서들을 사람 단위(rag.entities)로 나눔. 적재 다음 단계.

    python -X utf8 v2/entity/resolve.py --mode rule            규칙만
    python -X utf8 v2/entity/resolve.py --mode llm             규칙 + 애매한 쌍만 LLM 판단
    python -X utf8 v2/entity/resolve.py --mode rule --dry-run  LLM 판단 대상 쌍만 세어 봄

입력은 rag 테이블뿐 (가명 본문·토큰·언급 역할). 원문·정답 인물 ID는 쓰지 않음.
1. 문서별 프로필: 역할·부서, 같은 줄의 식별 토큰(그 줄에 이 사람만 있을 때)
2. 규칙: 식별 토큰 공유 → 합침 / 역할·부서·같은 종류 번호가 다르면 충돌(다른 사람) / 역할이 같고 충돌 없으면 합침
   단서 없는 문서는 그 이름의 단서 있는 묶음이 하나뿐일 때만 붙임
3. LLM(선택): 규칙 뒤 충돌도 증거도 없는 묶음 쌍만 판단. A·B / B·A 두 번 모두 "같음"·확신 0.8 이상·근거 인용이
   입력에 실제로 있을 때만 합침. "다름"은 충돌로 기록해 이후 합치기를 막음. 판단은 outputs/llm_judgments.jsonl 에 캐시
결과: rag.entities, rag.entity_links, rag.mentions.entity_id
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
import db  # noqa: E402

OUT = V2 / "outputs"
JUDGE_CACHE = OUT / "llm_judgments.jsonl"
ID_PREFIX = ("<PHONE_", "<RRN_", "<EMAIL_", "<ACCOUNT_", "<CARD_", "<ADDRESS_")
MAX_DOCS_PER_SIDE = 3
CONF_MIN = 0.8

SYSTEM = """너는 금융회사 내부 문서에서 가명처리된 인물이 같은 사람인지 판단한다.
- <PERSON_…>, <PHONE_…> 같은 꺾쇠 토큰은 가명이다. 같은 토큰은 같은 값(같은 이름·같은 번호)이라는 뜻이다.
- [대상]은 판단할 사람이다. 두 묶음의 [대상]은 이름이 같다. 이름이 같다는 것만으로는 같은 사람이라는 증거가 아니다.
- "같음": 같은 사람이라는 근거가 있을 때만 (같은 번호·이메일 토큰, 같은 사건이 이어짐, 같은 소속과 역할).
- "다름": 다른 사람이라는 근거가 있을 때 (고객과 직원처럼 역할이 다름, 소속이 다름, 같은 종류의 번호가 서로 다름).
  전화번호와 이메일처럼 종류가 다른 번호는 다르다는 근거가 아니다. 한쪽에만 정보가 있는 것도 근거가 아니다.
- "모름": 근거가 부족할 때. 추측하지 않는다.
- "근거"에는 입력에 있는 글자를 그대로 복사한 짧은 구절(토큰 포함)을 넣는다. 설명문을 쓰지 않는다.
JSON 한 줄만 출력한다: {"판단": "같음|다름|모름", "확신": 0~1 숫자, "근거": "입력에서 그대로 복사한 구절"}"""


# ───────────────────────── 입력 ─────────────────────────
def load():
    with db.ingest_conn() as conn:
        docs = {d: {"doc_type": t, "masked": m} for d, t, m in
                conn.execute("SELECT doc_id, doc_type, masked_text FROM rag.documents")}
        ments = conn.execute("SELECT mention_id, doc_id, chunk_id, token, label, line_no, role, org_unit, form, "
                             "own_role FROM rag.mentions ORDER BY mention_id").fetchall()
    return docs, ments


def profiles(ments):
    """token → doc → {roles, orgs, ids, chunk, mention_ids}"""
    line_tok = collections.defaultdict(set)
    for _, d, _, t, _, ln, *_ in ments:
        line_tok[(d, ln)].add(t)
    prof = collections.defaultdict(dict)
    for mid, d, ch, t, lab, ln, role, org, form, own in ments:
        if lab != "person":
            continue
        p = prof[t].setdefault(d, {"roles": set(), "orgs": set(), "ids": set(), "chunk": ch, "mids": [],
                                   "forms": set(), "own": set()})
        p["mids"].append(mid)
        p["forms"].add(form)
        if own:
            p["own"].add(own)
        if role:
            p["roles"].add(role)
        if org:
            p["orgs"].add(org)
        # 식별 토큰: 이 사람만 있는 같은 줄 + 사람이 없는 바로 앞뒤 줄 (주소·연락처가 다음 줄에 오는 서식)
        for k in (ln - 1, ln, ln + 1):
            others = line_tok.get((d, k), set())
            persons = {x for x in others if x.startswith("<PERSON_")}
            if persons == ({t} if k == ln else set()):
                p["ids"] |= {x for x in others if x.startswith(ID_PREFIX)}
    # 문서 안 동명이인 의심: 같은 문서 같은 토큰인데 언급마다 역할이 다르거나 부서가 둘 이상
    for by in prof.values():
        for p in by.values():
            p["mixed"] = len(p["own"]) > 1 or len(p["orgs"]) > 1
    return prof


# ───────────────────────── 규칙 ─────────────────────────
def conflict(a, b, no_link=frozenset()):
    if a.get("mixed") or b.get("mixed"):          # 문서 안 동명이인 의심 문서는 어느 쪽과도 합치지 않음
        return True
    if a["roles"] and b["roles"] and not (a["roles"] & b["roles"]):
        return True
    if a["orgs"] and b["orgs"] and not (a["orgs"] & b["orgs"]):
        return True
    for pre in ID_PREFIX:
        ia = {x for x in a["ids"] if x.startswith(pre)}
        ib = {x for x in b["ids"] if x.startswith(pre)}
        if ia and ib and not (ia & ib):
            return True
    return any((x, y) in no_link or (y, x) in no_link for x in a["docs"] for y in b["docs"])


def merge(a, b, method):
    return {"docs": a["docs"] | b["docs"], "roles": a["roles"] | b["roles"], "orgs": a["orgs"] | b["orgs"],
            "ids": a["ids"] | b["ids"], "method": method, "conf": min(a["conf"], b["conf"]),
            "mixed": a.get("mixed", False) or b.get("mixed", False)}


def rule_clusters(by_doc):
    clusters = [{"docs": {d}, "roles": set(p["roles"]), "orgs": set(p["orgs"]), "ids": set(p["ids"]),
                 "method": "single", "conf": 1.0, "mixed": p["mixed"]} for d, p in by_doc.items()]
    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a, b = clusters[i], clusters[j]
                if conflict(a, b):
                    continue
                if a["ids"] & b["ids"]:
                    method = "cooccur"
                elif a["roles"] and a["roles"] == b["roles"] and a["orgs"] & b["orgs"]:
                    # 역할과 부서가 둘 다 같을 때만 합침. "둘 다 고객"·"둘 다 직원"만으로는 같은 사람 근거가 아님
                    # (test에서 부서 없는 직원 박도윤 2명, 이름만 부른 서연·민준·성호를 합친 오류 → worklog 39번)
                    method = "role"
                else:
                    continue
                clusters[i] = merge(a, b, method)
                clusters.pop(j)
                changed = True
                break
            if changed:
                break
    # 충돌 없는 이름: 문서 사이에 충돌이 하나도 없으면 한 사람으로 보고 모두 합침.
    # 이름만 부른 표기(form=given)는 여러 사람이 같은 토큰을 쓰므로 제외
    given = any("given" in p["forms"] for p in by_doc.values())   # 한 번이라도 이름만 불렸으면 여러 사람일 수 있음
    if len(clusters) > 1 and not given and not any(
            conflict(clusters[i], clusters[j]) for i in range(len(clusters)) for j in range(i + 1, len(clusters))):
        merged = clusters[0]
        for c in clusters[1:]:
            merged = merge(merged, c, "role")
        return [merged]
    known = [c for c in clusters if c["roles"] or c["ids"] or c["orgs"]]
    if len(known) == 1 and not given and not known[0]["mixed"]:
        k = known[0]
        for c in clusters:
            if c is not k:
                k["docs"] |= c["docs"]
        clusters = [k]
    return clusters


def ambiguous_pairs(clusters):
    return [(i, j) for i in range(len(clusters)) for j in range(i + 1, len(clusters))
            if not conflict(clusters[i], clusters[j])]


# ───────────────────────── LLM 판단 ─────────────────────────
def evidence(token, cluster, docs):
    """묶음의 문서(최대 3개)에서 [대상]이 나오는 줄과 앞뒤 한 줄."""
    parts = []
    for d in sorted(cluster["docs"])[:MAX_DOCS_PER_SIDE]:
        lines = docs[d]["masked"].split("\n")
        keep = sorted({k for i, l in enumerate(lines) if token in l for k in (i - 1, i, i + 1) if 0 <= k < len(lines)})
        text = " / ".join(lines[k].strip() for k in keep if lines[k].strip()).replace(token, "[대상]")
        parts.append(f"- {d} {docs[d]['doc_type']}: {text[:400]}")
    info = (f"  추출 정보: 역할 {','.join(sorted(cluster['roles'])) or '미상'}, "
            f"소속 {','.join(sorted(cluster['orgs'])) or '미상'}, 식별 토큰 {' '.join(sorted(cluster['ids'])) or '없음'}")
    return "\n".join(parts) + "\n" + info


REASONING = None   # main 에서 설정 (None = 모델 기본 low)


def judge(token, a, b, docs, cache):
    from clients import chat
    user = f"[묶음 A]\n{evidence(token, a, docs)}\n\n[묶음 B]\n{evidence(token, b, docs)}\n\n두 묶음의 [대상]은 같은 사람입니까?"
    key = hashlib.sha1(f"{REASONING}|{user}".encode("utf-8")).hexdigest()
    if key not in cache:
        raw = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], REASONING)
        cache[key] = {"token": token, "a": sorted(a["docs"]), "b": sorted(b["docs"]), "reasoning": REASONING, "raw": raw}
        with open(JUDGE_CACHE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, **cache[key]}, ensure_ascii=False) + "\n")
    raw = cache[key]["raw"]
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        j = {}
    verdict, conf, quote = j.get("판단", "모름"), float(j.get("확신", 0) or 0), str(j.get("근거", "")).strip(" \"'")
    quote_ok = bool(quote) and re.sub(r"\s+", "", quote) in re.sub(r"\s+", "", user)
    return {"verdict": verdict, "conf": conf, "quote": quote, "quote_ok": quote_ok}


def llm_refine(token, clusters, docs, cache, log):
    no_link = set()
    pairs = ambiguous_pairs(clusters)
    decisions = []
    for i, j in pairs:
        r1 = judge(token, clusters[i], clusters[j], docs, cache)
        r2 = judge(token, clusters[j], clusters[i], docs, cache)
        same = all(r["verdict"] == "같음" and r["conf"] >= CONF_MIN and r["quote_ok"] for r in (r1, r2))
        diff = any(r["verdict"] == "다름" and r["quote_ok"] for r in (r1, r2))
        decisions.append((i, j, same, diff, min(r1["conf"], r2["conf"])))
        log.append({"token": token, "a": sorted(clusters[i]["docs"]), "b": sorted(clusters[j]["docs"]),
                    "r1": r1, "r2": r2, "merge": same, "conflict": diff})
        if diff:
            no_link |= {(x, y) for x in clusters[i]["docs"] for y in clusters[j]["docs"]}
    # 합치기: 묶음 번호 기준 union-find, "다름" 충돌이 생기면 합치지 않음
    groups = {i: {"members": {i}, **clusters[i]} for i in range(len(clusters))}
    find = {i: i for i in range(len(clusters))}
    for i, j, same, _, conf in sorted(decisions, key=lambda x: -x[4]):
        gi, gj = find[i], find[j]
        if not same or gi == gj or conflict(groups[gi], groups[gj], frozenset(no_link)):
            continue
        g = merge(groups[gi], groups[gj], "llm")
        g["conf"] = min(g["conf"], conf)
        g["members"] = groups[gi]["members"] | groups[gj]["members"]
        groups[gi] = g
        for k in g["members"]:
            find[k] = gi
        del groups[gj]
    return [{k: v for k, v in g.items() if k != "members"} for g in groups.values()]


# ───────────────────────── 저장 ─────────────────────────
def save(result, prof):
    with db.ingest_conn() as conn, conn.cursor() as cur:
        # TRUNCATE … CASCADE 는 entities 를 참조하는 mentions 까지 비우므로 DELETE 로 지움
        cur.execute("UPDATE rag.mentions SET entity_id = NULL")
        cur.execute("DELETE FROM rag.entity_links")
        cur.execute("DELETE FROM rag.entities")
        n = 0
        for token, clusters in result.items():
            for c in clusters:
                n += 1
                eid = f"E{n:04d}"
                role = collections.Counter(r for d in c["docs"] for r in prof[token][d]["roles"]).most_common(1)
                org = sorted(c["orgs"])[0] if c["orgs"] else None
                cur.execute("INSERT INTO rag.entities (entity_id, person_token, role, org_unit, confidence, method) "
                            "VALUES (%s, %s, %s, %s, %s, %s)",
                            (eid, token, role[0][0] if role else None, org, c["conf"], c["method"]))
                for d in c["docs"]:
                    p = prof[token][d]
                    cur.execute("UPDATE rag.mentions SET entity_id = %s WHERE mention_id = ANY(%s)", (eid, p["mids"]))
                    for t in p["ids"]:
                        cur.execute("INSERT INTO rag.entity_links (entity_id, token, evidence_chunk) VALUES (%s, %s, %s) "
                                    "ON CONFLICT DO NOTHING", (eid, t, p["chunk"]))
        conn.commit()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("rule", "llm"), default="rule")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reasoning", default=None, help="none | low (기본: 모델 기본값 low)")
    ap.add_argument("--tokens", nargs="*", help="LLM 판단을 이 토큰들에만 (설정 비교용)")
    a = ap.parse_args()
    global REASONING
    REASONING = a.reasoning
    docs, ments = load()
    prof = profiles(ments)
    rule = {t: rule_clusters(by) for t, by in prof.items()}
    pairs = {t: ambiguous_pairs(c) for t, c in rule.items()}
    n_pairs = sum(len(p) for p in pairs.values())
    print(f"이름 토큰 {len(prof)}개 → 규칙 개체 {sum(len(c) for c in rule.values())}개, "
          f"LLM 판단 대상 쌍 {n_pairs}개 (토큰 {sum(1 for p in pairs.values() if p)}개, 호출 {n_pairs * 2}회)")
    if a.dry_run:
        for t, p in sorted(pairs.items(), key=lambda x: -len(x[1]))[:10]:
            print(f"  {t}: 묶음 {len(rule[t])}개, 쌍 {len(p)}개")
        return
    result = rule
    if a.mode == "llm":
        cache = {}
        if JUDGE_CACHE.exists():
            for line in open(JUDGE_CACHE, encoding="utf-8"):
                r = json.loads(line)
                cache[r["key"]] = r
        log = []
        result = {t: (llm_refine(t, c, docs, cache, log) if pairs[t] and (not a.tokens or t in a.tokens) else c)
                  for t, c in rule.items()}
        (OUT / "llm_decisions.json").write_text(json.dumps(log, ensure_ascii=False, indent=1), encoding="utf-8")
        merged = sum(1 for x in log if x["merge"])
        print(f"LLM 판단 {len(log)}쌍: 합침 {merged} · 다름 {sum(1 for x in log if x['conflict'])} · "
              f"모름·불일치 {len(log) - merged - sum(1 for x in log if x['conflict'] and not x['merge'])}")
    n = save(result, prof)
    print(f"저장: 개체 {n}개 ({a.mode})")


if __name__ == "__main__":
    main()
