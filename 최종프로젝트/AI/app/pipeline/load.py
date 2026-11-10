"""문서 파일 한 번에 적재 (새 DB 용, worklog 11장). 사람 검수 없이 끝까지 — 안내문 "수작업 없이".

순서: 직원 목록·문서 메타데이터(org.infer) → 동명이인 구분(2주차 resolve, 파일 전체를 한꺼번에) → 가명처리 → guard
      → 청크(600자) → 임베딩(가명처리된 청크만) → rag·vault 저장
2주차 starter/ingest.py 의 적재 부분과 같은 계산 + 문서 권한 칸(owner_emp_id·team·visibility) + 직원 목록 + 사람 key
(person_keys: 나중에 화면 업로드(9장)가 이 사람들과 이어질 수 있게).
"""
import collections
import time

from app.core import db
from app.ingest.ingest import _anchors_of, _display, chunk
from app.ingest.linking import pid_of
from app.org import infer
from app.pii import person_resolve, pii_pipeline
from app.rag.embed import embed

MAX_CHARS = 600


class NotEmpty(RuntimeError):
    pass


def load(docs, spans_by_doc, reset=False, log=print):
    """docs: [{id, text, doc_type}], spans_by_doc: {id: [{label,start,end}]} (탐지 결과).
    반환: 요약 dict (+ employees, meta 는 결과 파일 저장용)"""
    t0 = time.perf_counter()
    with db.rag_conn() as c:
        n_old = c.execute("SELECT count(*) FROM rag.chunks").fetchone()[0]
    if n_old and not reset:
        raise NotEmpty(f"rag.chunks 에 {n_old}행이 있음. 새 DB 용 적재라 --reset 이 필요 (직원 번호가 겹치지 않게)")

    text_by = {d["id"]: d["text"] for d in docs}  # 캐시로 받은 탐지 결과도 detect() 와 같은 보완을 거침 (worklog 20장)
    spans_by_doc = {i: pii_pipeline.complete_spans(text_by[i], s) if i in text_by else s for i, s in spans_by_doc.items()}
    employees, meta, *_ = infer.build(docs, spans_by_doc)
    meta_by = {m["id"]: m for m in meta}
    t_org = time.perf_counter() - t0

    spans, _ = person_resolve.resolve(docs, spans_by_doc)
    rows, mapping, persons, mentions, keys = [], {}, {}, [], {}
    n_guard = 0
    for d in docs:
        masked, records = pii_pipeline.pseudonymize(d["text"], spans[d["id"]])
        masked, g = pii_pipeline.guard(masked)
        n_guard += g
        extra = collections.defaultdict(set)
        for r in records:
            extra[r["token"]].update(r["search_tokens"])
            m = mapping.setdefault(r["token"], {"label": r["label"], "normalized": r["normalized"],
                                                "originals": collections.Counter(), "pid": None})
            m["originals"][r["original"]] += 1
            if r["label"] == "person":
                key = r.get("person_key")
                pid = pid_of(r["normalized"], key)
                m["pid"] = pid
                keys[pid] = (r["normalized"], key)
                p = persons.setdefault(r["token"], {"pid": pid, "name": r["normalized"], "roles": collections.Counter(),
                                                    "affs": set(), "attrs": set(), "mentions": 0,
                                                    "status": "resolved" if key else "unresolved"})
                p["mentions"] += 1
                if r.get("role"):
                    p["roles"][r["role"]] += 1
                if r.get("aff"):
                    p["affs"].add(r["aff"])
                mentions.append((d["id"], r["start"], r["end"], r["token"], pid, r.get("basis", "off"), r.get("role")))
                if key:
                    for lab, v in _anchors_of(spans[d["id"]], r):
                        p["attrs"].add((lab, pii_pipeline.make_token(lab, v)))
        mt = meta_by[d["id"]]
        for i, c in enumerate(chunk(masked, MAX_CHARS)):
            body = set(pii_pipeline.TOKEN_PATTERN.findall(c))
            rows.append({"chunk_id": f"{d['id']}#{i}", "doc_id": d["id"], "chunk_idx": i, "doc_type": d.get("doc_type"),
                         "masked_text": c, "tokens": sorted(body | {x for t in body for x in extra[t]}),
                         "owner_emp_id": mt["owner_emp_id"], "team": mt["team"], "visibility": mt["visibility"]})
    t_mask = time.perf_counter() - t0 - t_org

    t1 = time.perf_counter()
    vecs = embed([r["masked_text"] for r in rows])  # 가명처리 + guard 된 글만 나감
    t_embed = time.perf_counter() - t1

    with db.rag_conn() as conn, conn.cursor() as cur:
        if reset:
            cur.execute("TRUNCATE rag.chunks, rag.documents, rag.employees, rag.person_attrs, rag.persons")
        cur.executemany(
            """INSERT INTO rag.documents (doc_id, doc_type, owner_emp_id, team, visibility, basis)
               VALUES (%s,%s,%s,%s,%s,%s::jsonb)""",
            [(d["id"], d.get("doc_type"), meta_by[d["id"]]["owner_emp_id"], meta_by[d["id"]]["team"],
              meta_by[d["id"]]["visibility"], _json(meta_by[d["id"]]["basis"])) for d in docs])
        cur.executemany(
            """INSERT INTO rag.chunks (chunk_id, doc_id, chunk_idx, doc_type, masked_text, tokens, embedding,
                                       owner_emp_id, team, visibility)
               VALUES (%(chunk_id)s, %(doc_id)s, %(chunk_idx)s, %(doc_type)s, %(masked_text)s, %(tokens)s, %(embedding)s,
                       %(owner_emp_id)s, %(team)s, %(visibility)s)""",
            [r | {"embedding": v} for r, v in zip(rows, vecs)])
        cur.executemany("INSERT INTO rag.employees (emp_id, team, position, role, n_docs) VALUES (%s,%s,%s,%s,%s)",
                        [(e["emp_id"], e["team"], e["position"], e["role"], e["n_docs"]) for e in employees])
        cur.executemany(
            """INSERT INTO rag.persons (pid, person_token, name_token, given_token, role, affiliations, status, mentions)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
            [(p["pid"], tok, pii_pipeline.name_token(p["name"]),
              pii_pipeline.given_token(pii_pipeline.split_name(p["name"])[1]),
              p["roles"].most_common(1)[0][0] if p["roles"] else None, sorted(p["affs"]), p["status"], p["mentions"])
             for tok, p in persons.items()])
        cur.executemany("INSERT INTO rag.person_attrs (pid, label, token) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                        [(p["pid"], lab, t) for p in persons.values() for lab, t in sorted(p["attrs"])])

    with db.vault_conn() as conn, conn.cursor() as cur:
        if reset:
            cur.execute("TRUNCATE vault.token_map, vault.person_mentions, vault.person_keys, vault.employees")
        cur.executemany(
            "INSERT INTO vault.token_map (token, label, normalized, originals, pid, display) VALUES (%s,%s,%s,%s,%s,%s)",
            [(t, m["label"], m["normalized"], sorted(m["originals"]), m["pid"], _display(m)) for t, m in mapping.items()])
        cur.executemany(
            """INSERT INTO vault.person_mentions (doc_id, start_pos, end_pos, token, pid, basis, role)
               VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""", mentions)
        cur.executemany("INSERT INTO vault.person_keys (pid, name, person_key) VALUES (%s,%s,%s)",
                        [(pid, name, key) for pid, (name, key) in keys.items()])
        cur.executemany(
            """INSERT INTO vault.employees (emp_id, name, team, position, role, n_mentions, n_docs)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            [(e["emp_id"], e["name"], e["team"], e["position"], e["role"], e["n_mentions"], e["n_docs"])
             for e in employees])

    n = len(docs)
    summary = {"docs": n, "chunks": len(rows), "tokens": len(mapping), "persons": len(persons),
               "employees": len(employees), "guard_masked": n_guard,
               "visibility": dict(collections.Counter(m["visibility"] for m in meta)),
               "ms_per_doc": {"org": round(t_org * 1000 / n, 1), "mask": round(t_mask * 1000 / n, 1),
                              "embed": round(t_embed * 1000 / n, 1)}}
    log(f"적재: {summary}")
    return summary, employees, meta


def _json(x):
    import json
    return json.dumps(x, ensure_ascii=False)
