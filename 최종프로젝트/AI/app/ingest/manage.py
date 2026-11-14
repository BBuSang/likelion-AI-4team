"""저장된 문서 보기·지우기 (관리자, worklog 15장).

문서 1건은 여러 곳에 나뉘어 있음:
  rag.chunks / rag.documents                   → 이 문서 것만 지움
  vault.person_mentions                        → 이 문서 것만 지움
  vault.token_map                              → 다른 문서에 안 남은 토큰만 지움 (같은 전화번호 = 같은 토큰이라 공유됨)
  rag.persons·person_attrs / vault.person_keys → 다른 문서에 안 나오는 사람만 지움
  vault.staging_doc (검수 원문)                 → 행은 기록용으로 남기고 원문·구간을 비움 (status = deleted)
지운 기록: vault.deletion_log (값 없이 개수만).
rag 를 먼저 지우고(검색에서 바로 빠짐) vault 를 지움. 중간에 실패하면 다시 실행해도 됨 (남은 것만 지움).
"""
from psycopg.types.json import Jsonb

from app.core import db


class NotFound(Exception):
    pass


def list_docs(q=None, visibility=None, source=None, limit=50, offset=0):
    where, params = [], {}
    if q:
        where.append("(d.doc_id ILIKE %(q)s OR d.doc_type ILIKE %(q)s OR d.team ILIKE %(q)s OR d.owner_emp_id = %(raw)s)")
        params |= {"q": f"%{q.strip()}%", "raw": q.strip()}
    if visibility:
        where.append("d.visibility = %(vis)s")
        params["vis"] = visibility
    if source == "upload":
        where.append("d.basis->>'source' = 'upload'")
    elif source == "batch":
        where.append("coalesce(d.basis->>'source', '') <> 'upload'")
    w = ("WHERE " + " AND ".join(where)) if where else ""
    params |= {"limit": max(1, min(int(limit), 200)), "offset": max(0, int(offset))}
    with db.rag_conn() as c:
        total = c.execute(f"SELECT count(*) FROM rag.documents d {w}", params).fetchone()[0]
        rows = c.execute(
            f"""SELECT d.doc_id, d.doc_type, d.owner_emp_id, d.team, d.visibility,
                       coalesce(d.basis->>'source', 'batch'), d.created_at,
                       (SELECT count(*) FROM rag.chunks c WHERE c.doc_id = d.doc_id)
                FROM rag.documents d {w} ORDER BY d.created_at DESC, d.doc_id LIMIT %(limit)s OFFSET %(offset)s""",
            params).fetchall()
    keys = ("doc_id", "doc_type", "owner_emp_id", "team", "visibility", "source", "created_at", "chunks")
    return {"total": total, "items": [dict(zip(keys, r)) for r in rows]}


def get_doc(doc_id):
    """가명처리된 글(토큰 상태)만. 원문은 안 보여 줌"""
    with db.rag_conn() as c:
        d = c.execute("SELECT doc_id, doc_type, owner_emp_id, team, visibility, basis, created_at "
                      "FROM rag.documents WHERE doc_id = %s", (doc_id,)).fetchone()
        chunks = c.execute("SELECT chunk_idx, masked_text, cardinality(tokens) FROM rag.chunks "
                           "WHERE doc_id = %s ORDER BY chunk_idx", (doc_id,)).fetchall()
    if d is None and not chunks:
        raise NotFound(doc_id)
    with db.vault_conn() as v:
        n_people = v.execute("SELECT count(DISTINCT pid) FROM vault.person_mentions WHERE doc_id = %s",
                             (doc_id,)).fetchone()[0]
    keys = ("doc_id", "doc_type", "owner_emp_id", "team", "visibility", "basis", "created_at")
    meta = dict(zip(keys, d)) if d else {"doc_id": doc_id}
    return meta | {"chunks": [{"idx": i, "masked_text": t, "n_tokens": n} for i, t, n in chunks], "people": n_people}


VISIBILITY = ("public", "team", "restricted")


def update_meta(doc_id, visibility, team, owner_emp_id):
    """공개 범위·팀·담당자 바꾸기. 문서 행과 청크(검색이 먼저 거르는 곳)에 같이 씀. 반환: {before, after}
    - team 공개는 팀이 있어야 함 (팀 없는 team 문서는 담당자·감사만 보게 되어 뜻이 안 맞음)
    - 담당자는 직원 목록에 있는 사람만"""
    if visibility not in VISIBILITY:
        raise ValueError(f"공개 범위는 {VISIBILITY} 중 하나")
    team = (team or "").strip() or None
    owner_emp_id = (owner_emp_id or "").strip() or None
    if visibility == "team" and team is None:
        raise ValueError("팀 공개로 하려면 팀을 정해 주세요")
    with db.rag_conn() as r:
        row = r.execute("SELECT visibility, team, owner_emp_id FROM rag.documents WHERE doc_id = %s FOR UPDATE",
                        (doc_id,)).fetchone()
        if row is None:
            raise NotFound(doc_id)
        if owner_emp_id and r.execute("SELECT 1 FROM rag.employees WHERE emp_id = %s", (owner_emp_id,)).fetchone() is None:
            raise ValueError(f"직원 목록에 없는 번호: {owner_emp_id}")
        r.execute("UPDATE rag.documents SET visibility = %s, team = %s, owner_emp_id = %s, "
                  "basis = basis || %s WHERE doc_id = %s",
                  (visibility, team, owner_emp_id, Jsonb({"edited": True}), doc_id))
        n = r.execute("UPDATE rag.chunks SET visibility = %s, team = %s, owner_emp_id = %s WHERE doc_id = %s",
                      (visibility, team, owner_emp_id, doc_id)).rowcount
        r.commit()
    keys = ("visibility", "team", "owner_emp_id")
    return {"doc_id": doc_id, "chunks": n, "before": dict(zip(keys, row)),
            "after": dict(zip(keys, (visibility, team, owner_emp_id)))}


def delete_doc(doc_id, actor, reason):
    if not (reason or "").strip():
        raise ValueError("지우는 사유가 필요합니다")
    # 1) rag: 이 문서 청크·메타 지우고, 다른 문서에 안 남은 토큰 목록 구하기
    with db.rag_conn() as r:
        toks = [t for (t,) in r.execute("SELECT DISTINCT unnest(tokens) FROM rag.chunks WHERE doc_id = %s",
                                        (doc_id,)).fetchall()]
        n_chunks = r.execute("DELETE FROM rag.chunks WHERE doc_id = %s", (doc_id,)).rowcount
        n_meta = r.execute("DELETE FROM rag.documents WHERE doc_id = %s", (doc_id,)).rowcount
        orphan = [t for (t,) in r.execute(
            "SELECT t FROM unnest(%s::text[]) t WHERE NOT EXISTS (SELECT 1 FROM rag.chunks WHERE tokens @> ARRAY[t])",
            (toks,)).fetchall()]
        r.commit()
    # 2) vault: 언급·토큰·사람·검수 원문
    with db.vault_conn() as v:
        pids = [p for (p,) in v.execute("SELECT DISTINCT pid FROM vault.person_mentions WHERE doc_id = %s",
                                        (doc_id,)).fetchall()]
        n_mentions = v.execute("DELETE FROM vault.person_mentions WHERE doc_id = %s", (doc_id,)).rowcount
        gone = [p for (p,) in v.execute(
            "SELECT p FROM unnest(%s::text[]) p WHERE NOT EXISTS (SELECT 1 FROM vault.person_mentions m WHERE m.pid = p)",
            (pids,)).fetchall()]
        n_tokens = v.execute("DELETE FROM vault.token_map WHERE token = ANY(%s)", (orphan,)).rowcount
        v.execute("DELETE FROM vault.person_keys WHERE pid = ANY(%s)", (gone,))
        staged = v.execute("""UPDATE vault.staging_doc SET status = 'deleted', text = '', detected = '[]',
                                     gold = NULL, final_spans = NULL, meta = '{}', error = NULL
                              WHERE doc_id = %s RETURNING id""", (doc_id,)).fetchall()
        for (sid,) in staged:
            v.execute("INSERT INTO vault.review_log (staging_doc_id, actor, action, detail) VALUES (%s, %s, 'delete', %s)",
                      (sid, actor, Jsonb({"reason_given": True})))
        if not (n_chunks or n_meta or n_mentions or staged):
            v.rollback()
            raise NotFound(doc_id)
        counts = {"chunks": n_chunks, "documents": n_meta, "mentions": n_mentions, "tokens": n_tokens,
                  "people": len(gone), "staging": len(staged)}
        v.execute("INSERT INTO vault.deletion_log (doc_id, actor, reason, counts) VALUES (%s, %s, %s, %s)",
                  (doc_id, actor, reason.strip()[:500], Jsonb(counts)))
        v.commit()
    # 3) rag: 아무 문서에도 안 나오는 사람 (person_attrs 는 CASCADE)
    if gone:
        with db.rag_conn() as r:
            r.execute("DELETE FROM rag.persons WHERE pid = ANY(%s)", (gone,))
            r.commit()
    return {"doc_id": doc_id} | counts


def deletion_log(limit=100):
    with db.vault_conn() as v:
        rows = v.execute("SELECT id, doc_id, actor, reason, counts, created_at FROM vault.deletion_log "
                         "ORDER BY id DESC LIMIT %s", (max(1, min(int(limit), 500)),)).fetchall()
    keys = ("id", "doc_id", "actor", "reason", "counts", "created_at")
    return [dict(zip(keys, r)) for r in rows]
