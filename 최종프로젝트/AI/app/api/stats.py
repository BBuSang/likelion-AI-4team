"""통계·기록 API (/internal/stats, /internal/restore-log). 스프링만 부름.
값(개인정보)은 없이 개수만. 복원 기록은 토큰 목록 대신 개수만 내보냄."""
from typing import Optional

from fastapi import APIRouter

from app.core import db

router = APIRouter(tags=["stats"])


def _rows(cur, sql, params=()):
    cur.execute(sql, params)
    return cur.fetchall()


@router.get("/stats")
def stats(days: int = 14):
    days = max(1, min(days, 90))
    with db.rag_conn() as rc, rc.cursor() as r:
        docs = dict(_rows(r, "SELECT visibility, count(*) FROM rag.documents GROUP BY visibility"))
        n_chunks = _rows(r, "SELECT count(*) FROM rag.chunks")[0][0]
        employees = dict(_rows(r, "SELECT role, count(*) FROM rag.employees GROUP BY role"))
    with db.vault_conn() as vc, vc.cursor() as v:
        staging = dict(_rows(v, "SELECT status, count(*) FROM vault.staging_doc GROUP BY status"))
        auto = _rows(v, "SELECT count(*) FROM vault.staging_doc WHERE approved_by = 'gold'")[0][0]
        review = _rows(v, """SELECT coalesce(sum(added),0), coalesce(sum(removed),0), coalesce(sum(relabeled),0),
                                    count(*) FILTER (WHERE action = 'approve'), count(*) FILTER (WHERE action = 'reject')
                             FROM vault.review_log""")[0]
        restore_role = _rows(v, """SELECT coalesce(role, '-'), count(*), coalesce(sum(cardinality(tokens)),0),
                                          coalesce(sum(cardinality(denied)),0)
                                   FROM vault.restore_log GROUP BY 1 ORDER BY 1""")
        daily = _rows(v, """SELECT to_char(d::date, 'MM-DD'),
                                   (SELECT count(*) FROM vault.restore_log WHERE created_at::date = d::date),
                                   (SELECT count(*) FROM vault.review_log
                                     WHERE created_at::date = d::date AND action IN ('approve', 'reject'))
                            FROM generate_series(current_date - (%s - 1), current_date, '1 day') d ORDER BY d""",
                      (days,))
    return {
        "documents": {"total": sum(docs.values()), "by_visibility": docs, "chunks": n_chunks},
        "employees": {"total": sum(employees.values()), "by_role": employees},
        "ingest": {"by_status": staging, "auto_approved": auto, "approved": review[3], "rejected": review[4],
                   "added": review[0], "removed": review[1], "relabeled": review[2]},
        "restore": [{"role": a, "count": b, "tokens": c, "denied": d} for a, b, c, d in restore_role],
        "daily": [{"day": a, "restores": b, "reviews": c} for a, b, c in daily],
    }


@router.get("/restore-log")
def restore_log(actor: Optional[str] = None, limit: int = 50):
    sql = ("SELECT id, actor, role, purpose, doc_ids, cardinality(tokens), cardinality(denied), created_at "
           "FROM vault.restore_log")
    params = []
    if actor:
        sql += " WHERE actor = %s"
        params.append(actor)
    sql += " ORDER BY id DESC LIMIT %s"
    params.append(max(1, min(limit, 500)))
    with db.vault_conn() as vc, vc.cursor() as v:
        rows = _rows(v, sql, params)
    keys = ("id", "actor", "role", "purpose", "doc_ids", "restored", "denied", "created_at")
    return [dict(zip(keys, r)) for r in rows]
