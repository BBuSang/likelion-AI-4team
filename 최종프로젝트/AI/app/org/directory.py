"""직원 목록 조회·추가 (스프링 계정 ↔ 직원 연결용, worklog 11-4).

- 팀 목록: rag 에서 (이름 없음) → 가입 화면 선택칸
- 후보 찾기·직원 목록: vault.employees 에서 (이름 있음) → 관리자 승인 화면에서만 쓰게 스프링이 막음
- 새 직원 번호 (안 A): 문서에 안 나오는 사람. 'U' 로 시작해서 문서에서 만든 'E001' 과 안 겹침
"""
import re

from app.core import access, db

_SPACE = re.compile(r"\s+")


class OrgError(ValueError):
    pass


def _name(s):
    return _SPACE.sub("", s or "")


def _team(s):
    s = _SPACE.sub("", s or "")
    return s or None


def _row(r, with_name):
    out = {"emp_id": r[0], "team": r[2], "position": r[3], "role": r[4], "n_docs": r[5]}
    if with_name:
        out["name"] = r[1]
    return out


_COLS = "emp_id, name, team, position, role, n_docs"


def teams():
    """문서에서 나온 팀 이름 (직원 팀 + 문서 팀). 문서 수 많은 순"""
    with db.rag_conn() as c:
        rows = c.execute(
            """SELECT team, count(*) FROM (
                   SELECT team FROM rag.documents WHERE team IS NOT NULL
                   UNION ALL SELECT team FROM rag.employees WHERE team IS NOT NULL) t
               GROUP BY team ORDER BY count(*) DESC, team""").fetchall()
    return [{"name": t, "n": n} for t, n in rows]


def match(name, team=None):
    """가입 정보(이름·팀)로 직원 후보. 이름이 같아야 하고, 팀이 같은 사람 → 팀 모르는 사람 → 다른 팀 순.
    same_team: 가입 팀과 문서 팀이 같은지 (관리자가 보고 고름)"""
    name, team = _name(name), _team(team)
    if not name:
        raise OrgError("이름이 비었습니다")
    with db.vault_conn() as c:
        rows = c.execute(f"SELECT {_COLS} FROM vault.employees WHERE name = %s", (name,)).fetchall()
    out = []
    for r in rows:
        e = _row(r, with_name=True)
        e["same_team"] = None if (team is None or e["team"] is None) else e["team"] == team
        out.append(e)
    rank = {True: 0, None: 1, False: 2}
    out.sort(key=lambda e: (rank[e["same_team"]], -e["n_docs"], e["emp_id"]))
    return out


def employees(q=None, limit=200):
    """관리자용 직원 목록 (이름 포함). q: 이름 또는 팀 일부"""
    sql, params = f"SELECT {_COLS} FROM vault.employees", {}
    if q:
        sql += " WHERE name LIKE %(q)s OR team LIKE %(q)s OR emp_id = %(raw)s"
        params = {"q": f"%{_name(q)}%", "raw": q.strip()}
    sql += " ORDER BY n_docs DESC, emp_id LIMIT %(limit)s"
    params["limit"] = max(1, min(int(limit), 1000))
    with db.vault_conn() as c:
        return [_row(r, with_name=True) for r in c.execute(sql, params).fetchall()]


def get(emp_id):
    with db.vault_conn() as c:
        r = c.execute(f"SELECT {_COLS} FROM vault.employees WHERE emp_id = %s", (emp_id,)).fetchone()
    return _row(r, with_name=True) if r else None


def create(name, team, position, role):
    """안 A: 문서에 없는 사람에게 새 직원 번호. rag(이름 없이)·vault(이름) 둘 다 넣음"""
    name, team = _name(name), _team(team)
    if not name:
        raise OrgError("이름이 비었습니다")
    if role not in access.ROLES:
        raise OrgError(f"역할은 {access.ROLES} 중 하나")
    position = (position or "").strip() or None
    with db.vault_conn() as v, db.rag_conn() as r:
        v.execute("SELECT pg_advisory_xact_lock(hashtext('org.create'))")
        last = v.execute("SELECT max(substring(emp_id from 2)::int) FROM vault.employees "
                         "WHERE emp_id ~ '^U[0-9]+$'").fetchone()[0]
        emp_id = f"U{(last or 0) + 1:04d}"
        # rag 먼저 (실패하면 vault 도 롤백) — 두 DB 계정이라 한 트랜잭션은 아님
        r.execute("INSERT INTO rag.employees (emp_id, team, position, role, n_docs) VALUES (%s,%s,%s,%s,0)",
                  (emp_id, team, position, role))
        r.commit()
        v.execute("INSERT INTO vault.employees (emp_id, name, team, position, role, n_mentions, n_docs) "
                  "VALUES (%s,%s,%s,%s,%s,0,0)", (emp_id, name, team, position, role))
        v.commit()
    return get(emp_id)
