"""실명 복원 (디코딩): 토큰 → 원래 값.  vault_user 계정만 쓴다 (검색·답변 쪽 rag_user 와 분리).

    from app.pii import restore
    text, info = restore.restore(text, actor="demo")

- 복원 값은 vault.token_map.display (적재 때 정한 대표 표기: 사람은 성 포함 전체 이름, 번호는 원래 서식)
- 복원하지 않는 것
    · 출력 검사에서 바뀐 '[확인 안 된 토큰]' (LLM 이 바꿔 쓰거나 지어낸 토큰)
    · 매핑 테이블에 없는 토큰 (형식만 맞는 가짜 토큰)
    · 검색 보조 토큰 <NAME_…>, <GIVEN_…>, <ADDRP_…> (본문에 나오지 않음)
- 복원할 때마다 vault.restore_log 에 기록 (누가·언제·어떤 토큰). 권한 확인은 아직 없음 → 종합에서 역할별로 추가
"""
from app.core import db
from app.pii import pii_pipeline


def lookup(tokens, conn=None):
    """토큰 → {"display", "label", "pid"}. 매핑에 없는 토큰은 빠진다."""
    tokens = sorted({t for t in tokens if not t[1:].startswith(pii_pipeline.SEARCH_ONLY_PREFIXES)})
    if not tokens:
        return {}
    own = conn is None
    conn = conn or db.vault_conn()
    try:
        rows = conn.execute("SELECT token, display, label, pid FROM vault.token_map WHERE token = ANY(%s)",
                            (tokens,)).fetchall()
    finally:
        if own:
            conn.close()
    return {t: {"display": d, "label": lab, "pid": pid} for t, d, lab, pid in rows}


def restore(text, actor="unknown", purpose=None):
    """text 속 토큰을 원래 값으로 바꾼다. (복원된 텍스트, {"restored": {토큰: 값}, "unknown": [매핑 없는 토큰]})"""
    found = pii_pipeline.TOKEN_PATTERN.findall(text or "")
    with db.vault_conn() as conn:
        m = lookup(found, conn)
        if m:
            conn.execute("INSERT INTO vault.restore_log (actor, purpose, tokens) VALUES (%s, %s, %s)",
                         (actor, purpose, sorted(m)))
    out = text
    for t, v in m.items():
        out = out.replace(t, v["display"])
    unknown = sorted({t for t in found if t not in m and not t[1:].startswith(pii_pipeline.SEARCH_ONLY_PREFIXES)})
    return out, {"restored": {t: v["display"] for t, v in m.items()}, "unknown": unknown}


# ---------------------------------------------------------------- (최종) 역할별 복원 (worklog 3-4-1, 11장)
def restore_for(text, user, doc_ids, purpose=None, actor=None):
    """답변 text 의 토큰을 user 가 복원해도 되는 문서(doc_ids 중)에서 나온 것만 실명으로 바꾼다.
    doc_ids: 답변 근거로 쓴 문서. 권한 판단은 app.core.access.can_restore (STAFF=담당 문서만, LEAD=볼 수 있는 문서,
    AUDITOR=전부·사유 필수). 나머지 토큰은 그대로 두고 denied 로 기록.
    반환: (텍스트, {"restored": {토큰: 값}, "denied": [토큰], "unknown": [토큰], "docs": [복원 허용 문서]})"""
    from app.core import access
    access.check_user(user)
    if user["role"] == "AUDITOR" and not (purpose or "").strip():
        raise access.AccessError("감사 역할의 실명 복원은 사유가 필요")
    found = sorted({t for t in pii_pipeline.TOKEN_PATTERN.findall(text or "")
                    if not t[1:].startswith(pii_pipeline.SEARCH_ONLY_PREFIXES)})
    doc_ids = sorted(set(doc_ids or []))
    with db.rag_conn() as rc:
        metas = rc.execute("SELECT doc_id, owner_emp_id, team, visibility FROM rag.documents WHERE doc_id = ANY(%s)",
                           (doc_ids,)).fetchall()
        ok_docs = sorted(d for d, o, t, v in metas
                         if access.can_restore(user, {"owner_emp_id": o, "team": t, "visibility": v}))
        rows = rc.execute("SELECT DISTINCT unnest(tokens) FROM rag.chunks WHERE doc_id = ANY(%s)", (ok_docs,)).fetchall()
    allowed = {r[0] for r in rows}
    want = [t for t in found if t in allowed]
    denied = [t for t in found if t not in allowed]
    with db.vault_conn() as conn:
        m = lookup(want, conn)
        conn.execute("""INSERT INTO vault.restore_log (actor, purpose, tokens, role, doc_ids, denied)
                        VALUES (%s, %s, %s, %s, %s, %s)""",
                     (actor or user.get("emp_id") or "system", purpose, sorted(m), user["role"], ok_docs, denied))
    out = text
    for t, v in m.items():
        out = out.replace(t, v["display"])
    unknown = [t for t in want if t not in m]
    return out, {"restored": {t: v["display"] for t, v in m.items()}, "denied": denied, "unknown": unknown,
                 "docs": ok_docs}
