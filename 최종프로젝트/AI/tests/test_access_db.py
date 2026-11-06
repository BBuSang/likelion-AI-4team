"""문서 권한이 실제 DB 검색·복원에서 지켜지는지 (holdout 을 새로 적재해서 확인).

    SAEBOM_IT=1 python -m pytest tests/test_access_db.py -q
**로컬 개발 DB 의 rag·vault 데이터를 지우고 holdout 을 다시 적재**함 (임베딩 local, 재순위 끔, 외부 API 안 씀).
"""
import json
import os
from pathlib import Path

import pytest

if os.getenv("SAEBOM_IT") != "1":
    pytest.skip("SAEBOM_IT=1 일 때만 (로컬 DB 필요)", allow_module_level=True)

os.environ["EMBED_PROVIDER"] = "local"
os.environ["RERANK"] = "0"

from app.core import access, db  # noqa: E402
from app.pii import restore  # noqa: E402
from app.pipeline import load  # noqa: E402
from app.rag import search as search_mod  # noqa: E402

W2 = Path(__file__).resolve().parents[3] / "실전프로젝트2"


def read(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


@pytest.fixture(scope="module")
def corpus():
    docs = [{"id": r["id"], "text": r["text"], "doc_type": r.get("doc_type")}
            for r in read(W2 / "data/holdout_labeled.jsonl")]
    spans = {r["id"]: r["spans"] for r in read(W2 / "outputs/detect_holdout_labeled.jsonl")}
    _, employees, meta = load.load(docs, spans, reset=True, log=lambda *_: None)
    return employees, {m["id"]: m for m in meta}


def users(employees):
    """역할마다 대표 사용자 몇 명 + 팀 없는 직원."""
    out = []
    for role in access.ROLES:
        es = [e for e in employees if e["role"] == role]
        out += [{"emp_id": e["emp_id"], "team": e["team"], "role": role} for e in es[:4]]
    out += [{"emp_id": e["emp_id"], "team": None, "role": "STAFF"} for e in employees if e["team"] is None][:2]
    return out


def test_sql_matches_python_for_every_employee(corpus):
    employees, meta = corpus
    with db.rag_conn() as c:
        for e in employees:
            u = {"emp_id": e["emp_id"], "team": e["team"], "role": e["role"]}
            cond, params = access.sql(u)
            got = {r[0] for r in c.execute(f"SELECT doc_id FROM rag.documents WHERE {cond}", params).fetchall()}
            want = {d for d, m in meta.items() if access.can_view(u, m)}
            assert got == want, e["emp_id"]


def test_search_never_returns_out_of_scope(corpus):
    employees, meta = corpus
    queries = [q["query"] for q in read(W2 / "data/holdout_retrieval_queries.jsonl")][:25]
    n_results = 0
    for u in users(employees):
        for q in queries:
            _, _, docs = search_mod.search(q, k=5, user=u)
            for d in docs:
                assert access.can_view(u, meta[d["doc_id"]]), (u, q, d["doc_id"])
            n_results += len(docs)
    assert n_results > 0


def test_staff_sees_fewer_docs_than_lead_than_auditor(corpus):
    employees, meta = corpus
    lead = next(e for e in employees if e["role"] == "LEAD" and e["team"])
    staff = {"emp_id": "E_TEST", "team": lead["team"], "role": "STAFF"}
    lead_u = {"emp_id": lead["emp_id"], "team": lead["team"], "role": "LEAD"}
    n = [sum(access.can_view(u, m) for m in meta.values()) for u in (staff, lead_u, access.SYSTEM)]
    assert n[0] <= n[1] < n[2] == len(meta)


def test_restore_by_role(corpus):
    employees, meta = corpus
    # 담당자가 있는 팀 문서 하나: 담당자(STAFF 로 가정)·다른 같은 팀 직원·팀장·감사
    doc_id, m = next((d, m) for d, m in meta.items() if m["visibility"] == "team" and m["owner_emp_id"])
    with db.rag_conn() as c:
        text = c.execute("SELECT masked_text FROM rag.chunks WHERE doc_id = %s ORDER BY chunk_idx LIMIT 1",
                         (doc_id,)).fetchone()[0]
    owner = {"emp_id": m["owner_emp_id"], "team": m["team"], "role": "STAFF"}
    peer = {"emp_id": "E_PEER", "team": m["team"], "role": "STAFF"}
    lead = {"emp_id": "E_LEAD", "team": m["team"], "role": "LEAD"}
    other_lead = {"emp_id": "E_OL", "team": "다른팀", "role": "LEAD"}
    auditor = {"emp_id": "E_AUD", "team": "정보보호팀", "role": "AUDITOR"}

    _, r_owner = restore.restore_for(text, owner, [doc_id])
    _, r_peer = restore.restore_for(text, peer, [doc_id])
    _, r_lead = restore.restore_for(text, lead, [doc_id])
    _, r_other = restore.restore_for(text, other_lead, [doc_id])
    assert r_owner["restored"] and not r_owner["denied"]
    assert not r_peer["restored"] and r_peer["denied"]      # 같은 팀이어도 담당자가 아니면 토큰 그대로
    assert r_lead["restored"] == r_owner["restored"]
    assert not r_other["restored"]                         # 다른 팀 팀장은 못 봄
    with pytest.raises(access.AccessError):
        restore.restore_for(text, auditor, [doc_id])       # 감사는 사유 필수
    _, r_aud = restore.restore_for(text, auditor, [doc_id], purpose="민원 감사 확인")
    assert r_aud["restored"] == r_owner["restored"]

    with db.vault_conn() as c:
        rows = c.execute("SELECT role, doc_ids, cardinality(tokens), cardinality(denied) FROM vault.restore_log "
                         "ORDER BY id DESC LIMIT 5").fetchall()
    assert [r[0] for r in rows] == ["AUDITOR", "LEAD", "LEAD", "STAFF", "STAFF"]
    assert rows[1][1] == [] and rows[1][2] == 0 and rows[1][3] > 0  # 다른 팀 팀장: 복원 0, 거부 기록
