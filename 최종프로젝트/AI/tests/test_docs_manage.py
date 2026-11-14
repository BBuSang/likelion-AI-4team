"""저장된 문서 지우기: 이 문서 것만 지우고, 다른 문서와 같이 쓰는 토큰·사람은 남기는지.

    SAEBOM_IT=1 python -m pytest tests/test_docs_manage.py -q
로컬 DB 에 시험 문서 2건(T-DEL-A/B)을 직접 넣고 지움. 다른 데이터는 건드리지 않음.
"""
import os

import pytest

if os.getenv("SAEBOM_IT") != "1":
    pytest.skip("SAEBOM_IT=1 일 때만 (로컬 DB 필요)", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from app.core import config, db  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app, headers={"X-Service-Key": config.AI_SERVICE_KEY})
SHARED, ONLY_A = "<PHONE_tdelshared>", "<PERSON_tdelonlya>"


def _clean():
    with db.rag_conn() as r, db.vault_conn() as v:
        r.execute("DELETE FROM rag.chunks WHERE doc_id LIKE 'T-DEL-%'")
        r.execute("DELETE FROM rag.documents WHERE doc_id LIKE 'T-DEL-%'")
        r.execute("DELETE FROM rag.persons WHERE pid = 'P_TDELA'")
        v.execute("DELETE FROM vault.person_mentions WHERE doc_id LIKE 'T-DEL-%'")
        v.execute("DELETE FROM vault.token_map WHERE token = ANY(%s)", ([SHARED, ONLY_A],))
        v.execute("DELETE FROM vault.person_keys WHERE pid = 'P_TDELA'")
        v.execute("DELETE FROM vault.deletion_log WHERE doc_id LIKE 'T-DEL-%'")
        r.commit()
        v.commit()


@pytest.fixture
def two_docs():
    _clean()
    with db.rag_conn() as r, db.vault_conn() as v:
        emb = r.execute("SELECT embedding FROM rag.chunks LIMIT 1").fetchone()
        if emb is None:
            pytest.skip("청크가 없음 (먼저 적재)")
        for doc, toks in (("T-DEL-A", [SHARED, ONLY_A]), ("T-DEL-B", [SHARED])):
            r.execute("INSERT INTO rag.documents (doc_id, doc_type, visibility) VALUES (%s, '시험', 'restricted')", (doc,))
            r.execute("INSERT INTO rag.chunks (chunk_id, doc_id, chunk_idx, doc_type, masked_text, tokens, embedding) "
                      "VALUES (%s, %s, 0, '시험', %s, %s, %s)", (f"{doc}#0", doc, " ".join(toks), toks, emb[0]))
        r.execute("INSERT INTO rag.persons (pid, person_token, name_token, given_token, status) "
                  "VALUES ('P_TDELA', %s, '<NAME_tdel>', '<GIVEN_tdel>', 'resolved')", (ONLY_A,))
        v.execute("INSERT INTO vault.token_map (token, label, normalized) VALUES (%s, 'phone_number', 'x'), (%s, 'person', 'y')",
                  (SHARED, ONLY_A))
        v.execute("INSERT INTO vault.person_mentions (doc_id, start_pos, end_pos, token, pid, basis) "
                  "VALUES ('T-DEL-A', 0, 3, %s, 'P_TDELA', 'new')", (ONLY_A,))
        v.execute("INSERT INTO vault.person_keys (pid, name) VALUES ('P_TDELA', '시험')")
        r.commit()
        v.commit()
    yield
    _clean()


def test_list_and_detail(two_docs):
    out = client.get("/internal/docs", params={"q": "T-DEL"}).json()
    assert out["total"] == 2 and {d["doc_id"] for d in out["items"]} == {"T-DEL-A", "T-DEL-B"}
    d = client.get("/internal/docs/T-DEL-A").json()
    assert d["chunks"][0]["masked_text"].startswith("<PHONE_") and d["people"] == 1
    assert client.get("/internal/docs/T-DEL-NONE").status_code == 404


def test_delete_keeps_shared(two_docs):
    r = client.post("/internal/docs/T-DEL-A/delete", json={"actor": "a@t", "reason": "시험 삭제"})
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["chunks"] == 1 and c["documents"] == 1 and c["mentions"] == 1 and c["tokens"] == 1 and c["people"] == 1
    with db.rag_conn() as rc, db.vault_conn() as v:
        assert rc.execute("SELECT count(*) FROM rag.chunks WHERE doc_id = 'T-DEL-A'").fetchone()[0] == 0
        assert rc.execute("SELECT count(*) FROM rag.persons WHERE pid = 'P_TDELA'").fetchone()[0] == 0
        left = {t for (t,) in v.execute("SELECT token FROM vault.token_map WHERE token = ANY(%s)", ([SHARED, ONLY_A],))}
        assert left == {SHARED}  # B 가 아직 쓰는 전화번호 토큰은 남음
        assert v.execute("SELECT count(*) FROM vault.person_keys WHERE pid = 'P_TDELA'").fetchone()[0] == 0
        log = v.execute("SELECT actor, reason FROM vault.deletion_log WHERE doc_id = 'T-DEL-A'").fetchone()
        assert log == ("a@t", "시험 삭제")
    assert client.post("/internal/docs/T-DEL-A/delete", json={"actor": "a@t", "reason": "또"}).status_code == 404
    # 사유 공백 → 400
    assert client.post("/internal/docs/T-DEL-B/delete", json={"actor": "a@t", "reason": "  "}).status_code == 400


def test_update_meta(two_docs):
    url = "/internal/docs/T-DEL-B/meta"
    assert client.post(url, json={"visibility": "team"}).status_code == 400            # 팀 없이 팀 공개
    assert client.post(url, json={"visibility": "public", "owner_emp_id": "E_NOPE"}).status_code == 400
    assert client.post(url, json={"visibility": "secret"}).status_code == 422
    with db.rag_conn() as c:
        emp = c.execute("SELECT emp_id, team FROM rag.employees WHERE team IS NOT NULL LIMIT 1").fetchone()
    r = client.post(url, json={"visibility": "team", "team": emp[1], "owner_emp_id": emp[0]})
    assert r.status_code == 200, r.text
    assert r.json()["before"]["visibility"] == "restricted" and r.json()["after"]["team"] == emp[1]
    with db.rag_conn() as c:  # 검색이 먼저 거르는 청크 값도 같이 바뀜
        assert c.execute("SELECT visibility, team, owner_emp_id FROM rag.chunks WHERE doc_id = 'T-DEL-B'").fetchone() \
            == ("team", emp[1], emp[0])
    assert client.post("/internal/docs/T-DEL-NONE/meta", json={"visibility": "public"}).status_code == 404
