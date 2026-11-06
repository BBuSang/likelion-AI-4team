"""직원 목록 API (/internal/org). 적재된 직원 목록이 있는 로컬 DB 필요 (test_access_db 또는 load_corpus 로 적재).

    SAEBOM_IT=1 python -m pytest tests/test_org_db.py -q
"""
import os

import pytest

if os.getenv("SAEBOM_IT") != "1":
    pytest.skip("SAEBOM_IT=1 일 때만 (로컬 DB 필요)", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from app.core import config, db  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app, headers={"X-Service-Key": config.AI_SERVICE_KEY})


@pytest.fixture(scope="module")
def known():
    with db.vault_conn() as c:
        row = c.execute("SELECT emp_id, name, team FROM vault.employees WHERE team IS NOT NULL "
                        "AND emp_id LIKE 'E%' ORDER BY n_docs DESC LIMIT 1").fetchone()
    if row is None:
        pytest.skip("직원 목록이 비어 있음 (먼저 적재)")
    return dict(zip(("emp_id", "name", "team"), row))


@pytest.fixture
def cleanup():
    yield
    with db.rag_conn() as r, db.vault_conn() as v:
        r.execute("DELETE FROM rag.employees WHERE emp_id LIKE 'U%' AND n_docs = 0 AND team = '테스트팀'")
        v.execute("DELETE FROM vault.employees WHERE emp_id LIKE 'U%' AND n_docs = 0 AND team = '테스트팀'")
        r.commit()
        v.commit()


def test_teams_have_no_names(known):
    ts = client.get("/internal/org/teams").json()
    assert known["team"] in [t["name"] for t in ts]
    assert all(set(t) == {"name", "n"} for t in ts)


def test_match_ranks_same_team_first(known):
    out = client.post("/internal/org/match", json={"name": known["name"], "team": known["team"]}).json()
    assert out[0]["emp_id"] == known["emp_id"] and out[0]["same_team"] is True
    # 띄어쓰기는 무시
    spaced = " ".join(known["name"])
    assert client.post("/internal/org/match", json={"name": spaced}).json()[0]["name"] == known["name"]
    assert client.post("/internal/org/match", json={"name": "없는사람이름"}).json() == []


def test_get_and_list(known):
    assert client.get(f"/internal/org/employees/{known['emp_id']}").json()["name"] == known["name"]
    assert client.get("/internal/org/employees/E99999").status_code == 404
    assert any(e["emp_id"] == known["emp_id"] for e in client.get("/internal/org/employees",
                                                                    params={"q": known["name"]}).json())


def test_create_new_employee(cleanup):
    a = client.post("/internal/org/employees", json={"name": "새 직원", "team": "테스트팀", "role": "STAFF"})
    b = client.post("/internal/org/employees", json={"name": "새직원2", "team": "테스트팀", "role": "LEAD"})
    assert a.status_code == b.status_code == 201
    a, b = a.json(), b.json()
    assert a["emp_id"].startswith("U") and a["emp_id"] != b["emp_id"] and a["name"] == "새직원"
    with db.rag_conn() as c:  # rag 에는 이름 없이
        assert c.execute("SELECT role FROM rag.employees WHERE emp_id = %s", (b["emp_id"],)).fetchone()[0] == "LEAD"
    assert client.post("/internal/org/employees", json={"name": "x", "role": "ADMIN"}).status_code == 422
