"""질문·복원 API (/internal/chat). 적재된 로컬 DB 필요 (test_access_db 가 local 임베딩으로 적재한 상태).

    SAEBOM_IT=1 python -m pytest tests/test_chat_api.py -q
LLM 은 부르지 않음 (LLM_API_KEY 를 비움 → 검색 결과까지만). 외부 API 안 씀.
"""
import os

import pytest

if os.getenv("SAEBOM_IT") != "1":
    pytest.skip("SAEBOM_IT=1 일 때만 (로컬 DB 필요)", allow_module_level=True)

os.environ["EMBED_PROVIDER"] = "local"
os.environ["RERANK"] = "0"

from fastapi.testclient import TestClient  # noqa: E402

from app.core import access, config, db  # noqa: E402
from app.main import app  # noqa: E402

config.LLM_API_KEY = ""
client = TestClient(app, headers={"X-Service-Key": config.AI_SERVICE_KEY})


@pytest.fixture(scope="module")
def meta():
    with db.rag_conn() as c:
        rows = c.execute("SELECT doc_id, owner_emp_id, team, visibility FROM rag.documents").fetchall()
    if not rows:
        pytest.skip("적재된 문서가 없음")
    return {d: {"owner_emp_id": o, "team": t, "visibility": v} for d, o, t, v in rows}


def test_ask_only_returns_visible_docs(meta):
    owner_doc = next(d for d, m in meta.items() if m["visibility"] == "team" and m["owner_emp_id"])
    m = meta[owner_doc]
    user = {"emp_id": m["owner_emp_id"], "team": m["team"], "role": "STAFF"}
    for q in ("대출 상담 내역", "카드 분실 신고", "민원 처리 결과"):
        r = client.post("/internal/chat/ask", json={"user": user, "query": q, "k": 5})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["answer"] is None and body["model"] is None  # LLM 키 없음
        for d in body["docs"]:
            assert access.can_view(user, meta[d["doc_id"]])
            assert len(d["snippet"]) <= 300


def test_ask_rejects_unlinked_user():
    r = client.post("/internal/chat/ask", json={"user": {"emp_id": None, "team": "고객센터", "role": "STAFF"},
                                                "query": "대출"})
    assert r.status_code == 403
    assert client.post("/internal/chat/ask", json={"user": {"role": "ADMIN"}, "query": "x"}).status_code == 422


def test_restore_by_role(meta):
    doc_id, m = next((d, m) for d, m in meta.items() if m["visibility"] == "team" and m["owner_emp_id"])
    with db.rag_conn() as c:
        text = c.execute("SELECT masked_text FROM rag.chunks WHERE doc_id = %s ORDER BY chunk_idx LIMIT 1",
                         (doc_id,)).fetchone()[0][:500]

    def call(user, purpose=None):
        return client.post("/internal/chat/restore", json={"user": user, "actor": "t@saebom.test", "text": text,
                                                           "doc_ids": [doc_id], "purpose": purpose})

    owner = call({"emp_id": m["owner_emp_id"], "team": m["team"], "role": "STAFF"}).json()
    peer = call({"emp_id": "E_PEER", "team": m["team"], "role": "STAFF"}).json()
    assert owner["restored"] > 0 and owner["docs"] == [doc_id] and owner["text"] != text
    assert peer["restored"] == 0 and peer["denied"] and peer["text"] == text
    assert call({"emp_id": None, "team": None, "role": "AUDITOR"}).status_code == 403  # 사유 필수
    assert call({"emp_id": None, "team": None, "role": "AUDITOR"}, "감사 확인").json()["restored"] == owner["restored"]
