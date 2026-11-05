"""업로드 → 탐지 → 검수 → 사람 잇기 → 저장 전체 흐름 (실제 DB + 실제 탐지기, 임베딩은 local).

로컬 DB(infra/docker-compose)가 떠 있을 때만:
    SAEBOM_IT=1 python -m pytest tests/test_ingest_db.py -q
만든 행은 끝에 지움 (다른 데이터는 건드리지 않음).
"""
import json
import os

import pytest

if os.getenv("SAEBOM_IT") != "1":
    pytest.skip("SAEBOM_IT=1 일 때만 (로컬 DB 필요)", allow_module_level=True)

os.environ["EMBED_PROVIDER"] = "local"  # 외부 API 안 씀
os.environ.setdefault("AI_SERVICE_KEY", "test-service-key")

from fastapi.testclient import TestClient  # noqa: E402

from app.core import config, db  # noqa: E402
from app.main import app  # noqa: E402
from app.pii import pii_pipeline  # noqa: E402

H = {"X-Service-Key": config.AI_SERVICE_KEY}
client = TestClient(app)
NAME = "도하람"
PHONE = "010-9871-2034"


@pytest.fixture(scope="module")
def created():
    jobs = []
    yield jobs
    with db.vault_conn() as vc, vc.cursor() as cur:
        cur.execute("SELECT doc_id FROM vault.staging_doc WHERE job_id = ANY(%s)", (jobs,))
        doc_ids = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT DISTINCT token, pid FROM vault.person_mentions WHERE doc_id = ANY(%s)", (doc_ids,))
        rows = cur.fetchall()
        pids = list({r[1] for r in rows})
        cur.execute("DELETE FROM vault.person_mentions WHERE doc_id = ANY(%s)", (doc_ids,))
        cur.execute("DELETE FROM vault.person_keys WHERE pid = ANY(%s)", (pids,))
        cur.execute("DELETE FROM vault.ingest_job WHERE id = ANY(%s)", (jobs,))
    with db.rag_conn() as rc, rc.cursor() as cur:
        cur.execute("SELECT DISTINCT unnest(tokens) FROM rag.chunks WHERE doc_id = ANY(%s)", (doc_ids,))
        tokens = [r[0] for r in cur.fetchall()]
        cur.execute("DELETE FROM rag.chunks WHERE doc_id = ANY(%s)", (doc_ids,))
        cur.execute("DELETE FROM rag.persons WHERE pid = ANY(%s)", (pids,))
    with db.vault_conn() as vc, vc.cursor() as cur:
        cur.execute("DELETE FROM vault.token_map WHERE token = ANY(%s)", (tokens,))


def upload(created, source, content):
    r = client.post("/internal/ingest/jobs", headers=H,
                    json={"uploader": "it@saebom.test", "source": source, "content": content, "filename": "t"})
    assert r.status_code == 202, r.text
    job_id = r.json()["job_id"]
    created.append(job_id)
    assert client.get(f"/internal/ingest/jobs/{job_id}", headers=H).json()["status"] == "ready"
    return job_id, client.get(f"/internal/ingest/jobs/{job_id}/docs", headers=H).json()


def open_doc(sid):
    r = client.get(f"/internal/ingest/docs/{sid}", params={"actor": "reviewer@saebom.test"}, headers=H)
    assert r.status_code == 200
    return r.json()


def approve(sid, spans, choices=None):
    return client.post(f"/internal/ingest/docs/{sid}/review", headers=H,
                       json={"actor": "reviewer@saebom.test", "action": "approve", "spans": spans, "choices": choices})


def person_link(result):
    [g] = [g for g in result["links"]]
    return g


def test_requires_service_key():
    assert client.get("/internal/ingest/jobs/1").status_code == 401


def test_flow(created):
    # 1. 첫 문서: 새 사람
    _, [d1] = upload(created, "text", f"고객명: {NAME}\n연락처: {PHONE}\n대출 한도 상담을 요청함.")
    assert d1["status"] == "pending"
    doc = open_doc(d1["id"])
    labels = {s["label"] for s in doc["spans"]}
    assert {"person", "phone_number"} <= labels
    r1 = approve(d1["id"], doc["spans"])
    assert r1.status_code == 200, r1.text
    g1 = person_link(r1.json())
    assert g1["basis"] == "new"

    # 같은 문서 다시 승인 → 409
    assert approve(d1["id"], doc["spans"]).status_code == 409

    # 2. 다른 업로드, 같은 전화번호 → 기존 인물과 이어짐 (같은 pid)
    _, [d2] = upload(created, "text", f"민원인: {NAME} (연락처 {PHONE})\n카드 분실 신고 접수.")
    r2 = approve(d2["id"], open_doc(d2["id"])["spans"]).json()
    g2 = person_link(r2)
    assert g2["basis"] == "anchor_link" and g2["pid"] == g1["pid"]

    # 3. 같은 이름, 다른 전화번호 → 다른 사람
    _, [d3] = upload(created, "text", f"고객명: {NAME}\n연락처: 010-5550-7781\n주소 변경 요청.")
    g3 = person_link(approve(d3["id"], open_doc(d3["id"])["spans"]).json())
    assert g3["basis"] == "new" and g3["pid"] != g1["pid"]

    # 4. 닻 없는 언급, 같은 이름 확정 인물이 2명 → 미확정 + 검수자가 고르면 그 사람
    text4 = f"{NAME} 고객이 지난 상담 결과를 다시 문의함."
    _, [d4] = upload(created, "text", text4)
    doc4 = open_doc(d4["id"])
    [p4] = [g for g in doc4["links"]]
    assert p4["basis"] == "unresolved" and set(p4["candidates"]) == {g1["pid"], g3["pid"]}
    g4 = person_link(approve(d4["id"], doc4["spans"], choices={p4["id"]: g1["pid"]}).json())
    assert g4["basis"] == "chosen_link" and g4["pid"] == g1["pid"]

    # 저장된 청크에는 원문 이름·번호가 없고, 1·2·4 문서는 같은 사람 토큰
    with db.rag_conn() as rc, rc.cursor() as cur:
        cur.execute("SELECT doc_id, masked_text FROM rag.chunks WHERE doc_id = ANY(%s)",
                    ([open_doc(x["id"])["doc_id"] for x in (d1, d2, d3, d4)],))
        chunks = dict(cur.fetchall())
    assert all(NAME not in t and "9871" not in t for t in chunks.values())
    tok = {doc_id: set(pii_pipeline.TOKEN_PATTERN.findall(t)) for doc_id, t in chunks.items()}
    person_tok = [{t for t in v if t.startswith("<PERSON_")} for v in tok.values()]
    assert len(set.union(*person_tok)) == 2  # 사람 2명 (1·2·4 = 한 명, 3 = 다른 한 명)


def test_reviewer_changes_and_reject(created):
    text = f"상담원 메모: {NAME} 고객 이메일 haram.do@example.com 로 서류 발송.\n비고 없음"
    _, [d] = upload(created, "text", text)
    doc = open_doc(d["id"])
    # 검수자: 이메일 구간을 지우고, '비고'를 주소로 (일부러 틀린 예) 추가 → 개수만 확인
    spans = [s for s in doc["spans"] if s["label"] != "email"]
    s = text.index("비고")
    spans.append({"label": "address", "start": s, "end": s + 2, "text": "비고"})
    r = approve(d["id"], spans).json()
    assert r["added"] == 1 and r["removed"] == 1
    # 위치와 글자가 다르면 400
    _, [d2] = upload(created, "text", text)
    bad = [{"label": "person", "start": 0, "end": 3, "text": "엉뚱함"}]
    assert approve(d2["id"], bad).status_code == 400
    r = client.post(f"/internal/ingest/docs/{d2['id']}/review", headers=H,
                    json={"actor": "reviewer@saebom.test", "action": "reject", "reason": "중복 문서"})
    assert r.json()["status"] == "rejected"
    assert approve(d2["id"], doc["spans"]).status_code == 409


def test_gold_auto_approve(created):
    text = f"송금인: {NAME}\n연락처: {PHONE}\n해외송금 접수."
    gold = pii_pipeline.detect(text)  # 탐지 결과와 같은 정답 → 자동 승인
    wrong = gold[:-1]                 # 하나 빠진 정답 → 검수 대기
    content = "\n".join(json.dumps({"id": i, "text": text, "entities": e}, ensure_ascii=False)
                        for i, e in (("A", gold), ("B", wrong)))
    job_id, docs = upload(created, "jsonl", content)
    st = {x["ext_id"]: (x["status"], x["approved_by"]) for x in docs}
    assert st["A"] == ("stored", "gold") and st["B"] == ("pending", None)
    assert client.get(f"/internal/ingest/jobs/{job_id}", headers=H).json()["auto_approved"] == 1
