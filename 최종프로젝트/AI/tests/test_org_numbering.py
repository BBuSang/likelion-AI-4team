"""새 직원 번호 (안 A) — 문서 재적재로 vault·rag 의 U 직원이 지워져도 스프링 계정 번호와 안 겹침"""
from app.api.org import CreateRequest
from app.org.directory import _next_u


def test_next_u_skips_taken_by_accounts():
    # 재적재 뒤 vault 에는 U0001 만, 계정은 U0001~U0003 사용 중 → U0004
    assert _next_u(["U0001"], ["U0001"], ["U0001", "U0002", "U0003"]) == "U0004"


def test_next_u_ignores_doc_ids_and_empty():
    assert _next_u([], [], []) == "U0001"
    assert _next_u(["E001"], [], [None, "", "E777"]) == "U0001"


def test_create_request_taken_optional():
    assert CreateRequest(name="홍길동", role="STAFF").taken == []
    assert CreateRequest(name="홍길동", role="STAFF", taken=["U0003"]).taken == ["U0003"]
