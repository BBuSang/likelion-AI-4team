"""서버가 뜨고, 서비스 키 없이는 내부 API 가 막히는지 확인 (DB·모델 필요 없음)."""
import os

os.environ.setdefault("AI_SERVICE_KEY", "test-service-key")  # .env 가 있으면 그 값

from fastapi.testclient import TestClient  # noqa: E402

from app.core import config  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)


def test_health_is_open():
    assert client.get("/health").status_code == 200


def test_internal_requires_service_key():
    assert client.get("/internal/ping").status_code == 401
    assert client.get("/internal/ping", headers={"X-Service-Key": "wrong"}).status_code == 401
    assert client.get("/internal/ping", headers={"X-Service-Key": config.AI_SERVICE_KEY}).status_code == 200


def test_docs_are_closed():
    assert client.get("/docs").status_code == 404
