"""문서 종류 자동 판단: 원문 개인정보가 임베딩으로 안 나감 · 1·2위가 가까우면 '애매' · 저장된 문서가 없으면 None (DB·모델 필요 없음)."""
import contextlib

import numpy as np

from app.ingest import service


class _Cur:
    def __init__(self, rows):
        self.rows, self.sql = rows, []

    def execute(self, sql, params=None):
        self.sql.append(sql)

    def fetchall(self):
        return self.rows


def _patch(monkeypatch, rows):
    sent = []
    cur = _Cur(rows)

    def fake_embed(texts):
        sent.extend(texts)
        return [np.ones(4, np.float32) for _ in texts]

    @contextlib.contextmanager
    def fake_conn():
        class C:
            def cursor(self):
                return contextlib.nullcontext(cur)
        yield C()

    monkeypatch.setattr(service, "embed", fake_embed)
    monkeypatch.setattr(service.db, "rag_conn", fake_conn)
    return sent, cur


TEXT = "김민수 고객님이 010-1234-5678 로 카드 분실을 신고했습니다. 이메일 minsu@example.com"
SPANS = [{"label": "person", "start": 0, "end": 3}]   # 일부러 전화·이메일은 탐지 못 한 걸로 — guard 가 가려야 함


def test_guess_masks_pii_before_embedding(monkeypatch):
    sent, _ = _patch(monkeypatch, [("카드분실상담", 0.82), ("민원처리", 0.70)])
    g = service.guess_doc_type(TEXT, SPANS)
    assert g["doc_type"] == "카드분실상담" and g["second"] == "민원처리" and not g["uncertain"]
    joined = " ".join(sent)
    assert "김민수" not in joined and "[person]" in joined
    assert "010-1234-5678" not in joined and "minsu@example.com" not in joined


def test_guess_uncertain_and_empty(monkeypatch):
    _patch(monkeypatch, [("카드분실상담", 0.801), ("민원처리", 0.800)])
    assert service.guess_doc_type(TEXT, SPANS)["uncertain"]
    _patch(monkeypatch, [("회의록", 0.9)])                     # 종류가 하나뿐이면 2위 없음
    g = service.guess_doc_type(TEXT, SPANS)
    assert g["doc_type"] == "회의록" and g["second"] is None and not g["uncertain"]
    _patch(monkeypatch, [])                                    # 저장된 문서가 아직 없음
    assert service.guess_doc_type(TEXT, SPANS) is None
    assert service.guess_doc_type("   ", []) is None
