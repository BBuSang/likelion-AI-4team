"""문서 ID 자릿수를 가정하지 않는지 (Holdout2 는 H20139 5자리, 화면 업로드는 J35-H20001)"""
import types

from app.rag import chat, search


def test_cite_any_length():
    assert not chat.is_not_found("[H20139] 에는 있고 나머지는 찾을 수 없습니다")
    assert not chat.is_not_found("[J35-H20001] 외에는 찾을 수 없습니다")
    assert chat.is_not_found("관련 문서에서 찾을 수 없습니다")
    assert chat._CITE_WORD.sub(r"[\1]", "[문서 H20139] [문서 T0139]") == "[H20139] [T0139]"


def test_rerank_reads_long_ids(monkeypatch):
    docs = [{"doc_id": i, "token_score": 1.0, "doc_type": "t", "masked_text": "x"} for i in ("H20139", "H20654", "H2065")]
    reply = {"choices": [{"message": {"content": "H2065, H20654, H20139"}}]}
    fake = types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: reply)
    monkeypatch.setattr(search.requests, "post", lambda *a, **k: fake)
    out = search._rerank("q", docs)
    assert [d["doc_id"] for d in out] == ["H2065", "H20654", "H20139"]
