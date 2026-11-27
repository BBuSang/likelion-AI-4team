"""금융상품 RAG — 에이전트 도구 search_product"""
import threading

from . import respond as _respond
from .index import CHUNKS, VECTORS, ProductIndex

SOURCE = "product"
_idx, _lock = None, threading.Lock()


def available():
    return all((CHUNKS / f).exists() for f in ("chunks.jsonl", "clauses.jsonl", "glossary.jsonl")) and VECTORS.exists()


def index():
    """색인은 한 번만 (약 10~20초: 데이터·벡터·키워드 색인)"""
    global _idx
    with _lock:
        if _idx is None:
            _idx = ProductIndex()
    return _idx


def search(query, k=5):
    """에이전트 도구: 상품명 찾기 → 좁힌 결합 순위 → 점수형 재순위 상위 k 를 Evidence 로"""
    idx = index()
    ids, _ = _respond.context_ids(idx, query)
    return [{"source": SOURCE, "cite_id": f"PRD-{n}", "title": idx.label(i), "text": idx.items[i]["text"][:_respond.CTX_CHARS], "display": idx.display(i),
             "score": float(idx.q_scores.get(query, {}).get(i, 0)), "meta": {"kind": idx.items[i]["kind"]}}
            for n, i in enumerate(ids[:max(k, 1)], 1)]


def answer(query):
    """금융상품만 필요한 질문: 답변 흐름 전체 (판매 상태·상품명 찾기·근거 머리·대상 상품 줄)"""
    return _respond.answer(index(), query)


__all__ = ["SOURCE", "available", "search", "answer", "index"]
