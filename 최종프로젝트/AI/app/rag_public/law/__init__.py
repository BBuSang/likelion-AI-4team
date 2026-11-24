"""법령 RAG — 에이전트 도구 search_law"""
from .respond import MIN_SCORE, answer, law_url
from .search import answer_candidates, article_texts, available

SOURCE = "law"


def search(query, k=5):
    """에이전트 도구: 답변용 후보 조문을 Evidence 로. 1위 점수가 MIN_SCORE 미만이면 관련 조문 없음 → []
    text = LLM 에 넘기는 글 (긴 조문은 걸린 조각만), display = 화면 근거 문서에 보여 줄 조문 전체"""
    cands = answer_candidates(query)
    if not cands or max(a["score"] for a in cands) < MIN_SCORE:
        return []
    cands = cands[:max(k, 1)]
    full = article_texts([a["doc_id"] for a in cands])
    return [{"source": SOURCE, "cite_id": f"LAW-{i}", "title": a["title"], "text": a["text"], "display": full.get(a["doc_id"], a["text"]),
             "score": round(a["score"], 4),
             "meta": {"law_name": a["law_name"], "article_label": a.get("article_label", ""), "law_level": a.get("law_level", ""),
                      "effective_date": a.get("effective_date", ""), "url": law_url(a), "linked_from": a.get("linked_from")}}
            for i, a in enumerate(cands, 1)]


__all__ = ["SOURCE", "available", "search", "answer"]
