"""금융당국 회신 선례(법령해석·비조치의견서·현장건의) RAG — 에이전트 도구 search_noaction"""
from . import search as S
from .respond import answer

SOURCE = "noaction"
available = S.available


def search(query, k=5):
    """에이전트 도구: 벡터 20 → 재순위 5 + 현장건의를 Evidence 로 (사례 글 = 답변 LLM 에 넘기는 것과 같은 모양)"""
    masked = S.mask_pii(query)
    with S.connect() as conn:
        cands, hits = S.rank(conn, masked)
        if not cands:
            return []
        cases = S.load_cases(conn, [h["doc_id"] for h in hits])
        fields = S.exmnt_search(conn, masked, max(c["score"] for c in cands))
    out = []
    for h in hits[:k]:
        c = cases[h["doc_id"]]
        out.append({"source": SOURCE, "cite_id": "", "title": f"{c['gubun']} {c['case_key']} {(c['title'] or '').strip()}"[:120],
                    "text": S.case_block(c), "score": round(h["score"], 4),
                    "meta": {"case_key": c["case_key"], "reply_date": str(c["reply_date"] or ""), "url": c["url"], "time_label": c["time_label"]}})
    for f in fields:
        out.append({"source": SOURCE, "cite_id": "", "title": f"현장건의 {f['case_key']} {(f['title'] or '').strip()}"[:120],
                    "text": S.exmnt_block(f), "display": S.exmnt_block(f, full=True), "score": round(f["score"], 4),
                    "meta": {"case_key": f["case_key"], "reply_date": str(f["reply_date"] or ""), "url": f["url"]}})
    return out


__all__ = ["SOURCE", "available", "search", "answer"]
