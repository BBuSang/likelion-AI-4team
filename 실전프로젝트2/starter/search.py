"""검색: 토큰 필터 + 의미 검색 결합.

    python -m starter.search "윤선영 고객 상담 내역 찾아줘"
    python -m starter.search "연체 안내 문서" --mode semantic

검색 방식 (--mode, 또는 환경변수 SEARCH_MODE. 기본 hybrid) — 연구계획서 E3 비교 조건
  semantic : 마스킹된 질의를 임베딩해 코사인 거리가 가까운 순서
  token    : 질의 토큰을 가진 문서만, 겹치는 토큰이 많은 순서 (임베딩 API 호출 없음)
  hybrid   : ① 질의 토큰을 가진 문서를 위로 (겹치는 토큰 수가 많을수록 위, 같으면 의미 점수 순)
             ② 그 아래를 의미 검색 결과로 채움
             ③ 질의에 토큰이 없거나 맞는 문서가 0건이면 의미 검색 결과 그대로
  토큰이 여러 개일 때 '모두 가진 문서만'으로 거르지 않고 겹치는 수로 순서만 매깁니다.
  모두 가진 문서가 맨 위에 오고, 일부만 가진 문서도 버리지 않습니다.

다른 코드(run_queries.py, chat.py)가 쓰는 약속
  search(query, k, conn=None) -> (masked_query, query_tokens, docs)
  docs: [{"doc_id", "chunk_id", "masked_text", "tokens", "doc_type", "score", "token_hits"}, ...]  순위 순서
검색 단계는 rag_user 계정만 씁니다 (매핑 테이블 접근 없음).
"""
import argparse
import os

from . import db, pii_pipeline
from .embed import embed

MODES = ("semantic", "token", "hybrid")
MODE = os.getenv("SEARCH_MODE", "hybrid")

_COLS = "doc_id, chunk_id, masked_text, tokens, doc_type"
_HITS = "cardinality(ARRAY(SELECT unnest(tokens) INTERSECT SELECT unnest(%(q)s::text[])))"


def mask_query(query):
    masked, records = pii_pipeline.pseudonymize(query, pii_pipeline.detect(query))
    return masked, sorted({r["token"] for r in records})


def _rows(cur, sql, params):
    cur.execute(sql, params)
    return [{"doc_id": d, "chunk_id": c, "masked_text": t, "tokens": tk, "doc_type": dt,
             "token_hits": int(h), "score": 1 - float(dist) if dist is not None else 0.0}
            for d, c, t, tk, dt, h, dist in cur.fetchall()]


def _semantic(cur, vec, q_tokens, n):
    return _rows(cur, f"""SELECT {_COLS}, {_HITS} AS hits, embedding <=> %(v)s AS dist
                          FROM rag.chunks ORDER BY dist LIMIT %(n)s""",
                 {"q": q_tokens, "v": vec, "n": n})


def _token(cur, vec, q_tokens, n):
    """질의 토큰을 하나라도 가진 청크 (GIN 인덱스). vec 이 없으면 의미 점수 없이 doc_id 순."""
    dist = "embedding <=> %(v)s" if vec is not None else "NULL::float"
    return _rows(cur, f"""SELECT {_COLS}, {_HITS} AS hits, {dist} AS dist
                          FROM rag.chunks WHERE tokens && %(q)s::text[]
                          ORDER BY hits DESC, dist, doc_id LIMIT %(n)s""",
                 {"q": q_tokens, "v": vec, "n": n})


def _dedupe(rows, k):
    """문서별 가장 앞 청크 하나만 남깁니다."""
    best = {}
    for r in rows:
        best.setdefault(r["doc_id"], r)
    return list(best.values())[:k]


def search(query, k=10, conn=None, mode=None):
    mode = mode or MODE
    if mode not in MODES:
        raise ValueError(f"mode 는 {MODES} 중 하나여야 합니다: {mode}")
    masked_q, q_tokens = mask_query(query)
    vec = None if mode == "token" else embed([masked_q])[0]
    own = conn is None
    conn = conn or db.rag_conn()
    try:
        with conn.cursor() as cur:
            if mode == "semantic":
                rows = _semantic(cur, vec, q_tokens, k * 3)
            elif mode == "token":
                rows = _token(cur, None, q_tokens, k * 3) if q_tokens else []
            else:
                matched = _token(cur, vec, q_tokens, k * 3) if q_tokens else []
                rows = matched + _semantic(cur, vec, q_tokens, k * 3)
    finally:
        if own:
            conn.close()
    return masked_q, q_tokens, _dedupe(rows, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--mode", choices=MODES, default=MODE)
    a = ap.parse_args()
    masked_q, q_tokens, docs = search(a.query, a.k, mode=a.mode)
    print(f"질의(가명처리): {masked_q}\n질의 토큰: {q_tokens or '없음'}\n")
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d['doc_id']}  score={d['score']:.3f}  token_hits={d['token_hits']}  tokens={d['tokens'][:4]}")
        print("   " + d["masked_text"][:120].replace("\n", " ") + " …")


if __name__ == "__main__":
    main()
