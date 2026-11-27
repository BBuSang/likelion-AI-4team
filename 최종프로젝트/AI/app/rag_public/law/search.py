"""법령 RAG 검색.

흐름: 질문 → KURE-v1 임베딩 → pgvector 조각 상위 50 (HNSW ef_search 100) → 조문 단위로 합침 (조문 점수 = 조각 최고점)
      → 답변 후보 = 1~10위 + 1~5위에 위임·준용으로 연결된 조문 → 조문 글 (여러 조각이면 2,000자까지 전체)
키워드(BM25)·질문 바꾸기·재순위는 쓰지 않는다 .

DB: 사공삼 DB 의 lawrag 스키마, 읽기 전용 계정 law_reader (사내 RAG 계정과 서로 못 읽음).
질문 임베딩은 사내 RAG 와 같은 모델을 같이 씀 (원 코드의 transformers CLS 512 와 벡터 같음, cosine 1.0 확인).
"""
import threading
from collections import defaultdict

import numpy as np
import psycopg

from app.core import config

EF_SEARCH = 100   # 예시 질문 12개에서 전수 비교와 top-5 가 같아지는 가장 작은 값
POOL = 50         # 조문으로 합치기 전에 가져올 조각 수
ANSWER_TOP = 10   # 답변 후보로 넘기는 검색 순위 조문 수
LINK_FROM = 5     # 연결 조문을 찾는 대상: 1~5위
LINK_PER, LINK_TOTAL = 2, 5   # 조문마다·전체 연결 조문 수
FULL_LIMIT = 2000  # 여러 조각 조문을 통째로 넘기는 글자 한도

_lock = threading.RLock()  # 연결 하나를 여러 스레드가 같이 씀
_conn = None
_links = None

SQL = """SELECT chunk_id, article_uid, law_uid, law_name, law_level, article_label, title, cite, text,
                effective_date, 1 - (embedding <=> %(q)s::vector) AS score
         FROM lawrag.chunks {where}
         ORDER BY embedding <=> %(q)s::vector LIMIT %(pool)s"""


def db():
    global _conn
    with _lock:
        if _conn is None or _conn.closed:
            _conn = psycopg.connect(host=config.DB_HOST, port=config.DB_PORT, dbname=config.DB_NAME,
                                    user="law_reader", password=config.LAW_PASSWORD, autocommit=True)
            _conn.execute(f"SET hnsw.ef_search = {EF_SEARCH}")
        return _conn


def available():
    if not config.LAW_PASSWORD:
        return False
    try:
        with _lock:
            return db().execute("SELECT EXISTS (SELECT 1 FROM lawrag.chunks LIMIT 1)").fetchone()[0]
    except Exception:
        return False


def embed_query(text):
    from app.rag.embed import embed
    return np.asarray(embed([text])[0], dtype=np.float32)


def _lit(v):
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


def search_vec(qv, k=5, pool=POOL, levels=None):
    """질문 벡터 → 조문 순위 (k=0 이면 전부). 같은 조문의 조각은 첫(최고점) 조각으로 합침"""
    where = "WHERE law_level = ANY(%(levels)s)" if levels else ""
    with _lock:
        rows = db().execute(SQL.format(where=where), {"q": _lit(qv), "pool": pool, "levels": levels}).fetchall()
    arts = {}
    for cid, auid, luid, lname, lvl, label, title, cite, text, ef, score in rows:
        a = arts.get(auid)
        if a is None:
            arts[auid] = {"doc_id": auid, "law_uid": luid, "law_name": lname, "law_level": lvl, "article_label": label,
                          "title": f"{lname} {label}" + (f"({title})" if title else ""), "cite": cite, "text": text,
                          "score": float(score), "chunk_id": cid, "effective_date": str(ef) if ef else "",
                          "hit_chunks": [cid]}
        else:
            a["hit_chunks"].append(cid)
    out = sorted(arts.values(), key=lambda a: -a["score"])
    for i, a in enumerate(out, 1):
        a["rank"] = i
    return out[:k] if k else out


def search(question, k=5, levels=None):
    """조문 상위 k (검색만)"""
    return search_vec(embed_query(legal_query(question)), k=k, levels=levels)


def neighbors():
    """조문 → {연결된 조문: 종류} 양방향 (위임·준용, lawrag.article_links). 처음 한 번 읽어 둠"""
    global _links
    with _lock:
        if _links is None:
            links = defaultdict(dict)
            for src, dst, kind in db().execute("SELECT src_uid, dst_uid, kind FROM lawrag.article_links").fetchall():
                links[src][dst] = kind
                links[dst].setdefault(src, kind)
            _links = links
    return _links


def attach_links(arts, k=LINK_FROM, per=LINK_PER, total=LINK_TOTAL):
    """1~k위는 그대로, 그 조문에 연결된 조문 중 같은 검색 후보 안에 있는 것을 점수 순으로 '함께 볼 조문'으로"""
    nb = neighbors()
    top = arts[:k]
    seen = {a["doc_id"] for a in top}
    cand = {a["doc_id"]: a for a in arts[k:]}
    extra = []
    for a in top:
        got = sorted((cand[d] for d in nb.get(a["doc_id"], {}) if d in cand and d not in seen), key=lambda x: -x["score"])[:per]
        for g in got:
            if len(extra) >= total:
                break
            seen.add(g["doc_id"])
            extra.append(dict(g, linked_from=a["doc_id"], link_kind=nb[a["doc_id"]][g["doc_id"]]))
    return top, extra


def with_full_text(arts, limit=FULL_LIMIT):
    """LLM 에 넘길 조문 글: 조각 하나 → 그대로 / 여러 조각 + limit 자 이하 → 조문 전체 / 초과 → 검색에 걸린 조각만 (순서대로)"""
    with _lock:
        rows = db().execute("SELECT article_uid, chunk_no, chunk_id, text FROM lawrag.chunks WHERE article_uid = ANY(%s) "
                            "ORDER BY article_uid, chunk_no", ([a["doc_id"] for a in arts],)).fetchall()
    by = defaultdict(list)
    for uid, no, cid, text in rows:
        by[uid].append((no, cid, text))
    out = []
    for a in arts:
        cs = by.get(a["doc_id"], [])
        if len(cs) <= 1:
            out.append(a)
        elif sum(len(t) for _, _, t in cs) <= limit:
            out.append(dict(a, text="\n".join(t for _, _, t in cs)))
        else:
            hit, parts, prev = set(a["hit_chunks"]), [], None
            for no, cid, t in cs:
                if cid in hit:
                    if prev is not None and no != prev + 1:
                        parts.append("(중략)")
                    parts.append(t)
                    prev = no
            out.append(dict(a, text="\n".join(parts)))
    return out


def article_texts(uids):
    """조문 → 조문 전체 글 (조각을 순서대로 이음). 화면 근거 문서용 — LLM 에 넘기는 글은 with_full_text 그대로"""
    if not uids:
        return {}
    with _lock:
        rows = db().execute("SELECT article_uid, text FROM lawrag.chunks WHERE article_uid = ANY(%s) ORDER BY article_uid, chunk_no",
                            (list(uids),)).fetchall()
    by = defaultdict(list)
    for uid, text in rows:
        by[uid].append(text)
    return {uid: "\n".join(ts) for uid, ts in by.items()}


# 업무에서 흔히 쓰는 말 → 조문에 실제로 적힌 말. 조문엔 '보이스피싱'이 없고 '전기통신금융사기'라
# "보이스피싱 관련 법"이 1위 점수 0.50 으로 기준(MIN_SCORE 0.5338)에 못 미쳐 "찾지 못함"이 됨.
# 검색어에 법률 용어를 덧붙이기만 함 (원래 말은 그대로 둠)
LEGAL_TERMS = {
    "보이스피싱": "전기통신금융사기",
    "스미싱": "전기통신금융사기",
    "대포통장": "사기이용계좌 접근매체 양도",
    "몸캠": "전기통신금융사기",
}


def legal_query(question):
    extra = [v for k, v in LEGAL_TERMS.items() if k in question and v not in question]
    return question + (" " + " ".join(dict.fromkeys(extra)) if extra else "")


def answer_candidates(question, levels=None):
    """답변 LLM 에 넘길 후보 (순서 = 번호): 1~10위 + 1~5위에 연결된 조문 중 10위 안에 없는 것, 조문 글 규칙 적용"""
    arts = search_vec(embed_query(legal_query(question)), k=0, levels=levels)
    _, extra = attach_links(arts)
    head = arts[:ANSWER_TOP]
    ids = {a["doc_id"] for a in head}
    return with_full_text(head + [e for e in extra if e["doc_id"] not in ids])
