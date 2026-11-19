"""법령 RAG 리트리버 (기획안 4-1, 5-6) — 베이스라인: 벡터 검색만

    from retriever.search import search
    search("예금자보호 한도는 1인당 얼마야?", k=5)
    → [{doc_id, title, cite, text, score, law_name, law_level, effective_date, chunk_id, n_hits}, ...]  (조문 단위)

흐름: 질문 → KURE-v1 임베딩(로컬 CPU, 조각과 같은 계산: 최대 512토큰 → CLS → 정규화)
      → pgvector 에서 조각 top-`pool`(기본 50, HNSW ef_search=100) → 조문 단위로 합치기(조문 점수 = 그 조문 조각 중 최고점)
      → 조문 top-k
키워드(BM25)·질문 바꾸기·재순위는 쓰지 않는다 (팀 결정, 기획안 4-1).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import psycopg
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
ENV = dotenv_values(ROOT / ".env")
MODEL, MAX_TOK = "nlpai-lab/KURE-v1", 512
EF_SEARCH = 100   # 5-5: 예시 질문 12개에서 전수 비교와 top-5가 같아지는 가장 작은 값
POOL = 50         # 조문으로 합치기 전에 가져올 조각 수

_model = _conn = None


def _encoder():
    global _model
    if _model is None:
        # 모델이 이미 받아져 있으면 인터넷 확인 없이(빠르게), 없으면 처음 한 번 받는다 (약 2.2GB)
        hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
        if (hub / ("models--" + MODEL.replace("/", "--"))).exists():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(MODEL).eval()
        _model = (tok, model, torch)
    return _model


def embed(texts):
    tok, model, torch = _encoder()
    with torch.inference_mode():
        x = tok(list(texts), padding=True, truncation=True, max_length=MAX_TOK, return_tensors="pt")
        h = model(**x).last_hidden_state[:, 0].float()
        return torch.nn.functional.normalize(h, dim=-1).numpy()


def _db():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg.connect(host=ENV["DB_HOST"], port=ENV["DB_PORT"], dbname=ENV["POSTGRES_DB"],
                                user="law_reader", password=ENV["LAW_PASSWORD"], autocommit=True)
        _conn.execute(f"SET hnsw.ef_search = {EF_SEARCH}")
    return _conn


def _lit(v):
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


SQL = """SELECT chunk_id, article_uid, law_uid, law_name, law_level, article_label, title, cite, text,
                effective_date, 1 - (embedding <=> %(q)s::vector) AS score
         FROM lawrag.chunks {where}
         ORDER BY embedding <=> %(q)s::vector LIMIT %(pool)s"""


def search_vec(qv, k=5, pool=POOL, levels=None):
    """질문 벡터 → 조문 top-k (평가에서 질문 벡터를 미리 묶어 만들 때 씀)"""
    where = "WHERE law_level = ANY(%(levels)s)" if levels else ""
    rows = _db().execute(SQL.format(where=where), {"q": _lit(qv), "pool": pool, "levels": levels}).fetchall()
    arts = {}
    for cid, auid, luid, lname, lvl, label, title, cite, text, ef, score in rows:
        a = arts.get(auid)
        if a is None:
            arts[auid] = {"doc_id": auid, "law_uid": luid, "law_name": lname, "law_level": lvl, "article_label": label,
                          "title": f"{lname} {label}" + (f"({title})" if title else ""), "cite": cite, "text": text,
                          "score": float(score), "chunk_id": cid, "effective_date": str(ef) if ef else "", "n_hits": 1,
                          "hit_chunks": [cid]}
        else:
            a["n_hits"] += 1  # 같은 조문의 다른 조각 (점수는 이미 더 높은 첫 조각)
            a["hit_chunks"].append(cid)
    out = sorted(arts.values(), key=lambda a: -a["score"])
    for i, a in enumerate(out, 1):
        a["rank"] = i
    return out[:k] if k else out


def search(question, k=5, pool=POOL, levels=None):
    """팀 공통 모양: search(question, k) → [{doc_id, title, text, score, ...}] (조문 단위)"""
    return search_vec(embed([question])[0], k=k, pool=pool, levels=levels)


# ── 위임·준용 연결 (기획안 5-6-6, v0.30 채택 · v0.31 DB 표 `lawrag.article_links`) — 기본 search()는 바꾸지 않음 ──
_links = None


def _neighbors():
    """조문 → {연결된 조문: 종류}. 양방향 (아래 → 위, 위 → 아래). 처음 부를 때 DB 에서 한 번 읽어 메모리에 둔다"""
    global _links
    if _links is None:
        from collections import defaultdict
        links = defaultdict(dict)
        for src, dst, kind in _db().execute("SELECT src_uid, dst_uid, kind FROM lawrag.article_links ORDER BY src_uid, dst_uid").fetchall():
            links[src][dst] = kind
            links[dst].setdefault(src, kind)
        _links = links
    return _links


def attach_links(arts, k=5, per=2, total=5):
    """조문 순위(arts, search_vec(k=0) 결과)는 그대로 두고, 1~k위에 연결된 조문 중
    같은 검색 후보 안에 있는 것만 점수 순으로 조문마다 per개·전체 total개까지 '함께 볼 조문'으로 고른다."""
    nb = _neighbors()
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


def search_linked(question, k=5, pool=POOL, levels=None, per=2, total=5):
    """1~k위(순위 그대로) + 함께 볼 조문"""
    return attach_links(search_vec(embed([question])[0], k=0, pool=pool, levels=levels), k=k, per=per, total=total)


# ── LLM 에 넘기는 조문 글 (기획안 5-6-8, 시험 중 — search() 결과는 바꾸지 않음) ──
FULL_LIMIT = 2000  # 여러 조각 조문을 통째로 넘기는 글자 한도


def with_full_text(arts, limit=FULL_LIMIT):
    """후보 조문(arts)의 text 를 LLM 에 넘길 글로 바꾼 새 목록을 돌려준다 (순위·점수는 그대로).
    조각 하나 → 그대로 / 여러 조각 + 전체 limit 자 이하 → 조문 전체 / 초과 → 검색에 걸린 조각 전부(순서대로)"""
    rows = _db().execute("SELECT article_uid, chunk_no, chunk_id, text FROM lawrag.chunks WHERE article_uid = ANY(%s) "
                         "ORDER BY article_uid, chunk_no", ([a["doc_id"] for a in arts],)).fetchall()
    by = {}
    for uid, no, cid, text in rows:
        by.setdefault(uid, []).append((no, cid, text))
    out = []
    for a in arts:
        cs = by.get(a["doc_id"], [])
        if len(cs) <= 1:
            out.append(dict(a, text_mode="조각 하나"))
        elif sum(len(t) for _, _, t in cs) <= limit:
            out.append(dict(a, text="\n".join(t for _, _, t in cs), text_mode=f"전체({len(cs)}조각)"))
        else:
            hit = set(a.get("hit_chunks") or [a["chunk_id"]])
            parts, prev = [], None
            for no, cid, t in cs:
                if cid in hit:
                    if prev is not None and no != prev + 1:
                        parts.append("(중략)")
                    parts.append(t)
                    prev = no
            out.append(dict(a, text="\n".join(parts), text_mode=f"걸린 조각 {len(hit)}/{len(cs)}"))
    return out


# ── 채택된 답변용 후보 (기획안 5-6-7·5-6-8·5-6-9, v0.30) ──
ANSWER_TOP = 10   # 검색 순위에서 그대로 넘기는 조문 수
LINK_FROM = 5     # 연결 조문을 찾는 대상: 1~5위


def union_candidates(arts):
    """합친 구성: 1~ANSWER_TOP위 + 1~LINK_FROM위에 연결된 조문 중 1~ANSWER_TOP위에 없는 것 (점수 순)"""
    _, extra = attach_links(arts, k=LINK_FROM)
    head = arts[:ANSWER_TOP]
    ids = {a["doc_id"] for a in head}
    return head + [e for e in extra if e["doc_id"] not in ids]


def answer_candidates(question, pool=POOL, levels=None):
    """답변 LLM 에 넘길 후보 조문 목록 (순서 = 번호). 글은 조문 글 규칙을 거친 것.
    → [{doc_id, title, cite, text, text_mode, score, rank, law_name, law_level, effective_date, linked_from?, ...}]"""
    arts = search_vec(embed([question])[0], k=0, pool=pool, levels=levels)
    return with_full_text(union_candidates(arts))


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "예금자보호 한도는 1인당 얼마야?"
    t0 = time.time()
    for a in search(q):
        print(f"{a['rank']}. {a['score']:.3f}  {a['cite']}  ({a['law_level']}, 조각 {a['n_hits']}개)")
    print(f"{time.time() - t0:.2f}초 (첫 호출은 모델 로드 포함)")
