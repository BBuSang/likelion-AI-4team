"""v2 DB 연결. 계정 3개를 용도별로 씀.

- rag_conn()    : 검색·답변 (rag 읽기, ops 쓰기)
- ingest_conn() : 적재 (rag 쓰기)
- vault_conn()  : 적재 중 매핑 저장 (vault 만)
"""
import psycopg
from pgvector.psycopg import register_vector

import config


def _conn(dsn, vector=True):
    conn = psycopg.connect(dsn, autocommit=False)
    if vector:
        register_vector(conn)
    return conn


def rag_conn():
    return _conn(config.RAG_DSN)


def ingest_conn():
    return _conn(config.INGEST_DSN)


def vault_conn():
    return _conn(config.VAULT_DSN, vector=False)
