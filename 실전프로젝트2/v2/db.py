"""v2 DB 연결. 계정 3개를 용도별로 씀.

- rag_conn()    : 검색·답변 (rag 읽기, ops 쓰기)
- ingest_conn() : 적재 (rag 쓰기)
- vault_conn()  : 적재 중 매핑 저장 (vault 만)
"""
import psycopg
from pgvector.psycopg import register_vector

import config


def _conn(dsn, vector=True, autocommit=False):
    conn = psycopg.connect(dsn, autocommit=autocommit)
    if vector:
        register_vector(conn)
    return conn


def rag_conn():
    # 자동 커밋: 검색은 읽기만 해서 트랜잭션을 열어 둘 필요가 없음. 열어 두면 데모가 켜진 동안
    # 재적재(TRUNCATE)가 잠금을 기다리며 멈춤 (worklog 46번)
    return _conn(config.RAG_DSN, autocommit=True)


def ingest_conn():
    return _conn(config.INGEST_DSN)


def vault_conn():
    return _conn(config.VAULT_DSN, vector=False)
