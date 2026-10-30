"""v2 DB 계정 권한 점검 (plan_v2.md P1).

계정마다 "되어야 하는 것"과 "막혀야 하는 것"을 실제로 실행해 보고 표로 출력합니다.
쓰기 시험은 트랜잭션 안에서 하고 항상 되돌립니다(ROLLBACK). 비밀 값은 출력하지 않습니다.

실행 (실전프로젝트2/v2 폴더): python -X utf8 check_permissions.py
"""
import sys

import psycopg

import config

# (계정, DSN, 설명, SQL, 기대: True=성공해야 함 / False=거부돼야 함)
CASES = [
    ("rag_user", config.RAG_DSN, "rag 읽기", "SELECT count(*) FROM rag.chunks", True),
    ("rag_user", config.RAG_DSN, "rag 쓰기", "INSERT INTO rag.tokens(token, label) VALUES ('<T_TEST>', 't')", False),
    ("rag_user", config.RAG_DSN, "rag 삭제", "DELETE FROM rag.documents", False),
    ("rag_user", config.RAG_DSN, "vault 읽기", "SELECT count(*) FROM vault.token_map", False),
    ("rag_user", config.RAG_DSN, "ops 쓰기", "INSERT INTO ops.query_log(query_id, masked_query) VALUES ('T', 't')", True),
    ("ingest_user", config.INGEST_DSN, "rag 쓰기", "INSERT INTO rag.tokens(token, label) VALUES ('<T_TEST>', 't')", True),
    ("ingest_user", config.INGEST_DSN, "vault 읽기", "SELECT count(*) FROM vault.token_map", False),
    ("ingest_user", config.INGEST_DSN, "ops 읽기", "SELECT count(*) FROM ops.query_log", False),
    ("vault_user", config.VAULT_DSN, "vault 쓰기", "INSERT INTO vault.token_map(token, label, normalized) VALUES ('<T_TEST>', 't', 't')", True),
    ("vault_user", config.VAULT_DSN, "rag 읽기", "SELECT count(*) FROM rag.chunks", False),
    ("vault_user", config.VAULT_DSN, "ops 읽기", "SELECT count(*) FROM ops.query_log", False),
]


def run(dsn, sql):
    """성공하면 (True, ''), 권한 오류면 (False, 오류 종류)."""
    with psycopg.connect(dsn) as conn:
        try:
            conn.execute(sql)
            return True, ""
        except psycopg.errors.InsufficientPrivilege as e:
            return False, type(e).__name__
        finally:
            conn.rollback()


def main():
    failed = 0
    print(f"DB {config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}\n")
    print(f"{'계정':<12} {'시험':<10} {'기대':<6} {'결과':<6} 판정")
    for user, dsn, desc, sql, expect in CASES:
        ok, err = run(dsn, sql)
        good = ok == expect
        failed += not good
        print(f"{user:<12} {desc:<10} {'허용' if expect else '거부':<6} {'허용' if ok else '거부':<6} "
              f"{'OK' if good else 'FAIL'} {err}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} 통과")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
