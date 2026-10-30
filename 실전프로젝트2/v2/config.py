"""v2 환경변수 설정. 비밀 값은 상위 폴더(실전프로젝트2) .env 를 v1과 같이 씁니다 (수정 없이).

v2에서 선택으로 쓰는 값 (없으면 기본값)
- INGEST_PASSWORD : 적재 전용 계정 (rag 쓰기). 없으면 RAG_PASSWORD. 챗봇은 rag_user 로 읽기만 함
- V2_DB_PORT      : v2 DB 포트 (기본 5433, v1 DB 5432와 함께 띄우기 위해)
"""
import os
from pathlib import Path

from dotenv import load_dotenv

V2_DIR = Path(__file__).resolve().parent
ROOT = V2_DIR.parent
load_dotenv(ROOT / ".env")


def env(name, default=None):
    return os.getenv(name, default)


DB_HOST = env("DB_HOST", "localhost")
DB_PORT = int(env("V2_DB_PORT", "5433"))
DB_NAME = env("POSTGRES_DB", "pii_rag")


def dsn(user, password):
    return f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={user} password={password}"


RAG_DSN = dsn("rag_user", env("RAG_PASSWORD", ""))                                    # 검색·답변 (rag 읽기, ops 쓰기)
INGEST_DSN = dsn("ingest_user", env("INGEST_PASSWORD") or env("RAG_PASSWORD", ""))  # 적재 (rag 쓰기)
VAULT_DSN = dsn("vault_user", env("VAULT_PASSWORD", ""))                              # 적재 중 매핑 저장만

EMBED_API_BASE = env("EMBED_API_BASE", "")
EMBED_API_KEY = env("EMBED_API_KEY", "")
EMBED_MODEL = env("EMBED_MODEL", "bge-m3")
EMBED_DIM = int(env("EMBED_DIM", "1024"))
LLM_API_BASE = env("LLM_API_BASE", "")
LLM_API_KEY = env("LLM_API_KEY", "")
LLM_MODEL = env("LLM_MODEL", "HCX-007")
HMAC_KEY = env("HMAC_KEY", "")

DATA_DIR = ROOT / "data"          # 데이터는 v1과 같은 파일을 읽기만 함
OUT_DIR = V2_DIR / "outputs"
