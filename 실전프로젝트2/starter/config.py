"""환경변수 설정. 값은 프로젝트 루트의 .env 에서 읽습니다."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def env(name, default=None):
    return os.getenv(name, default)


DB_HOST = env("DB_HOST", "localhost")
DB_PORT = int(env("DB_PORT", "5432"))
DB_NAME = env("POSTGRES_DB", "pii_rag")
RAG_DSN = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user=rag_user password={env('RAG_PASSWORD', '')}"
VAULT_DSN = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user=vault_user password={env('VAULT_PASSWORD', '')}"

EMBED_PROVIDER = env("EMBED_PROVIDER", "local")  # api | local
EMBED_API_BASE = env("EMBED_API_BASE", "")
EMBED_API_KEY = env("EMBED_API_KEY", "")
EMBED_MODEL = env("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(env("EMBED_DIM", "1536"))

LLM_API_BASE = env("LLM_API_BASE", "")
LLM_API_KEY = env("LLM_API_KEY", "")
LLM_MODEL = env("LLM_MODEL", "gpt-4o-mini")
# HCX-007 추론 단계 (none | low | medium | high). 비워 두면 보내지 않음 = 모델 기본값(low)
# none은 응답 약 3초로 빠르지만 규칙을 지키는 정도가 실행마다 흔들려 기본은 low (worklog 25·26번)
LLM_REASONING = env("LLM_REASONING", "")

HMAC_KEY = env("HMAC_KEY", "")

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
