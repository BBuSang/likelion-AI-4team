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
EMBED_BATCH = int(env("EMBED_BATCH", "64"))  # 요청 1건에 넣을 문장 수 (CLOVA Studio 는 1)

LLM_API_BASE = env("LLM_API_BASE", "")
LLM_API_KEY = env("LLM_API_KEY", "")
LLM_MODEL = env("LLM_MODEL", "gpt-4o-mini")
RERANK = env("RERANK", "0") == "1"           # 검색 상위 결과를 LLM 으로 다시 정렬할지 (search.py)
RERANK_MODEL = env("RERANK_MODEL", "HCX-005")  # 재순위용 (빠른 모델)

# LLM 제공자 (worklog 17장). clova = 위 LLM_API_BASE/KEY(.env), openai = OpenAI API.
# OpenAI 키는 OPENAI_API_KEY 또는 상위 폴더 .env 의 openai-api-key. 임베딩(bge-m3)은 제공자와 상관없이 그대로
LLM_PROVIDER = env("LLM_PROVIDER", "clova")
if LLM_PROVIDER == "openai":
    LLM_API_BASE = env("OPENAI_API_BASE", "https://api.openai.com/v1")
    LLM_API_KEY = env("OPENAI_API_KEY", "")
    if not LLM_API_KEY and (ROOT.parent / ".env").exists():
        for _line in (ROOT.parent / ".env").read_text(encoding="utf-8").splitlines():
            if _line.strip().lower().startswith(("openai-api-key=", "openai_api_key=")):
                LLM_API_KEY = _line.split("=", 1)[1].strip().strip("\"'")
    LLM_MODEL = env("OPENAI_LLM_MODEL", "gpt-4.1")
    RERANK_MODEL = env("OPENAI_RERANK_MODEL", "gpt-4.1-mini")
# 검색 개선 단계별 켜고 끄기 (비교 실험용, 기본은 모두 켬)
SEARCH_DOCTYPE = env("SEARCH_DOCTYPE", "1") == "1"   # 질의가 가리키는 문서 종류 가산점
DOC_TYPE_MODE = env("DOC_TYPE_MODE", "both")  # dict = 손으로 만든 단어 사전, centroid = 종류별 평균 임베딩, both = 둘 다 (search.centroid_doc_types)
DOC_TYPE_MARGIN = float(env("DOC_TYPE_MARGIN", "0"))  # centroid: 1위 종류가 2위보다 이만큼 가까울 때만
QUERY_REGION_FILTER = env("QUERY_REGION_FILTER", "1") == "1"  # 질의에서 지역명을 이름으로 잡은 것 거르기
SELECT_FULL_MATCH = env("SELECT_FULL_MATCH", "1") == "1"  # LLM 에 조건을 다 맞춘 문서만 넘기기 (chat.py)
API_GUARD = env("API_GUARD", "1") == "1"  # API 로 보내기 직전 남은 번호·이메일 모양 가리기 (pii_pipeline.guard)
ASK_BACK = env("ASK_BACK", "1") == "1"  # 동명이인을 질의 단서로 못 고르면 LLM 대신 되묻기 (chat.py)
SOFT_HOMONYM = env("SOFT_HOMONYM", "0") == "1"  # 다른 사람이라는 증거가 없는 같은 이름 확정 인물은 0.3 대신 0.7 (search._choose_person). 17장: 효과 없어 기본 끔

HMAC_KEY = env("HMAC_KEY", "")

DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
