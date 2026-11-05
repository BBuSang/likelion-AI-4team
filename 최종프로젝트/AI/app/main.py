"""AI 서버 (FastAPI). 스프링 서버만 부르는 내부 서버.

실행 (최종프로젝트/AI 에서):
    uvicorn app.main:app --port 8000 --reload

- 외부 LLM·임베딩 API 를 부르는 곳은 여기뿐이고, 보내는 글은 가명처리된 것만 (pii_pipeline.guard 로 마지막 차단)
- vault(원문 개인정보) 계정은 이 서버만 가짐
- /health 말고는 모두 X-Service-Key 필요. 기능 라우터는 app/api 에 만들어 아래 internal 에 붙임
"""
import logging

from fastapi import APIRouter, Depends, FastAPI

from app.api import ingest
from app.core import config
from app.core.security import require_service_key

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ai")

if not config.AI_SERVICE_KEY:
    raise RuntimeError("AI_SERVICE_KEY 가 비어 있습니다. .env 에 넣어 주세요 (.env.example 참고)")
if not config.HMAC_KEY:
    logger.warning("HMAC_KEY 가 비어 있습니다. 가명처리 토큰을 만들 수 없습니다")

# 외부에 열 서버가 아니라서 문서 화면(/docs)은 끔
app = FastAPI(title="Saebom AI", docs_url=None, redoc_url=None, openapi_url=None)

# 브라우저가 직접 부르지 않으므로 CORS 미들웨어는 넣지 않음 (스프링이 중계)
internal = APIRouter(prefix="/internal", dependencies=[Depends(require_service_key)])


@internal.get("/ping")
def ping():
    """서비스 키 확인용"""
    return {"ok": True}


internal.include_router(ingest.router)
app.include_router(internal)


@app.get("/health")
def health():
    return {"status": "ok", "embed_provider": config.EMBED_PROVIDER, "llm_provider": config.LLM_PROVIDER}
