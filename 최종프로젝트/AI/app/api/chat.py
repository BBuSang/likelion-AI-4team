"""질문·실명 복원 API (/internal/chat/*). 스프링만 부름.

user = {emp_id, team, role} 는 스프링이 요청마다 DB 에서 읽은 계정 값 (worklog 11-4, 12장).
- ask    : 검색(권한 밖 문서는 검색 단계에서 거름) → LLM 답변(토큰 상태 그대로) → 출력 검사
- restore: 답변 속 토큰 중, 이 사람이 복원해도 되는 문서에서 나온 것만 실명으로. 매번 vault.restore_log 에 기록
"""
import logging
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core import access
from app.pii import restore as restore_mod
from app.rag import chat as chat_mod

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger("ai.chat")

SNIPPET = 300  # 화면 근거 문서 미리보기 글자 수 (가명처리된 글)


class User(BaseModel):
    emp_id: Optional[str] = Field(default=None, max_length=20)
    team: Optional[str] = Field(default=None, max_length=50)
    role: Literal["STAFF", "LEAD", "AUDITOR"]


class AskRequest(BaseModel):
    user: User
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=3, ge=1, le=10)


class RestoreRequest(BaseModel):
    user: User
    actor: str = Field(min_length=1, max_length=100)  # 계정 이메일 (기록용)
    text: str = Field(min_length=1, max_length=20_000)
    doc_ids: List[str] = Field(max_length=50)
    purpose: Optional[str] = Field(default=None, max_length=500)


def _doc(d):
    return {"doc_id": d["doc_id"], "doc_type": d.get("doc_type"), "score": round(float(d.get("score", 0)), 4),
            "snippet": (d.get("masked_text") or "")[:SNIPPET]}


@router.post("/ask")
def ask(req: AskRequest):
    user = req.user.model_dump()
    try:
        access.check_user(user)
        r = chat_mod.answer(req.query, k=req.k, user=user)
    except access.AccessError as e:
        raise HTTPException(403, str(e))
    except Exception as e:  # 임베딩·LLM API 실패 등 — 내용(질문 원문 포함 가능)은 응답·로그에 안 남김
        logger.warning("ask 실패: %s", type(e).__name__)
        raise HTTPException(502, "검색 또는 답변 생성에 실패했습니다")
    return {
        "masked_query": r["masked_query"],
        "answer": r["answer"],          # 토큰 상태. LLM 키가 없으면 None (검색 결과만)
        "ask_back": r["ask_back"],
        "docs": [_doc(d) for d in r["docs"]],
        "check": r["check"],            # 출력 검사: 확인 안 된 토큰·원문 개인정보 (가린 뒤)
        "model": r["model"] if r["answer"] and not r["ask_back"] else None,
        "timing": r["timing"],
    }


@router.post("/restore")
def restore(req: RestoreRequest):
    user = req.user.model_dump()
    try:
        text, info = restore_mod.restore_for(req.text, user, req.doc_ids, purpose=req.purpose, actor=req.actor)
    except access.AccessError as e:
        raise HTTPException(403, str(e))
    return {"text": text, "restored": len(info["restored"]), "denied": info["denied"],
            "unknown": info["unknown"], "docs": info["docs"]}
