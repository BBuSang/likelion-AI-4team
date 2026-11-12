"""질문·실명 복원 API (/internal/chat/*). 스프링만 부름.

user = {emp_id, team, role} 는 스프링이 요청마다 DB 에서 읽은 계정 값 (worklog 11-4, 12장).
- ask    : (AGENT_CHAT) 에이전트 분류 → 검색(권한 밖 문서는 검색 단계에서 거름) → LLM 답변(토큰 상태 그대로) → 출력 검사
- restore: 답변 속 토큰 중, 이 사람이 복원해도 되는 문서에서 나온 것만 실명으로. 매번 vault.restore_log 에 기록
"""
import logging
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents import agent as agent_mod
from app.core import access, config
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


def _ambiguous(answer):
    try:
        return chat_mod.ambiguous_people(answer)
    except Exception as e:  # DB 일시 오류 — 경고만 빠짐
        logger.warning("동명이인 확인 실패: %s", type(e).__name__)
        return []


@router.post("/ask")
def ask(req: AskRequest):
    user = req.user.model_dump()
    try:
        access.check_user(user)
        r = (agent_mod.ask(req.query, user, k=req.k) if config.AGENT_CHAT
             else chat_mod.answer(req.query, k=req.k, user=user))
    except access.AccessError as e:
        raise HTTPException(403, str(e))
    except Exception as e:  # 임베딩·LLM API 실패 등 — 내용(질문 원문 포함 가능)은 응답·로그에 안 남김
        logger.warning("ask 실패: %s", type(e).__name__)
        raise HTTPException(502, "검색 또는 답변 생성에 실패했습니다")
    return {
        "masked_query": r["masked_query"],
        "answer": r["answer"],          # 토큰 상태. LLM 키가 없으면 None (검색 결과만)
        "ask_back": r["ask_back"],
        # 못 찾았다고 답했으면 근거 문서 없음 (검색 후보를 '근거 문서 N건'으로 보이면 오해, worklog 26장)
        "docs": [] if r.get("not_found") else [_doc(d) for d in r["docs"]],
        "not_found": bool(r.get("not_found")),
        "check": r["check"],            # 출력 검사: 확인 안 된 토큰·원문 개인정보 (가린 뒤)
        "model": r["model"] if r["answer"] and not r["ask_back"] else None,
        "timing": r["timing"],
        "agent": r.get("agent"),        # 에이전트 분류 결과 (경로·출처·아직 없는 RAG). AGENT_CHAT=0 이면 None
        # 답변 속 사람 중 같은 이름이 여럿인 사람 → 화면이 '동명이인 가능' 경고 (37장). 실패해도 답변은 그대로
        "ambiguous_people": _ambiguous((r.get("answer") or "") + " " + (r.get("masked_query") or "")),  # 답변·질문 속 사람
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
