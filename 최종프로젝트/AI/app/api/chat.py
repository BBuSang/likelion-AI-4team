"""질문·실명 복원 API (/internal/chat/*). 스프링만 부름.

user = {emp_id, team, role} 는 스프링이 요청마다 DB 에서 읽은 계정 값.
- ask    : (AGENT_CHAT) 에이전트 분류 → 검색(권한 밖 문서는 검색 단계에서 거름) → LLM 답변(토큰 상태 그대로) → 출력 검사
- ask/stream: 같은 흐름을 SSE 로 — 단계(stage) · 출력 검사를 거친 답변 조각(delta) · 마지막 결과(done, ask 와 같은 모양)
- restore: 답변 속 토큰 중, 이 사람이 복원해도 되는 문서에서 나온 것만 실명으로. 매번 vault.restore_log 에 기록
"""
import json
import logging
import queue
import threading
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agents import agent as agent_mod
from app.core import access, config
from app.pii import restore as restore_mod
from app.rag import chat as chat_mod

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger("ai.chat")

SNIPPET = 16000  # 화면 근거 문서 글 (가명처리된 글). 전에는 300자라 글이 중간에 잘림 → 원문 전체 (가장 긴 법령 조문 15,930자), 화면이 접어서 보여 줌


class User(BaseModel):
    emp_id: Optional[str] = Field(default=None, max_length=20)
    team: Optional[str] = Field(default=None, max_length=50)
    role: Literal["STAFF", "LEAD", "AUDITOR"]


class Turn(BaseModel):
    """같은 대화의 앞 턴. Spring 이 DB 에서 꺼내 넘김 — 화면이 보낸 값이 아님"""
    query: Optional[str] = Field(default=None, max_length=1000)   # 원문 질문 (이 서버 안에서 토큰 되돌리기에만)
    masked_query: str = Field(max_length=2000)
    answer: Optional[str] = Field(default=None, max_length=20_000)  # 토큰 상태 답변 (다시 쓰기엔 앞 300자만)


class AskRequest(BaseModel):
    user: User
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=3, ge=1, le=10)
    history: List[Turn] = Field(default_factory=list, max_length=5)


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


def _query_only_names(query):
    """질문에 쓴 이름 중 **저장된 데이터에 없는 사람**의 토큰 목록. 화면이 이 토큰만 질문 원문의 그 자리 말로 보여 줌
    (본인이 친 이름이고 데이터에도 없어 새로 드러나는 정보가 없음). 데이터에 있는 이름은 그대로 가림.
    이름 원문은 내보내지 않고 토큰만 — 답변 결과는 스프링이 평문으로 저장하므로.
    데이터에 있다 = 같은 이름(성 없는 이름은 이름 부분)의 인물이 rag.persons 에 있거나, 그 토큰이 문서 청크에 있음"""
    try:
        from app.core import db
        from app.pii import pii_pipeline
        from app.rag.search import mask_query
        _, records = mask_query(query)
        people = [r for r in records if r["label"] == "person"]
        if not people:
            return []
        out = set()
        with db.rag_conn() as c:
            for r in people:
                name = r["normalized"]
                surname, _ = pii_pipeline.split_name(name)
                names = [pii_pipeline.name_token(name)]
                givens = [] if surname else [pii_pipeline.given_token(name)]
                hit = c.execute("SELECT 1 FROM rag.persons WHERE name_token = ANY(%s) OR given_token = ANY(%s) LIMIT 1",
                                (names, givens)).fetchone() \
                    or c.execute("SELECT 1 FROM rag.chunks WHERE tokens @> ARRAY[%s] LIMIT 1", (r["token"],)).fetchone()
                if not hit:
                    out.add(r["token"])
        return sorted(out)
    except Exception as e:  # 실패하면 전처럼 모두 가린 채로
        logger.warning("질문 이름 확인 실패: %s", type(e).__name__)
        return []


FAILED = "검색 또는 답변 생성에 실패했습니다"


def _run(req, user, on_event=None):
    """질문 하나 실행 (ask·ask/stream 공통). on_event: 스트리밍 단계·검사한 답변 조각"""
    if config.AGENT_CHAT:
        return agent_mod.ask(req.query, user, k=req.k, on_event=on_event,
                             history=[t.model_dump() for t in req.history])
    on_delta = (lambda t: on_event("delta", text=t)) if on_event else None
    return chat_mod.answer(req.query, k=req.k, user=user, on_delta=on_delta)


@router.post("/ask")
def ask(req: AskRequest):
    user = req.user.model_dump()
    try:
        access.check_user(user)
        r = _run(req, user)
    except access.AccessError as e:
        raise HTTPException(403, str(e))
    except Exception as e:  # 임베딩·LLM API 실패 등 — 내용(질문 원문 포함 가능)은 응답·로그에 안 남김
        logger.warning("ask 실패: %s", type(e).__name__)
        raise HTTPException(502, FAILED)
    return _response(r, req.query)


@router.post("/ask/stream")
def ask_stream(req: AskRequest):
    """SSE. event: stage {stage, sources?} / delta {text: 출력 검사를 거친 조각} / done {result: ask 와 같은 모양}
    / error {status, message}. 권한 문제는 흐름을 열기 전에 403 으로. 화면은 done 의 answer 로 마지막에 바꿈 (전체 글 기준 검사)"""
    user = req.user.model_dump()
    try:
        access.check_user(user)
    except access.AccessError as e:
        raise HTTPException(403, str(e))
    q = queue.Queue()

    def work():
        try:
            q.put(("done", {"result": _response(_run(req, user, on_event=lambda kind, **d: q.put((kind, d))), req.query)}))
        except access.AccessError as e:
            q.put(("error", {"status": 403, "message": str(e)}))
        except Exception as e:  # 내용은 남기지 않음 (ask 와 같음)
            logger.warning("ask/stream 실패: %s", type(e).__name__)
            q.put(("error", {"status": 502, "message": FAILED}))
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()

    def events():
        while (item := q.get()) is not None:
            kind, data = item
            yield f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _response(r, query=None):
    """화면으로 가는 답 (ask 응답 · ask/stream 의 done). query = 사용자가 친 질문 원문 (이름 확인에만, 내보내지 않음)"""
    return {
        "masked_query": r["masked_query"],
        "answer": r["answer"],          # 토큰 상태. LLM 키가 없으면 None (검색 결과만)
        "ask_back": r["ask_back"],
        # 못 찾았다고 답했으면 근거 문서 없음 (검색 후보를 '근거 문서 N건'으로 보이면 오해)
        "docs": [] if r.get("not_found") else [_doc(d) for d in r["docs"]],
        "not_found": bool(r.get("not_found")),
        "check": r["check"],            # 출력 검사: 확인 안 된 토큰·원문 개인정보 (가린 뒤)
        "model": r["model"] if r["answer"] and not r["ask_back"] else None,
        "timing": r["timing"],
        "agent": r.get("agent"),        # 에이전트 분류 결과 (경로·출처·아직 없는 RAG). AGENT_CHAT=0 이면 None
        # 답변 속 사람 중 같은 이름이 여럿인 사람 → 화면이 '동명이인 가능' 경고 (37장). 실패해도 답변은 그대로
        "ambiguous_people": _ambiguous((r.get("answer") or "") + " " + (r.get("masked_query") or "")),  # 답변·질문 속 사람
        # 질문에 쓴 이름 중 데이터에 없는 사람의 토큰 → 화면이 질문 원문의 그 말로 보여 줌 (데이터에 있는 이름은 가린 채로)
        "query_only_tokens": _query_only_names(query) if query else [],
    }


@router.post("/restore")
def restore(req: RestoreRequest):
    user = req.user.model_dump()
    try:
        text, info = restore_mod.restore_for(req.text, user, req.doc_ids, purpose=req.purpose, actor=req.actor)
    except access.AccessError as e:
        raise HTTPException(403, str(e))
    return {"text": text, "restored": len(info["restored"]), "denied": info["denied"],
            "unknown": info["unknown"], "docs": info["docs"],
            "warnings": info.get("warnings", [])}  # 이 실명 보기가 걸린 이상 접근 규칙 (본인에게 알림, 39장)
