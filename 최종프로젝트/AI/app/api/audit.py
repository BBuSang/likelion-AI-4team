"""이상 접근 탐지 API (/internal/audit/anomalies, worklog 38장). 스프링(관리자)만 부름. 실명 없이 토큰·문서 ID·개수만."""
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.audit import anomaly
from app.core import access
from app.pii import restore as restore_mod

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/anomalies")
def anomalies(days: int = 14):
    return anomaly.run(days)


class RefusedUser(BaseModel):
    emp_id: str | None = None
    team: str | None = None
    role: str


class RefusedRequest(BaseModel):
    user: RefusedUser
    actor: str = Field(min_length=1, max_length=100)
    doc_ids: List[str] = Field(default_factory=list, max_length=50)
    reason: str = Field(min_length=1, max_length=40)


@router.post("/refused")
def refused(req: RefusedRequest):
    """백엔드가 거절한 실명 보기 시도(사유 없음 등)를 복원 기록에 남김 (39장) — 권한 밖 시도 반복 규칙이 보게"""
    user = req.user.model_dump()
    try:
        access.check_user(user)
    except access.AccessError as e:
        raise HTTPException(403, str(e))
    restore_mod._log(req.actor, user, None, [], sorted(set(req.doc_ids)), [], refused=req.reason)
    return {"ok": True}
