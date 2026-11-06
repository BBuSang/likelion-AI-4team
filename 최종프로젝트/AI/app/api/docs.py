"""저장된 문서 관리 API (/internal/docs/*). 스프링이 관리자에게만 열어 줌. 원문은 안 내보냄 (토큰 상태 글만)."""
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ingest import manage

router = APIRouter(prefix="/docs", tags=["docs"])


class DeleteRequest(BaseModel):
    actor: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=500)


@router.get("")
def docs(q: Optional[str] = None, visibility: Optional[Literal["public", "team", "restricted"]] = None,
         source: Optional[Literal["upload", "batch"]] = None, limit: int = 50, offset: int = 0):
    return manage.list_docs(q, visibility, source, limit, offset)


@router.get("/deletion-log")
def deletion_log(limit: int = 100):
    return manage.deletion_log(limit)


@router.get("/{doc_id}")
def doc(doc_id: str):
    try:
        return manage.get_doc(doc_id)
    except manage.NotFound:
        raise HTTPException(404, "문서 없음")


class MetaRequest(BaseModel):
    visibility: Literal["public", "team", "restricted"]
    team: Optional[str] = Field(default=None, max_length=50)
    owner_emp_id: Optional[str] = Field(default=None, max_length=20)


@router.post("/{doc_id}/meta")
def update_meta(doc_id: str, req: MetaRequest):
    try:
        return manage.update_meta(doc_id, req.visibility, req.team, req.owner_emp_id)
    except manage.NotFound:
        raise HTTPException(404, "문서 없음")
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/{doc_id}/delete")
def delete(doc_id: str, req: DeleteRequest):
    try:
        return manage.delete_doc(doc_id, req.actor, req.reason)
    except manage.NotFound:
        raise HTTPException(404, "문서 없음")
    except ValueError as e:
        raise HTTPException(400, str(e))
