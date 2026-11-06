"""적재·검수 API (/internal/ingest/*). 스프링만 부름 — 로그인·review_ingest 권한 확인은 스프링이 하고
누가 했는지(actor·uploader)를 본문에 담아 넘긴다 (worklog 4-1-1, 3-7)."""
from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.ingest import service

router = APIRouter(prefix="/ingest", tags=["ingest"])

MAX_UPLOAD_CHARS = 20_000_000


class UploadRequest(BaseModel):
    uploader: str = Field(min_length=1, max_length=100)
    source: Literal["jsonl", "text"]
    content: str = Field(max_length=MAX_UPLOAD_CHARS)
    filename: Optional[str] = Field(default=None, max_length=255)
    doc_type: Optional[str] = Field(default=None, max_length=50)


class Span(BaseModel):
    label: str
    start: int
    end: int
    text: Optional[str] = None  # 화면이 고른 글자. 보내면 서버가 위치와 맞는지 확인


class PreviewRequest(BaseModel):
    spans: List[Span]
    choices: Optional[Dict[str, str]] = None


class ReviewRequest(BaseModel):
    actor: str = Field(min_length=1, max_length=100)
    action: Literal["approve", "reject"]
    spans: Optional[List[Span]] = None
    reason: Optional[str] = Field(default=None, max_length=500)
    choices: Optional[Dict[str, str]] = None  # 사람 잇기: {그룹 id: 기존 pid | "new" | "unresolved"}


def _spans(spans):
    return None if spans is None else [s.model_dump(exclude_none=True) for s in spans]


@router.post("/jobs", status_code=202)
def upload(req: UploadRequest, background: BackgroundTasks):
    """업로드 → job 생성. 탐지는 응답 뒤 백그라운드로 (모델 첫 로딩 1~2분). 진행은 GET /jobs/{id}"""
    try:
        job_id, docs = service.create_job(req.uploader, req.source, req.filename, req.content, req.doc_type)
    except service.IngestError as e:
        raise HTTPException(400, str(e))
    background.add_task(service.process_job, job_id, docs)
    return {"job_id": job_id, "total": len(docs), "status": "processing"}


@router.get("/jobs")
def jobs(limit: int = 50):
    return service.list_jobs(limit)


@router.get("/jobs/{job_id}")
def job(job_id: int):
    s = service.job_summary(job_id)
    if s is None:
        raise HTTPException(404, "job 없음")
    return s


@router.get("/jobs/{job_id}/docs")
def job_docs(job_id: int):
    return service.list_docs(job_id)


@router.get("/docs/{staging_id}")
def doc(staging_id: int, actor: str):
    d = service.get_doc(staging_id, actor)
    if d is None:
        raise HTTPException(404, "문서 없음")
    return d


@router.post("/docs/{staging_id}/preview-links")
def preview(staging_id: int, req: PreviewRequest):
    try:
        out = service.preview_links(staging_id, _spans(req.spans), req.choices)
    except service.IngestError as e:
        raise HTTPException(400, str(e))
    if out is None:
        raise HTTPException(404, "문서 없음")
    return out


@router.post("/docs/{staging_id}/review")
def review(staging_id: int, req: ReviewRequest):
    try:
        out = service.review(staging_id, req.actor, req.action, _spans(req.spans), req.reason, req.choices)
    except service.IngestError as e:
        raise HTTPException(400, str(e))
    except service.Conflict as e:
        raise HTTPException(409, str(e))
    if out is None:
        raise HTTPException(404, "문서 없음")
    return out
