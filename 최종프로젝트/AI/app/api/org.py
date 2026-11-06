"""직원 목록 API (/internal/org/*). 스프링만 부름.
이름이 나가는 API(match, employees, employees/{id}, POST employees)는 스프링이 관리자에게만 열어 줌."""
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.org import directory

router = APIRouter(prefix="/org", tags=["org"])


class MatchRequest(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    team: Optional[str] = Field(default=None, max_length=50)


class CreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    team: Optional[str] = Field(default=None, max_length=50)
    position: Optional[str] = Field(default=None, max_length=30)
    role: Literal["STAFF", "LEAD", "AUDITOR"]


@router.get("/teams")
def teams():
    return directory.teams()


@router.post("/match")
def match(req: MatchRequest):
    try:
        return directory.match(req.name, req.team)
    except directory.OrgError as e:
        raise HTTPException(400, str(e))


@router.get("/employees")
def employees(q: Optional[str] = None, limit: int = 200):
    return directory.employees(q, limit)


@router.get("/employees/{emp_id}")
def employee(emp_id: str):
    e = directory.get(emp_id)
    if e is None:
        raise HTTPException(404, "직원 없음")
    return e


@router.post("/employees", status_code=201)
def create(req: CreateRequest):
    try:
        return directory.create(req.name, req.team, req.position, req.role)
    except directory.OrgError as e:
        raise HTTPException(400, str(e))
