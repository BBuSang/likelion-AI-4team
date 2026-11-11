"""문서 권한 (worklog 3-4-1, 10-4). 검색 SQL 조건과 파이썬 판단을 한 곳에서 정의한다.

사용자 = {"emp_id", "team", "role"}  (로그인 계정에 연결된 직원. 스프링이 확인해서 넘김)
  STAFF   : 전사 공개 + 자기 팀 문서 중 담당자가 정해진 것
  LEAD    : 전사 공개 + 자기 팀 문서 전부 (담당자 모르는 것 포함)
  AUDITOR : 전부 (restricted = 팀 모름·감사 보고서 포함)
  공통    : 담당자 본인은 자기 문서를 항상 봄 (사용자 결정 10-4)
실명 복원
  STAFF   : 자기가 담당자인 문서만
  LEAD    : 볼 수 있는 문서 전부, 단 사유(purpose) 필수 (2026-11-11 사용자 결정, worklog 25-9)
  AUDITOR : 전부, 단 사유(purpose) 필수
  STAFF 는 자기 담당 문서라 사유 없이
"""
ROLES = ("STAFF", "LEAD", "AUDITOR")
PURPOSE_REQUIRED = {"LEAD", "AUDITOR"}  # 실명 복원 때 사유가 필요한 역할

# 평가 스크립트처럼 사람이 아닌 실행에서만 쓰는 '전부 보기'. 서버 API 는 항상 실제 사용자를 넘김
SYSTEM = {"emp_id": None, "team": None, "role": "AUDITOR", "system": True}


class AccessError(PermissionError):
    pass


def check_user(user):
    if not isinstance(user, dict) or user.get("role") not in ROLES:
        raise AccessError("역할을 모르는 사용자")
    if user["role"] != "AUDITOR" and not user.get("emp_id"):
        raise AccessError("직원(emp_id)이 연결되지 않은 계정")
    return user


def sql(user, alias=""):
    """검색 WHERE 에 붙일 조건과 값. alias: 'c.' 처럼 테이블 별칭."""
    check_user(user)
    a = alias
    if user["role"] == "AUDITOR":
        return "TRUE", {}
    own = f"{a}owner_emp_id = %(acl_emp)s"
    if user["role"] == "LEAD":
        cond = f"({a}visibility = 'public' OR ({a}visibility = 'team' AND {a}team = %(acl_team)s) OR {own})"
    else:
        cond = (f"({a}visibility = 'public' OR ({a}visibility = 'team' AND {a}team = %(acl_team)s "
                f"AND {a}owner_emp_id IS NOT NULL) OR {own})")
    return cond, {"acl_emp": user["emp_id"], "acl_team": user.get("team")}


def can_view(user, doc):
    """doc = {owner_emp_id, team, visibility} — sql() 과 같은 판단 (테스트로 둘이 같은지 확인)."""
    check_user(user)
    if user["role"] == "AUDITOR":
        return True
    if user.get("emp_id") and doc.get("owner_emp_id") == user["emp_id"]:
        return True
    if doc["visibility"] == "public":
        return True
    if doc["visibility"] != "team" or doc.get("team") is None or doc["team"] != user.get("team"):
        return False
    return user["role"] == "LEAD" or doc.get("owner_emp_id") is not None


def can_restore(user, doc):
    check_user(user)
    if user["role"] == "AUDITOR":
        return True
    if not can_view(user, doc):
        return False
    return user["role"] == "LEAD" or doc.get("owner_emp_id") == user["emp_id"]


def scope_text(user):
    """이 사용자가 볼 수 있는 범위 설명 (못 찾았을 때 안내용). 권한 밖에 문서가 '있는지'는 말하지 않음."""
    check_user(user)
    if user["role"] == "AUDITOR":
        return "전체 문서"
    team = user.get("team") or "소속 팀"
    if user["role"] == "LEAD":
        return f"전사 공개 문서, {team} 문서, 본인 담당 문서"
    return f"전사 공개 문서, {team} 문서 중 담당자가 정해진 것, 본인 담당 문서"


def not_found_text(user):
    """'관련 문서에서 찾을 수 없습니다' 대신 보여 줄 고정 문구 (worklog 26장)"""
    scope = scope_text(user)
    if user["role"] == "AUDITOR":
        return f"{scope}에서 질문과 관련된 기록을 찾지 못했습니다. 고객 이름, 처리 건 종류, 날짜 등을 바꿔서 다시 물어봐 주세요."
    return (f"볼 수 있는 범위({scope})에서 질문과 관련된 기록을 찾지 못했습니다. "
            "다른 팀 담당 기록이라면 그 팀 담당자나 감사 담당자에게 확인을 요청해 주세요.")
