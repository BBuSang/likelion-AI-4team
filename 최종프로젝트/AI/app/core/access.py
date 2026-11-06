"""문서 권한 (worklog 3-4-1, 10-4). 검색 SQL 조건과 파이썬 판단을 한 곳에서 정의한다.

사용자 = {"emp_id", "team", "role"}  (로그인 계정에 연결된 직원. 스프링이 확인해서 넘김)
  STAFF   : 전사 공개 + 자기 팀 문서 중 담당자가 정해진 것
  LEAD    : 전사 공개 + 자기 팀 문서 전부 (담당자 모르는 것 포함)
  AUDITOR : 전부 (restricted = 팀 모름·감사 보고서 포함)
  공통    : 담당자 본인은 자기 문서를 항상 봄 (사용자 결정 10-4)
실명 복원
  STAFF   : 자기가 담당자인 문서만
  LEAD    : 볼 수 있는 문서 전부
  AUDITOR : 전부, 단 사유(purpose) 필수
"""
ROLES = ("STAFF", "LEAD", "AUDITOR")

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
