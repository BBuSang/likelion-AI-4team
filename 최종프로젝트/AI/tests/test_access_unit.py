"""문서 권한 판단 (app/core/access.py). DB 없이."""
import pytest

from app.core import access

STAFF = {"emp_id": "E1", "team": "고객센터", "role": "STAFF"}
LEAD = {"emp_id": "E2", "team": "고객센터", "role": "LEAD"}
AUDIT = {"emp_id": "E3", "team": "정보보호팀", "role": "AUDITOR"}

DOCS = {
    "public": {"owner_emp_id": None, "team": None, "visibility": "public"},
    "team_owned": {"owner_emp_id": "E9", "team": "고객센터", "visibility": "team"},
    "team_no_owner": {"owner_emp_id": None, "team": "고객센터", "visibility": "team"},
    "other_team": {"owner_emp_id": "E8", "team": "인사팀", "visibility": "team"},
    "restricted": {"owner_emp_id": None, "team": None, "visibility": "restricted"},
    "restricted_mine": {"owner_emp_id": "E1", "team": None, "visibility": "restricted"},  # 담당자 본인
    "audit": {"owner_emp_id": None, "team": "고객센터", "visibility": "restricted"},     # 감사 보고서
}


@pytest.mark.parametrize("user,visible,restorable", [
    (STAFF, {"public", "team_owned", "restricted_mine"}, {"restricted_mine"}),
    (LEAD, {"public", "team_owned", "team_no_owner"}, {"public", "team_owned", "team_no_owner"}),
    (AUDIT, set(DOCS), set(DOCS)),
])
def test_matrix(user, visible, restorable):
    assert {k for k, d in DOCS.items() if access.can_view(user, d)} == visible
    assert {k for k, d in DOCS.items() if access.can_restore(user, d)} == restorable


def test_rejects_unknown_users():
    for bad in (None, {}, {"role": "ADMIN", "emp_id": "E1"}, {"role": "STAFF", "team": "고객센터"}):
        with pytest.raises(access.AccessError):
            access.sql(bad)
