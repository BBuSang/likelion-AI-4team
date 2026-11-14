"""직원 목록·문서 메타데이터 추론 (app/org/infer.py). 탐지기 없이 이름 구간을 직접 줌."""
from app.org import infer


def doc(doc_id, text, names, doc_type=None):
    spans = []
    for n in names:
        i = text.index(n)
        spans.append({"label": "person", "start": i, "end": i + len(n)})
    return {"id": doc_id, "text": text, "doc_type": doc_type}, spans


def run(*docs):
    ds = [d for d, _ in docs]
    sp = {d["id"]: s for d, s in docs}
    employees, meta, *_ = infer.build(ds, sp)
    return {e["emp_id"]: e for e in employees}, {m["id"]: m for m in meta}


def test_team_from_other_doc_fills_customer_doc():
    # 1번 문서에 팀이 나오면, 팀이 없는 고객 응대 문서의 같은 직원 → 그 팀
    emps, meta = run(
        doc("A", "- 담당자: 금은우 과장(고객센터)\n- 상태: 접수", ["금은우"]),
        doc("B", "고객명: 우채원 고객님\n상담 담당자: 금은우 과장\n상담 완료", ["우채원", "금은우"]),
    )
    [e] = emps.values()  # 고객(우채원)은 직원이 아님
    assert (e["name"], e["team"], e["position"], e["role"]) == ("금은우", "고객센터", "과장", "STAFF")
    assert meta["B"]["owner_emp_id"] == e["emp_id"] and meta["B"]["team"] == "고객센터"
    assert meta["B"]["visibility"] == "team"


def test_formats_table_slash_inline_signature():
    emps, meta = run(
        doc("T", "| 성명 | 소속 |\n| 김재원 | 고객센터 |\n| 정현우 | 미르보안 |", ["김재원", "정현우"]),
        doc("S", "소명자: 리스크관리팀 / 부장 / 김성호\n금액: 5만원\n확인자: 팀장 이지영", ["김성호", "이지영"]),
        doc("I", "정태오 고객 문의. 심사역 안윤이 조건 안내 완료.", ["정태오", "안윤"]),
        doc("M", "Dear 전예은 책임님,\nThanks.\nBest regards,\n김미영", ["전예은", "김미영"]),
    )
    by_name = {e["name"]: e for e in emps.values()}
    assert by_name["김재원"]["team"] == "고객센터" and "정현우" not in by_name  # 협력사 사람은 빠짐
    assert (by_name["김성호"]["team"], by_name["김성호"]["position"], by_name["김성호"]["role"]) == ("리스크관리팀", "부장", "LEAD")
    assert meta["S"]["owner_emp_id"] == by_name["김성호"]["emp_id"]  # 확인자(승인)보다 소명자(작성)
    assert meta["I"]["owner_emp_id"] == by_name["안윤"]["emp_id"]
    assert meta["M"]["owner_emp_id"] == by_name["김미영"]["emp_id"]  # 서명


def test_homonyms_in_different_teams_are_split():
    emps, _ = run(
        doc("A", "담당: 여신심사팀 박도윤 대리", ["박도윤"]),
        doc("B", "IT운영팀 박도윤 사원에게 원인 분석 의뢰", ["박도윤"]),
        doc("C", "처리자: 박도윤", ["박도윤"]),  # 팀 단서 없음 → 어느 쪽인지 모름
    )
    teams = sorted((e["team"] or "-") for e in emps.values())
    assert teams == ["-", "IT운영팀", "여신심사팀"]


def test_visibility_public_team_restricted_and_team_cleanup():
    _, meta = run(
        doc("P", "[부고] 고객센터 김민준 대리 장인상\n삼가 알려드립니다.", ["김민준"], "경조사공지"),
        doc("R", "고객의 소리 - 칭찬 접수\n상담해 주신 분 덕분에 해결했어요.", []),
        doc("F", "부서/직급: 인사팀팀/과장\n신청자 성명: 문시우", ["문시우"]),
        doc("A", "내부 감사 지적 사항\n\n감사 결과 리스크관리팀 소속 김성호가 고객 정보를 메신저로 전송함.", ["김성호"]),
    )
    assert meta["A"]["visibility"] == "restricted"  # 감사 보고서는 지적받은 팀이 아니라 감사만
    assert meta["P"]["visibility"] == "public"
    assert meta["R"]["visibility"] == "restricted" and meta["R"]["team"] is None  # 끝까지 모르면 감사만
    assert meta["F"]["team"] == "인사팀"  # 오타 정리 + 소속 항목 → 신청자 팀
