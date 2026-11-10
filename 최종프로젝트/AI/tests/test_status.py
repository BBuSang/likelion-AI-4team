"""처리 상태 추출 (worklog 19장)."""
from app.rag import status as st


def test_doc_status_from_status_slots():
    assert st.doc_status("요청금액: 300,000원\n상태: 접수\n담당: 대리") == "접수"
    assert st.doc_status("현재 상태는 '보류'입니다.") == "보류"
    assert st.doc_status("Subject: [처리 완료] 카드론 상담") == "처리 완료"
    assert st.doc_status("5. 접수 현황: 처리 중") == "처리 중"
    assert st.doc_status("환불 건이 처리 완료되었습니다.") == "처리 완료"
    assert st.doc_status("담당자가 확인 후 보류로 처리 중입니다.") == "보류"


def test_doc_status_ignores_non_status_words():
    assert st.doc_status("접수일: 2026-06-04\n접수자: <PERSON_ab12cd34>\n후속: 연락처 변경 완료") is None
    assert st.doc_status("2026-04-24까지 처리 완료 예정") is None


def test_table_rows_have_their_own_status():
    t = "VOC-1 | <PERSON_aaaa1111> 고객 | 앱 오류 | 처리 완료\nVOC-2 | <PERSON_bbbb2222> 고객 | 포인트 | 처리 중"
    assert st.doc_status(t) is None  # 줄마다 다르면 문서 상태는 모름
    rows = dict((s, line) for line, s in st.line_statuses(t))
    assert "<PERSON_bbbb2222>" in rows["처리 중"] and "<PERSON_aaaa1111>" in rows["처리 완료"]


def test_query_status():
    assert st.query_status("송진우 고객 불만 아직 처리 안 끝난 거 있어?") == st.UNFINISHED
    assert st.query_status("윤선영 VOC 해결 완료 건") == {"처리 완료"}
    assert st.query_status("김민준 고객 문의 중에 마무리된 거 보여줘") == {"처리 완료"}
    assert st.query_status("한도심사결과 접수") == {"접수"}
    assert st.query_status("가족카드발급 처리 중인 건 목록") == {"처리 중"}
    assert st.query_status("보류된 결제취소 건") == {"보류"}
    assert st.query_status("보이스피싱 피해 접수 기록 찾아줘") is None  # '피해 접수'는 문서 종류
    assert st.query_status("윤선영 고객 대출 상담 내역") is None
