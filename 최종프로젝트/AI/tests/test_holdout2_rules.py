"""Holdout2 첫 실행 뒤 더한 탐지·사람 잇기 규칙 (pii_pipeline ⑥, person_resolve 표·고객 라벨). 탐지 모델 없이."""
from app.pii import person_resolve as r
from app.pii import pii_pipeline as p


def sp(text, word, label="person", nth=0):
    i = -1
    for _ in range(nth + 1):
        i = text.index(word, i + 1)
    return {"label": label, "start": i, "end": i + len(word)}


def labels(text, spans):
    return [(s["label"], text[s["start"]:s["end"]]) for s in p.complete_spans(text, spans)]


def test_form_and_org_words_are_not_names():
    t = "| 기안 | 검토 |\n[양도인] 성명: 김하늘 (양도인의 자녀)\n문서번호: 소보-2026-3148\n청솔은행 302-11\n카드가 안 돼요"
    spans = [sp(t, "기안"), sp(t, "양도인"), sp(t, "김하늘"), sp(t, "양도", nth=1), sp(t, "소보"), sp(t, "청솔"), sp(t, "안 돼요")]
    assert [x for x in labels(t, spans) if x[0] == "person"] == [("person", "김하늘")]


def test_title_keeps_name():
    t = "기안 님께 보고"  # 뒤에 호칭이 붙으면 이름으로 남김
    assert labels(t, [sp(t, "기안")]) == [("person", "기안")]


def test_corp_number_ip_and_address_paren():
    t = "법인등록번호: 118023-3131332 서버: 10.78.215.206 주소: 광주 서구 마루로 85 (치평동))"
    spans = [sp(t, "118023-3131332", "resident_registration_number"), sp(t, "10.78.215.206", "account_number"),
             sp(t, "광주 서구 마루로 85 (치평동))", "address")]
    assert labels(t, spans) == [("address", "광주 서구 마루로 85 (치평동)")]


def test_address_before_card_field_kept():
    t = "주소: 서울 종로구 사직로 11 카드: 1218-4423"
    assert ("address", "서울 종로구 사직로 11") in labels(t, [sp(t, "서울 종로구 사직로 11", "address")])


def test_korean_read_phone():
    t = "사용자: 공일공 팔사칠칠 일이공구\n봇: 확인되었습니다."
    assert labels(t, []) == [("phone_number", "공일공 팔사칠칠 일이공구")]
    assert p.normalize("phone_number", "공일공 팔사칠칠 일이공구") == "01084771209"


def test_roman_names_full_span():
    t = "Dear Saebom Card,\n\nThis is Yumin Koo. My card...\n\nBest regards,\nYumin Koo"
    got = labels(t, [sp(t, "Yumin"), sp(t, "Saebom")])
    assert got == [("person", "Yumin Koo"), ("person", "Yumin Koo")]


def test_one_char_names():
    t = "새봄카드 채권관리팀 혁입니다. 알겠습니다. 윤 고객님, 서류 접수 후 연락드립니다. 총무 율 님께 연락. 전 고객님께 안내"
    got = [x for x in labels(t, []) if x[0] == "person"]
    assert got == [("person", "혁"), ("person", "윤"), ("person", "율")]


def test_table_given_name_spread():
    t = "| 1 | 장예결 | 010-1234-5678 | 김영결 |\n| 2 | 도채슬 | 010-2234-5678 | 영결 |"
    spans = [sp(t, "장예결"), sp(t, "김영결"), sp(t, "도채슬")]
    assert ("person", "영결") in [(lab, w) for lab, w in labels(t, spans)]


def test_table_customer_column_role_and_anchor():
    t = "| 순번 | 고객명 | 연락처 |\n|---|---|---|\n| 1 | 조수현 | 010-8342-0857 |"
    name, phone = sp(t, "조수현"), sp(t, "010-8342-0857", "phone_number")
    f = r.mention_features(t, [name, phone], 0)
    assert f["role"] == "customer" and f["anchors"] == [("phone_number", "01083420857")]


def test_staff_table_not_customer():
    t = "| 성명 | 소속 |\n| --- | --- |\n| 강연우 | 디지털전략팀 |"
    assert r._table_role(t, t.index("강연우")) is None


def test_claimant_is_customer():
    t = "청구인: 도재호\n"
    assert r.mention_features(t, [sp(t, "도재호")], 0)["role"] == "customer"


def test_recipient_gwiha_is_customer():
    t = "수신: 은진연 귀하\n"
    assert r.mention_features(t, [sp(t, "은진연")], 0)["role"] == "customer"


def test_roman_signature_not_employee():
    from app.org import infer
    t = "Dear Saebom Card,\n\nThis is Yumin Koo. My card was declined.\n\nBest regards,\nYumin Koo"
    spans = [sp(t, "Yumin Koo"), sp(t, "Yumin Koo", nth=1)]
    assert infer.staff_mentions({"id": "X", "text": t}, spans) == []
