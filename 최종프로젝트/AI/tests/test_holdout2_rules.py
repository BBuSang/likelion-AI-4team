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


def test_sender_header_in_customer_mail_is_customer():
    t = "안녕하세요, 표태혁 고객님. 문의하신 건 안내드립니다.\n----- 원본 메시지 -----\n보낸 사람: 표태혁\n수수료가 궁금합니다."
    spans = [sp(t, "표태혁"), sp(t, "표태혁", nth=1)]
    out, _ = r.resolve([{"id": "D1", "text": t}], {"D1": spans})
    assert [s["role"] for s in out["D1"]] == ["customer", "customer"]


def test_log_with_many_customers_does_not_borrow_other_ids():
    log = ("09:00 표태혁 고객 결제 계좌 변경 요청\n09:10 김가람 고객 카드 재발급 (기존 카드 4518509179343600 정지)\n"
           "09:20 이하늘 고객 한도 문의\n")
    mail = "고객명: 표태혁\n카드번호: 9440-3474-8532-8722\n"
    docs = [{"id": "L", "text": log}, {"id": "M", "text": mail}]
    spans = {"L": [sp(log, "표태혁"), sp(log, "김가람"), sp(log, "4518509179343600", "card_number"), sp(log, "이하늘")],
             "M": [sp(mail, "표태혁"), sp(mail, "9440-3474-8532-8722", "card_number")]}
    out, _ = r.resolve(docs, spans)
    key = lambda d: next(s["person_key"] for s in out[d] if s.get("normalized") == "표태혁")
    assert key("L") == key("M")


def test_ask_back_only_when_answer_docs_mix_people():
    from app.rag import chat
    amb = [{"query_token": "<PERSON_q>", "candidates": [("<PERSON_a>", "customer"), ("<PERSON_b>", "staff")]}]
    d1 = {"doc_id": "H1", "doc_type": "콜백요청목록", "tokens": ["<PERSON_a>"], "ambiguous_people": amb}
    d2 = {"doc_id": "H2", "doc_type": "해외결제차단해제", "tokens": ["<PERSON_a>"]}
    d3 = {"doc_id": "H3", "doc_type": "지점업무일지", "tokens": ["<PERSON_b>"]}
    topic = "<PERSON_q> 님 콜백 요청 기록 혹시 있나요?"
    got, _ = chat.ambiguity([d1, d2, d3], [d1, d2], topic)    # 주제가 맞고 답에 쓸 문서는 한 사람뿐 → 되묻지 않음
    assert got == []
    got, docs = chat.ambiguity([d1, d2, d3], [d1, d3], topic)  # 두 사람이 섞임 → 되묻고, 목록 문서를 근거로
    assert got == amb and [d["doc_id"] for d in docs] == ["H1", "H2", "H3"]
    got, _ = chat.ambiguity([d1, d2, d3], [d1, d2], "<PERSON_q> 고객 문서만")  # 주제 없이 사람만 → 후보 전체로 판단
    assert got == amb
