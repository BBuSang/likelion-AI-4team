"""탐지 보완 (worklog 20장): 라벨 없는 긴 숫자 → 번호, 법 줄임말·상태 말 이름 오탐 빼기. 탐지 모델 없이."""
from app.pii import pii_pipeline as p


def labels(text, spans):
    return [(s["label"], text[s["start"]:s["end"]]) for s in p.complete_spans(text, spans)]


def test_bare_numbers_become_tokens_by_shape():
    t = "(464039625067 신규 정보) 주민 9001011234567 연락처 01012345678 카드 9411897341382214"
    assert labels(t, []) == [("account_number", "464039625067"), ("resident_registration_number", "9001011234567"),
                             ("phone_number", "01012345678"), ("card_number", "9411897341382214")]


def test_dates_amounts_short_numbers_and_detected_values_untouched():
    t = "2026-09-16 18:14 기준 1,900,000원, 접수번호 REQ-92736, 연락처 010-1234-5678"
    phone = {"label": "phone_number", "start": t.index("010"), "end": t.index("010") + 13}
    assert labels(t, [phone]) == [("phone_number", "010-1234-5678")]


def test_law_abbreviations_and_status_words_are_not_names():
    assert labels("신정법 바뀐 거", [{"label": "person", "start": 0, "end": 3}]) == []
    assert labels("금소법 6대 원칙", [{"label": "person", "start": 0, "end": 3}]) == []
    assert labels("상태: 보류", [{"label": "person", "start": 4, "end": 6}]) == []


def test_titled_names_kept_and_idempotent():
    t = "이수 님께 안내, 김민준 고객"
    spans = [{"label": "person", "start": 0, "end": 2}, {"label": "person", "start": 10, "end": 13}]
    once = p.complete_spans(t, spans)
    assert [t[s["start"]:s["end"]] for s in once] == ["이수", "김민준"]
    assert p.complete_spans(t, once) == once


def test_given_name_with_title():
    assert labels("보람대리님 확인", []) == [("person", "보람")]
    assert labels("회원다운고객님께 전달", []) == [("person", "다운")]
    for t in ["우수고객님 대상", "신규 고객 안내", "담당 과장에게", "해외 거주 고객 중", "결과에 따라 고객님의", "접수 내역\n고객: 홍"]:
        assert [x for x in labels(t, []) if x[0] == "person"] == [], t


def test_given_name_spread_in_doc():
    t = "담당: 김하늘\n하늘은 오후에 연락, 홍보팀하늘이 진행"
    spans = [{"label": "person", "start": 4, "end": 7}]
    assert [w for l, w in labels(t, spans) if l == "person"] == ["김하늘", "하늘", "하늘"]
    # 성 포함 이름이 없거나 사람 자리가 아니면 일반 단어로 둠
    assert [x for x in labels("하늘은 맑음", []) if x[0] == "person"] == []
    assert [w for l, w in labels("김하늘 확인, 하늘색 표지", [{"label": "person", "start": 0, "end": 3}]) if l == "person"] == ["김하늘"]


def test_one_char_names():
    person = lambda t: [w for l, w in labels(t, []) if l == "person"]
    assert person("린님 안녕하세요") == ["린"]
    assert person("담당 최 씨가 확인") == ["최"]
    assert person("담당 장께서 예약") == ["장"]
    assert person("필요시 진이한테 연락") == ["진"]
    for t in ["손님께 안내", "형님께서", "함께 처리", "그에게 전달", "아이한테 설명", "고객님께 안내", "박 대리와 연결", "주차장 이용"]:
        assert person(t) == [], t
