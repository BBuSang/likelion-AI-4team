"""v2 가명처리 모듈 테스트. 팀 HMAC 키가 아닌 시험용 키로 토큰 성질만 확인.

    python -X utf8 -m pytest v2/tests -v          (실전프로젝트2 폴더)
    SKIP_DETECTOR=1 이면 탐지기 연결 테스트를 건너뜀 (모델 로딩 수십 초)
"""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pseudo import pipeline, roles  # noqa: E402

KEY = "test-key-not-for-production"


def spans_of(text, *items):
    """(label, 문자열) → 원문에서 위치를 찾아 span 으로."""
    out, start = [], 0
    for label, s in items:
        i = text.index(s, start)
        out.append({"label": label, "start": i, "end": i + len(s)})
        start = i + len(s)
    return out


# ───────── 정규화 ─────────
@pytest.mark.parametrize("label,a,b", [
    ("person", "윤 선영", "윤선영"),
    ("phone_number", "010-0349-2063", "01003492063"),
    ("phone_number", "+82 10-0349-2063", "01003492063"),
    ("phone_number", "010 0349 2063", "01003492063"),
    ("account_number", "110-234-567890", "110234567890"),
    ("email", "SY.Yoon@Example.net", "sy.yoon@example.net"),
    ("address", "대구 수성구 달구벌대로 1", "대구광역시 수성구 달구벌대로 1"),
    ("address", "(42110) 대구광역시  수성구 달구벌대로 1", "대구광역시 수성구 달구벌대로 1"),
])
def test_normalize_same(label, a, b):
    assert pipeline.normalize(label, a) == pipeline.normalize(label, b)


def test_normalize_keeps_different_names():
    assert pipeline.normalize("person", "윤선영") != pipeline.normalize("person", "윤서영")


# ───────── 토큰 ─────────
def test_token_same_for_variants():
    t = lambda label, v: pipeline.make_token(label, pipeline.normalize(label, v), KEY)  # noqa: E731
    assert t("person", "윤 선영") == t("person", "윤선영")
    assert t("phone_number", "+82 10-0349-2063") == t("phone_number", "010-0349-2063")


def test_token_label_separates_same_digits():
    assert pipeline.make_token("phone_number", "01003492063", KEY) != pipeline.make_token("account_number", "01003492063", KEY)


def test_token_depends_on_key():
    assert pipeline.make_token("person", "윤선영", KEY) != pipeline.make_token("person", "윤선영", KEY + "x")


def test_token_format():
    assert pipeline.TOKEN_RE.fullmatch(pipeline.make_token("resident_registration_number", "9001011234567", KEY))


def test_empty_key_raises():
    with pytest.raises(RuntimeError):
        pipeline.make_token("person", "윤선영", "")


# ───────── 치환·위치 ─────────
TEXT = "상담원: 김은영\n윤 선영 고객(010-0349-2063)이 문의함. 주소는 대구 수성구 달구벌대로 1."
SPANS = spans_of(TEXT, ("person", "김은영"), ("person", "윤 선영"), ("phone_number", "010-0349-2063"),
                 ("address", "대구 수성구 달구벌대로 1"))


def test_masked_offsets_point_to_token():
    r = pipeline.process(TEXT, SPANS, KEY)
    for m in r["mentions"]:
        assert r["masked"][m["m_start"]:m["m_end"]] == m["token"]
        assert TEXT[m["o_start"]:m["o_end"]] == m["raw"]


def test_no_raw_left_in_masked():
    r = pipeline.process(TEXT, SPANS, KEY)
    for s in ("김은영", "윤 선영", "010-0349-2063", "달구벌대로"):
        assert s not in r["masked"]


def test_line_no():
    r = pipeline.process(TEXT, SPANS, KEY)
    assert [m["line_no"] for m in r["mentions"]] == [0, 1, 1, 1]


def test_overlap_keeps_longer():
    text = "대구광역시 수성구 달구벌대로 1"
    spans = [{"label": "address", "start": 0, "end": 9}, {"label": "address", "start": 0, "end": len(text)}]
    r = pipeline.process(text, spans, KEY)
    assert len(r["mentions"]) == 1 and r["mentions"][0]["o_end"] == len(text)


# ───────── 언급 정보 ─────────
def test_roles_and_form():
    r = pipeline.process(TEXT, SPANS, KEY)
    kim, yoon = r["mentions"][0], r["mentions"][1]
    assert kim["role"] == "staff" and yoon["role"] == "customer"
    assert kim["form"] == "full" and yoon["form"] == "spaced"


@pytest.mark.parametrize("name,form", [("윤선영", "full"), ("시우", "given"), ("민준", "given"),
                                       ("전린", "full"), ("남궁나연", "full"), ("린", "given")])
def test_name_form(name, form):
    assert pipeline.name_form(name, name) == form


@pytest.mark.parametrize("name,after,form", [("서연", "아, 오늘", "given"), ("성호", " 씨가", "given"),
                                             ("민준", "이랑 같이", "given"), ("서연", " 고객", "full"),
                                             ("이서연", "아", "full")])
def test_name_form_vocative(name, after, form):
    assert pipeline.name_form(name, name, after) == form


def test_region_generalizes_to_sigungu():
    assert pipeline.region("충청남도 천안시 서북구 불당대로 1") == ("충청남도", "천안시 서북구")
    assert pipeline.region("대구광역시 수성구 달구벌대로 1") == ("대구광역시", "수성구")
    assert pipeline.region("어딘가 1번지") == (None, None)


def test_role_shared_within_doc():
    text = "담당 정예린이 확인함.\n정예린은 다음 날 회신함."
    spans = [{"start": 3, "end": 6, "key": "A"}, {"start": 13, "end": 16, "key": "A"}]
    assert [r["role"] for r in roles.extract(text, spans)] == ["staff", "staff"]


def test_external_vendor():
    text = "3. 수탁사 담당자: 문은정 매니저"
    i = text.index("문은정")
    assert roles.extract(text, [{"start": i, "end": i + 3, "key": "A"}])[0]["role"] == "external"


# ───────── 탐지기 연결 ─────────
@pytest.mark.skipif(os.getenv("SKIP_DETECTOR") == "1", reason="SKIP_DETECTOR=1")
def test_detector_end_to_end():
    r = pipeline.process("윤 선영 고객(01003492063)이 추가 한도를 문의함.", key=KEY)
    labels = {m["label"] for m in r["mentions"]}
    assert {"person", "phone_number"} <= labels
    assert "윤 선영" not in r["masked"] and "01003492063" not in r["masked"]
