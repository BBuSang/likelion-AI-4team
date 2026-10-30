"""가명처리 모듈(starter/pii_pipeline.py) 테스트.

    python -X utf8 -m pytest tests -v                          (실전프로젝트2 폴더에서)
    python -X utf8 -m unittest tests.test_pii_pipeline -v      (pytest 없이)

- 실제 HMAC_KEY(.env)를 쓰지 않고 시험용 키로 바꿔서 확인합니다. 토큰 값 자체가 아니라 성질(같음·다름·형식)을 봅니다.
- 탐지기 연결 테스트(TestDetector)는 모델 로딩에 몇 초 걸립니다. 건너뛰려면 SKIP_DETECTOR=1
"""
import os
import unittest

from starter import config, pii_pipeline as p

TEST_KEY = "unit-test-key-not-the-team-key"


class Base(unittest.TestCase):
    def setUp(self):
        self._saved = (config.HMAC_KEY, p.NORMALIZE, p.TOKEN_MODE)
        config.HMAC_KEY, p.NORMALIZE, p.TOKEN_MODE = TEST_KEY, True, "hmac"

    def tearDown(self):
        config.HMAC_KEY, p.NORMALIZE, p.TOKEN_MODE = self._saved

    def tok(self, label, value):
        return p.make_token(label, p.normalize(label, value))


class TestNormalize(Base):
    def test_person_spaces(self):
        self.assertEqual(p.normalize("person", "윤 선영"), "윤선영")
        self.assertEqual(p.normalize("person", "남궁 나연"), "남궁나연")

    def test_phone_formats(self):
        for v in ["010-0349-2063", "01003492063", "010.0349.2063", "010 0349 2063", "+82 10-0349-2063"]:
            self.assertEqual(p.normalize("phone_number", v), "01003492063", v)

    def test_number_labels_digits_only(self):
        self.assertEqual(p.normalize("resident_registration_number", "770802 1978865"), "7708021978865")
        self.assertEqual(p.normalize("account_number", "607-960055-16"), "60796005516")
        self.assertEqual(p.normalize("card_number", "3565-3016-6131-8603"), "3565301661318603")

    def test_email_lower(self):
        self.assertEqual(p.normalize("email", " SY.Yoon07@Example.net "), "sy.yoon07@example.net")

    def test_address_sido_alias(self):
        full = "대구광역시 수성구 한빛로 174, 502호 (범어동)"
        self.assertEqual(p.normalize("address", "대구 수성구 한빛로 174, 502호 (범어동)"), full)
        self.assertEqual(p.normalize("address", "대구시  수성구 한빛로 174, 502호 (범어동)"), full)
        self.assertEqual(p.normalize("address", "충남 천안시 서북구 마루길 164"), "충청남도 천안시 서북구 마루길 164")

    def test_address_postcode_removed(self):
        self.assertEqual(p.normalize("address", "(05299) 서울특별시 강남구 한빛로 9, 2층"), "서울특별시 강남구 한빛로 9, 2층")

    def test_address_alias_only_first_word(self):
        # 주소 중간의 '광주'는 시·도가 아니므로 바꾸지 않음 (경기도 광주시)
        self.assertEqual(p.normalize("address", "경기 광주시 오포읍 1"), "경기도 광주시 오포읍 1")

    def test_normalize_off_for_b1(self):
        p.NORMALIZE = False
        self.assertEqual(p.normalize("person", "윤 선영"), "윤 선영")


class TestToken(Base):
    def test_format(self):
        t = self.tok("person", "윤선영")
        self.assertRegex(t, r"^<PERSON_[0-9a-f]{8}>$")
        self.assertTrue(p.TOKEN_PATTERN.fullmatch(t))

    def test_deterministic(self):
        self.assertEqual(self.tok("person", "윤선영"), self.tok("person", "윤선영"))

    def test_variants_same_token(self):
        self.assertEqual(self.tok("person", "윤 선영"), self.tok("person", "윤선영"))
        self.assertEqual(self.tok("phone_number", "+82 10-0349-2063"), self.tok("phone_number", "01003492063"))

    def test_label_in_input(self):
        # 같은 숫자라도 전화번호와 계좌번호는 다른 토큰
        self.assertNotEqual(p.make_token("phone_number", "01003492063"), p.make_token("account_number", "01003492063"))

    def test_key_changes_token(self):
        a = self.tok("person", "윤선영")
        config.HMAC_KEY = TEST_KEY + "-other"
        self.assertNotEqual(a, self.tok("person", "윤선영"))

    def test_empty_key_raises(self):
        config.HMAC_KEY = ""
        with self.assertRaises(RuntimeError):
            p.make_token("person", "윤선영")

    def test_without_normalize_variants_split(self):
        p.NORMALIZE = False
        self.assertNotEqual(self.tok("person", "윤 선영"), self.tok("person", "윤선영"))

    def test_label_mode_for_b0(self):
        p.TOKEN_MODE = "label"
        self.assertEqual(p.make_token("person", "윤선영"), "[PERSON]")
        self.assertIsNone(p.TOKEN_PATTERN.fullmatch("[PERSON]"))  # 토큰 필터에 안 걸림


class TestPseudonymize(Base):
    TEXT = "윤 선영 고객(010-0349-2063)이 문의함. 윤선영은 확인함."

    @staticmethod
    def _span(label, value, text=TEXT):
        s = text.index(value)
        return {"label": label, "start": s, "end": s + len(value)}

    SPANS = [_span("person", "윤 선영"), _span("phone_number", "010-0349-2063"), _span("person", "윤선영")]

    def test_records_use_original_positions(self):
        _, records = p.pseudonymize(self.TEXT, self.SPANS)
        for r in records:
            self.assertEqual(self.TEXT[r["start"]:r["end"]], r["original"])

    def test_masked_text(self):
        masked, records = p.pseudonymize(self.TEXT, self.SPANS)
        person = records[0]["token"]
        self.assertEqual(masked, f"{person} 고객({records[1]['token']})이 문의함. {person}은 확인함.")
        self.assertNotIn("선영", masked)
        self.assertNotIn("0349", masked)

    def test_same_person_same_token_in_doc(self):
        _, records = p.pseudonymize(self.TEXT, self.SPANS)
        self.assertEqual(records[0]["token"], records[2]["token"])

    def test_query_matches_document(self):
        _, doc = p.pseudonymize(self.TEXT, self.SPANS)
        q = "윤선영 고객 상담 내역"
        masked_q, q_rec = p.pseudonymize(q, [{"label": "person", "start": 0, "end": 3}])
        self.assertEqual(q_rec[0]["token"], doc[0]["token"])
        self.assertTrue(masked_q.startswith("<PERSON_"))

    def test_overlaps_keep_longer(self):
        spans = [{"label": "person", "start": 0, "end": 4}, {"label": "person", "start": 2, "end": 4}]
        _, records = p.pseudonymize(self.TEXT, spans)
        self.assertEqual([(r["start"], r["end"]) for r in records], [(0, 4)])

    def test_no_spans(self):
        self.assertEqual(p.pseudonymize("연체 안내 문서", []), ("연체 안내 문서", []))


@unittest.skipIf(os.getenv("SKIP_DETECTOR") == "1", "SKIP_DETECTOR=1")
class TestDetector(Base):
    def test_detect_and_pseudonymize(self):
        text = "윤 선영 고객(01003492063)이 추가 한도를 문의함. 윤선영은 sy.yoon07@example.net로 서류를 받음."
        spans = p.detect(text)
        self.assertEqual({s["label"] for s in spans}, {"person", "phone_number", "email"})
        masked, records = p.pseudonymize(text, spans)
        for raw in ("선영", "01003492063", "sy.yoon07"):
            self.assertNotIn(raw, masked)
        persons = {r["token"] for r in records if r["label"] == "person"}
        self.assertEqual(len(persons), 1)  # '윤 선영'과 '윤선영'이 같은 토큰


if __name__ == "__main__":
    unittest.main()
