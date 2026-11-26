"""상담 녹음: 업로드 확인 · 통화 규칙(질문 다음 답) · 이름 오탐 빼기 (DB·탐지 모델·Whisper 필요 없음, Kiwi 만)."""
import io
import wave

import pytest

from app.ingest import service
from app.stt import call_pii


def _wav(sec, ch=2, rate=8000, width=2):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(b"\0" * int(sec * rate) * ch * width)
    return buf.getvalue()


def test_audio_upload_checks():
    for data in (b"", b"not a wav", _wav(0.5), _wav(2, width=1), _wav(31 * 60, ch=1)):
        with pytest.raises(service.IngestError):
            service.create_audio_job("a@b.c", "x.wav", data)


TX = """[00:00] 상담원: 새봄은행 고객센터 이지은입니다.
[00:15] 상담원: 본인 확인을 위해 성함과 생년월을 말씀해 주시겠어요?
[00:21] 고객: 박준호고요.
[00:22] 고객: 86년 3월 12일생입니다.
[00:37] 상담원: 분실하신 카드 끝 내 자리가 어떻게 되시나요?
[00:42] 고객: 7703이요.
[00:48] 상담원: 그리고 계약하실 집주소는요.
[00:52] 고객: 서울시 구로구 디지털로 3201로 보내주세요.
[00:58] 상담원: 이메일 불러주시겠어요.
[01:00] 고객: 네 두심 고 70 골뱅이 다음 닷네요.
[01:05] 상담원: 연 0.5%가 적용되어 약 2만 3천원 입니다."""


def _masked(spans):
    return {(s["label"], TX[s["start"]:s["end"]]) for s in spans}


def test_context_answers():
    got = _masked({"label": l, "start": a, "end": z} for l, a, z in call_pii._context_spans(TX))
    assert ("person", "박준호") in got
    assert ("resident_registration_number", "86년 3월 12일") in got
    assert ("card_number", "7703") in got           # '네 자리'가 '내 자리'로 받아써져도
    assert any(l == "address" and "디지털로" in t for l, t in got)  # 묻는 말 끝이 '는요.' 여도
    assert any(l == "email" and "골뱅이" in t for l, t in got)     # 말로 불러 준 이메일
    assert not any("2만" in t or "천원" in t for _, t in got)        # 묻지 않은 금액은 안 가림


def _drop(text, word):
    i = text.index(word)
    return call_pii._drop_reason(text, {"label": "person", "start": i, "end": i + len(word)})


@pytest.mark.parametrize("text,word", [
    ("정채원 과장에게 전달", "정채원"), ("이도윤이고 010", "이도윤"), ("담당 정하은인데요.", "정하은"),
    ("노 예린 고객의 보험", "노 예린"), ("나은 씨가 왔어요", "나은"), ("이국민 고객, 국", "이국민"),
    ("2) 박별이 입원 중", "박별"), ("아린께 10시", "아린"), ("마크토마스하고 관련", "마크토마스"),
])
def test_real_names_kept(text, word):
    assert _drop(text, word) is None


@pytest.mark.parametrize("text,word", [
    ("세보문행 고객센터", "세보문행"), ("잔액은 12만 4천원입니다.", "천원"), ("외환 상담 노은비", "외환"),
    ("보내신 분 계좌번호 말씀해", "보내신 분 계좌번호"), ("인터넷뱅킹 상담 유나래", "인터넷뱅킹"),
    ("네 그럼 좋으나요", "좋으나"),
])
def test_not_names_dropped(text, word):
    assert _drop(text, word) is not None
