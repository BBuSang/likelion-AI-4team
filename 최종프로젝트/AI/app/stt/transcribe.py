"""상담 녹음 → 받아쓴 글 (CPU, 로컬 Whisper). 개인정보 판단은 여기서 하지 않음 — 받아쓰기만.

- 모델: faster-whisper (int8). 기본 medium (small 과 비교해 개인정보·글자 오류가 적고 CPU 로도 실시간의 절반 정도)
- 스테레오면 채널을 나눠 받아씀 (왼쪽 상담원 · 오른쪽 고객). 콜센터 녹음이 보통 이렇게 저장됨 → 화자가 저절로 붙음
- 모노면 화자 없이 받아씀
- 문장 단위로 자름 (단어 시각 기준, 마침표·물음표 또는 0.6초 넘는 쉼) → 채널을 합쳐도 순서가 맞고, 뒤의 대화 맥락 규칙이 '질문 다음 답'을 볼 수 있음
- 힌트 단어(config.STT_HOTWORDS): 회사 이름·업무 말을 미리 알려 줌
- 지금은 wav 만 받음 (faster-whisper 1.2.1 + av 19 조합에서 파일 디코딩이 깨져 직접 읽음)
"""
import io
import re
import threading
import wave

import numpy as np

from app.core import config

SPEAKERS = ("상담원", "고객")
PAUSE = 0.6
_model, _lock = None, threading.Lock()


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(config.STT_MODEL, device="cpu", compute_type="int8", cpu_threads=config.STT_THREADS)
    return _model


def read_wav(data):
    """wav 바이트/경로 → (채널별 float32 배열 목록, 표본율)"""
    src = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else str(data)
    with wave.open(src) as w:
        ch, rate, width = w.getnchannels(), w.getframerate(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("16비트 wav 만 받습니다")
    x = np.frombuffer(raw, np.int16).reshape(-1, ch).astype(np.float32) / 32768
    return [x[:, i] for i in range(ch)], rate


def _to16k(x, rate):
    if rate == 16000:
        return x.astype(np.float32)
    n = int(len(x) * 16000 / rate)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def _sentences(model, audio, speaker, hotwords):
    segs, _ = model.transcribe(audio, language="ko", vad_filter=True, beam_size=5, word_timestamps=True,
                               condition_on_previous_text=False, hotwords=hotwords or None)
    out, cur = [], []
    for s in segs:
        for w in s.words or []:
            # 말 사이가 0.6초 넘게 비면 문장 끝으로 봄 — 마침표가 안 찍혀도 질문과 답이 한 줄로 붙지 않게
            if cur and w.start - cur[-1].end >= PAUSE:
                out.append(cur)
                cur = []
            cur.append(w)
            if re.search(r"[.?!]$", w.word.strip()):
                out.append(cur)
                cur = []
        if cur and s.words:  # 구간 끝도 문장 끝으로 봄
            out.append(cur)
            cur = []
    return [{"start": round(ws[0].start, 2), "end": round(ws[-1].end, 2), "speaker": speaker,
             "text": "".join(w.word for w in ws).strip()} for ws in out if "".join(w.word for w in ws).strip()]


def transcribe(data, split_channels=True, hotwords=None):
    """→ [{start, end, speaker(상담원/고객/None), text}] 시간순
    hotwords: 받아쓰기 힌트 단어 (쉼표 구분 문자열). None 이면 config.STT_HOTWORDS, '' 이면 힌트 없음"""
    hw = config.STT_HOTWORDS if hotwords is None else hotwords
    # 낱말만 나열하면 받아쓰기도 마침표 없이 나와서, 쉼표·마침표 찍은 한 문장으로 줌
    words = [w.strip() for w in hw.split(",") if w.strip()]
    hw = ", ".join(words) + "." if words else ""
    chans, rate = read_wav(data)
    with _lock:  # 모델 하나를 여러 요청이 동시에 쓰지 않게 (CPU 를 다른 서비스와 나눠 씀)
        model = _get_model()
        if len(chans) == 2 and split_channels:
            turns = []
            for x, spk in zip(chans, SPEAKERS):
                turns += _sentences(model, _to16k(x, rate), spk, hw)
        else:
            turns = _sentences(model, _to16k(np.mean(chans, 0), rate), None, hw)
    return sorted(turns, key=lambda t: t["start"])


def to_text(turns):
    """적재용 문서 본문: 한 줄에 한 문장, '[분:초] 화자: 말'"""
    lines = []
    for t in turns:
        m, s = divmod(int(t["start"]), 60)
        who = f"{t['speaker']}: " if t["speaker"] else ""
        lines.append(f"[{m:02d}:{s:02d}] {who}{t['text']}")
    return "\n".join(lines)
