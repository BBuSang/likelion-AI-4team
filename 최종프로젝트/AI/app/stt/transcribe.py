"""상담 녹음 → 받아쓴 글 (CPU, 로컬 Whisper). 개인정보 판단은 여기서 하지 않음 — 받아쓰기만.

- 모델: faster-whisper (int8). 기본 medium (small 과 비교해 개인정보·글자 오류가 적고 CPU 로도 실시간의 절반 정도)
- 상담(kind='call'): 스테레오면 채널을 나눠 받아씀 (왼쪽 상담원 · 오른쪽 고객). 콜센터 녹음이 보통 이렇게 저장됨 → 화자가 저절로 붙음
  모노면 목소리로 두 사람을 나누고(diarize.py) 상담원 말투가 많은 쪽을 상담원으로
- 회의(kind='meeting'): 채널을 합쳐 받아쓰고 목소리로 사람을 나눔 → '화자1·화자2…' (이름은 검수 때 사람이 붙임)
- 문장 단위로 자름 (단어 시각 기준, 마침표·물음표 또는 0.6초 넘는 쉼) → 채널을 합쳐도 순서가 맞고, 뒤의 대화 맥락 규칙이 '질문 다음 답'을 볼 수 있음
- 힌트 단어(config.STT_HOTWORDS): 회사 이름·업무 말을 미리 알려 줌
- 파일 형식: wav·mp3·m4a·aac·flac·ogg·opus·webm 등 (av 로 직접 풀어 16kHz 로). faster-whisper 1.2.1 자체 디코더는
  av 19 와 안 맞아(open() 인자 metadata_errors 없어짐) 쓰지 않음
"""
import io
import re
import threading

import numpy as np

from app.core import config
from app.core.loading import MODEL_LOAD_LOCK

SPEAKERS = ("상담원", "고객")
RATE = 16000
PAUSE = 0.6
# 상담원이 주로 하는 말 — 모노 상담 녹음에서 두 사람 중 누가 상담원인지 정할 때
AGENT_CUES = ("새봄은행", "고객님", "도와드릴", "확인해 드리", "확인 도와", "말씀해 주", "알려 주시", "감사합니다", "상담원", "잠시만 기다려")
_model, _lock = None, threading.Lock()


def _get_model():
    """처음 부를 때 모델을 읽음 (없으면 허깅페이스에서 받아 HF_HOME 에 둠, medium 약 1.5GB).
    서버 시작 때 미리 읽는 중에 받아쓰기 요청이 와도 두 번 읽지 않게 모델 읽기 잠금 안에서"""
    global _model
    if _model is None:
        with MODEL_LOAD_LOCK:
            if _model is None:
                from faster_whisper import WhisperModel
                _model = WhisperModel(config.STT_MODEL, device="cpu", compute_type="int8", cpu_threads=config.STT_THREADS)
    return _model


def warmup():
    """서버 시작 때 백그라운드에서 부름 — 첫 받아쓰기 요청이 모델 받기를 기다리다 시간 초과 나지 않게"""
    _get_model()


def read_audio(data, max_sec=None):
    """녹음 바이트/경로 → (채널별 float32 배열 목록, 16000). 스테레오는 두 채널 그대로, 셋 이상이면 스테레오로 섞음.
    max_sec: 넘으면 다 풀기 전에 ValueError (작은 파일이 아주 길게 풀리는 경우 막기)"""
    import av
    src = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else str(data)
    with av.open(src) as c:
        if not c.streams.audio:
            raise ValueError("소리가 없는 파일")
        s = c.streams.audio[0]
        rs = av.AudioResampler(format="fltp", layout="stereo" if (s.channels or 1) >= 2 else "mono", rate=RATE)
        parts, n = [], 0
        for f in c.decode(s):
            for g in rs.resample(f):
                parts.append(g.to_ndarray())
                n += parts[-1].shape[1]
            if max_sec and n > max_sec * RATE:
                raise ValueError("too long")
        for g in rs.resample(None):
            parts.append(g.to_ndarray())
    if not parts:
        raise ValueError("소리가 없는 파일")
    x = np.concatenate(parts, 1).astype(np.float32)
    return [x[i] for i in range(x.shape[0])], RATE


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


def _name_call(turns, labels):
    """모노 상담: 두 무리 중 상담원 말투(AGENT_CUES)가 많은 쪽 = 상담원. 같으면 먼저 말한 쪽 (상담원이 먼저 인사)"""
    score = {}
    for t, lb in zip(turns, labels):
        score[lb] = score.get(lb, 0) + sum(c in t["text"].replace(" ", "") for c in (c.replace(" ", "") for c in AGENT_CUES))
    first = labels[0]
    agent = max(score, key=lambda k: (score[k], k == first))
    return [SPEAKERS[0] if lb == agent else SPEAKERS[1] for lb in labels]


def transcribe(data, kind="call", speakers=None, split_channels=True, hotwords=None, diarize=None):
    """→ [{start, end, speaker, text}] 시간순
    kind: 'call' (상담원/고객) · 'meeting' (화자1·화자2…)
    speakers: 회의 참석자 수를 알면 (모르면 None → 목소리 차이로 추정). 상담 모노는 늘 2
    diarize: 목소리로 사람 나누기 (None 이면 config.STT_DIARIZE)
    hotwords: 받아쓰기 힌트 단어 (쉼표 구분 문자열). None 이면 config.STT_HOTWORDS, '' 이면 힌트 없음"""
    hw = config.STT_HOTWORDS if hotwords is None else hotwords
    # 낱말만 나열하면 받아쓰기도 마침표 없이 나와서, 쉼표·마침표 찍은 한 문장으로 줌
    words = [w.strip() for w in hw.split(",") if w.strip()]
    hw = ", ".join(words) + "." if words else ""
    use_dia = config.STT_DIARIZE if diarize is None else diarize
    chans, _ = read_audio(data) if not isinstance(data, tuple) else data
    with _lock:  # 모델 하나를 여러 요청이 동시에 쓰지 않게 (CPU 를 다른 서비스와 나눠 씀)
        model = _get_model()
        if kind == "call" and len(chans) == 2 and split_channels:
            turns = []
            for x, spk in zip(chans, SPEAKERS):
                turns += _sentences(model, x, spk, hw)
            return sorted(turns, key=lambda t: t["start"])
        mono = np.mean(chans, 0).astype(np.float32)
        turns = _sentences(model, mono, None, hw)
        if use_dia and turns:
            from app.stt import diarize as D
            labels = D.diarize(mono, turns, 2 if kind == "call" else speakers)
            names = _name_call(turns, labels) if kind == "call" and len(set(labels)) == 2 else \
                [f"화자{lb + 1}" for lb in labels] if len(set(labels)) > 1 or kind == "meeting" else [None] * len(turns)
            for t, nm in zip(turns, names):
                t["speaker"] = nm
    return sorted(turns, key=lambda t: t["start"])


def to_text(turns):
    """적재용 문서 본문: 한 줄에 한 문장, '[분:초] 화자: 말'"""
    lines = []
    for t in turns:
        m, s = divmod(int(t["start"]), 60)
        who = f"{t['speaker']}: " if t["speaker"] else ""
        lines.append(f"[{m:02d}:{s:02d}] {who}{t['text']}")
    return "\n".join(lines)
