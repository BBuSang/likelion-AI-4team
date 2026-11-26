"""녹음 읽기(여러 형식) · 화자 묶기 · 상담원/고객 이름 붙이기 (DB·Whisper·목소리 모델 필요 없음)."""
import io
import wave

import av
import numpy as np
import pytest

from app.stt import diarize as D
from app.stt import transcribe as T


def _tone(sec=2.0, rate=8000, ch=2):
    t = np.arange(int(sec * rate)) / rate
    x = np.stack([0.3 * np.sin(2 * np.pi * 440 * t), 0.3 * np.sin(2 * np.pi * 660 * t)][:ch], 1)
    return (x * 32767).astype(np.int16), rate


def _wav(x, rate):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(x.shape[1])
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(x.tobytes())
    return buf.getvalue()


def _encode(x, rate, codec, fmt):
    buf = io.BytesIO()
    layout = "stereo" if x.shape[1] == 2 else "mono"
    with av.open(buf, "w", format=fmt) as out:
        out_rate = 48000 if codec == "libopus" else rate if codec != "aac" else 16000
        st = out.add_stream(codec, rate=out_rate)
        st.layout = layout
        rs = av.AudioResampler(format=st.format.name, layout=layout, rate=out_rate)
        fr = av.AudioFrame.from_ndarray(x.reshape(1, -1), format="s16", layout=layout)
        fr.sample_rate = rate
        for f in rs.resample(fr) + rs.resample(None):
            for p in st.encode(f):
                out.mux(p)
        for p in st.encode(None):
            out.mux(p)
    return buf.getvalue()


@pytest.mark.parametrize("codec,fmt", [(None, "wav"), ("libmp3lame", "mp3"), ("aac", "ipod"), ("flac", "flac"), ("libopus", "webm")])
def test_read_audio_formats(codec, fmt):
    x, rate = _tone()
    data = _wav(x, rate) if codec is None else _encode(x, rate, codec, fmt)
    chans, r = T.read_audio(data)
    assert r == 16000 and len(chans) == 2                       # 스테레오는 두 채널 그대로 (상담원/고객 채널 나누기)
    assert abs(len(chans[0]) / r - 2.0) < 0.15                    # 길이 유지 (압축 형식은 앞뒤 여백이 조금 붙음)
    assert 0.1 < np.abs(chans[0]).max() < 1.01


def test_read_audio_mono_and_limits():
    x, rate = _tone(ch=1)
    chans, _ = T.read_audio(_wav(x, rate))
    assert len(chans) == 1
    with pytest.raises(ValueError, match="too long"):
        T.read_audio(_wav(*_tone(sec=5, ch=1)), max_sec=2)
    with pytest.raises(Exception):
        T.read_audio(b"not audio at all")


def _fake_emb(groups, dim=32, noise=0.15, seed=0):
    """사람마다 다른 방향 + 흔들림 → 길이 1"""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(max(groups) + 1, dim))
    e = centers[groups] + noise * rng.normal(size=(len(groups), dim)) * np.linalg.norm(centers[0])
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def test_cluster_known_count_and_first_speaker_is_zero():
    g = [2, 0, 1, 0, 2, 1, 1, 0, 2, 0]
    lab = D.cluster(_fake_emb(g), [3.0] * len(g), n_speakers=3)
    assert lab[0] == 0                                            # 먼저 말한 사람이 0
    mapping = {}
    for a, b in zip(g, lab):
        assert mapping.setdefault(a, b) == b                      # 같은 사람은 늘 같은 번호
    assert len(set(lab)) == 3


def test_cluster_unknown_count_and_short_turns_join_nearest():
    g = [0, 1, 0, 1, 2, 0, 2, 1, 0, 2, 1]
    dur = [3.0] * len(g)
    dur[3] = 0.4                                                  # 짧은 '네' 는 묶을 때 빠졌다가 가까운 무리로
    lab = D.cluster(_fake_emb(g, noise=0.05), dur, n_speakers=None)
    assert len(set(lab)) == 3
    assert lab[3] == lab[1]


def test_cluster_edge_cases():
    assert D.cluster(_fake_emb([0]), [2.0]) == [0]
    assert D.cluster(_fake_emb([0, 1, 0]), [2.0] * 3, n_speakers=1) == [0, 0, 0]


def test_name_call_picks_agent_by_cues():
    turns = [{"text": "네 여보세요"}, {"text": "새봄은행 고객센터입니다. 무엇을 도와드릴까요?"},
             {"text": "카드를 잃어버렸어요"}, {"text": "고객님 확인해 드리겠습니다"}]
    assert T._name_call(turns, [0, 1, 0, 1]) == ["고객", "상담원", "고객", "상담원"]
    # 말투로 가릴 수 없으면 먼저 말한 쪽이 상담원
    assert T._name_call([{"text": "음"}, {"text": "네"}], [1, 0]) == ["상담원", "고객"]
