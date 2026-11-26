"""모노 녹음 화자 나누기 (CPU). 받아쓴 문장마다 목소리 특징을 뽑아 비슷한 것끼리 묶음.

- 목소리 특징: config.STT_SPK_MODEL (기본 microsoft/wavlm-base-plus-sv, 화자 확인용 x-vector 512차원). 처음 쓸 때 허깅페이스에서 받음
- 묶기: 사람 수를 알면 k-means, 모르면 응집 군집(코사인 거리, 평균 연결, 거리 기준 config.STT_SPK_THRESHOLD)
- 짧은 문장('네', '음' 등 MIN_SEC 미만)은 특징이 흔들려서 묶을 때 빼고, 다 묶은 뒤 가장 가까운 무리에 붙임
- 결과는 문장별 번호 0,1,2… (먼저 말한 순서). 이름(상담원/고객, 화자N)은 transcribe.py 가 붙임
- 한계: 두 사람이 겹쳐 말하는 곳, 한 문장 안에서 사람이 바뀌는 곳(쉼 없이 이어 말함)은 못 나눔
"""
import numpy as np

from app.core import config
from app.core.loading import MODEL_LOAD_LOCK

RATE = 16000
MIN_SEC = 1.0    # 이보다 짧은 문장은 묶을 때 빼고 나중에 붙임
MAX_SEC = 10.0   # 긴 문장은 가운데 10초만 (속도)
PAD = 0.1        # 문장 앞뒤 여유 (단어 시각이 조금 안쪽으로 잡힘)
MAX_AUTO = 8     # 사람 수를 모를 때 최대

_fe = _model = None


def _get():
    global _fe, _model
    if _model is None:
        with MODEL_LOAD_LOCK:
            if _model is None:
                import torch
                from transformers import AutoFeatureExtractor, WavLMForXVector
                torch.set_num_threads(config.STT_THREADS)
                _fe = AutoFeatureExtractor.from_pretrained(config.STT_SPK_MODEL)
                _model = WavLMForXVector.from_pretrained(config.STT_SPK_MODEL).eval()
    return _fe, _model


def warmup():
    _get()


def _clip(audio, t):
    a, b = max(0, int((t["start"] - PAD) * RATE)), min(len(audio), int((t["end"] + PAD) * RATE))
    x = audio[a:b]
    if len(x) > MAX_SEC * RATE:
        m = (len(x) - int(MAX_SEC * RATE)) // 2
        x = x[m:m + int(MAX_SEC * RATE)]
    if len(x) < 0.4 * RATE:  # 너무 짧으면 모델이 못 받음 → 양옆 소리까지 넓힘
        c = (a + b) // 2
        x = audio[max(0, c - int(0.2 * RATE)):c + int(0.2 * RATE)]
    return x


def embed(audio, turns):
    """문장별 목소리 특징 (n, d), 길이 1로 맞춤"""
    import torch
    fe, model = _get()
    out = []
    with torch.inference_mode():
        for t in turns:
            x = _clip(audio, t)
            inp = fe(x, sampling_rate=RATE, return_tensors="pt")
            out.append(model(**inp).embeddings[0].numpy())
    e = np.stack(out)
    return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)


def cluster(emb, durations, n_speakers=None, threshold=None):
    """→ 문장별 무리 번호 (먼저 나온 순서로 0,1,2…)"""
    from sklearn.cluster import AgglomerativeClustering
    n = len(emb)
    th = config.STT_SPK_THRESHOLD if threshold is None else threshold
    long_ = np.array([d >= MIN_SEC for d in durations])
    if long_.sum() < 2:   # 긴 문장이 거의 없으면 짧은 것까지 다 써서 묶음
        long_ = np.ones(n, bool)
    if n < 2 or (n_speakers is not None and n_speakers < 2):
        return [0] * n
    idx = np.where(long_)[0]
    k = min(n_speakers, len(idx)) if n_speakers else None
    if len(idx) < 2:
        return [0] * n
    if k:   # 사람 수를 알면 k-means (합성 2~4인 녹음 비교에서 응집 군집보다 나음, worklog 75장)
        from sklearn.cluster import KMeans
        lab = KMeans(k, n_init=20, random_state=0).fit_predict(emb[idx])
    else:   # 모르면 응집 군집 + 거리 기준. 조금 잘게 나누는 쪽 (검수 때 합치기가 잘못 섞인 걸 떼기보다 쉬움)
        lab = AgglomerativeClustering(n_clusters=None, distance_threshold=th, metric="cosine", linkage="average").fit_predict(emb[idx])
    if not k and len(set(lab)) > MAX_AUTO:   # 기준이 너무 잘게 나눴으면 최대 수로 다시
        lab = AgglomerativeClustering(n_clusters=MAX_AUTO, metric="cosine", linkage="average").fit_predict(emb[idx])
    cent = np.stack([emb[idx][lab == c].mean(0) for c in range(lab.max() + 1)])
    cent /= np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-9)
    full = np.empty(n, int)
    full[idx] = lab
    rest = np.where(~long_)[0]
    if len(rest):
        full[rest] = (emb[rest] @ cent.T).argmax(1)
    order = {}
    for c in full:   # 먼저 말한 사람이 0
        order.setdefault(int(c), len(order))
    return [order[int(c)] for c in full]


def diarize(audio, turns, n_speakers=None, threshold=None):
    """audio: 16kHz 모노 float32, turns: transcribe 의 문장들 (시간순 아니어도 됨) → 문장별 무리 번호"""
    if not turns:
        return []
    emb = embed(audio, turns)
    return cluster(emb, [t["end"] - t["start"] for t in turns], n_speakers, threshold)
