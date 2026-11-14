"""임베딩. 세 가지 (config.EMBED_PROVIDER)
  api    OpenAI 호환 /embeddings API (CLOVA Studio bge-m3). 받은 결과는 outputs/embed_cache_<모델>.jsonl 에 캐시
  hf     내 컴퓨터·서버에서 직접 돌리는 모델 (sentence-transformers, 예: nlpai-lab/KURE-v1). 외부로 아무것도 안 나감. 같은 캐시 사용
  local  테스트용 문자 n-gram 해시 (검색 품질 낮음)

주의: 이 함수에는 반드시 가명처리된(마스킹된) 텍스트만 넣어야 합니다.
"""
import hashlib
import json
import re
import threading
import time

import numpy as np
import requests

from app.core import config

# 한 번에 보낼 문장 수. CLOVA Studio 의 OpenAI 호환 /embeddings 는 리스트 입력을 받지 않아 1 로 둔다 (.env 의 EMBED_BATCH)
_BATCH = config.EMBED_BATCH
_RETRIES = 8
_MAX_WAIT = 60

# API 로 받은 임베딩을 모델별로 저장해 두고 재사용한다 (재적재·검색 실험 때 API 호출·속도 제한을 줄임)
# 키는 텍스트의 해시라 마스킹된 텍스트도 파일에 남지 않는다
_cache = None


def _cache_path():
    return config.OUT_DIR / f"embed_cache_{re.sub(r'[^0-9A-Za-z_.-]', '_', config.EMBED_MODEL)}.jsonl"


def _cache_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache():
    global _cache
    if _cache is None:
        _cache = {}
        if _cache_path().exists():
            with open(_cache_path(), encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    _cache[row["key"]] = np.array(row["vec"], dtype=np.float32)
    return _cache


def _save_cache(text, vec):
    config.OUT_DIR.mkdir(exist_ok=True)
    with open(_cache_path(), "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": _cache_key(text), "vec": vec.tolist()}) + "\n")
    _cache[_cache_key(text)] = vec


def _local(texts):
    """API 없이 파이프라인 동작만 확인하는 문자 n-gram 해시 임베딩 (검색 품질 낮음)."""
    out = []
    for t in texts:
        v = np.zeros(config.EMBED_DIM, dtype=np.float32)
        s = re.sub(r"\s+", " ", t)
        for n in (2, 3):
            for i in range(len(s) - n + 1):
                h = int(hashlib.md5(s[i:i + n].encode()).hexdigest()[:8], 16)
                v[h % config.EMBED_DIM] += 1.0
        norm = np.linalg.norm(v)
        out.append(v / norm if norm else v)
    return out


def _api(texts, retries=_RETRIES):
    if not config.EMBED_API_KEY:
        raise RuntimeError("EMBED_API_KEY 가 비어 있습니다. .env 를 확인하거나 EMBED_PROVIDER=local 로 테스트하세요.")
    cache = _load_cache()
    todo = list(dict.fromkeys(t for t in texts if _cache_key(t) not in cache))
    for i in range(0, len(todo), _BATCH):
        batch = todo[i:i + _BATCH]
        r = _post(batch[0] if _BATCH == 1 else batch, retries)
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        for t, d in zip(batch, data):
            _save_cache(t, np.array(d["embedding"], dtype=np.float32))
    out = [cache[_cache_key(t)] for t in texts]
    if out and len(out[0]) != config.EMBED_DIM:
        raise RuntimeError(f"임베딩 차원 {len(out[0])} 이 EMBED_DIM={config.EMBED_DIM} 과 다릅니다.")
    return out


def _wait_seconds(r, attempt):
    """서버가 알려주는 대기 시간. CLOVA 는 x-ratelimit-reset-requests: '6s' (분당 60회 한도, 키를 여럿이 같이 쓰면 금방 참)."""
    for h in ("Retry-After", "x-ratelimit-reset-requests"):
        m = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s)?", r.headers.get(h, "").strip())
        if m:
            sec = float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)
            return min(sec + 0.5, _MAX_WAIT)
    return min(2 ** (attempt + 1), _MAX_WAIT)


def _post(inp, retries=_RETRIES):
    """요청 1건. 속도 제한(429)·일시 오류(5xx)는 잠시 쉬었다가 다시 보낸다."""
    for attempt in range(retries):
        r = requests.post(
            config.EMBED_API_BASE.rstrip("/") + "/embeddings",
            headers={"Authorization": f"Bearer {config.EMBED_API_KEY}"},
            json={"model": config.EMBED_MODEL, "input": inp},
            timeout=60,
        )
        if r.status_code != 429 and r.status_code < 500:
            break
        if attempt < retries - 1:
            time.sleep(_wait_seconds(r, attempt))
    r.raise_for_status()
    return r


_hf_model = None
_hf_lock = threading.Lock()  # 서버 시작 때 미리 읽기와 첫 질문이 겹쳐도 모델을 한 번만 읽게


def _hf_load():
    global _hf_model
    with _hf_lock:
        if _hf_model is None:
            from sentence_transformers import SentenceTransformer
            _hf_model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    return _hf_model


def warmup():
    """서버 시작 때 로컬 모델을 미리 읽음 (안 하면 첫 질문이 수십 초). api·local 이면 할 일 없음"""
    if config.EMBED_PROVIDER == "hf":
        _hf_load().encode(["준비"], normalize_embeddings=True)
        _load_cache()


def _hf(texts):
    """로컬 모델 (worklog 17장). 기본 BAAI/bge-m3 — CLOVA 의 bge-m3 와 같은 벡터 (17-2). KURE-v1 도 같은 1024차원.
    모델은 처음 쓸 때 읽음 (CPU 에서 수십 초, 서버는 시작 때 warmup). 정규화한 벡터 → pgvector 코사인 거리와 맞음"""
    cache = _load_cache()
    todo = list(dict.fromkeys(t for t in texts if _cache_key(t) not in cache))
    if todo:
        vecs = _hf_load().encode(todo, batch_size=config.EMBED_BATCH, normalize_embeddings=True,
                                convert_to_numpy=True, show_progress_bar=len(todo) > 50)
        for t, v in zip(todo, vecs):
            _save_cache(t, v.astype(np.float32))
    out = [cache[_cache_key(t)] for t in texts]
    if out and len(out[0]) != config.EMBED_DIM:
        raise RuntimeError(f"임베딩 차원 {len(out[0])} 이 EMBED_DIM={config.EMBED_DIM} 과 다릅니다.")
    return out


def embed(texts, retries=_RETRIES):
    """retries: 대화형(질의 1건)에서는 적게 줘서 오래 멈추지 않게 한다 (api 만)."""
    if config.EMBED_PROVIDER == "api":
        return _api(texts, retries)
    if config.EMBED_PROVIDER == "hf":
        return _hf(texts)
    return _local(texts)
