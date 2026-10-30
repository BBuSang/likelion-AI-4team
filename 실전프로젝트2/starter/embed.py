"""임베딩. OpenAI 호환 /embeddings API 또는 테스트용 로컬 해시 임베딩.

주의: 이 함수에는 반드시 가명처리된(마스킹된) 텍스트만 넣어야 합니다.
"""
import hashlib
import re
import time

import numpy as np
import requests

from . import config

# CLOVA Studio OpenAI 호환 /embeddings 는 input 에 리스트를 넣으면 400(convert error) → 문장 1개씩 보냄
_BATCH = 1
_RETRY = 5
# 호출 한도: 분당 60회 (응답 헤더 x-ratelimit-limit-requests) → 요청 사이 최소 간격
_MIN_INTERVAL = 1.05
_last_call = 0.0


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


def _api(texts):
    global _last_call
    if not config.EMBED_API_KEY:
        raise RuntimeError("EMBED_API_KEY 가 비어 있습니다. .env 를 확인하거나 EMBED_PROVIDER=local 로 테스트하세요.")
    out = []
    for i in range(0, len(texts), _BATCH):
        batch = texts[i:i + _BATCH]
        for attempt in range(_RETRY):
            time.sleep(max(0.0, _last_call + _MIN_INTERVAL - time.monotonic()))
            _last_call = time.monotonic()
            r = requests.post(
                config.EMBED_API_BASE.rstrip("/") + "/embeddings",
                headers={"Authorization": f"Bearer {config.EMBED_API_KEY}"},
                json={"model": config.EMBED_MODEL, "input": batch[0] if _BATCH == 1 else batch},
                timeout=60,
            )
            if r.status_code != 429:  # 호출 한도 초과면 한도가 풀릴 때까지 쉬고 다시
                break
            # CLOVA는 남은 대기 시간을 x-ratelimit-reset-requests: "45s" 로 알려 줌
            reset = r.headers.get("retry-after") or r.headers.get("x-ratelimit-reset-requests", "")
            time.sleep(float(reset.rstrip("s") or 0) + 1 if reset.rstrip("s").replace(".", "").isdigit() else 10 * (attempt + 1))
        if r.status_code == 429:
            raise RuntimeError("임베딩 API 호출 한도(분당 60회)를 넘었습니다. 같은 키를 쓰는 다른 작업이 있는지 확인하고 1분 뒤 다시 시도하세요.")
        if not r.ok:
            raise RuntimeError(f"임베딩 API {r.status_code}: {r.text[:300]}")
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        out.extend(np.array(d["embedding"], dtype=np.float32) for d in data)
    if out and len(out[0]) != config.EMBED_DIM:
        raise RuntimeError(f"임베딩 차원 {len(out[0])} 이 EMBED_DIM={config.EMBED_DIM} 과 다릅니다.")
    return out


def embed(texts):
    return _api(texts) if config.EMBED_PROVIDER == "api" else _local(texts)
