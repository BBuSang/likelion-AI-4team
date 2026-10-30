"""외부 API 클라이언트 (CLOVA Studio OpenAI 호환). 가명처리된 텍스트만 보낸다.

- embed(text)  : bge-m3 임베딩 1건 (CLOVA는 input 리스트를 받지 않음)
- chat(messages, reasoning=None) : HCX-007 답변 문자열
두 API 모두 분당 60회 한도 → 호출 간격 1.05초, 429면 서버가 알려 준 시간만큼 기다렸다 다시.
"""
import time

import numpy as np
import requests

import config

MIN_INTERVAL = 1.05
RETRY = 4
_last = {"embed": 0.0, "chat": 0.0}


def _post(kind, url, key, payload, timeout):
    for attempt in range(RETRY + 1):
        time.sleep(max(0.0, _last[kind] + MIN_INTERVAL - time.monotonic()))
        _last[kind] = time.monotonic()
        r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=timeout)
        if r.status_code != 429 or attempt == RETRY:
            break
        reset = (r.headers.get("retry-after") or r.headers.get("x-ratelimit-reset-requests", "")).rstrip("s")
        time.sleep(float(reset) + 1 if reset.replace(".", "").isdigit() else 10 * (attempt + 1))
    if r.status_code == 429:
        raise RuntimeError(f"{kind} API 호출 한도(분당 60회) 초과. 같은 키를 쓰는 다른 작업을 확인하세요.")
    if not r.ok:
        raise RuntimeError(f"{kind} API {r.status_code}: {r.text[:300]}")
    return r.json()


def embed(text):
    if not config.EMBED_API_KEY:
        raise RuntimeError("EMBED_API_KEY 가 비어 있습니다.")
    data = _post("embed", config.EMBED_API_BASE.rstrip("/") + "/embeddings", config.EMBED_API_KEY,
                 {"model": config.EMBED_MODEL, "input": text}, 60)
    v = np.asarray(data["data"][0]["embedding"], dtype=np.float32)
    if len(v) != config.EMBED_DIM:
        raise RuntimeError(f"임베딩 차원 {len(v)} ≠ EMBED_DIM {config.EMBED_DIM}")
    return v


def chat(messages, reasoning=None):
    payload = {"model": config.LLM_MODEL, "temperature": 0, "messages": messages}
    if reasoning:
        payload["reasoning_effort"] = reasoning
    data = _post("chat", config.LLM_API_BASE.rstrip("/") + "/chat/completions", config.LLM_API_KEY, payload, 120)
    return data["choices"][0]["message"]["content"]
