"""에이전트용 LLM 호출 (질문 분류·계획). JSON 만 받는다.

- 넘기는 글은 가명처리된 질문뿐 (agent.protect 를 거친 뒤). 이 파일은 원문을 받지 않는다
- 응답 시간·토큰·비용을 같이 돌려준다 (멘토 안내: 모델·이유·응답 시간·비용을 리포트에)
"""
import json
import re
import time

import requests

from app.core import config

# 1M 토큰당 달러 (입력, 출력). 비용 계산용 — OpenAI 가격표 기준, 바뀌면 여기만 고침
PRICE = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-6-luna": (0.10, 0.50),  # 2026-11 가격 (제3자 가격표 기준, OpenAI 가격표에서 다시 확인 필요)
}

_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def _endpoint(provider):
    if provider == "openai":
        return config.OPENAI_API_BASE, config._openai_key()
    return config.CLOVA_API_BASE, config.CLOVA_API_KEY


def parse_json(text):
    """```json ... ``` 이나 앞뒤 설명이 붙어 와도 첫 { 부터 마지막 } 까지 읽는다. 못 읽으면 None"""
    m = _JSON_OBJ.search(text or "")
    if not m:
        return None
    try:
        out = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    return out if isinstance(out, dict) else None


def chat_json(system, user, model=None, provider=None, timeout=60):
    """(dict 또는 None, meta) 반환. meta = {model, latency, in_tokens, out_tokens, cost_usd, raw}"""
    provider = provider or config.AGENT_PROVIDER
    model = model or config.AGENT_MODEL
    base, key = _endpoint(provider)
    if not key:
        raise RuntimeError(f"에이전트 LLM 키가 없습니다 (provider={provider})")
    payload = {"model": model, **config.sampling_params(model),
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if provider == "openai":
        payload["response_format"] = {"type": "json_object"}
    elif model == "HCX-007":
        payload["reasoning_effort"] = config.LLM_REASONING_EFFORT or "none"
    t0 = time.perf_counter()
    r = requests.post(base.rstrip("/") + "/chat/completions", headers={"Authorization": f"Bearer {key}"},
                      json=payload, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    latency = time.perf_counter() - t0
    text = body["choices"][0]["message"]["content"]
    usage = body.get("usage") or {}
    tin, tout = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    pin, pout = PRICE.get(model, (0.0, 0.0))
    meta = {"model": model, "latency": round(latency, 3), "in_tokens": tin, "out_tokens": tout,
            "cost_usd": (tin * pin + tout * pout) / 1e6, "raw": text}
    return parse_json(text), meta
