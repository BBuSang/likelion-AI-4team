"""파이프라인 공통: 경로, .env, 국가법령정보 API 호출(캐시·간격·인증값 가림)"""
import hashlib
import json
import time
from datetime import date
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
ENV = dotenv_values(ROOT / ".env")
SOURCE = ROOT / "data" / "source"
WORK = ROOT / "data" / "work"
REPORTS = ROOT / "reports"

API = "https://www.law.go.kr/DRF/"
PAUSE = 1.0  # 기획안 5-1: 요청 간격 1초 이상
TODAY = date.today().strftime("%Y%m%d")  # 기준일 = 실행일


def offline_if_cached(model="nlpai-lab/KURE-v1"):
    """모델이 이미 받아져 있으면 인터넷 확인 없이 쓰고, 없으면 처음 한 번 받게 둔다 (새 컴퓨터에서 첫 실행 오류 방지)"""
    import os
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    if (hub / ("models--" + model.replace("/", "--"))).exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _mask(s):
    oc = ENV.get("LAW_API_OC") or ""
    return str(s).replace(oc, "***") if oc else str(s)


def api_get(path, cache_dir, log_path, **params):
    """API 호출. 같은 요청은 cache_dir 의 파일을 돌려준다 (다시 보내지 않음).
    인증값(OC)은 캐시 키·로그·오류 메시지에 남기지 않는다."""
    oc = ENV.get("LAW_API_OC")
    if not oc:
        raise SystemExit(".env 에 LAW_API_OC 가 없습니다")
    key = json.dumps({"path": path, **params}, ensure_ascii=False, sort_keys=True)
    cache = Path(cache_dir) / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    last = None
    for attempt in range(3):  # 실패 시 3번까지
        try:
            r = requests.get(API + path, params={"OC": oc, "type": "JSON", **params}, timeout=60)
            r.raise_for_status()
            body = r.json()
            time.sleep(PAUSE)
            cache.parent.mkdir(parents=True, exist_ok=True)
            text = _mask(json.dumps(body, ensure_ascii=False))
            cache.write_text(text, encoding="utf-8")
            _log(log_path, path, params, "ok", cache)
            return json.loads(text)
        except Exception as e:
            last = _mask(e)
            time.sleep(2 ** attempt + PAUSE)
    _log(log_path, path, params, "fail: " + last[:200], None)
    raise RuntimeError(last)


def _log(log_path, path, params, status, file):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": time.strftime("%Y-%m-%d %H:%M:%S"), "path": path, "params": params,
                            "status": status, "file": str(file) if file else None}, ensure_ascii=False) + "\n")


def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def content(x):
    """API 값이 {'content': ...} 모양이면 content 만"""
    return x.get("content") if isinstance(x, dict) else x
