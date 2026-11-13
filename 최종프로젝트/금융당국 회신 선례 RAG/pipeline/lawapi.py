"""국가법령정보 OPEN API 호출. 인증값(OC)은 환경 변수나 .env에서 불러오기만 하고 절대 출력·저장하지 않는다.
응답 원본은 data/raw/law/ 에 캐시해서 같은 요청을 다시 보내지 않는다 (캐시 파일 이름·내용에 OC 없음)."""
import hashlib
import json
import os
import time

import requests
from dotenv import load_dotenv

from common import RAW, ROOT

load_dotenv(ROOT / ".env")
_OC = os.getenv("LAW_OC", "").strip()
CACHE = RAW / "law"
CACHE.mkdir(parents=True, exist_ok=True)
DELAY = 1.0
_last = 0.0
_s = requests.Session()
_s.headers["User-Agent"] = "Mozilla/5.0 (education project; non-commercial)"
n_requests = 0


def mask(text):
    return text.replace(_OC, "****") if _OC else text


def call(service, params, cache=True):
    """service: 'lawSearch' | 'lawService'. params: OC 제외한 요청 변수. JSON으로 받는다."""
    global _last, n_requests
    p = {k: v for k, v in sorted(params.items())}
    tag = hashlib.sha1(json.dumps([service, p], ensure_ascii=False).encode()).hexdigest()[:16]
    path = CACHE / f"{service}_{p.get('target', '')}_{tag}.json"
    if cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    if not _OC:
        raise RuntimeError("LAW_OC가 비어 있음 (.env 확인)")
    for attempt in range(3):
        wait = DELAY - (time.time() - _last)
        if wait > 0:
            time.sleep(wait)
        _last = time.time()
        n_requests += 1
        try:
            r = _s.get(f"http://www.law.go.kr/DRF/{service}.do", params={"OC": _OC, "type": "JSON", **p}, timeout=30)
            r.raise_for_status()
            # 응답 안 링크에 OC 값이 들어 있으므로 가린 뒤 저장·반환한다
            data = json.loads(mask(r.text))
            path.write_text(json.dumps({"request": p, "response": data}, ensure_ascii=False), encoding="utf-8")
            return {"request": p, "response": data}
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(mask(f"{service} {p} 실패: {e}"))
            time.sleep(5 * (attempt + 1))
