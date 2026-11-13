"""파이프라인 공통: 경로, 포털 요청(간격·재시도), jsonl·csv 입출력, 실행 기록."""
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"          # 원문 데이터. 저장소에 올리지 않는다 (포털 저작권정책)
RAW = DATA / "raw"
OUT = DATA / "processed"
OUT.mkdir(parents=True, exist_ok=True)

BASE = "https://better.fsc.go.kr/fsc_new"
DELAY = float(os.getenv("CRAWL_DELAY", "1.0"))
RULES_VERSION = "2026-11-09"
SECS = ("question", "answer", "reason")

_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0 (education project; non-commercial)"
_last = 0.0
n_requests = 0


def request(method, url, delay=None, retries=3, **kw):
    """요청 사이 간격을 두고, 실패하면 간격을 늘려 다시 시도한다."""
    global _last, n_requests
    for attempt in range(retries):
        wait = (DELAY if delay is None else delay) - (time.time() - _last)
        if wait > 0:
            time.sleep(wait)
        _last = time.time()
        n_requests += 1
        try:
            r = _session.request(method, url, timeout=60, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def pg_connect(dsn, **kw):
    """DB 접속. Windows에서 localhost는 IPv6를 먼저 시도해 약 130초를 기다리므로 IPv4로 고정한다."""
    import psycopg
    from urllib.parse import urlsplit
    if urlsplit(dsn).hostname == "localhost":
        kw.setdefault("hostaddr", "127.0.0.1")
    return psycopg.connect(dsn, **kw)


def case_key(gubun, idx):
    return ("L" if gubun == "법령해석" else "O") + str(idx)


def read_jsonl(name):
    with open(OUT / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(name, rows):
    with open(OUT / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(name, rows, cols):
    """엑셀에서 한글이 깨지지 않도록 UTF-8 BOM으로 저장한다. 목록 항목은 '; '로 잇는다."""
    def cell(key, v):
        if key == "attachments":
            return "; ".join(a["org_file_name"] for a in v)
        if isinstance(v, list):
            return "; ".join(map(str, v))
        return "" if v is None else v
    with open(OUT / name, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows([cell(k, r.get(k)) for k in cols] for r in rows)


def manifest(stage, **counts):
    """실행마다 단계, 시각, 규칙 버전, 건수를 남긴다."""
    row = {"stage": stage, "at": datetime.now().isoformat(timespec="seconds"), "rules_version": RULES_VERSION, **counts}
    with open(OUT / "manifest.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False))
