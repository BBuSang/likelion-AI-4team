"""13 임베딩: chunks의 embed_text를 벡터로 바꾼다. 모델명+글 해시로 캐시해서 끊겨도 이어서 받고 같은 글은 다시 호출하지 않는다.
입력: processed/chunks.jsonl   출력: processed/embedding_cache.sqlite, embeddings.npy (chunks.jsonl과 같은 순서)
키·주소는 .env에서 읽기만 하고 출력하지 않는다.
"""
import hashlib
import os
import sqlite3
import sys
import time

import numpy as np
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from common import OUT, ROOT, manifest, read_jsonl

load_dotenv(ROOT / ".env", override=True)
_KEY, _URL = os.getenv("OPENAI_API_KEY", "").strip(), os.getenv("OPENAI_BASE_URL", "").strip()
MODEL, DIM = os.getenv("EMBED_MODEL", "").strip(), int(os.getenv("EMBED_DIM", "0"))
BATCH, RETRIES = 100, 6
client = OpenAI(api_key=_KEY, base_url=_URL, timeout=120, max_retries=0)


def mask(s):
    return s.replace(_KEY, "****").replace(_URL, "****") if _KEY and _URL else s


def embed(texts):
    """글 목록을 벡터로 바꾼다. 일시적 오류는 간격을 늘려 다시 시도한다. (벡터 목록, 사용 토큰)"""
    for attempt in range(RETRIES):
        try:
            r = client.embeddings.create(model=MODEL, input=texts, encoding_format="float")
            data = sorted(r.data, key=lambda d: d.index)
            if len(data) != len(texts) or any(len(d.embedding) != DIM for d in data):
                raise RuntimeError(f"응답 개수·차원 불일치: {len(data)}/{len(texts)}")
            return [d.embedding for d in data], r.usage.total_tokens
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            err = e
        except APIStatusError as e:
            if e.status_code < 500:
                raise SystemExit(mask(f"요청 오류 {e.status_code}: {e}")[:400])
            err = e
        time.sleep(min(60, 2 ** (attempt + 1)))
        print(f"재시도 {attempt + 1}/{RETRIES}: {type(err).__name__}", flush=True)
    raise SystemExit(mask(f"반복 실패: {type(err).__name__}")[:300])


def text_hash(text):
    return hashlib.sha256(f"{MODEL}\n{text}".encode()).hexdigest()


def main():
    chunks = read_jsonl("chunks.jsonl")
    hashes = [text_hash(c["embed_text"]) for c in chunks]
    db = sqlite3.connect(OUT / "embedding_cache.sqlite")
    db.execute("create table if not exists cache (hash text primary key, model text, dim int, vec blob)")
    have = {h for (h,) in db.execute("select hash from cache")}
    todo, seen = [], set()
    for h, c in zip(hashes, chunks):
        if h not in have and h not in seen:
            seen.add(h)
            todo.append((h, c["embed_text"]))
    print(f"청크 {len(chunks)}, 고유 글 {len(set(hashes))}, 캐시 있음 {len(set(hashes)) - len(todo)}, 호출 대상 {len(todo)}", flush=True)

    tokens, started = 0, time.time()
    for i in range(0, len(todo), BATCH):
        part = todo[i:i + BATCH]
        vecs, used = embed([t for _, t in part])
        tokens += used
        db.executemany("insert or replace into cache values (?,?,?,?)",
                       [(h, MODEL, DIM, np.asarray(v, dtype=np.float32).tobytes()) for (h, _), v in zip(part, vecs)])
        db.commit()
        n = i + len(part)
        if n % 1000 == 0 or n == len(todo):
            print(f"[임베딩] {n}/{len(todo)} 토큰 {tokens} {time.time() - started:.0f}초", flush=True)

    got = {h: b for h, b in db.execute("select hash, vec from cache where model = ?", (MODEL,))}
    missing = [h for h in hashes if h not in got]
    if missing:
        sys.exit(f"캐시에 없는 글 {len(missing)}개")
    arr = np.vstack([np.frombuffer(got[h], dtype=np.float32) for h in hashes])
    np.save(OUT / "embeddings.npy", arr)
    manifest("13_embeddings", chunks=len(chunks), unique_texts=len(set(hashes)), called=len(todo), tokens=tokens,
             shape=list(arr.shape), seconds=round(time.time() - started))


if __name__ == "__main__":
    main()
