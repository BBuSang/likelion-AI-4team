"""D3 후속: 로컬 임베딩 모델(KURE-v1, bge-m3 등)로 D3 청크와 평가 질문을 임베딩한다. API를 쓰지 않는다 (모델 내려받기만 외부 통신).
- 캐시: processed/embedding_cache_local.sqlite, 키는 모델명 + 글 해시. 끊겨도 이어서 받는다
- 청크 글은 chunks_d3.jsonl의 embed_text, 질문은 개인정보 마스킹 후 그대로 (질문 접두어 없음)
- 글 길이순으로 묶어 패딩을 줄이고, 최대 512토큰으로 자른다
사용: python d3b_local_embed.py <모델> [--limit N]     예: python d3b_local_embed.py nlpai-lab/KURE-v1
      --limit N: 속도 측정용. 고르게 뽑은 N개 청크만 임베딩하고 .npy는 만들지 않는다
출력: processed/embeddings_d3_<모델 짧은 이름>.npy (chunks_d3.jsonl과 같은 순서)
"""
import argparse
import hashlib
import json
import sqlite3
import time

import numpy as np

from common import DATA, OUT, manifest, read_jsonl
from rag import mask_pii

MAX_LEN, BATCH = 512, 16
EVAL = DATA / "eval"
QSETS = ("rewrite_150", "dev1", "dev2", "unanswerable_20")


def key(model, text):
    return hashlib.sha256(f"{model}\n{text}".encode()).hexdigest()


def query_texts():
    out = []
    for n in QSETS:
        with open(EVAL / f"{n}.jsonl", encoding="utf-8") as f:
            out += [mask_pii(json.loads(l)["question"]) for l in f if l.strip()]
    return list(dict.fromkeys(out))


def encode_missing(model, name, db, texts, label):
    have = {h for (h,) in db.execute("select hash from cache where model = ?", (name,))}
    todo, seen = [], set()
    for t in texts:
        h = key(name, t)
        if h not in have and h not in seen:
            seen.add(h)
            todo.append((h, t))
    todo.sort(key=lambda x: len(x[1]))
    print(f"[{label}] 글 {len(texts)}개, 고유 {len(set(key(name, t) for t in texts))}, 호출 대상 {len(todo)}", flush=True)
    started, done = time.time(), 0
    for i in range(0, len(todo), BATCH):
        part = todo[i:i + BATCH]
        vecs = model.encode([t for _, t in part], batch_size=BATCH, normalize_embeddings=True, show_progress_bar=False)
        db.executemany("insert or replace into cache values (?,?,?,?)", [(h, name, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes()) for (h, _), v in zip(part, vecs)])
        db.commit()
        done += len(part)
        if (i // BATCH) % 25 == 0 or done == len(todo):
            el = time.time() - started
            print(f"[{label}] {done}/{len(todo)} {el:.0f}초 (남은 시간 약 {el / done * (len(todo) - done):.0f}초)", flush=True)
    return len(todo), time.time() - started


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    model = SentenceTransformer(a.model, device="cpu")
    model.max_seq_length = MAX_LEN
    print(f"모델 로드 {time.time() - t0:.0f}초, 차원 {model.get_sentence_embedding_dimension()}", flush=True)
    chunks = read_jsonl("chunks_d3.jsonl")
    texts = [c["embed_text"] for c in chunks]
    db = sqlite3.connect(OUT / "embedding_cache_local.sqlite")
    db.execute("create table if not exists cache (hash text primary key, model text, dim int, vec blob)")
    if a.limit:
        step = max(1, len(texts) // a.limit)
        n, sec = encode_missing(model, a.model, db, texts[::step][:a.limit], "속도 측정")
        print(f"청크 {n}개 {sec:.1f}초 → 청크당 {sec / max(n, 1):.3f}초, 전체 {len(texts)}개 예상 약 {sec / max(n, 1) * len(texts) / 60:.0f}분")
        return
    n_c, s_c = encode_missing(model, a.model, db, texts, "청크")
    n_q, s_q = encode_missing(model, a.model, db, query_texts(), "질문")
    got = {h: b for h, b in db.execute("select hash, vec from cache where model = ?", (a.model,))}
    arr = np.vstack([np.frombuffer(got[key(a.model, t)], dtype=np.float32) for t in texts])
    short = a.model.split("/")[-1].lower()
    np.save(OUT / f"embeddings_d3_{short}.npy", arr)
    manifest("d3b_local_embed", model=a.model, chunks=len(texts), called_chunks=n_c, called_queries=n_q, shape=list(arr.shape), seconds=round(s_c + s_q), max_len=MAX_LEN)


if __name__ == "__main__":
    main()
