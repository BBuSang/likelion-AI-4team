"""Colab 결과를 로컬 평가 형식으로 옮긴다.
입력: processed/colab_output_<짧은 이름>.npz  (chunks: N x 차원, queries: Q x 차원), processed/colab_input.json
출력: processed/embeddings_d3_<짧은 이름>.npy, embedding_cache_local.sqlite에 질문 벡터 추가 (d3b_eval.py가 읽는 형식)
사용: python d3b_colab_import.py nlpai-lab/KURE-v1
"""
import json
import sqlite3
import sys

import numpy as np

from common import OUT, manifest
from d3b_local_embed import key


def main(model):
    short = model.split("/")[-1].lower()
    src = json.load(open(OUT / "colab_input.json", encoding="utf-8"))
    z = np.load(OUT / f"colab_output_{short}.npz")
    ch, qu = z["chunks"].astype(np.float32), z["queries"].astype(np.float32)
    assert ch.shape[0] == len(src["chunks"]) and qu.shape[0] == len(src["queries"]), "개수 불일치"
    assert np.isfinite(ch).all() and np.isfinite(qu).all(), "NaN/Inf"
    norms = np.linalg.norm(ch, axis=1)
    assert norms.min() > 0.99 and norms.max() < 1.01, f"정규화 안 됨 {norms.min()}~{norms.max()}"
    np.save(OUT / f"embeddings_d3_{short}.npy", ch)
    db = sqlite3.connect(OUT / "embedding_cache_local.sqlite")
    db.execute("create table if not exists cache (hash text primary key, model text, dim int, vec blob)")
    db.executemany("insert or replace into cache values (?,?,?,?)", [(key(model, t), model, int(v.shape[0]), v.tobytes()) for t, v in zip(src["queries"], qu)])
    db.commit()
    manifest("d3b_colab_import", model=model, chunks=list(ch.shape), queries=list(qu.shape))


if __name__ == "__main__":
    main(sys.argv[1])
