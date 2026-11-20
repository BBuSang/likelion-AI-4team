"""금융상품 RAG 벡터 만들기 (worklog 53장) — 처음 한 번, 데이터가 바뀌면 다시.

    python scripts/build_product_vectors.py

입력 data/public_rag/product/chunks/*.jsonl, meta/*.csv → 출력 data/public_rag/product/vectors.npz (git 제외)
대상: 색인 항목 2,457개 (청크·공통 조항·용어) + 상품 이름·판매 중지 이름 (상품명 찾기 후보)
모델: 서버와 같은 KURE-v1 (config.EMBED_MODEL). 문서는 최대 2,048토큰, 정규화.
CPU 에서 10~20분 걸릴 수 있음. 이미 있는 벡터는 다시 계산하지 않음.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config  # noqa: E402
from app.rag_public.product.index import VECTORS, items_texts, name_lists, text_hash  # noqa: E402


def main():
    texts = [i["text"] for i in items_texts()]
    on, disc, _ = name_lists()
    texts += on + disc
    have = {}
    if VECTORS.exists():
        old = np.load(VECTORS)
        have = dict(zip(old["hashes"], old["vectors"]))
    todo = list(dict.fromkeys(t for t in texts if text_hash(t) not in have))
    print(f"항목 {len(texts)}개, 새로 계산 {len(todo)}개 (모델 {config.EMBED_MODEL})", flush=True)
    if todo:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
        model.max_seq_length = 2048
        t0 = time.time()
        vecs = model.encode(todo, batch_size=8, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
        for t, v in zip(todo, vecs):
            have[text_hash(t)] = v.astype(np.float32)
        print(f"계산 {time.time() - t0:.0f}초")
    keys = sorted(have)
    VECTORS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(VECTORS, hashes=np.array(keys), vectors=np.stack([have[k] for k in keys]))
    print(f"저장 {VECTORS} ({len(keys)}개)")


if __name__ == "__main__":
    main()
