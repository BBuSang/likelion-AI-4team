"""X4 Colab 결과 가져오기·점검: exmnt_colab_output.npz → processed/embeddings_exmnt_kure-v1.npy (exmnt_chunks.jsonl과 같은 순서). DB에는 넣지 않는다.
점검 (49·53번 적재 전 점검과 같은 항목)
- 개수, NaN/Inf, 정규화(노름 0.99~1.01), 1024차원
- 순서·내용: Colab이 읽은 입력의 sha256 = 지금 exmnt_chunks.jsonl로 다시 계산한 값
- 로컬 비교: 무작위 조각 SAMPLE개를 이 PC에서 fp32로 임베딩해 Colab(fp16) 결과와 코사인 비교 (최소 0.999 미만이면 멈춤)
사용: python x4_exmnt_colab_import.py   (data/processed/exmnt_colab_output.npz 필요)
"""
import random
import sys

import numpy as np

import rag
from common import OUT, manifest, read_jsonl
from x4_exmnt_colab_export import texts_sha

SAMPLE, MIN_COS = 60, 0.999


def main():
    path = OUT / "exmnt_colab_output.npz"
    if not path.exists():
        sys.exit(f"{path} 가 없습니다. Colab에서 내려받은 파일을 data/processed/ 에 두세요.")
    z = np.load(path)
    vec = z["chunks"].astype(np.float32)
    texts = [c["embed_text"] for c in read_jsonl("exmnt_chunks.jsonl")]
    problems = []
    if vec.shape != (len(texts), 1024):
        problems.append(f"모양 {vec.shape} (기대 {(len(texts), 1024)})")
    if str(z["sha256"]) != texts_sha(texts):
        problems.append("Colab 입력과 지금 청크 파일의 글·순서가 다릅니다")
    if str(z["model"]) != rag.QUERY_MODEL:
        problems.append(f"모델 {z['model']} (질문 임베딩은 {rag.QUERY_MODEL})")
    if not np.isfinite(vec).all():
        problems.append("NaN/Inf")
    norms = np.linalg.norm(vec, axis=1)
    if norms.min() < 0.99 or norms.max() > 1.01:
        problems.append(f"정규화 안 됨 {norms.min():.4f}~{norms.max():.4f}")
    if problems:
        sys.exit("점검 실패: " + " / ".join(problems))

    idx = sorted(random.Random(20261111).sample(range(len(texts)), SAMPLE))
    rag.embed("준비")
    local = rag._local_model.encode([texts[i] for i in idx], normalize_embeddings=True, batch_size=8)
    cos = (local * vec[idx]).sum(axis=1)
    print(f"로컬 fp32 vs Colab fp16 코사인 ({SAMPLE}개): 최소 {cos.min():.5f}, 평균 {cos.mean():.5f}")
    if cos.min() < MIN_COS:
        sys.exit(f"로컬 비교 실패: 최소 코사인 {cos.min():.5f} < {MIN_COS}")
    np.save(OUT / "embeddings_exmnt_kure-v1.npy", vec)
    manifest("x4_exmnt_colab_import", shape=list(vec.shape), norm=[round(float(norms.min()), 4), round(float(norms.max()), 4)],
             local_cos_min=round(float(cos.min()), 5), local_cos_mean=round(float(cos.mean()), 5), sample=SAMPLE)


if __name__ == "__main__":
    main()
