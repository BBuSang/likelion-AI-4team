"""X4 현장건의 청크 임베딩용 Colab 입력 파일 (49번 d3b_colab_export와 같은 방식, 질문은 넣지 않는다: 질문은 검색할 때 로컬에서 임베딩한다).
출력: processed/exmnt_colab_input.json  {"model", "max_len", "n", "sha256", "chunks": [embed_text, ...]}  (순서 = exmnt_chunks.jsonl 순서 = 결과 순서)
이 파일에는 현장건의 원문(개인정보는 가린 상태)이 들어 있다. 저장소에 올리지 않고 본인 Colab에만 올린다 (data/ 아래라 git 제외).
다음: pipeline/x4_exmnt_colab.ipynb 를 Colab(T4 GPU)에서 실행 → exmnt_colab_output.npz 를 data/processed/ 에 두고 python x4_exmnt_colab_import.py
"""
import hashlib
import json

from common import OUT, manifest, read_jsonl
from rag import QUERY_MODEL

MAX_LEN = 512          # rag.embed()의 질문 임베딩과 같은 최대 길이


def texts_sha(texts):
    return hashlib.sha256("\n␞\n".join(texts).encode("utf-8")).hexdigest()


def main():
    texts = [c["embed_text"] for c in read_jsonl("exmnt_chunks.jsonl")]
    path = OUT / "exmnt_colab_input.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"model": QUERY_MODEL, "max_len": MAX_LEN, "n": len(texts), "sha256": texts_sha(texts), "chunks": texts}, f, ensure_ascii=False)
    manifest("x4_exmnt_colab_export", chunks=len(texts), model=QUERY_MODEL, bytes=path.stat().st_size)


if __name__ == "__main__":
    main()
