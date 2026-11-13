"""Colab용 입력 파일 만들기: D3 청크의 임베딩 글과 평가 질문(개인정보 마스킹 후)을 글 목록 하나로 묶는다.
출력: processed/colab_input.json  {"chunks": [...], "queries": [...]}  (순서가 곧 결과 순서)
이 파일에는 포털 회신 원문이 들어 있다. 저장소에 올리지 않고 본인 Colab에만 올린다 (data/ 아래라 git 제외).
"""
import json

from common import OUT, manifest, read_jsonl
from d3b_local_embed import query_texts


def main():
    chunks = [c["embed_text"] for c in read_jsonl("chunks_d3.jsonl")]
    queries = query_texts()
    with open(OUT / "colab_input.json", "w", encoding="utf-8") as f:
        json.dump({"chunks": chunks, "queries": queries}, f, ensure_ascii=False)
    size = (OUT / "colab_input.json").stat().st_size
    manifest("d3b_colab_export", chunks=len(chunks), queries=len(queries), bytes=size)


if __name__ == "__main__":
    main()
