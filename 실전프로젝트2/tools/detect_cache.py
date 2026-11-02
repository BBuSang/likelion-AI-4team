"""문서 파일 전체를 탐지기로 한 번 돌려 결과를 저장한다. 동명이인 규칙·가명처리를 바꿔 가며 실험할 때
모델을 매번 돌리지 않으려고 쓴다 (ingest.py --detect-cache 로도 읽음).

  outputs/detect_<파일명>.jsonl : {"id", "spans": [{"label", "start", "end"}, ...], "ms"}

사용법
    python tools/detect_cache.py data/holdout_labeled.jsonl data/week1/dev_docs.jsonl data/week1/test_docs.jsonl
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from starter import config, pii_pipeline  # noqa: E402


def cache_path(data_path):
    return config.OUT_DIR / f"detect_{Path(data_path).stem}.jsonl"


def main():
    config.OUT_DIR.mkdir(exist_ok=True)
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            docs = [json.loads(l) for l in f if l.strip()]
        pii_pipeline.detect("워밍업 문장 010-0000-0000")  # 모델 로딩은 시간 측정에서 뺀다
        with open(cache_path(path), "w", encoding="utf-8") as f:
            for d in docs:
                t = time.perf_counter()
                spans = pii_pipeline.detect(d["text"])
                ms = (time.perf_counter() - t) * 1000
                f.write(json.dumps({"id": d["id"], "spans": spans, "ms": round(ms, 1)}, ensure_ascii=False) + "\n")
        print(f"{path}: {len(docs)}건 → {cache_path(path)}")


if __name__ == "__main__":
    main()
