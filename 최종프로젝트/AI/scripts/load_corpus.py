"""문서 파일 → 새 DB 적재 (app/pipeline/load.py). 탐지·직원 목록·가명처리·적재를 한 번에.

    python scripts/load_corpus.py --split holdout --reset --detect-cache            # 2주차 탐지 캐시 사용
    python scripts/load_corpus.py --data path/to/docs.jsonl --reset                 # 탐지기 실행 (1~2분 로딩)
    EMBED_PROVIDER=local python scripts/load_corpus.py --split holdout --reset --detect-cache   # 외부 API 없이

--reset 은 rag·vault 데이터를 지우고 새로 넣음 (새 DB 용).
결과: outputs/load_<이름>/employees.jsonl, doc_meta.jsonl, summary.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
W2 = ROOT.parents[1] / "실전프로젝트2"

from app.core import config  # noqa: E402
from app.pii import pii_pipeline  # noqa: E402
from app.pipeline import load  # noqa: E402

SPLITS = {"holdout": ("data/holdout_labeled.jsonl", "detect_holdout_labeled.jsonl"),
          "synth2": ("data/synth2/synth2_labeled.jsonl", "detect_synth2_labeled.jsonl"),
          "synth3": ("data/synth3/synth3_labeled.jsonl", "detect_synth3_labeled.jsonl")}


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--split", choices=SPLITS)
    g.add_argument("--data", help="jsonl: 한 줄에 {id, text, doc_type?} (entities 는 읽지 않음)")
    ap.add_argument("--detect-cache", action="store_true", help="2주차 outputs/detect_*.jsonl 사용 (--split 일 때)")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()

    path = W2 / SPLITS[a.split][0] if a.split else Path(a.data)
    name = a.split or path.stem
    docs = [{"id": r["id"], "text": r["text"], "doc_type": r.get("doc_type")} for r in read(path)]  # 정답 라벨은 버림
    if a.split and a.detect_cache:
        spans = {r["id"]: r["spans"] for r in read(W2 / "outputs" / SPLITS[a.split][1])}
    else:
        print("탐지 중 (모델 첫 로딩 1~2분)...")
        spans = {d["id"]: pii_pipeline.detect(d["text"]) for d in docs}
    summary, employees, meta = load.load(docs, spans, reset=a.reset)

    out = config.OUT_DIR / f"load_{name}"
    out.mkdir(parents=True, exist_ok=True)
    for fname, rows in (("employees", employees), ("doc_meta", meta)):
        with open(out / f"{fname}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(json.dumps(summary | {"embed_provider": config.EMBED_PROVIDER},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"결과 파일: {out}")


if __name__ == "__main__":
    main()
