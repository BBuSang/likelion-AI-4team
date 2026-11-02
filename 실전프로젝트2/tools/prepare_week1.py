"""실전 1 dev/test 데이터를 실전 2 형식으로 준비한다 (동명이인 구분 규칙을 holdout 이 아닌 dev/test 로 설계하기 위해).

  data/week1/<split>_docs.jsonl           : {"id", "doc_type", "text"}            (적재·탐지 입력)
  data/week1/<split>_persons_gold.jsonl   : {"id", "persons": [{start, end, text, person_id, form}]}  (채점 전용)

사용법
    python tools/prepare_week1.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT.parent / "실전프로젝트1" / "data"
OUT = ROOT / "data" / "week1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("dev", "test"):
        with open(SRC / f"pii_synth_finance_{split}.jsonl", encoding="utf-8") as f:
            docs = [json.loads(l) for l in f if l.strip()]
        with open(OUT / f"{split}_docs.jsonl", "w", encoding="utf-8") as fd, \
                open(OUT / f"{split}_persons_gold.jsonl", "w", encoding="utf-8") as fg:
            for d in docs:
                fd.write(json.dumps({"id": d["id"], "doc_type": d.get("doc_type"), "text": d["text"]},
                                    ensure_ascii=False) + "\n")
                persons = [{k: e.get(k) for k in ("start", "end", "text", "person_id", "form")}
                           for e in d["entities"] if e["label"] == "person" and e.get("person_id")]
                fg.write(json.dumps({"id": d["id"], "persons": persons}, ensure_ascii=False) + "\n")
        n = sum(1 for d in docs for e in d["entities"] if e["label"] == "person")
        print(f"{split}: 문서 {len(docs)}건, 이름 {n}회 → {OUT}")


if __name__ == "__main__":
    main()
