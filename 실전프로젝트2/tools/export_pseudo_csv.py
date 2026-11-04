"""가명처리 결과(outputs/pseudo_*.jsonl)를 사람이 한눈에 보기 좋은 CSV 두 개로 정리한다.

  outputs/pseudo_spans.csv : 토큰으로 바꾼 구간 1개당 1행 (원문 표기 · 정규화 값 · 토큰 · 앞뒤 문맥)
  outputs/pseudo_docs.csv  : 문서 1건당 1행 (원문 ↔ 마스킹 텍스트 나란히)

DB 나 API 는 쓰지 않는다. 원문은 data 파일에서, 토큰은 pseudo 파일에서 읽고,
마스킹 텍스트와 정규화 값은 pii_pipeline 과 같은 규칙으로 다시 만든다 (모델 로딩 없음).

사용법
    python tools/export_pseudo_csv.py
    python tools/export_pseudo_csv.py --data data/holdout_labeled.jsonl --pseudo outputs/pseudo_holdout_labeled.jsonl
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from starter.pii_pipeline import normalize  # noqa: E402

CONTEXT = 15  # 앞뒤 문맥 글자 수


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def mask(text, ents):
    out, last = [], 0
    for e in sorted(ents, key=lambda e: e["start"]):
        out += [text[last:e["start"]], e["token"]]
        last = e["end"]
    return "".join(out + [text[last:]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "holdout_labeled.jsonl"))
    ap.add_argument("--pseudo", default=str(ROOT / "outputs" / "pseudo_holdout_labeled.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    a = ap.parse_args()

    docs = {d["id"]: d for d in load_jsonl(a.data)}
    pseudo = load_jsonl(a.pseudo)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(exist_ok=True)

    spans, doc_rows = [], []
    for p in pseudo:
        d = docs[p["id"]]
        text = d["text"]
        for e in sorted(p["entities"], key=lambda e: e["start"]):
            original = text[e["start"]:e["end"]]
            spans.append({
                "doc_id": p["id"], "doc_type": d.get("doc_type", ""), "label": e["label"],
                "original": original, "normalized": normalize(e["label"], original), "token": e["token"],
                "start": e["start"], "end": e["end"],
                "context": (text[max(0, e["start"] - CONTEXT):e["start"]] + "[" + original + "]"
                            + text[e["end"]:e["end"] + CONTEXT]).replace("\n", " "),
            })
        doc_rows.append({
            "doc_id": p["id"], "doc_type": d.get("doc_type", ""), "n_tokens": len(p["entities"]),
            "tokens": " ".join(sorted({e["token"] for e in p["entities"]})),
            "original_text": text, "masked_text": mask(text, p["entities"]),
        })

    # 엑셀에서 한글이 깨지지 않도록 utf-8-sig (BOM)
    for name, rows in (("pseudo_spans.csv", spans), ("pseudo_docs.csv", doc_rows)):
        with open(out_dir / name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"{out_dir / name}: {len(rows)}행")


if __name__ == "__main__":
    main()
