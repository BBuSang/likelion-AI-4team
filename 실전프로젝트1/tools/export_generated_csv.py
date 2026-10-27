"""생성 데이터 5세트를 사람이 보기 쉬운 CSV 2개로 합친다 (엑셀로 열기용, 채점에는 jsonl 을 씀).

결과
  data/generated/generated_documents.csv  문서 1건 = 1줄 (세트, 카테고리, 문체, 본문, 표시 포함 본문)
  data/generated/generated_entities.csv   정답 1개 = 1줄 (라벨, 값, 위치, 앞뒤 문맥)
사용법
  python -X utf8 tools/export_generated_csv.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 맨 위 폴더의 common 사용

from common import DATA

SETS = {
    "synth_dev": "1차 일상 문서 · 개선용",
    "synth_holdout": "1차 일상 문서 · 평가용",
    "synth_fresh": "2차 금융 업무 문서 · 평가 전용",
    "synth_hard_dev": "어려운 표기 · 개선용",
    "synth_hard_eval": "어려운 표기 · 평가용",
}


def marked(text, ents):
    out, last = [], 0
    for e in sorted(ents, key=lambda e: e["start"]):
        out += [text[last:e["start"]], f"[[{text[e['start']:e['end']]}|{e['label']}]]"]
        last = e["end"]
    return "".join(out) + text[last:]


def main():
    gen = DATA / "generated"
    docs_rows, ent_rows = [], []
    for split, desc in SETS.items():
        for line in open(gen / f"{split}.jsonl", encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            ents, text = d["entities"], d["text"]
            base = {"set": desc, "split": split, "doc_id": d["id"], "category": d["template"], "style": d["doc_type"]}
            docs_rows.append({**base, "pii_count": len(ents),
                              "labels": ", ".join(sorted({e["label"] for e in ents})),
                              "text": text, "text_with_labels": marked(text, ents)})
            for e in sorted(ents, key=lambda e: e["start"]):
                s, t = e["start"], e["end"]
                ctx = (text[max(0, s - 20):s] + "【" + text[s:t] + "】" + text[t:t + 20]).replace("\n", " ⏎ ")
                ent_rows.append({**base, "label": e["label"], "value": text[s:t], "start": s, "end": t, "context": ctx})
    for name, rows in (("generated_documents.csv", docs_rows), ("generated_entities.csv", ent_rows)):
        with open(gen / name, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{name}: {len(rows)}줄")


if __name__ == "__main__":
    main()
