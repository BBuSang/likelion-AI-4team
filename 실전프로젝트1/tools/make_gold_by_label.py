"""정답 개인정보를 라벨별 CSV 로 모은다 (팀 데이터 분석용).

결과: data/<출력 폴더>/<라벨>.csv 7개 + summary.csv
엑셀에서 한글이 깨지지 않도록 utf-8-sig 로 저장한다. holdout 처럼 template·person_id 가 없는 데이터는 빈칸.

사용법
  python -X utf8 tools/make_gold_by_label.py                                         # dev·test → data/gold_by_label/
  python -X utf8 tools/make_gold_by_label.py --splits holdout --out gold_by_label_holdout
"""
import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 맨 위 폴더의 common·run_compare·detectors 사용

from collections import Counter

from common import DATA, LABELS, ROOT, load_docs

HEADER = ["split", "doc_id", "doc_type", "template", "text", "shape", "form", "person_id",
          "hard_cases", "prev_char", "next_char", "context"]


def shape(value: str) -> str:
    """값의 모양: 숫자는 9, 한글은 가, 영문은 a 로 바꿔 형식 분포를 보기 쉽게 한다."""
    s = re.sub(r"\d", "9", value)
    s = re.sub(r"[가-힣]", "가", s)
    return re.sub(r"[A-Za-z]", "a", s)


def context(text, s, e, w=20):
    return (text[max(0, s - w):s] + "【" + text[s:e] + "】" + text[e:e + w]).replace("\n", " ⏎ ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--out", default="gold_by_label")
    a = ap.parse_args()
    splits_used = a.splits.split(",")
    out_dir = DATA / a.out
    out_dir.mkdir(exist_ok=True)
    rows = {lab: [] for lab in LABELS}
    n_docs = 0
    for split in splits_used:
        for d in load_docs(split):
            n_docs += 1
            t = d["text"]
            for e in d["entities"]:
                s, end = e["start"], e["end"]
                rows[e["label"]].append([
                    split, d["id"], d["doc_type"], d.get("template", ""), e["text"], shape(e["text"]),
                    e.get("form", ""), e.get("person_id", ""), ", ".join(d["hard_cases"]),
                    t[s - 1:s].replace("\n", "⏎"), t[end:end + 1].replace("\n", "⏎"), context(t, s, end),
                ])

    summary = []
    for lab in LABELS:
        with open(out_dir / f"{lab}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(rows[lab])
        r = rows[lab]
        splits = Counter(x[0] for x in r)
        top = Counter(x[5] for x in r).most_common(5)
        summary.append([lab, len(r)] + [splits[sp] for sp in splits_used] + [len({x[4] for x in r}),
                        len({x[7] for x in r} - {""}), " / ".join(f"{k} ({v})" for k, v in top)])
        print(f"{lab:<30} {len(r):>5}개 (" + ", ".join(f"{sp} {splits[sp]}" for sp in splits_used) + ")")

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["label", "total"] + splits_used + ["unique_values", "unique_persons", "top_shapes"])
        w.writerows(summary)
    print(f"문서 {n_docs}건, 정답 {sum(len(v) for v in rows.values())}개 → {out_dir.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
