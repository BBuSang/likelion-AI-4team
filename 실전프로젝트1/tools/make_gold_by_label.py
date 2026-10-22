"""dev·test 300건의 정답 개인정보를 라벨별 CSV 로 모은다 (팀 데이터 분석용).

결과: data/gold_by_label/<라벨>.csv 7개 + summary.csv
엑셀에서 한글이 깨지지 않도록 utf-8-sig 로 저장한다.

사용법
  python -X utf8 tools/make_gold_by_label.py
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 맨 위 폴더의 common·run_compare·detectors 사용

from collections import Counter

from common import DATA, LABELS, ROOT, load_docs

OUT = DATA / "gold_by_label"
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
    OUT.mkdir(exist_ok=True)
    rows = {lab: [] for lab in LABELS}
    n_docs = 0
    for split in ("dev", "test"):
        for d in load_docs(split):
            n_docs += 1
            t = d["text"]
            for e in d["entities"]:
                s, end = e["start"], e["end"]
                rows[e["label"]].append([
                    split, d["id"], d["doc_type"], d["template"], e["text"], shape(e["text"]),
                    e.get("form", ""), e.get("person_id", ""), ", ".join(d["hard_cases"]),
                    t[s - 1:s].replace("\n", "⏎"), t[end:end + 1].replace("\n", "⏎"), context(t, s, end),
                ])

    summary = []
    for lab in LABELS:
        with open(OUT / f"{lab}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(rows[lab])
        r = rows[lab]
        splits = Counter(x[0] for x in r)
        top = Counter(x[5] for x in r).most_common(5)
        summary.append([lab, len(r), splits["dev"], splits["test"], len({x[4] for x in r}),
                        len({x[7] for x in r}), " / ".join(f"{k} ({v})" for k, v in top)])
        print(f"{lab:<30} {len(r):>5}개 (dev {splits['dev']}, test {splits['test']})")

    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["label", "total", "dev", "test", "unique_values", "unique_persons", "top_shapes"])
        w.writerows(summary)
    print(f"문서 {n_docs}건, 정답 {sum(len(v) for v in rows.values())}개 → {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
