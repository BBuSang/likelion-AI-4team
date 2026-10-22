"""두 개 이상의 예측 파일을 합집합으로 합친다 (한쪽만 잡아도 가린다 → 재현율·mask_rec ↑, 정밀도 ↓).

규칙: 위치가 겹치는 예측은 앞에 적은 파일의 것을 우선한다 (예: 번호는 규칙, 나머지는 모델).
처리 시간(ms)은 두 탐지기를 순서대로 돌린다고 보고 더한다.

사용 예
  python -X utf8 make_union.py --out preds/union_rules_koelectra_dev.jsonl preds/rules_dev.jsonl preds/koelectra_pii_dev.jsonl
"""
import argparse
import json
from pathlib import Path

from common import save_preds


def overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefer", choices=["first", "longer"], default="first",
                    help="겹칠 때 남길 쪽: first=앞 파일 우선, longer=더 긴 구간 우선(더 많이 가림)")
    ap.add_argument("preds", nargs="+")
    a = ap.parse_args()

    merged = {}
    for path in a.preds:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            m = merged.setdefault(r["id"], {"id": r["id"], "entities": [], "ms": 0.0})
            m["ms"] = round(m["ms"] + r.get("ms", 0.0), 1)
            for e in r["entities"]:
                hits = [x for x in m["entities"] if overlap(e, x)]
                if not hits:
                    m["entities"].append(e)
                elif a.prefer == "longer" and all(e["end"] - e["start"] > x["end"] - x["start"] for x in hits):
                    # 규칙이 '서울시 종로구 종로3길 18'까지, 모델이 '…, 혜화빌딩 6층'까지 잡았으면 모델 쪽을 남긴다
                    for x in hits:
                        m["entities"].remove(x)
                    m["entities"].append(e)
    rows = list(merged.values())
    for r in rows:
        r["entities"].sort(key=lambda e: e["start"])
    save_preds(Path(a.out), rows)
    print(f"합집합 {len(rows)}건 → {a.out}")


if __name__ == "__main__":
    main()
