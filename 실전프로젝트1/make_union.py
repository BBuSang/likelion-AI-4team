"""두 개 이상의 예측 파일을 합집합으로 합친다 (한쪽만 잡아도 가린다 → 재현율·mask_rec ↑, 정밀도 ↓).

규칙: 위치가 겹치는 예측은 앞에 적은 파일의 것을 우선한다 (예: 번호는 규칙, 나머지는 모델).
처리 시간(ms)은 두 탐지기를 순서대로 돌린다고 보고 더한다.
문서 1건을 바로 합칠 때는 union_entities() 를 쓴다 (pii_detector.py 가 사용).

사용 예
  python -X utf8 make_union.py --out preds/union_rules_koelectra_dev.jsonl preds/rules_dev.jsonl preds/koelectra_pii_dev.jsonl
"""
import argparse
import json
from pathlib import Path

from common import save_preds


def overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def union_entities(entity_lists, prefer="first", names_only=False, longer_labels=("address",)):
    """한 문서의 예측 목록 여러 개를 합친다. 앞 목록이 우선."""
    merged = []
    for k, ents in enumerate(entity_lists):
        for e in ents:
            if names_only and k > 0 and (e["label"] != "person" or e["end"] - e["start"] < 2):
                continue
            hits = [x for x in merged if overlap(e, x)]
            if not hits:
                merged.append(e)
            # 라벨별 긴 쪽 우선은 겹치는 것이 하나일 때만: 모델이 주소 두 개를 하나로 이어 잡은 경우는 규칙 쪽을 둔다
            elif (prefer == "longer" or (e["label"] in longer_labels and len(hits) == 1
                                         and hits[0]["label"] == e["label"])) \
                    and all(e["end"] - e["start"] > x["end"] - x["start"] for x in hits):
                # 규칙이 '서울시 종로구 종로3길 18'까지, 모델이 '…, 혜화빌딩 6층'까지 잡았으면 모델 쪽을 남긴다
                for x in hits:
                    merged.remove(x)
                merged.append(e)
    return sorted(merged, key=lambda e: e["start"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefer", choices=["first", "longer"], default="first",
                    help="겹칠 때 남길 쪽: first=앞 파일 우선, longer=더 긴 구간 우선(더 많이 가림)")
    ap.add_argument("--names-only", action="store_true",
                    help="두 번째 파일부터는 두 글자 이상 이름(person)만 가져옴 (팀원 방식: 정형 라벨은 규칙만, 이름은 규칙 ∪ 모델)")
    ap.add_argument("--longer-labels", default="address",
                    help="이 라벨끼리 겹치면 앞 파일이어도 더 긴 쪽을 남김 (v9: 규칙이 '…로 261(보람동)'에서 끊고 모델이 "
                         "'…(보람동)삼산프라자8층801호'까지 잡는 경우). 빈 값이면 끔")
    ap.add_argument("preds", nargs="+")
    a = ap.parse_args()
    longer_labels = set(filter(None, a.longer_labels.split(",")))

    docs = {}  # id → ([파일별 예측 목록], [ms 합])
    for k, path in enumerate(a.preds):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            lists, ms = docs.setdefault(r["id"], ([[] for _ in a.preds], [0.0]))
            lists[k] = r["entities"]
            ms[0] = round(ms[0] + r.get("ms", 0.0), 1)
    rows = [{"id": i, "entities": union_entities(lists, a.prefer, a.names_only, longer_labels), "ms": ms[0]}
            for i, (lists, ms) in docs.items()]
    save_preds(Path(a.out), rows)
    print(f"합집합 {len(rows)}건 → {a.out}")


if __name__ == "__main__":
    main()
