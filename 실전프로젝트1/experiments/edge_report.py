"""엣지 케이스 종류(태그)별로 시스템 점수를 비교한다.

사용 예
  python -X utf8 experiments/edge_report.py --pred rules=preds/rules_edge.jsonl --pred model=preds/koelectra_pii_edge.jsonl
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 맨 위 폴더의 common·run_compare·detectors 사용

from collections import defaultdict

from common import DATA
from run_compare import error_rows, load_gold, load_pred, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True, help="이름=예측.jsonl")
    ap.add_argument("--errors", action="store_true", help="마지막 시스템의 오류 목록 출력")
    a = ap.parse_args()
    docs = load_gold(DATA / "edge_cases.jsonl")
    systems = {}
    for arg in a.pred:
        name, _, path = arg.partition("=")
        systems[name] = load_pred(path, docs)[0]

    groups = defaultdict(list)
    for i, d in enumerate(docs):
        groups[d["template"]].append(i)

    names = list(systems)
    print(f"{'케이스 (문서/정답)':<28}" + "".join(f"{k[:18]:>22}" for k in names))
    print(f"{'':<28}" + "".join(f"{'재현율 / 오탐':>22}" for _ in names))
    for tag, idx in groups.items():
        sub = [docs[i] for i in idx]
        n_gold = sum(len(d["gold"]) for d in sub)
        cells = []
        for k in names:
            P = [systems[k][i] for i in idx]
            s, _ = summarize(sub, P)
            fp = sum(1 for d, ps in zip(sub, P) for p in ps
                     if not any(p["start"] < g["end"] and g["start"] < p["end"] for g in d["gold"]))
            cells.append(f"{s['partial_r']:.2f} / {fp}" if n_gold else f"   - / {fp}")
        print(f"{tag + f' ({len(idx)}/{n_gold})':<28}" + "".join(f"{c:>22}" for c in cells))

    print("\n전체")
    for k in names:
        s, _ = summarize(docs, systems[k])
        print(f"  {k:<18} strict F1 {s['strict_f1']:.3f}  partial R {s['partial_r']:.3f}  mask_rec {s['mask_rec']:.3f}")

    if a.errors:
        k = names[-1]
        print(f"\n[{k}] 오류")
        for r in error_rows(docs, systems[k], k):
            print(f"  {r[1]} {r[2]:<22} 정답={r[4]!r:<28} 예측={r[5]!r:<28} {r[7]}")


if __name__ == "__main__":
    main()
