"""규칙 / 토큰 / 규칙+토큰 비교표 (정확도 + 처리 시간).

시스템
  rules                      규칙 단독
  <모델>                      토큰 분류 모델 단독 (우리 후처리 포함)
  rules+<모델>               전체 합집합: 모든 라벨을 합치고 겹치면 규칙 우선 (make_union 기본)
  rules+<모델>:names         이름만 결합: 정형 라벨은 규칙만, 이름은 규칙 ∪ 모델 (make_union --names-only, 팀원 방식)
처리 시간: 규칙+토큰은 두 탐지기를 순서대로 돌린다고 보고 문서당 시간을 더함. 1만 건은 순차 처리 환산.

사용법 (예측 파일이 preds/ 에 있어야 함: run_detect.py 로 먼저 생성)
  python -X utf8 experiments/compare_systems.py
결과: results/system_comparison.csv, 화면 출력
"""
import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 맨 위 폴더의 common·run_compare 사용

from common import DATA, ROOT
from run_compare import load_gold, load_pred, summarize

GOLD = {"dev": "pii_synth_finance_dev.jsonl", "test": "pii_synth_finance_test.jsonl", "edge": "edge_cases.jsonl",
        "synth_dev": "generated/synth_dev.jsonl", "synth_holdout": "generated/synth_holdout.jsonl",
        "synth_fresh": "generated/synth_fresh.jsonl",
        "synth_hard_dev": "generated/synth_hard_dev.jsonl", "synth_hard_eval": "generated/synth_hard_eval.jsonl"}
# 표시 이름 → 예측 파일 이름 (atonlee 는 우리 pipeline 버전 koelectra_pii).
# 팀 결정(2026-10-23): 토큰 모델은 atonlee 하나만 사용. 다른 모델을 비교하려면 여기에 추가 (예: "veil": "veil")
#   atonlee: 팀원 래퍼(detectors/hf_ner.py)로 문서를 한 번에 처리. pipeline 방식(koelectra_pii)과의 비교는 worklog 15-2
MODELS = {"atonlee": "atonlee"}
TMP = Path(tempfile.mkdtemp())  # 비교용 '이름만 결합' 결과는 preds/ 에 남기지 않음


def union(split, model_file, names_only):
    # 전체 합집합(운영 탐지기)만 preds/combined/ 에 저장
    out = (TMP if names_only else ROOT / "preds" / "combined") / f"rules+{model_file}{'_names' if names_only else ''}_{split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-X", "utf8", str(ROOT / "make_union.py"), "--out", str(out),
           str(ROOT / "preds" / f"rules_{split}.jsonl"), str(ROOT / "preds" / f"{model_file}_{split}.jsonl")]
    if names_only:
        cmd.insert(4, "--names-only")
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="dev,test,edge,synth_dev,synth_holdout")
    a = ap.parse_args()

    rows = []
    for split in a.splits.split(","):
        docs = load_gold(DATA / GOLD[split])
        systems = {"rules": ROOT / "preds" / f"rules_{split}.jsonl"}
        for show, f in MODELS.items():
            systems[show] = ROOT / "preds" / f"{f}_{split}.jsonl"
        for show, f in MODELS.items():
            systems[f"rules+{show}"] = union(split, f, names_only=False)
            systems[f"rules+{show}:names"] = union(split, f, names_only=True)
        print(f"\n=== {split} (문서 {len(docs)}, 정답 {sum(len(d['gold']) for d in docs)})")
        print(f"{'시스템':<24}{'strict P':>9}{'strict R':>9}{'strict F1':>10}{'partial R':>10}{'mask_rec':>9}"
              f"{'ms/문서':>9}{'1만건(분)':>10}")
        for name, path in systems.items():
            P, ms = load_pred(path, docs)
            s, _ = summarize(docs, P)
            ms = ms or 0.0
            rows.append({"split": split, "system": name, **{k: round(v, 4) for k, v in s.items()},
                         "ms_per_doc": round(ms, 1), "min_per_10k": round(ms * 10000 / 60000, 1)})
            print(f"{name:<24}{s['strict_p']:>9.3f}{s['strict_r']:>9.3f}{s['strict_f1']:>10.3f}{s['partial_r']:>10.3f}"
                  f"{s['mask_rec']:>9.3f}{ms:>9.1f}{ms * 10000 / 60000:>10.1f}")

    out = ROOT / "results" / "system_comparison.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
