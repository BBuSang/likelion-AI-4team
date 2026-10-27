"""부트스트랩으로 지표의 95% 신뢰구간과 시스템 간 차이의 신뢰구간을 구한다.

문서를 복원 추출로 다시 뽑아(같은 문서가 여러 번 뽑힐 수 있음) 점수를 1,000번 계산하고,
그 분포의 2.5%~97.5% 구간을 신뢰구간으로 본다. 정답이 같은 문서 안에서 묶여 있으므로
개체가 아닌 '문서' 단위로 뽑는다. 채점 로직은 run_compare.py 의 것을 그대로 쓴다.

사용 예
  python -X utf8 experiments/bootstrap_ci.py --gold data/pii_synth_finance_test.jsonl --pred rules=preds/rules_test.jsonl --pred llm=preds/llm_test_r1.jsonl
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 맨 위 폴더의 common·run_compare·detectors 사용


from run_compare import load_gold, load_pred, summarize

METRICS = [("strict_f1", "strict F1"), ("partial_r", "partial R"), ("mask_rec", "mask_rec")]


def ci(values):
    v = sorted(values)
    return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", action="append", required=True, help="이름=예측.jsonl")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    docs = load_gold(a.gold)
    systems = {}
    for arg in a.pred:
        name, _, path = arg.partition("=")
        systems[name] = load_pred(path, docs)[0]

    rng = random.Random(a.seed)
    samples = {k: {m: [] for m, _ in METRICS} for k in systems}
    for _ in range(a.n):
        idx = [rng.randrange(len(docs)) for _ in docs]
        sub = [docs[i] for i in idx]
        for k, P in systems.items():
            s, _ = summarize(sub, [P[i] for i in idx])
            for m, _ in METRICS:
                samples[k][m].append(s[m])

    print(f"문서 {len(docs)}건, 부트스트랩 {a.n}회 (문서 단위 복원 추출)\n")
    print(f"{'시스템':<16}" + "".join(f"{label:>26}" for _, label in METRICS))
    for k, P in systems.items():
        full, _ = summarize(docs, P)
        cells = []
        for m, _ in METRICS:
            lo, hi = ci(samples[k][m])
            cells.append(f"{full[m]:.3f} [{lo:.3f}, {hi:.3f}]")
        print(f"{k:<16}" + "".join(f"{c:>26}" for c in cells))

    # 첫 시스템 대비 차이: 같은 부트스트랩 표본에서 짝지어 뺀다. 구간이 0을 포함하면 '차이 있다고 말하기 어려움'
    names = list(systems)
    base = names[0]
    print(f"\n기준 '{base}' 대비 차이 (다른 시스템 - 기준), 95% 구간")
    for k in names[1:]:
        cells = []
        for m, _ in METRICS:
            diffs = [x - y for x, y in zip(samples[k][m], samples[base][m])]
            lo, hi = ci(diffs)
            mark = "  " if lo <= 0 <= hi else " *"
            cells.append(f"[{lo:+.3f}, {hi:+.3f}]{mark}")
        print(f"{k:<16}" + "".join(f"{c:>26}" for c in cells))
    print("\n* = 구간이 0을 포함하지 않음 (차이가 있다고 볼 수 있음)")


if __name__ == "__main__":
    main()
