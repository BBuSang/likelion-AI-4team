"""2-5 섹션 분할 평가: 정답 경계(data/eval/section_gold.json) 대비 경계 정밀도·재현율·F1

사용: python analysis/s5_eval.py [--gold=section_gold_holdout.json] [섹션 폴더 ...]   (기본: 개발용 정답, data/sections)
예측 경계 = 섹션 파일의 n_lines 누적값(섹션 시작 줄 번호), 0번 제외.
exact: 같은 줄일 때만 정답 / ±1: 한 줄 차이까지 정답 (두 줄짜리 행 이름 등)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLD = {}


def pred_starts(folder: Path, doc: str) -> set[int]:
    starts, pos = set(), 0
    for line in open(folder / f"{doc}.jsonl"):
        s = json.loads(line)
        if pos:
            starts.add(pos)
        pos += s["n_lines"]
    return starts


def score(pred: set, gold: set, tol: int):
    tp_g = {g for g in gold if any(abs(g - p) <= tol for p in pred)}
    tp_p = {p for p in pred if any(abs(g - p) <= tol for g in gold)}
    return len(tp_p), len(pred), len(tp_g), len(gold)


def evaluate(folder: Path, verbose=False, gold_file="section_gold.json"):
    GOLD = json.loads((ROOT / "data" / "eval" / gold_file).read_text())
    tot = {0: [0, 0, 0, 0], 1: [0, 0, 0, 0]}
    rows = []
    for doc, gold in GOLD.items():
        if doc.startswith("_"):
            continue
        pred, g = pred_starts(folder, doc), set(gold)
        for tol in (0, 1):
            for i, v in enumerate(score(pred, g, tol)):
                tot[tol][i] += v
        tp_p, n_p, tp_g, n_g = score(pred, g, 0)
        rows.append((doc, n_g, n_p, tp_g, sorted(g - pred), sorted(pred - g)))
    out = {}
    for tol, (tpp, np_, tpg, ng) in tot.items():
        p, r = tpp / max(np_, 1), tpg / max(ng, 1)
        out[tol] = (p, r, 2 * p * r / max(p + r, 1e-9))
    if verbose:
        for doc, ng, np_, tpg, miss, extra in rows:
            print(f"  {doc}: 정답 {ng} 예측 {np_} 맞음 {tpg} | 놓침 {miss} | 잘못 나눔 {extra}")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    gold_file = "section_gold.json"
    if args and args[0].startswith("--gold="):
        gold_file = args.pop(0).split("=", 1)[1]
    folders = [Path(f) for f in args] or [ROOT / "data" / "sections"]
    for f in folders:
        r = evaluate(f, verbose=True, gold_file=gold_file)
        print(f"[{f.name} / {gold_file}] exact P {r[0][0]:.3f} R {r[0][1]:.3f} F1 {r[0][2]:.3f} | ±1 P {r[1][0]:.3f} R {r[1][1]:.3f} F1 {r[1][2]:.3f}\n")
