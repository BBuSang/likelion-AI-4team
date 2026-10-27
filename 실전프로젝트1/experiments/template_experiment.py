"""처음 보는 템플릿에서의 성능 추정 실험.

dev 와 test 가 같은 템플릿 23종을 써서 test 점수는 '처음 보는 문서' 성능이 아니다. 그래서 두 가지를 본다.
1) 템플릿별 점수: 시스템마다 어떤 템플릿에서 약한지, 템플릿 간 편차가 얼마나 큰지
2) 규칙 기반 문맥 규칙 제거(ablation): 호칭·직함·메신저 말투처럼 dev 문구를 보고 만든 이름 규칙을 끄면
   점수가 얼마나 떨어지나 → 문구가 다른 새 템플릿에서 생길 수 있는 하락 폭의 추정치

사용법
  python -X utf8 experiments/template_experiment.py --split test --pred rules=preds/rules_test.jsonl --pred koelectra_pii=preds/koelectra_pii_test.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 맨 위 폴더의 common·run_compare·detectors 사용

from collections import defaultdict

from common import DATA, ROOT, dedupe, load_docs, save_preds
from run_compare import load_gold, load_pred, summarize


def per_template(docs, preds):
    groups = defaultdict(list)
    for i, d in enumerate(docs):
        groups[d["template"]].append(i)
    out = {}
    for t, idx in groups.items():
        s, _ = summarize([docs[i] for i in idx], [preds[i] for i in idx])
        n_gold = sum(len(docs[i]["gold"]) for i in idx)
        out[t] = (len(idx), n_gold, s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--pred", action="append", required=True)
    a = ap.parse_args()
    gold_path = DATA / f"pii_synth_finance_{a.split}.jsonl"
    docs = load_gold(gold_path)

    # 2) 문맥 규칙을 끈 규칙 기반 예측 만들기
    from detectors.rule_based import RuleBasedDetector
    det = RuleBasedDetector(context_rules=False)
    rows = [{"id": d["id"], "entities": dedupe(d["text"], det.detect(d["text"]))} for d in load_docs(a.split)]
    no_ctx = ROOT / "preds" / f"rules_no_context_{a.split}.jsonl"
    save_preds(no_ctx, rows)

    systems = {}
    for arg in a.pred + [f"rules_no_context={no_ctx}"]:
        name, _, path = arg.partition("=")
        systems[name] = load_pred(path, docs)[0]

    print(f"=== 규칙 기반 문맥 규칙 제거 실험 ({a.split}) ===")
    print(f"{'시스템':<20}{'strict F1':>10}{'partial R':>11}{'mask_rec':>10}")
    for k in ("rules", "rules_no_context"):
        if k in systems:
            s, _ = summarize(docs, systems[k])
            print(f"{k:<20}{s['strict_f1']:>10.3f}{s['partial_r']:>11.3f}{s['mask_rec']:>10.3f}")

    print(f"\n=== 템플릿별 mask_rec / strict F1 ({a.split}) ===")
    tables = {k: per_template(docs, P) for k, P in systems.items()}
    names = list(systems)
    print(f"{'템플릿 (문서/정답)':<26}" + "".join(f"{k[:16]:>18}" for k in names))
    for t in sorted(tables[names[0]]):
        n, g, _ = tables[names[0]][t]
        cells = []
        for k in names:
            s = tables[k][t][2]
            cells.append(f"{s['mask_rec']:.2f} / {s['strict_f1']:.2f}" if g else "    - / -")
        print(f"{t + f' ({n}/{g})':<26}" + "".join(f"{c:>18}" for c in cells))

    print("\n=== 템플릿 간 편차 (정답이 있는 템플릿만) ===")
    summary = {}
    for k in names:
        vals = [s["strict_f1"] for n, g, s in tables[k].values() if g]
        worst = min(((s["strict_f1"], t) for t, (n, g, s) in tables[k].items() if g))
        summary[k] = {"min": round(min(vals), 3), "max": round(max(vals), 3), "worst_template": worst[1]}
        print(f"{k:<20} strict F1 최저 {min(vals):.3f} ({worst[1]}) ~ 최고 {max(vals):.3f}")
    out = ROOT / "results" / f"template_experiment_{a.split}.json"
    out.write_text(json.dumps({k: {t: {"docs": n, "gold": g, **{m: round(v, 4) for m, v in s.items()}}
                                   for t, (n, g, s) in tables[k].items()} for k in names} | {"_summary": summary},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
