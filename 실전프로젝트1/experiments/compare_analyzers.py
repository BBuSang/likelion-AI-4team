"""규칙 기반의 형태소 분석기만 바꿔 끼워 비교한다 (Kiwi · Komoran · Kkma · Hannanum · Okt).

정규식(번호·이메일·주소)은 분석기와 상관없으므로 차이는 이름(person)에서만 난다.
두 가지 설정으로 본다.
- full : 문맥 규칙(필드명·호칭·직급·서명·이름 퍼뜨리기) 포함 = 실제 사용 설정
- core : 문맥 규칙 끔 = 분석기의 '명사 덩어리' 판단만으로 이름을 찾을 때 → 분석기 자체의 차이가 더 잘 드러남

사용법
  python -X utf8 experiments/compare_analyzers.py                  # dev·test·edge 전부, 분석기 5종
  python -X utf8 experiments/compare_analyzers.py --analyzers kiwi,komoran --splits dev
결과: preds/analyzers/, results/analyzer_comparison.csv, 화면 출력
"""
import argparse
import csv
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 맨 위 폴더의 common·run_compare·detectors 사용


from common import DATA, ROOT, dedupe, load_docs, save_preds
from detectors.analyzers import ANALYZERS, Analyzer
from detectors.rule_based import RuleBasedDetector
from run_compare import load_gold, load_pred, prf, summarize

GOLD = {"dev": "pii_synth_finance_dev.jsonl", "test": "pii_synth_finance_test.jsonl", "edge": "edge_cases.jsonl"}
SAMPLES = ["윤선영은", "윤 선영 고객", "신승현 과장", "손하은 책임", "최채원 고객은", "노성호님은",
           "남궁다은", "한결 플러스 카드", "배송지", "서비스", "연회비", "인사팀"]


def run(det, docs):
    rows = []
    det.detect(docs[0]["text"])  # 워밍업
    for d in docs:
        t0 = time.perf_counter()
        ents = det.detect(d["text"])
        rows.append({"id": d["id"], "entities": dedupe(d["text"], ents),
                     "ms": round((time.perf_counter() - t0) * 1000, 1)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyzers", default=",".join(ANALYZERS))
    ap.add_argument("--splits", default="dev,test,edge")
    a = ap.parse_args()
    names, splits = a.analyzers.split(","), a.splits.split(",")

    print("=== 분석기별 쪼개기 예시 (조각/품사, *는 사전 밖 단어) ===")
    for n in names:
        an = Analyzer(n)
        print(f"[{n}]")
        for w in SAMPLES:
            print(f"  {w:<14} " + " ".join(f"{t.form}/{t.tag}{'' if t.known else '*'}" for t in an.tokenize(w)))

    out_rows = []
    print("\n=== 성능 (person 은 partial 기준) ===")
    print(f"{'분석기':<10}{'설정':<6}{'split':<6}{'strict F1':>10}{'mask_rec':>10}{'이름 R':>8}{'이름 P':>8}"
          f"{'이름만 R':>9}{'ms/문서':>9}")
    for n in names:
        for ctx in (True, False):
            det = RuleBasedDetector(context_rules=ctx, analyzer=n)
            for sp in splits:
                docs = load_docs(sp)
                rows = run(det, docs)
                path = ROOT / "preds" / "analyzers" / f"rules_{n}_{'full' if ctx else 'core'}_{sp}.jsonl"
                save_preds(path, rows)
                gold = load_gold(DATA / GOLD[sp])
                P, ms = load_pred(path, gold)
                s, c = summarize(gold, P)
                pr = c["person"]
                p_p, p_r, _ = prf(pr["partial_tp"], pr["pred"], pr["gold"])
                given = [(g, Pd) for d, Pd in zip(gold, P) for g in d["gold"] if g.get("form") == "given"]
                given_r = (sum(any(p["label"] == "person" and p["start"] < g["end"] and g["start"] < p["end"]
                                   for p in Pd) for g, Pd in given) / len(given)) if given else None
                row = {"analyzer": n, "setting": "full" if ctx else "core", "split": sp,
                       "strict_f1": round(s["strict_f1"], 4), "mask_rec": round(s["mask_rec"], 4),
                       "person_recall": round(p_r, 4), "person_precision": round(p_p, 4),
                       "given_recall": round(given_r, 4) if given_r is not None else "",
                       "ms_per_doc": round(ms, 1)}
                out_rows.append(row)
                g = f"{given_r:.3f}" if given_r is not None else "-"
                print(f"{n:<10}{row['setting']:<6}{sp:<6}{s['strict_f1']:>10.3f}{s['mask_rec']:>10.3f}"
                      f"{p_r:>8.3f}{p_p:>8.3f}{g:>9}{ms:>9.1f}", flush=True)

    out = ROOT / "results" / "analyzer_comparison.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
