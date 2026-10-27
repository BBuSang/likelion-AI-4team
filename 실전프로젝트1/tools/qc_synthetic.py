"""생성 데이터(synth_dev)의 정답 품질 검수 목록을 만든다. holdout 에는 쓰지 않는다.

생성 모델이 개인정보를 쓰고도 [[ ]] 표시를 빠뜨리면, 탐지기가 맞게 잡아도 오탐으로 채점된다.
그래서 사람이 확인할 후보를 뽑는다.
1) 표시 누락 후보: 서로 다른 두 탐지기(규칙·모델)가 같은 곳을 잡았는데 정답과 겹치지 않는 것
2) 라벨·모양 불일치: 정답 값의 모양이 라벨과 안 맞는 것 (전화인데 숫자 7개 미만, 이메일인데 @ 없음 등)

사용법
  python -X utf8 tools/qc_synthetic.py --pred preds/rules_synth_dev.jsonl --pred preds/koelectra_pii_synth_dev.jsonl
결과: results/synth_qc_candidates.csv (엑셀로 열어 '판정' 칸을 채우면 정답 오류율을 계산할 수 있음)
"""
import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 맨 위 폴더의 common·run_compare·detectors 사용


from common import DATA, ROOT
from run_compare import ctx, load_gold, load_pred


def shape_problem(label, value):
    digits = len(re.sub(r"\D", "", value))
    if label == "email" and "@" not in value:
        return "이메일인데 @ 없음"
    if label == "phone_number" and not 8 <= digits <= 13:
        return f"전화인데 숫자 {digits}개"
    if label == "resident_registration_number" and digits + value.count("*") != 13:
        return f"주민번호인데 숫자 {digits}개"
    if label == "card_number" and not 14 <= digits + value.count("*") <= 16:
        return f"카드인데 숫자 {digits}개"
    if label == "account_number" and not 9 <= digits <= 16:
        return f"계좌인데 숫자 {digits}개"
    if label == "person" and (re.search(r"\d", value) or len(value) > 20):
        return "이름에 숫자가 있거나 너무 김"
    if label == "address" and len(value) < 6:
        return "주소가 너무 짧음"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", action="append", required=True)
    a = ap.parse_args()
    docs = load_gold(DATA / "generated" / "synth_dev.jsonl")
    systems = [load_pred(p, docs)[0] for p in a.pred]

    rows = []
    for i, d in enumerate(docs):
        T = d["text"]
        for g in d["gold"]:
            why = shape_problem(g["label"], g["text"])
            if why:
                rows.append([d["id"], d["template"], "라벨·모양 불일치", g["label"], g["text"], why,
                             ctx(T, g["start"], g["end"], 25), ""])
        # 모든 탐지기가 겹치게 잡았고 정답과는 안 겹치는 구간
        first, others = systems[0][i], [s[i] for s in systems[1:]]
        for p in first:
            if any(p["start"] < g["end"] and g["start"] < p["end"] for g in d["gold"]):
                continue
            if all(any(p["start"] < q["end"] and q["start"] < p["end"] for q in o) for o in others):
                rows.append([d["id"], d["template"], "표시 누락 후보", p["label"], p["text"],
                             "탐지기 모두 잡았지만 정답에 없음", ctx(T, p["start"], p["end"], 25), ""])

    out = ROOT / "results" / "synth_qc_candidates.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "category", "종류", "label", "value", "이유", "context", "판정(정답오류/탐지기오류)"])
        w.writerows(rows)
    print(f"검수 후보 {len(rows)}건 (표시 누락 후보 {sum(r[2] == '표시 누락 후보' for r in rows)}, "
          f"라벨·모양 불일치 {sum(r[2] == '라벨·모양 불일치' for r in rows)}) → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
