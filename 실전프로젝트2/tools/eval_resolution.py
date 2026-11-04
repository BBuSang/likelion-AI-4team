"""동명이인 구분(person_resolve) 효과를 DB·API 없이 바로 채점한다. 탐지는 tools/detect_cache.py 결과를 읽는다.

    python tools/eval_resolution.py dev            # data/week1/dev_*  (규칙 설계용)
    python tools/eval_resolution.py test           # data/week1/test_* (중간 확인)
    python tools/eval_resolution.py holdout        # 최종 확인 (규칙을 맞추는 데 쓰지 않음)
    python tools/eval_resolution.py dev --off      # 구분 끔 = 이름 문자열 기준 (v1)

출력: 토큰 일관성(미탐·쪼개짐·합쳐짐) + 언급별 판단 근거(basis) 분포. 사례는 outputs/resolution_<split>.csv
"""
import argparse
import collections
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from starter import config, pii_pipeline, person_resolve  # noqa: E402

SPLITS = {
    "dev": ("data/week1/dev_docs.jsonl", "data/week1/dev_persons_gold.jsonl"),
    "test": ("data/week1/test_docs.jsonl", "data/week1/test_persons_gold.jsonl"),
    "holdout": ("data/holdout_labeled.jsonl", "data/holdout_persons_gold.jsonl"),
    "synth2": ("data/synth2/synth2_labeled.jsonl", "data/synth2/synth2_persons_gold.jsonl"),
    "synth3": ("data/synth3/synth3_labeled.jsonl", "data/synth3/synth3_persons_gold.jsonl"),  # 최종 1회 (16장)
}


def load(path):
    with open(ROOT / path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split", choices=SPLITS)
    ap.add_argument("--off", action="store_true", help="동명이인 구분 끄기 (이름 문자열 기준)")
    a = ap.parse_args()
    data_path, gold_path = SPLITS[a.split]
    docs = load(data_path)
    cache = {r["id"]: r["spans"] for r in load(f"outputs/detect_{Path(data_path).stem}.jsonl")}
    spans = {d["id"]: cache[d["id"]] for d in docs}
    if not a.off:
        spans, _ = person_resolve.resolve(docs, spans)

    tag = f"{a.split}{'_off' if a.off else ''}"
    pred_path = config.OUT_DIR / f"pseudo_resolve_{tag}.jsonl"
    basis = collections.Counter()
    rows = []
    with open(pred_path, "w", encoding="utf-8") as f:
        for d in docs:
            _, recs = pii_pipeline.pseudonymize(d["text"], spans[d["id"]])
            f.write(json.dumps({"id": d["id"], "entities": [
                {k: r[k] for k in ("label", "start", "end", "token")} for r in recs]}, ensure_ascii=False) + "\n")
            for r in recs:
                if r["label"] == "person":
                    basis[r.get("basis", "off")] += 1
                    rows.append([d["id"], r["original"], r["normalized"], r.get("role"), r.get("aff"),
                                 r.get("basis", "off"), r.get("person_key"), r["token"]])
    with open(config.OUT_DIR / f"resolution_{tag}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "original", "normalized", "role", "aff", "basis", "person_key", "token"])
        w.writerows(rows)

    print(f"[{tag}] 판단 근거:", dict(basis.most_common()))
    subprocess.run([sys.executable, str(ROOT / "eval/check_token_consistency.py"), "--gold", str(ROOT / gold_path),
                    "--pred", str(pred_path), "--out", str(config.OUT_DIR / f"token_consistency_cases_{tag}.csv")],
                   check=True)


if __name__ == "__main__":
    main()
