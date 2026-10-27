"""generate_synthetic.py 결과를 정답 jsonl 로 바꾸고, 절반은 개선용(synth_dev)·절반은 holdout(synth_holdout)으로 나눈다.

문서 내용은 출력하지 않는다 (규칙 작성자가 holdout 을 보지 않도록 개수·형식 오류 통계만 보여 줌).
형식 오류가 있는 문서(닫히지 않은 표시, 모르는 라벨, 빈 값)는 정답이 불확실하므로 버리고 개수만 센다.

사용법
  python -X utf8 tools/parse_synthetic.py               # 1차: synth_dev / synth_holdout
  python -X utf8 tools/parse_synthetic.py --set fresh   # 2차 평가 세트: raw_fresh/ → synth_fresh_all.jsonl (나누지 않음)
  python -X utf8 tools/parse_synthetic.py --set hard    # 어려운 표기 세트: raw_hard_fixed/(표시 보정 후) → synth_hard_all.jsonl
결과: data/generated/synth_dev.jsonl, synth_holdout.jsonl, parse_report.json
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 맨 위 폴더의 common·run_compare·detectors 사용

from collections import Counter, defaultdict

from common import DATA, LABELS

HEADER = re.compile(r"^=== *(\S+?) *\| *(.+?) *\| *(.+?) *$")
MARK = re.compile(r"\[\[([^\[\]|]+?)\|([a-z_]+)\]\]")


def parse_doc(body):
    """[[값|라벨]] 을 지우고 정답 위치를 계산한다. 형식 오류면 (None, 이유)."""
    text, ents, last, pos = [], [], 0, 0
    for m in MARK.finditer(body):
        value, label = m.group(1), m.group(2)
        if label not in LABELS:
            return None, f"모르는 라벨({label})"
        if not value.strip() or value != value.strip():
            return None, "값 앞뒤 공백·빈 값"
        text.append(body[last:m.start()])
        pos += m.start() - last
        ents.append({"label": label, "text": value, "start": pos, "end": pos + len(value)})
        text.append(value)
        pos += len(value)
        last = m.end()
    text.append(body[last:])
    plain = "".join(text)
    if "[[" in plain or "]]" in plain:
        return None, "닫히지 않은 표시"
    return (plain, ents), None


def main(seed=42, kind="v1"):
    fresh = kind in ("fresh", "hard")  # 평가 전용 세트: 나누지 않고 전부 synth_<kind>_all 로
    work = DATA / "generated" / "_work"  # 생성 원문·중간 파일·보고서
    raw_dir = work / {"fresh": "raw_fresh", "hard": "raw_hard_fixed"}.get(kind, "raw")
    docs, errors, seen = [], Counter(), set()
    for path in sorted(raw_dir.glob("*.txt")):
        model = path.stem.rsplit("_", 1)[0]
        cur, body = None, []

        def flush():
            if cur is None:
                return
            text = "\n".join(body).strip()
            if not text:
                errors["빈 문서"] += 1
                return
            parsed, err = parse_doc(text)
            if err:
                errors[err] += 1
                return
            plain, ents = parsed
            doc_id = f"{ {'fresh': 'F', 'hard': 'H'}.get(kind, 'S') }{cur[0]}"
            if doc_id in seen:
                errors["중복 번호"] += 1
                return
            seen.add(doc_id)
            docs.append({"id": doc_id, "split": "", "doc_type": cur[2], "template": cur[1], "source_model": model,
                         "text": plain, "entities": ents, "hard_cases": []})

        for line in path.read_text(encoding="utf-8").splitlines():
            m = HEADER.match(line)
            if m:
                flush()
                cur, body = (m.group(1), m.group(2), m.group(3)), []
            elif cur is not None:
                body.append(line)
        flush()

    by_cat = defaultdict(list)
    for d in docs:
        by_cat[d["template"]].append(d)
    if fresh:
        parts = ((f"synth_{kind}_all", docs),)  # 표시 검수(qc_fresh.py) 후 synth_<kind>.jsonl
    else:
        # 카테고리별로 섞어서 절반씩 나눈다 (카테고리 비율 유지)
        rng = random.Random(seed)
        dev, hold = [], []
        for cat in sorted(by_cat):
            group = by_cat[cat]
            rng.shuffle(group)
            half = len(group) // 2
            dev += group[:half]
            hold += group[half:]
        parts = (("synth_dev", dev), ("synth_holdout", hold))
    for name, part in parts:
        # 평가 전용 세트의 _all 은 검수 전 중간 파일 → _work/, 1차 synth_dev/holdout 은 바로 채점용 → generated/
        with open((work if fresh else DATA / "generated") / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for d in part:
                d["split"] = name
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    labels = Counter(e["label"] for d in docs for e in d["entities"])
    report = {
        "문서": len(docs), "형식 오류로 버린 문서": dict(errors),
        **{name: len(part) for name, part in parts},
        "개인정보 없는 문서": sum(not d["entities"] for d in docs),
        "라벨별 개수": dict(labels.most_common()),
        "카테고리별 문서 수": {k: len(v) for k, v in sorted(by_cat.items())},
        "문체별 문서 수": dict(Counter(d["doc_type"] for d in docs)),
    }
    (work / f"parse_report{'_' + kind if fresh else ''}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(kind=sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "v1")
