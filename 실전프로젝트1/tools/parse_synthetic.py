"""generate_synthetic.py 결과를 정답 jsonl 로 바꾸고, 절반은 개선용(synth_dev)·절반은 holdout(synth_holdout)으로 나눈다.

문서 내용은 출력하지 않는다 (규칙 작성자가 holdout 을 보지 않도록 개수·형식 오류 통계만 보여 줌).
형식 오류가 있는 문서(닫히지 않은 표시, 모르는 라벨, 빈 값)는 정답이 불확실하므로 버리고 개수만 센다.

사용법
  python -X utf8 tools/parse_synthetic.py
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


def main(seed=42):
    raw_dir = DATA / "generated" / "raw"
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
            doc_id = f"S{cur[0]}"
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

    # 카테고리별로 섞어서 절반씩 나눈다 (카테고리 비율 유지)
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for d in docs:
        by_cat[d["template"]].append(d)
    dev, hold = [], []
    for cat in sorted(by_cat):
        group = by_cat[cat]
        rng.shuffle(group)
        half = len(group) // 2
        dev += group[:half]
        hold += group[half:]
    for name, part in (("synth_dev", dev), ("synth_holdout", hold)):
        with open(DATA / "generated" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for d in part:
                d["split"] = name
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    labels = Counter(e["label"] for d in docs for e in d["entities"])
    report = {
        "문서": len(docs), "형식 오류로 버린 문서": dict(errors),
        "synth_dev": len(dev), "synth_holdout": len(hold),
        "개인정보 없는 문서": sum(not d["entities"] for d in docs),
        "라벨별 개수": dict(labels.most_common()),
        "카테고리별 문서 수": {k: len(v) for k, v in sorted(by_cat.items())},
        "문체별 문서 수": dict(Counter(d["doc_type"] for d in docs)),
    }
    (DATA / "generated" / "parse_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
