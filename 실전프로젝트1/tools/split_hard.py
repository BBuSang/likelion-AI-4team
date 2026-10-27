"""synth_hard 를 개선용(synth_hard_dev)·평가용(synth_hard_eval) 반으로 나눈다 (문서 유형·문체별로 섞어서 반씩).

개선은 synth_hard_dev 오류만 보고 하고, synth_hard_eval 은 점수만 확인한다.
사용법
  python -X utf8 tools/split_hard.py
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 맨 위 폴더의 common 사용

from common import DATA


def main(seed=42):
    gen = DATA / "generated"
    docs = [json.loads(l) for l in open(gen / "_work" / "synth_hard.jsonl", encoding="utf-8") if l.strip()]
    groups = defaultdict(list)
    for d in docs:
        groups[(d["template"], d["doc_type"])].append(d)
    rng = random.Random(seed)
    dev, ev = [], []
    for key in sorted(groups):
        g = groups[key]
        rng.shuffle(g)
        dev += g[:len(g) // 2]
        ev += g[len(g) // 2:]
    for name, part in (("synth_hard_dev", dev), ("synth_hard_eval", ev)):
        with open(gen / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for d in part:
                f.write(json.dumps({**d, "split": name}, ensure_ascii=False) + "\n")
        print(name, len(part), "건, 정답", sum(len(d["entities"]) for d in part), "개")


if __name__ == "__main__":
    main()
