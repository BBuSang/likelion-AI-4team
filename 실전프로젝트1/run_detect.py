"""탐지기 실행기 (run_detect.py).

detectors/<이름>.py 의 detect(text)를 문서마다 실행하고, run_compare.py 예측 형식으로 저장합니다.
탐지기에는 원문 text만 넘깁니다 (정답 필드는 넘기지 않음).

사용법
    python run_detect.py --detector rules --split dev
    python run_detect.py --detector llm --split dev --runs 3 --workers 8  # 반복 실행 (일관성 측정용)
    python run_detect.py --detector rules --data data/holdout.jsonl     # 다른 데이터 파일

출력
    preds/<이름>_<split>.jsonl   (--runs 2 이상이면 preds/<이름>_<split>_r1.jsonl, _r2.jsonl ...)
"""
import argparse
import importlib
import json
import os
import sys
import time
from collections import Counter

from detectors.base import postprocess, surname_title
from run_compare import load_jsonl

DATA = {"dev": "data/pii_synth_finance_dev.jsonl", "test": "data/pii_synth_finance_test.jsonl"}


def load_texts(path):
    """문서 id와 원문만 꺼냄."""
    return [{"id": d["id"], "text": d["text"]} for d in load_jsonl(path)]


def run(detect, docs, workers=1, surname_title_on=False):
    """문서별 탐지. workers > 1이면 문서를 동시에 처리 (API 탐지기용, 문서별 시간은 각자 측정).
    surname_title_on이면 성+직함('김 대리') 구간을 person으로 더함 (팀 기준 변경 실험용)."""
    def one(d):
        t0 = time.perf_counter()
        ents = detect(d["text"])
        if surname_title_on:
            ents = ents + surname_title(d["text"])
        ents = postprocess(d["text"], ents)
        ms = (time.perf_counter() - t0) * 1000
        return {"id": d["id"], "entities": ents, "ms": round(ms, 2)}

    if workers <= 1:
        return [one(d) for d in docs]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, docs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, help="detectors/ 안의 모듈 이름 (예: empty, rules)")
    ap.add_argument("--split", choices=list(DATA), default="dev")
    ap.add_argument("--data", help="데이터 파일 직접 지정 (지정하면 --split 대신 파일 이름으로 저장)")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--out", default="preds")
    ap.add_argument("--workers", type=int, default=1, help="동시 처리 문서 수 (LLM 등 API 탐지기용)")
    ap.add_argument("--surname-title", action="store_true",
                    help="성+직함('김 대리')도 person으로 잡음. 출력 파일 이름에 _st가 붙음")
    a = ap.parse_args()
    if a.surname_title:
        os.environ["PII_SURNAME_TITLE"] = "1"  # LLM 프롬프트도 이 기준으로 바뀜

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    path = a.data or DATA[a.split]
    tag = (os.path.splitext(os.path.basename(a.data))[0] if a.data else a.split) + ("_st" if a.surname_title else "")
    docs = load_texts(path)
    module = importlib.import_module(f"detectors.{a.detector}")
    if hasattr(module, "load"):
        t0 = time.perf_counter()
        module.load()
        print(f"로딩 시간: {time.perf_counter() - t0:.1f}초 (문서별 시간에서 제외)")

    os.makedirs(a.out, exist_ok=True)
    outs = []
    for i in range(1, a.runs + 1):
        rows = run(module.detect, docs, a.workers, a.surname_title)
        out = os.path.join(a.out, f"{a.detector}_{tag}" + (f"_r{i}" if a.runs > 1 else "") + ".jsonl")
        with open(out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        labels = Counter(e["label"] for r in rows for e in r["entities"])
        avg = sum(r["ms"] for r in rows) / len(rows)
        print(f"[{out}] 문서 {len(rows)}건, 탐지 {sum(labels.values())}개, 평균 {avg:.1f}ms/문서"
              f" (1만 건 약 {avg * 10000 / 60000:.1f}분) {dict(labels.most_common())}")
        outs.append(out)

    gold = path.replace("\\", "/")
    print(f"\n채점: python run_compare.py --gold {gold} --pred {a.detector}={','.join(outs)}")


if __name__ == "__main__":
    main()
