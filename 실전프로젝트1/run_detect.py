"""탐지기를 실행해 run_compare.py 가 읽는 예측 파일(preds/<이름>_<split>.jsonl)을 만든다.

사용 예
  python -X utf8 run_detect.py --detector rules --split dev
  python -X utf8 run_detect.py --detector koelectra_pii --split dev
  python -X utf8 run_detect.py --detector llm --split dev --runs 3      # 일관성 측정용 3회 → _r1, _r2, _r3
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from common import ROOT, dedupe, load_docs, load_env, save_preds


def build(name):
    if name == "rules":
        from detectors.rule_based import RuleBasedDetector
        return RuleBasedDetector()
    if name == "llm":
        from detectors.llm import LLMDetector
        return LLMDetector()
    from detectors.hf_ner import MODELS as HF_MODELS
    from detectors.token_cls import HFNerDetector, TokenClsDetector
    if name in HF_MODELS:
        return HFNerDetector(name)  # 팀원 래퍼: veil, seongyeon, dlp, kpf, e5, atonlee
    return TokenClsDetector(name)   # pipeline 방식: koelectra_pii, piilot


def detect_one(det, doc):
    t0 = time.perf_counter()
    ents = det.detect(doc["text"])
    ms = (time.perf_counter() - t0) * 1000
    return {"id": doc["id"], "entities": dedupe(doc["text"], ents), "ms": round(ms, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, help="rules | koelectra_pii | piilot | llm | veil | seongyeon | dlp | kpf | e5 | atonlee")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "edge", "synth_dev", "synth_holdout"])
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8, help="LLM API 동시 호출 수")
    a = ap.parse_args()
    load_env()

    docs = load_docs(a.split)
    det = build(a.detector)
    if a.detector != "llm":
        det.detect(docs[0]["text"])  # 워밍업: 모델 로딩 시간은 문서당 처리 시간에서 뺀다

    for run in range(1, a.runs + 1):
        t0 = time.perf_counter()
        if a.detector == "llm":
            # API 대기 시간이 대부분이라 병렬 호출. ms 는 문서 1건의 응답 시간
            with ThreadPoolExecutor(a.workers) as ex:
                rows = list(ex.map(lambda d: detect_one(det, d), docs))
        else:
            rows = [detect_one(det, d) for d in docs]
        suffix = f"_r{run}" if a.runs > 1 else ""
        out = ROOT / "preds" / f"{a.detector}_{a.split}{suffix}.jsonl"
        save_preds(out, rows)
        print(f"[{a.detector}] {a.split} {len(docs)}건, 예측 {sum(len(r['entities']) for r in rows)}개, "
              f"전체 {time.perf_counter() - t0:.1f}초 → {out.relative_to(ROOT)}")
    if a.detector == "llm":
        print(f"[llm] 모델 {det.model}, 토큰 사용량 {det.usage}")


if __name__ == "__main__":
    main()
