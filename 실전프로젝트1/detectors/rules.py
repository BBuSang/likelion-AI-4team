"""규칙 기반 탐지기 — 팀장님 규칙 기반(rule_based.py)을 부르는 입구 (2026-10-23부터 이것이 '규칙 기반').

출처: 팀 저장소 BBuSang/likelion-AI-4team, hosung 브랜치, 커밋 482466b
      실전프로젝트1/detectors/rule_based.py (+ detectors/analyzers.py, postprocess.py) 를 그대로 복사해 씀.
      팀 코드는 고치지 않는다. 바뀌면 팀 저장소에서 다시 복사한다.
이전에 우리가 만든 규칙(v1~v5)은 detectors/old/ 에 보관 (rules_v4.py, rules_v5.py 등).

사용법 (전과 같음)
    python detectors/rules.py data/pii_synth_finance_dev.jsonl preds/rules_dev.jsonl
    python detectors/rules.py data/team/synth_holdout.jsonl preds/rules_synth_holdout.jsonl

팀 run_detect.py 와 같은 방식으로 저장: 첫 문서로 한 번 먼저 돌려(Kiwi 사전 로딩) 처리 시간에서 로딩을 빼고,
결과는 앞뒤 공백 제거 + 중복 제거(dedupe) 후 저장 → 팀장님 예측 파일과 결과가 같다.
"""
import json
import os
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)  # 'detectors' 패키지와 postprocess.py 를 찾기 위해

from detectors.rule_based import RuleBasedDetector  # noqa: E402

LABELS = ["person", "phone_number", "email", "address",
          "resident_registration_number", "account_number", "card_number"]
_DETECTOR = None


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = RuleBasedDetector()
    return _DETECTOR


def dedupe(text, ents):
    """팀 common.dedupe 와 같음: 앞뒤 공백 제거 + 중복 제거 (채점기는 중복 예측을 오탐으로 셈)."""
    seen, out = set(), []
    for e in ents:
        s, t = e["start"], e["end"]
        while s < t and text[s].isspace():
            s += 1
        while t > s and text[t - 1].isspace():
            t -= 1
        k = (e["label"], s, t)
        if s < t and e["label"] in LABELS and k not in seen:
            seen.add(k)
            out.append({**e, "start": s, "end": t, "text": text[s:t]})
    return sorted(out, key=lambda e: (e["start"], e["end"]))


def detect(text):
    """다른 도구(tools/check_variants.py 등)에서 쓰는 함수: 문서 한 건 → [{label, start, end, text}]"""
    return dedupe(text, _detector().detect(text))


def run(in_path, out_path):
    docs = [json.loads(l) for l in open(in_path, encoding="utf-8") if l.strip()]
    if docs:
        detect(docs[0]["text"])  # 워밍업: 사전 로딩 시간은 문서당 처리 시간에서 뺀다
    with open(out_path, "w", encoding="utf-8") as out:
        for d in docs:
            t0 = time.perf_counter()
            ents = detect(d["text"])
            ms = (time.perf_counter() - t0) * 1000
            out.write(json.dumps({"id": d["id"], "entities": ents, "ms": round(ms, 1)}, ensure_ascii=False) + "\n")
    print(f"저장: {out_path}")


if __name__ == "__main__":
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(paths[0], paths[1])
