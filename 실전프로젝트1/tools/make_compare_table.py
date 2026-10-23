"""탐지 방식 비교표 — 규칙 기반과 LLM을 팀 비교표 형식(strict P/R/F1, mask_rec, 미끼 문서 오탐)에 속도·일관성을 더해 정리.

채점은 run_compare.py 함수를 그대로 사용. 일관성은 같은 설정으로 여러 번 돌린 예측 파일이 있을 때만 계산.

표에 넣는 것은 19개 방식뿐: 규칙 기반·옛 규칙(비교용)·LLM 단독·규칙 기반 + LLM 하이브리드. 모두 후처리 없음.
테스트 오류를 먼저 보고 만든 결과(rules v5, 하이브리드 후처리 ②)와 후처리(형식 정리) 결과는 뺐음 (WORKLOG 63·67번).

사용법
    python tools/make_compare_table.py            # test
    python tools/make_compare_table.py dev
    python tools/make_compare_table.py test --rank      # 성능 순위로 정렬 (mask_rec → 재현율 → 정밀도 → 미끼 오탐 → 일관성 → 속도)
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_compare as rc  # noqa: E402

DECOY_TAGS = ("decoy_company_phone", "decoy_words", "decoy_numbers", "no_pii")
# (표에 쓸 이름, 예측 파일 이름 앞부분, 같은 설정 반복 실행 파일 뒷부분들)
# 후처리를 하나도 안 넣은 결과 (LLM은 기본 처리만, 하이브리드는 팀장님 규칙에 LLM을 후처리 없이 합침)
# 예전 rules v4 기반 하이브리드(preds/hybridraw_*)는 2026-10-23부터 목록에서 뺌 (WORKLOG 65번)
SYSTEMS = [
    ("규칙 기반", "rules", []),
    ("rules v1 (정규식만)", "rules_v1", []),
    ("rules v4 (정규식 + 이름·주소 규칙)", "rules_v4", []),
    ("rules v4 + Kiwi", "rules_v4_kiwi", []),
    ("llm DASH-002 + 검토", "llm_dash", []),
    ("llm HCX-007", "llm_hcx007", ["_run2", "_run3"]),
    ("llm gpt-4.1-mini", "llm_gpt41mini", ["_run2", "_run3"]),
    ("llm gpt-5.4-mini 추론 끄기", "llm_gpt54mini", ["_run2", "_run3"]),
    ("llm gpt-5.5 추론 끄기", "llm_gpt55", ["_run2", "_run3"]),
    ("llm gpt-5.6-luna 추론 끄기", "llm_gpt56luna", ["_run2", "_run3"]),
    ("llm gpt-6-luna 추론 끄기", "llm_gpt6luna", ["_run2", "_run3"]),
    ("hybrid: 규칙 기반 + HCX-007", "hybridteam_hcx007", ["_run2", "_run3"]),
    ("hybrid: 규칙 기반 + gpt-4.1-mini", "hybridteam_gpt41mini", ["_run2", "_run3"]),
    ("hybrid: 규칙 기반 + gpt-5.4-mini", "hybridteam_gpt54mini", ["_run2", "_run3"]),
    ("hybrid: 규칙 기반 + gpt-5.5", "hybridteam_gpt55", ["_run2", "_run3"]),
    ("hybrid: 규칙 기반 + gpt-5.6-luna", "hybridteam_gpt56luna", ["_run2", "_run3"]),
    ("hybrid: 규칙 기반 + gpt-6-luna", "hybridteam_gpt6luna", ["_run2", "_run3"]),
    ("hybrid_union: 규칙 기반 ∪ GPT 3개", "hybridteam_gpt3_union", ["_run2", "_run3"]),
    ("hybrid_vote: 규칙 기반 + GPT 2개 이상 공통", "hybridteam_gpt3_vote", ["_run2", "_run3"]),
]
# 예전 이름 (eval_customer.py가 씀)
SYSTEMS_NO_POST = SYSTEMS


def decoy_fp(docs, preds):
    n = 0
    for d, P in zip(docs, preds):
        if any(t in d.get("hard_cases", []) for t in DECOY_TAGS):
            n += sum(1 for p in P if not any(rc.overlap(g, p) for g in d["gold"]))
    return n


def load(path, docs):
    with contextlib.redirect_stdout(io.StringIO()):  # 7종 밖 라벨 경고 등은 표에서 숨김
        return rc.load_pred(path, docs)


def table(split, systems=None, rank=False):
    docs = rc.load_gold(f"data/pii_synth_finance_{split}.jsonl")
    rows = []
    for name, key, reruns in systems or SYSTEMS:
        path = f"preds/{key}_{split}.jsonl"
        if not os.path.exists(path):
            continue
        P, ms = load(path, docs)
        s, _ = rc.summarize(docs, P)
        runs = [P] + [load(f"preds/{key}_{split}{r}.jsonl", docs)[0] for r in reruns
                      if os.path.exists(f"preds/{key}_{split}{r}.jsonl")]
        same = rc.consistency(runs)[1] if len(runs) >= 3 else (1.0 if key.startswith("rules") else None)
        cons = "-" if same is None else f"{same:.1%}" + (" (규칙은 매번 같음)" if key.startswith("rules") else "")
        rows.append((name, s, decoy_fp(docs, P), ms, same, cons))
    if rank:  # 이 프로젝트의 우선순위: 안 새어 나가기 → 안 놓치기 → 엉뚱한 것 안 잡기 → 미끼 → 일관성 → 속도
        rows.sort(key=lambda r: (-round(r[1]["mask_rec"], 3), -round(r[1]["strict_r"], 3), -round(r[1]["strict_p"], 3),
                                 r[2], -(r[4] or 0), r[3]))
    head = "| 순위 " if rank else ""
    lines = [head + "| 방식 | mask_rec | strict R | strict P | strict F1 | 오탐(미끼 문서) | ms/문서 | 일관성 (3회 모두 같은 문서) |",
             ("| --- " if rank else "") + "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i, (name, s, fp, ms, _, cons) in enumerate(rows, 1):
        lines.append((f"| {i} " if rank else "") + f"| {name} | {s['mask_rec']:.3f} | {s['strict_r']:.3f} | "
                     f"{s['strict_p']:.3f} | {s['strict_f1']:.3f} | {fp} | {ms:,.1f} | {cons} |")
    return "\n".join(lines)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    split = args[0] if args else "test"
    print(table(split, SYSTEMS, rank="--rank" in sys.argv))
