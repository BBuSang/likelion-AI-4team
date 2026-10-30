"""HCX-007 추론 설정(기본 low vs none)을 시연 시나리오 5개로 비교합니다: 응답 시간 · 사실 확인 · 말투.

    python -X utf8 eval/compare_reasoning.py            (실전프로젝트2 폴더에서, DB 적재 후)

사실 확인은 시나리오마다 근거 문서에서 직접 확인한 값(VOC 번호, 금액, 사건 번호)이 답변에 있는지, 틀린 값이 없는지 봅니다.
결과: outputs/reasoning_compare.json (답변 전문 포함, 사람이 다시 읽어 볼 것)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from starter import config, pii_pipeline  # noqa: E402
from starter.chat import answer  # noqa: E402

# (번호, 질문, 있어야 할 값, 있으면 틀린 값)
CASES = [
    ("①", "윤선영 고객 상담 내역 알려줘", ["3110", "8673", "5585"], []),
    ("②", "윤 선영 님 한도 상향 심사 결과 알려줘", ["300", "200"], []),  # 300만 → 1200만(1,200만)
    ("③", "01009335186 번호 쓰는 고객 관련 기록 찾아줘", ["H0113", "8090", "3798"], []),
    # 8293(H0023)·5744(H0159)는 같은 VOC 목록에 있는 다른 고객(임우진·남궁나연)의 건
    ("④", "김민준 고객 관련 문서 보여줘", ["5693", "8090"], ["8293", "5744"]),
    ("⑤", "보이스피싱 피해 접수 기록 알려줘", ["921245", "280674", "623736"], []),
]
SETTINGS = [("low (기본)", None), ("none", "none")]
TONE_BAD = ["귀하", "고객님", "[문서]"]


def main():
    pii_pipeline._get_detector()
    pii_pipeline.detect("워밍업 윤선영")
    rows = []
    for name, effort in SETTINGS:
        for no, q, must, wrong in CASES:
            t = time.perf_counter()
            r = answer(q, k=3, reasoning=effort or "")
            sec = time.perf_counter() - t
            a = r["answer"] or ""
            row = {"설정": name, "no": no, "질문": q, "초": round(sec, 1), "LLM초": round(r.get("llm_sec", 0), 1),
                   "빠진 값": [m for m in must if m not in a], "틀린 값": [w for w in wrong if w in a],
                   "말투 문제": [b for b in TONE_BAD if b in a], "인용": r["cited_docs"],
                   "지어낸 토큰": r["unknown_tokens"], "글자 수": len(a), "답변": a}
            rows.append(row)
            print(f"[{name:9}] {no} 전체 {sec:5.1f}초 (LLM {row['LLM초']:4.1f}초)  빠진값 {row['빠진 값'] or '-'}  틀린값 {row['틀린 값'] or '-'}  "
                  f"말투 {row['말투 문제'] or '-'}  인용 {r['cited_docs']}  지어낸토큰 {len(r['unknown_tokens'])}  {len(a)}자")
    for name, _ in SETTINGS:
        rs = [x for x in rows if x["설정"] == name]
        print(f"\n{name}: LLM 평균 {sum(x['LLM초'] for x in rs) / len(rs):.1f}초 (전체 {sum(x['초'] for x in rs) / len(rs):.1f}초) · 사실 오류(빠짐+틀림) "
              f"{sum(len(x['빠진 값']) + len(x['틀린 값']) for x in rs)}건 · 말투 문제 {sum(bool(x['말투 문제']) for x in rs)}건")
    out = config.OUT_DIR / "reasoning_compare.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
