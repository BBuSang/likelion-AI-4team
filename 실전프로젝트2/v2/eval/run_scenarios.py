"""시연 시나리오 실행 + 점검 (plan_v2 P5·P6). 결과: v2/outputs/scenarios.json

    python -X utf8 v2/eval/run_scenarios.py [--reasoning none]

외부 전송 텍스트(가명 질의 + 문맥)는 토큰을 지운 뒤 같은 탐지기로 다시 검사 (원문이 남았는지).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2))
from chat import TOKEN_RE, Chat  # noqa: E402
from pseudo import pipeline  # noqa: E402

SCENARIOS = [
    ("① 이름 검색", "윤선영 고객 상담 내역 알려줘"),
    ("② 띄어쓰기 변형", "윤 선영 님 한도 상향 심사 결과 알려줘"),
    ("③ 번호 형식 변형", "01009335186 번호 쓰는 고객 관련 기록 찾아줘"),
    ("④ 동명이인 (고객)", "김민준 고객 관련 문서 보여줘"),
    ("⑤ 개인정보 없는 질문", "보이스피싱 피해 접수 기록 알려줘"),
    ("⑥ 동명이인 (직원)", "직원 김민준 관련 문서 알려줘"),
    ("⑦ 주소 일부", "대구 수성구 고객 문서 알려줘"),
    ("⑧ 처리 상태", "남궁나연 고객 아직 처리 안 끝난 불만 있어?"),
    ("⑨ 문서에 없는 내용", "주택담보대출 금리인하요구권 안내 문서 알려줘"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoning", default=None)
    ap.add_argument("--out", default="scenarios.json")
    a = ap.parse_args()
    bot = Chat()
    out = []
    for name, q in SCENARIOS:
        r = bot.answer(q, a.reasoning)
        sent = TOKEN_RE.sub(" ", r["masked_query"] + "\n" + r["context"])
        recheck = [{"label": e["label"], "text": sent[e["start"]:e["end"]]} for e in pipeline.detect(sent)]
        out.append({"name": name, "query": q, "masked_query": r["masked_query"], "docs": r["docs"],
                    "labels": r["labels"], "context": r["context"], "answer": r["answer"], "checks": r["checks"],
                    "recheck_detected": recheck, "search_sec": round(r["search_sec"], 2), "llm_sec": round(r["llm_sec"], 1)})
        c = r["checks"]
        print(f"\n{name} | {q}\n  가명: {r['masked_query']}\n  문서: {r['docs']}  인물표시: {list(r['labels'].values())}")
        print("  답변:\n    " + r["answer"].replace("\n", "\n    "))
        print(f"  점검: 인용 {c['cited_docs']} · 지어낸 토큰 {len(c['unknown_tokens'])} · 줄 대응 오류 {len(c['row_mismatch'])} "
              f"{c['row_mismatch']} · 재검사 탐지 {recheck} · 검색 {r['search_sec']:.1f}초 · 답변 {r['llm_sec']:.1f}초")
    (V2 / "outputs" / a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n답변 평균 {statistics.mean(x['llm_sec'] for x in out):.1f}초 · 검색 평균 "
          f"{statistics.mean(x['search_sec'] for x in out):.2f}초 · 지어낸 토큰 합 "
          f"{sum(len(x['checks']['unknown_tokens']) for x in out)} · 줄 대응 오류 합 "
          f"{sum(len(x['checks']['row_mismatch']) for x in out)} · 재검사 탐지 합 {sum(len(x['recheck_detected']) for x in out)}")


if __name__ == "__main__":
    main()
