"""기획안 5-7 판단 기준 점검 (화면을 Streamlit AppTest 로 실제 실행)

    python 최종프로젝트/금융권 법령 RAG/demo/check_demo.py

① 예시 질문 12개 버튼을 하나씩 눌러 답변·근거가 오류 없이 나오는지
② 표시된 출처가 `법령명 제O조(조문제목)[ 제O항…]` 형식인지 (+ 링크 모양)
③ 화면의 후보 목록 = 같은 질문의 answer() 후보 목록
④ 질문 → 답변 10초 이내
⑤ 범위 밖 질문에 '관련 조문을 찾지 못했습니다'
결과 → eval/demo_check.json
"""
import json
import re
import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retriever.answer import NOT_FOUND, answer_candidates  # noqa: E402

APP = str(ROOT / "demo" / "app.py")
CITE = re.compile(r"^.+ (제\d+(-\d+)?조(의\d+)?|\d+\.)(\(.+?\))?( (제\d+항|제\d+호)(~(제\d+항|제\d+호))?( 제\d+호(~제\d+호)?)?)?$")


def main():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    buttons = [b for b in at.sidebar.button]
    print("예시 버튼", len(buttons), flush=True)
    rows = []
    for i in range(len(buttons)):
        q = at.sidebar.button[i].label
        t0 = time.perf_counter()
        at.sidebar.button[i].click().run()
        wall = time.perf_counter() - t0
        r = at.session_state["last"]
        ok_box = bool(at.success) and not at.error
        cites = [s["cite"] or s["title"] for s in r["sources"]]
        bad_cite = [c for c in cites if not CITE.match(c)]
        bad_url = [s["url"] for s in r["sources"] if not s["url"].startswith("https://www.law.go.kr/")]
        same = [c["title"] for c in r["candidates"]] == [a["title"] for a in answer_candidates(q)]
        rows.append({"질문": q, "오류": bool(at.exception) or bool(r["error"]), "답변 표시": ok_box, "근거 수": len(cites),
                     "출처": cites, "출처 형식 어긋남": bad_cite, "링크 어긋남": bad_url, "후보 = answer()": same,
                     "답변 초": r["timing"]["total"], "화면 다시 그리기 포함 초": round(wall, 2)})
        print(f"{i + 1:2}. {r['timing']['total']:.2f}초 근거 {len(cites)} 형식OK={not bad_cite} 같음={same} | {q}", flush=True)
    at.session_state["q"] = "내일 서울 날씨 어때?"
    at.button[0].click().run()  # 폼 제출 버튼
    na = at.session_state["last"]
    res = {"① 12개 오류 없이 답변·근거": all(not x["오류"] and x["답변 표시"] and x["근거 수"] > 0 for x in rows),
           "② 출처 형식·링크": all(not x["출처 형식 어긋남"] and not x["링크 어긋남"] for x in rows),
           "③ 화면 후보 = answer()": all(x["후보 = answer()"] for x in rows),
           "④ 최대 답변 초": max(x["답변 초"] for x in rows), "④ 10초 이내": max(x["답변 초"] for x in rows) <= 10,
           "⑤ 범위 밖 질문": {"답변": na["answer"], "LLM 안 부름": na.get("below_min_score"), "통과": na["answer"] == NOT_FOUND},
           "질문별": rows}
    json.dump(res, open(ROOT / "eval" / "demo_check.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "질문별"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
