"""문서 생성: 질문과_답변_예시.md (저장된 data/eval/f4_answer_eval.json에서). 실행: python pipeline/doc_qa_examples.py
(원래 임시 폴더에서 만든 스크립트를 보관용으로 옮긴 것이다. 경로가 이 PC 기준 절대 경로로 들어 있다.)"""
import contextlib
import io
import json
import sys

sys.path.insert(0, r"C:\Users\USER\Desktop\likelion-AI-4team\최종프로젝트\금융당국 회신 선례 RAG\pipeline")
sys.stdout.reconfigure(encoding="utf-8")
import rag
import time_marks as tm
from articles_lookup import Lookup, show

ROOT = r"C:\Users\USER\Desktop\likelion-AI-4team\최종프로젝트\금융당국 회신 선례 RAG"
E = ROOT + r"\data\eval"
res = json.load(open(E + r"\f4_answer_eval.json", encoding="utf-8"))
items = {(x["set"], x["qid"]): x for x in res["items"]}
questions = {}
for name in ("rewrite_150", "dev1", "dev2", "dev3", "dev4", "test1", "test2", "test3", "test4", "final_test", "unanswerable_20"):
    for l in open(f"{E}\\{name}.jsonl", encoding="utf-8"):
        x = json.loads(l)
        questions[(name, x["qid"])] = x["question"]

ok = lambda x: x["judge"] and x["judge"]["correctness"] == "일치" and x["judge"]["grounded"] == "충실" and x["rank"] == 1
first = lambda sets, tag, cond=ok: next(x for x in res["items"] if x["set"] in sets and (tag is None or x["tag"] == tag) and cond(x))
good = [
    ("쉬운 질문 (합성 질의셋)", first(("rewrite_150",), None)),
    ("규제 효력형 (dev2)", first(("dev2",), "규제 효력형", lambda x: x["judge"] and x["judge"]["correctness"] == "일치")),
    ("우회 서술형 (dev3) 법령 이름 없이 상황만 설명", first(("dev3",), "우회 서술형")),
    ("구어체형 (dev3)", first(("dev3",), "구어체형")),
    ("경계 조건형 (dev3)", first(("dev3",), "경계 조건형", lambda x: x["judge"] and x["judge"]["correctness"] == "일치")),
    ("복합 쟁점형 (dev4)", first(("dev4",), "복합 쟁점형", lambda x: x["judge"] and x["judge"]["correctness"] == "일치")),
    ("조건 포함형 (dev4) '2016년 이후 법령해석만' 조건이 문장에 든 질문", first(("dev4",), "조건 포함형", lambda x: x["judge"] and x["judge"]["correctness"] == "일치")),
    ("비조치 신청형 (dev4)", first(("dev4",), "비조치 신청형", lambda x: x["judge"] and x["judge"]["correctness"] == "일치")),
    ("비조치 신청형 (final_test)", first(("final_test",), "비조치 신청형")),
    ("조문+상황형 (final_test)", first(("final_test",), "조문+상황형")),
    ("구어체형 (test3)", first(("test3",), "구어체형", lambda x: x["judge"] and x["judge"]["correctness"] == "일치")),
]
notfound = [("답변불가 질문 1: 금융 회신과 무관", items[("unanswerable_20", "NA01")]), ("답변불가 질문 2", items[("unanswerable_20", "NA02")]),
            ("정답이 검색 상위 5 밖이라 \"찾지 못했습니다\" (dev4)", next(x for x in res["items"] if x["set"] == "dev4" and x["check"]["not_found"] and x["gold_key"])),
            ("정답이 검색 상위 5 밖이라 \"찾지 못했습니다\" (test4)", items[("test4", "TEST4-005")])]
problems = [("문제 1: 인용 사례에 없는 단서를 덧붙임 (dev1)", items[("dev1", "DEV050")]),
            ("문제 2: 한 답변 안에서 문단끼리 모순 (dev3)", items[("dev3", "DEV3-011")]),
            ("문제 3: 정답 사례가 6위라 전달되지 않아 한쪽 결론으로만 답함 (test4)", items[("test4", "TEST4-002")]),
            ("문제 4: 질문의 두 번째 쟁점을 놓침 (dev2)", items[("dev2", "DEV2-022")])]

conn = rag.connect()
lines = []
add = lines.append


def quote(text):
    return "\n".join("> " + l if l.strip() else ">" for l in text.strip().splitlines())


def entry(no, title, x):
    key = (x["set"], x["qid"])
    q = questions[key]
    add(f"### {no}. {title}")
    add(f"`{x['set']}` · `{x['qid']}`" + (f" · 유형 {x['tag']}" if x.get("tag") else "") + (f" · {x['gubun']}" if x.get("gubun") else ""))
    add("")
    add("**질문**")
    add(quote(q))
    add("")
    add("**시스템 답변** (실제 출력 그대로, 마지막 안내 문구는 시스템이 붙임)")
    add(quote(x["answer"]))
    add("")
    cited = x["check"]["cited"]
    if cited:
        marks = tm.load_marks(conn, cited)
        titles = {r["case_key"]: (r["title"], r["reply_date"], r["gubun"], r["answer_type"]) for r in conn.execute("select case_key, title, reply_date, gubun, answer_type from interp.cases where case_key = any(%s)", (cited,)).fetchall()}
        add("**답변이 근거로 쓴 사례와 시점 표시**")
        add("")
        add("| 사례 | 제목 | 회신일 | 시점 표시 |")
        add("|---|---|---|---|")
        for c in cited:
            t = titles[c]
            m = marks[c]
            detail = m["label"] if m["mark"] else "조문 인용 없음"
            add(f"| {c} | {(t[0] or '').strip()[:60]} | {t[1]} | {m['mark'] or '-'} {detail} |")
        add("")
    if x["gold_key"]:
        g = conn.execute("select title from interp.cases where case_key = %s", (x["gold_key"],)).fetchone()
        rank = f"{x['rank']}위" if x["rank"] else "10위 밖"
        j = x["judge"]
        add(f"**평가용 정보** — 이 질문의 정답 사례: {x['gold_key']} “{(g['title'] or '').strip()[:60]}” / 검색 순위 {rank}" + (f" / LLM 심사: 정확도 {j['correctness']}, 충실도 {j['grounded']} — {j['reason']}" if j else ""))
        add("")
    else:
        add(f"**평가용 정보** — 정답이 없는 질문(답변불가 세트). 시스템이 \"찾지 못했습니다\"라고 답해야 정상이다. 검색 최고 점수는 아래 참고.")
        add("")
    add("---")
    add("")


add("# 질문과 답변 예시 (실제 실행 결과)")
add("")
add("2026-11-10에 LLM 포함 RAG(KURE-v1 검색 → 상위 5개 묶음 → gpt-6-luna 답변)를 돌린 결과에서 뽑은 실제 질문과 답변이다. 글은 저장된 결과 파일(`data/eval/f4_answer_eval.json`)을 코드로 그대로 옮겼다. 평가 방법과 점수는 [질문_방식과_결과_표시.md](질문_방식과_결과_표시.md)와 [worklog.md](worklog.md) 65번을 본다.")
add("")
add("**읽는 법**")
add("- 질문은 실제 직원이 쓴 것이 아니라, 회신 원문을 보고 LLM(OpenAI)이 업무 말투로 만든 평가용 질문이다. 질문마다 정답 사례가 하나 정해져 있다.")
add("- `[L2587]`처럼 대괄호 안은 근거 사례번호다. L은 법령해석, O는 비조치의견서다.")
add("- 시점 표시: ⚠️ 회신일 이후 조문 개정·폐지, ✅ 이후 개정 없음, 🔶 규정 개정됨(조문 변경 미확인), 🕒 시행 예정 개정, ❔ 확인 불가. ⚠️는 회신이 무효라는 뜻이 아니라 \"현재 적용 여부 확인 필요\"라는 뜻이다.")
add("- 시험 세트(test3, test4, final_test)의 문항은 한 번만 평가하는 절차를 이미 마친 뒤의 결과라 여기에 실었다. 이 문서를 보고 시스템을 고치지 않는다.")
add("- 예시는 대표 사례를 골라 실었다. 전체 문항의 답변은 결과 파일에 있다. 아래 4부는 일부러 고른 실패·한계 사례다.")
add("")
add("## 1부. 잘 된 예")
add("")
n = 0
for title, x in good:
    n += 1
    entry(n, title, x)
add("## 2부. 답이 없을 때: \"찾지 못했습니다\"")
add("")
for title, x in notfound:
    n += 1
    entry(n, title, x)
add("## 3부. 조문으로 조회하기 (검색·LLM 없이 DB 조회)")
add("")
for q in ("은행법 제27조의2 해석 사례 최근 것부터", "전자금융거래법 제42조", "제27조의2 알려줘"):
    n += 1
    buf = io.StringIO()
    lk = Lookup(conn)
    with contextlib.redirect_stdout(buf):
        print(f"질문: {q}")
        show(lk.lookup(q))
    add(f"### {n}. 조문 조회: {q}")
    add("")
    add("```")
    add(buf.getvalue().strip())
    add("```")
    add("")
    add("---")
    add("")
add("## 4부. 문제가 보인 답변 (한계를 숨기지 않기 위해 일부러 고른 사례)")
add("")
for title, x in problems:
    n += 1
    entry(n, title, x)
open(ROOT + r"\질문과_답변_예시.md", "w", encoding="utf-8", newline="").write("\n".join(lines) + "\n")
print("생성 완료", n, "개 예시,", len(lines), "줄")
