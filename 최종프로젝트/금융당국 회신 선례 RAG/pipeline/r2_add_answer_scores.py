"""R2 검색 점수 CSV(`r1_retrieval_by_question.csv`)에 답변 판정(정확도·충실도)을 더하고 틀린 답변이 위에 오게 정렬한다 (사용자 요청 2026-11-11, worklog 95번).
새로 답변·심사를 돌리지 않는다. 이미 있는 평가 캐시에서 문항별 판정을 가져온다. 잠긴 시험 세트 캐시(test5·test6)는 읽지 않는다.
세트별 출처 (답변 평가 설정)
- 합성 150, dev1~4: f10 (리랭킹 + 프롬프트 V2, 현장건의 없음)
- dev5: f8의 A (이전 설정: 벡터 상위 5 + 처음 프롬프트). 지금 데모 설정으로는 재지 않았다
- dev6: f11의 A (리랭킹 + V2). 88번 조건으로 현장건의가 붙는 2문항은 f12
- dev7, test7: f13 (지금 데모 설정: 리랭킹 + V2 + 현장건의 88번 조건)
- 답변불가 20: f10 (88번 조건으로 현장건의가 붙는 2문항은 f12). 정답이 없으므로 "찾지 못했습니다"로 거절했는지만 본다
- devF: 답변 평가를 한 적이 없다(검색만) → 빈칸
정렬: 불일치 → 답변없음(답변불가 세트에서 답해 버린 경우 포함) → 부분일치 → 일치·일부추정 → 일치·충실 → 정상 거절 → 미평가. 같은 등급 안에서는 세트, qid 순
답변 내용: 시스템 답변 전문과 정답 사례(제목, 질의·건의, 회답·검토)를 함께 넣는다
주의: 검색 점수 칸은 벡터 검색(리랭킹 전) 값이고, 답변 판정은 위 설정으로 만든 답변의 것이다. 답변에 쓴 문맥은 리랭킹 뒤 5건이라 벡터 순위와 다를 수 있다
"""
import csv
import json
import sys

import rag
from common import DATA

EVAL = DATA / "eval"
CSV = EVAL / "r1_retrieval_by_question.csv"
ADDED = ["정확도", "충실도", "심사 사유", "답변이 인용한 사례", "답변 평가 설정", "시스템 답변", "정답 사례 제목", "정답 사례 질의(건의)", "정답 사례 회답(검토)"]
rd = lambda name: [json.loads(l) for l in open(EVAL / name, encoding="utf-8") if l.strip()]


def load():
    A, J, cfg = {}, {}, {}

    def put(ans, jdg, sets, label, mode=None):
        for r in rd(ans):
            if r["set"] in sets and (mode == "any" or r.get("mode") == mode):
                A[(r["set"], r["qid"])] = r
                cfg[(r["set"], r["qid"])] = label
        for r in rd(jdg) if jdg else []:
            if r["set"] in sets and r.get("mode") == mode:
                J[(r["set"], r["qid"])] = r
    put("_f10_answers.jsonl", "_f10_judge.jsonl", ("rewrite_150", "dev1", "dev2", "dev3", "dev4", "unanswerable_20"), "리랭킹 + V2 (f10)")
    put("_f8_answers.jsonl", "_f8_judge.jsonl", ("dev5",), "이전 설정: 벡터 5 + 처음 프롬프트 (f8)", "A")
    put("_f11_answers.jsonl", "_f11_judge.jsonl", ("dev6",), "리랭킹 + V2 (f11)", "A")
    put("_f12_answers.jsonl", "_f12_judge.jsonl", ("dev6", "unanswerable_20"), "리랭킹 + V2 + 현장건의 (f12)", "B")      # f12 답변은 mode B, 심사에는 mode가 없다
    for r in rd("_f12_judge.jsonl"):
        J[(r["set"], r["qid"])] = r
    for r in rd("_f13_judge.jsonl"):
        J[(r["set"], r["qid"])] = r
    put("_f13_answers.jsonl", None, ("dev7", "test7"), "지금 데모 설정 (f13)", "any")          # f13 답변에는 mode(A/B: 현장건의 붙음 여부)가 있고 문항당 하나다
    return A, J, cfg


def main():
    rows = list(csv.reader(open(CSV, encoding="utf-8-sig")))
    head, body = rows[0], rows[1:]
    if ADDED[0] in head:                                     # 다시 돌려도 칸이 겹치지 않게
        k = head.index(ADDED[0])
        head, body = head[:k], [r[:k] for r in body]
    A, J, cfg = load()
    keys = sorted({r[head.index("정답 사례")] for r in body} - {""})
    with rag.connect() as conn:                              # 정답 사례 글 (회신은 회답 + 이유, 현장건의는 검토결과 + 사유)
        G = {x["case_key"]: x for x in conn.execute("select case_key, title, question, coalesce(answer, '') || case when coalesce(reason, '') <> '' then E'\n[이유] ' || reason else '' end as answer from interp.cases where case_key = any(%s)", (keys,)).fetchall()}
        G.update({x["case_key"]: x for x in conn.execute("select case_key, title, question, '검토결과: ' || coalesce(answer_type, '') || E'\n' || coalesce(answer, '') as answer from interp.exmnt_cases where case_key = any(%s)", (keys,)).fetchall()})
    i_gold = head.index("정답 사례")
    i_set, i_qid, i_kind = head.index("세트"), head.index("qid"), head.index("정답 종류")
    out = []
    for r in body:
        k = (r[i_set], r[i_qid])
        a, j = A.get(k), J.get(k)
        if r[i_kind] == "답변불가":
            ok = a and a["check"]["not_found"]
            add, grade = ["거절(정상)" if ok else "답변함(오류)", "해당없음", "", ", ".join(a["check"]["cited"]) if a else "", cfg.get(k, "")], 5 if ok else 1
        elif j:
            add = [j["correctness"], j["grounded"], j.get("reason") or "", ", ".join(a["check"]["cited"]), cfg[k]]
            grade = {"불일치": 0, "답변없음": 1, "부분일치": 2}.get(j["correctness"], 3 if j["grounded"] != "충실" else 4)
        else:
            add, grade = ["", "", "", "", "답변 평가 안 함"], 6
        g = G.get(r[i_gold])
        add += [a["answer"] if a else "", (g["title"] or "").strip() if g else "", (g["question"] or "").strip() if g else "", (g["answer"] or "").strip() if g else ""]
        out.append((grade, r + add))
    out.sort(key=lambda x: (x[0], x[1][i_set], x[1][i_qid]))
    with open(CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(head + ADDED)
        w.writerows(r for _, r in out)
    names = ["불일치", "답변없음·답변불가에 답함", "부분일치", "일치·일부추정", "일치·충실", "정상 거절", "미평가"]
    print({names[g]: sum(1 for x, _ in out if x == g) for g in range(7)})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
