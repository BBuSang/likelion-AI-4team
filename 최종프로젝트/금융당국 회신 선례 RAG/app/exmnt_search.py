"""현장건의 별도 후보군 검색 (worklog 84·85번 현장건의 작업). 기존 회신 검색(`rag.search`, interp.chunks)과 따로 interp.exmnt_chunks를 찾는다. `rag.py`는 고치지 않는다.
- 붙이는 조건: 현장건의 묶음 점수가 THRESH 이상이면서 기존 회신 1위 벡터 점수(ref_score)보다 높은 것만, 최대 MAX_N개
  (88번 사용자 결정 "1번으로". 87번에서 0.65만 쓰면 기존 회신 질문의 60~68%에 붙어 test6 정확도·충실도가 떨어졌다.
   84번 수치: 현장건의 1위 > 회신 1위인 비율이 현장건의 정답 질문 98%, 회신 정답 질문 2.6%)
  ref_score를 주지 않으면(None) 예전처럼 THRESH만 본다
- LLM에 넘기는 글(block): 검토결과의 뜻을 함께 적는다. "이후 제도가 바뀌었을 수 있다"는 본문에 넣지 않고 화면 카드 배지로만 보인다
  (85번 시안에서 '수용'이 해석 확인인 경우에도 엉뚱한 경고가 붙었음)
"""
THRESH, MAX_N, FETCH = 0.65, 2, 50
MEANING = {"불수용": "건의가 받아들여지지 않음 → 회신 당시 기존 규제·해석 유지",
           "수용": "건의가 받아들여짐 → 제도 개선 또는 해석 확인",
           "일부수용": "건의 일부만 받아들여짐",
           "기조치(수용)": "회신 당시 이미 그렇게 운영되고 있음"}
EXMNT_RULES = """
현장건의 규칙 (머리줄에 '현장건의'로 표시된 사례)
- 현장건의는 업계가 규제 개선을 건의하고 당국이 검토한 결과입니다. 개별 행위에 대한 법령해석·비조치의견이 아닙니다.
- 법령해석·비조치의견서 사례를 먼저 쓰고, 현장건의는 그 뒤에 보충 근거로 씁니다. 현장건의만 관련 있으면 현장건의로 답해도 됩니다.
- 현장건의를 근거로 쓸 때는 "현장건의 검토에서"를 넣고, 무엇을 건의했는지와 그 건의가 받아들여졌는지를 함께 씁니다(예: "○○를 완화해 달라는 건의를 받아들이지 않아 기존 규제가 유지된다고 보았습니다").
- 현장건의 이후 제도가 실제로 바뀌었는지는 추측하지 않습니다."""


def search(conn, qvec, ref_score=None):
    """질문 벡터 → 붙일 현장건의 묶음 목록(대표 사례 정보 포함). 조건을 넘지 못하면 []와 1위 점수를 돌려준다.
    ref_score: 같은 질문의 기존 회신(interp.chunks) 1위 벡터 점수. 이보다 높은 현장건의만 붙인다."""
    v = "[" + ",".join(map(str, qvec)) + "]"
    conn.execute("SET hnsw.ef_search = 200")
    rows = conn.execute("select group_id, 1 - (embedding <=> %(v)s::vector) s from interp.exmnt_chunks order by embedding <=> %(v)s::vector limit %(n)s",
                        {"v": v, "n": FETCH}).fetchall()
    best = {}
    for r in rows:
        best.setdefault(r["group_id"], float(r["s"]))
    ranked = list(best.items())
    top1 = ranked[0][1] if ranked else None
    floor = THRESH if ref_score is None else max(THRESH, ref_score + 1e-9)
    pick = [(g, s) for g, s in ranked[:MAX_N] if s >= floor]
    if not pick:
        return [], top1
    info = {r["group_id"]: r for r in conn.execute(
        """select g.group_id, g.n_cases, c.case_key, c.title, c.reply_date, c.category, c.answer_type, c.question, c.answer, c.url, c.changed_after, c.already_in_place
           from interp.exmnt_groups g join interp.exmnt_cases c on c.case_key = g.representative_case_key where g.group_id = any(%s)""", ([g for g, _ in pick],)).fetchall()}
    return [{**info[g], "score": s, "n_same": info[g]["n_cases"] - 1} for g, s in pick], top1


def block(r):
    """LLM 문맥용 사례 글."""
    meaning = MEANING.get(r["answer_type"] or "", "")
    return (f"[사례 {r['case_key']}] 구분: 현장건의(업계 건의에 대한 당국 검토) | 회신일: {r['reply_date'] or '미상'} | 검토결과: {r['answer_type']}"
            + (f"({meaning})" if meaning else "") + f"\n제목: {r['title']}\n건의내용: {(r['question'] or '')[:900]}\n당국 검토: {r['answer'] or ''}")
