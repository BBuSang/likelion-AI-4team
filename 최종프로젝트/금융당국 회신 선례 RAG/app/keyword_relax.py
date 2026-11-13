"""키워드 순차 완화 검색: 벡터 상위 5개에 '키워드로 넓혀 찾은 사례'를 최대 3개 더한다. `rag.py`는 고치지 않는다.
1. 질문에서 핵심 개념 2~4개를 LLM으로 뽑는다. 개념마다 회신 본문에 쓰일 표현(동의어 포함)을 둔다(개념 안은 OR, 개념끼리는 AND).
2. 검색 대상 회신(제목·질의요지·회답·이유)에서 개념이 몇 개 일치하는지 센다(띄어쓰기 무시).
3. 모든 개념이 일치하는 회신부터 찾고, 모자라면 개념을 하나씩 덜 요구한다(n개 → n-1개 → … → 1개).
4. 같은 일치 수 안에서는 벡터 점수로 순서를 정하고, 벡터 상위 5개에 이미 있는 묶음은 뺀다.
화면(`demo.py`)과 개발셋 평가(`pipeline/f6_keyword_relax_eval.py`)가 같은 함수를 쓴다.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import rag  # noqa: E402

N_EXTRA = 3
EXTRACT_SYSTEM = """금융당국 회신(법령해석·비조치의견서) 검색을 돕습니다. 질문에서 회신 본문을 글자로 찾을 핵심 개념을 뽑아 JSON으로만 답하세요.
- 개념은 2~4개. 질문의 쟁점을 이루는 상품·행위·규제 용어를 고릅니다(예: 상품 "적금", 행위 "캐시백", 규제 "특별이익").
- 개념마다 회신 본문에 실제로 쓰일 짧은 명사 표현을 1~4개 둡니다. 같은 뜻의 다른 표현이나 법령 용어를 함께 둘 수 있습니다(예: ["캐시백", "청구할인"]).
- "금융회사", "가능 여부", "질의", "해당", "관련", "규정", "법령"처럼 거의 모든 회신에 나오는 말은 쓰지 않습니다.
- 질문이 금융 규제와 무관하면 빈 목록을 냅니다.
출력: {"concepts": [["표현", "표현"], ["표현"]]}"""

_cases = None


def _norm(s):
    return re.sub(r"\s+", "", s or "")


def extract_concepts(question):
    """질문 → [[표현, ...], ...]. 개인정보 형식은 가린 뒤 보낸다."""
    r = rag.client.chat.completions.create(model=rag.CHAT_MODEL, reasoning_effort="low", response_format={"type": "json_object"},
                                           messages=[{"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": rag.mask_pii(question)}])
    try:
        raw = json.loads(r.choices[0].message.content or "{}").get("concepts") or []
    except json.JSONDecodeError:
        raw = []
    concepts = []
    for c in raw[:4]:
        terms = [t.strip() for t in (c if isinstance(c, list) else [c]) if isinstance(t, str) and len(_norm(t)) >= 2][:4]
        if terms:
            concepts.append(terms)
    return concepts, {"input": r.usage.prompt_tokens, "output": r.usage.completion_tokens}


def _load_cases(conn):
    global _cases
    if _cases is None:
        rows = conn.execute("select case_key, group_id, title, question, answer, reason from interp.cases where indexable and group_id is not null").fetchall()
        _cases = [(r["case_key"], r["group_id"], _norm(" ".join(x or "" for x in (r["title"], r["question"], r["answer"], r["reason"])))) for r in rows]
    return _cases


def relax_search(conn, concepts, qvec, exclude_groups, n_extra=N_EXTRA):
    """키워드 순차 완화. 반환: {"extra": [...], "trail": [{"need": k, "groups": 건수}, ...]}"""
    if not concepts:
        return {"extra": [], "trail": []}
    terms = [[_norm(t) for t in c] for c in concepts]
    level = {}                                              # group_id → (일치 개념 수, 일치 개념 번호)
    for key, gid, text in _load_cases(conn):
        hit = tuple(i for i, ts in enumerate(terms) if any(t in text for t in ts))
        if hit and len(hit) > level.get(gid, (0,))[0]:
            level[gid] = (len(hit), hit)
    n = len(terms)
    trail = [{"need": k, "groups": sum(1 for v in level.values() if v[0] >= k)} for k in range(n, 0, -1)]
    pool = [g for g in level if g not in exclude_groups]
    if not pool:
        return {"extra": [], "trail": trail}
    v = "[" + ",".join(map(str, qvec)) + "]"
    score = {r["group_id"]: float(r["s"]) for r in conn.execute(
        "select group_id, max(1 - (embedding <=> %s::vector)) s from interp.chunks where group_id = any(%s) group by group_id", (v, pool)).fetchall()}
    pool.sort(key=lambda g: (-level[g][0], -score.get(g, 0)))
    pick = pool[:n_extra]
    reps = {r["group_id"]: r for r in conn.execute(
        "select g.group_id, g.n_cases, g.representative_case_key, c.title from interp.case_groups g join interp.cases c on c.case_key = g.representative_case_key where g.group_id = any(%s)", (pick,)).fetchall()}
    extra = [{"doc_id": reps[g]["representative_case_key"], "group_id": g, "title": reps[g]["title"], "score": round(score.get(g, 0), 4), "matched": level[g][0], "of": n,
              "matched_terms": [concepts[i][0] for i in level[g][1]], "missing_terms": [concepts[i][0] for i in range(n) if i not in level[g][1]], "n_same": reps[g]["n_cases"] - 1} for g in pick]
    return {"extra": extra, "trail": trail}


def extra_block(case, ex):
    """LLM에 넘기는 사례 글: 머리줄에 '키워드로 넓혀 찾은 사례'와 빠진 개념을 밝힌다."""
    block = rag.case_block(case)
    head, _, rest = block.partition("\n")
    note = f"키워드로 넓혀 찾은 사례(질문의 핵심 개념 {ex['of']}개 중 {ex['matched']}개 일치"
    note += f", 없는 개념: {', '.join(ex['missing_terms'])})" if ex["missing_terms"] else ")"
    return f"{head} | {note}\n{rest}"
