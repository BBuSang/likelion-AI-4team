"""법령 RAG 답변 단계 (기획안 5-8, v0.31) — 데모·팀 서비스·평가가 같이 쓰는 답변 함수

    from retriever.answer import answer, warmup
    warmup()                                   # 서버 시작 때 한 번 (모델·DB 연결·연결 자료를 미리 올림)
    r = answer("예금자보호 한도는 1인당 얼마야?")
    r["answer"], r["sources"], r["timing"]

    python retriever/answer.py "질문"          # 명령줄에서 한 번 물어보기

흐름 (채택된 구성, 기획안 5-6-7·5-6-8·5-6-9):
  answer_candidates(question)  검색(KURE-v1 + pgvector) → 합친 후보(1~10위 + 1~5위 연결 조문) → 조문 글 규칙
  → LLM 한 번 (추론 low, JSON {"근거": [번호...], "답변": "..."})
  → 고른 번호 순서대로 근거 조문(출처·시행일자·본문·원문 링크)
  검색 1위 점수가 MIN_SCORE(5-8-1) 미만이면 LLM 을 부르지 않고 '관련 조문을 찾지 못했습니다'.
LLM 호출은 질문당 한 번. JSON 이 깨지거나 오류가 나도 다시 부르지 않는다 (429·5xx 응답만 두 번까지 다시 보냄).
"""
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from retriever.search import _neighbors, answer_candidates, embed, search_vec  # noqa: E402

ENV = dotenv_values(ROOT / ".env")
EFFORT = "low"  # 기획안 5-6-7 v0.24 채택. "default" = 모델 기본값(요청에 넣지 않음)
LLM_TIMEOUT = 30  # 초. 한 번의 요청이 이보다 길면 오류로 돌려준다
NOT_FOUND = "관련 조문을 찾지 못했습니다."
# 기획안 5-8-1 (v0.32): 검색 1위 점수가 이보다 낮으면 LLM 을 부르지 않고 NOT_FOUND.
# = 답변 가능 dev1~5 250문항 1위 점수 최솟값 0.5438 − 0.01. dev6~10 250문항에서 막힌 문항 0 (최솟값 0.548)
MIN_SCORE = 0.5338
FAILED = "답변을 만들지 못했습니다. 잠시 뒤 다시 시도해 주세요."

_HEAD = """너는 한국 금융 법령 질문에 답하는 도우미다. 아래 [후보 조문]만 근거로 쓴다.
1. 질문의 조건(대상, 금액, 기간, 예외 등)에 직접 답하는 조문을 근거로 고른다. 가장 직접적인 것부터 최대 3개.
2. 법률과 그 시행령·시행규칙·감독규정이 함께 있으면, 질문이 묻는 구체적 기준(숫자, 범위, 절차)을 정한 쪽을 먼저 둔다.
"""
_TAIL = """
출력은 JSON 하나만: {"근거": [후보 번호, ...], "답변": "2~4문장, 근거 조문을 '법령명 제N조' 형태로 밝힘"}"""
_RULE3 = "3. 후보에 답이 없으면 근거를 비우고 답변에 그렇게 쓴다."
_TAIL_BY_ARTICLE = """
출력은 JSON 하나만: {"근거": [후보 번호, ...], "답변": "질문에 대한 결론 1~2문장", "조문별": [{"번호": 근거 번호, "설명": "그 조문이 이 질문에 대해 정한 내용 1~2문장"}]}
조문별은 근거 순서대로 근거마다 하나씩 쓴다. 근거가 없으면 조문별도 비운다."""
SYSTEMS = {  # 기획안 5-8-2: 3번만 다름 / 5-8-4: v3 는 v1 의 출력 형식만 조문별로
    "v1": _HEAD + _RULE3 + _TAIL,
    "v2": _HEAD + "3. 질문에 직접 답하는 조문이 후보에 없으면 근거를 비운다. 비슷한 조문을 참고로 설명하더라도 근거에는 넣지 않는다." + _TAIL,
    "v3": _HEAD + _RULE3 + _TAIL_BY_ARTICLE,
    # 5-8-5: v1 그대로 + 핵심 구절 칸 (화면 강조용)
    "v4": _HEAD + _RULE3 + """
출력은 JSON 하나만: {"근거": [후보 번호, ...], "답변": "2~4문장, 근거 조문을 '법령명 제N조' 형태로 밝힘", "핵심": ["답변에 쓴 글 그대로의 핵심 단어·구절 1~3개"]}
핵심은 질문이 물은 값(금액, 기간, 비율, 대상, 기관, 해야 할 일 등)을 답변 글에서 그대로 옮긴 짧은 구절이다. 근거가 없으면 핵심도 비운다.""",
    # 5-8-6: v4 + 답변 길이 제한 없음, 정확성·가독성 지시
    "v5": _HEAD + _RULE3 + """
4. 답변은 길이 제한 없이 질문이 물은 것에 빠짐없이 답한다.
   - 정확성: 숫자·기간·금액·비율과 '이상/초과/이하/미만', 주체·대상·조건·예외는 조문 표현을 그대로 쓴다. 조문에 목록이 있고 질문이 그 목록을 물으면 항목을 빠뜨리지 않는다. 후보 조문에 없는 내용은 쓰지 않는다.
   - 가독성: 결론을 첫 문장에 쓴다. 항목이 셋 이상이면 줄을 바꿔 "- "로 나열한다. 문장 끝에 근거 조문을 (법령명 제N조) 형태로 밝힌다.
출력은 JSON 하나만: {"근거": [후보 번호, ...], "답변": "위 4번에 맞춘 답변", "핵심": ["답변에 쓴 글 그대로의 핵심 단어·구절 1~3개"]}
핵심은 질문이 물은 값(금액, 기간, 비율, 대상, 기관, 해야 할 일 등)을 답변 글에서 그대로 옮긴 짧은 구절이다. 근거가 없으면 핵심도 비운다.""",
}
PROMPT = "v5"  # 지금 쓰는 지시문 = v4 + 길이 제한 없음·정확성·가독성 (5-8-6, v0.49). v2·v3 는 채택 안 함. test10 최종 점수는 v1 기준
SYSTEM = SYSTEMS[PROMPT]


def build(question, cands):
    """LLM 에 넘기는 입력 글 (후보 번호 = 목록 순서, 1부터)"""
    lines = [f"[질문]\n{question}\n\n[후보 조문]"]
    for i, a in enumerate(cands, 1):
        lines.append(f"({i}) {a['title']}\n{a['text']}")
    return "\n\n".join(lines)


def call_llm(user, effort=EFFORT, timeout=LLM_TIMEOUT, prompt=None):
    """→ (응답 글, usage). 429·5xx 는 두 번까지 다시 보내고, 그 밖의 실패는 RuntimeError. prompt = SYSTEMS 의 키 (없으면 PROMPT)"""
    url = ENV["LLM_API_BASE"].rstrip("/") + "/chat/completions"
    system = SYSTEMS[prompt or PROMPT]
    body = {"model": ENV["LLM_MODEL"], "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"}}
    if effort and effort != "default":
        body["reasoning_effort"] = effort
    for attempt in range(3):
        r = requests.post(url, headers={"Authorization": f"Bearer {ENV['LLM_API_KEY']}"}, json=body, timeout=timeout)
        if r.status_code == 200:
            j = r.json()
            return j["choices"][0]["message"]["content"], j.get("usage", {})
        if r.status_code in (429, 500, 502, 503) and attempt < 2:
            time.sleep(5 * (attempt + 1))
            continue
        raise RuntimeError(f"LLM 오류 {r.status_code}: {r.text[:300]}")


def parse(txt, n):
    """LLM 응답 JSON → (고른 번호 목록, 답변). 범위 밖·중복 번호는 버림. 형식이 틀리면 ValueError"""
    picks, text, _ = parse_full(txt, n)
    return picks, text


def key_phrases(txt):
    """LLM 응답 JSON 의 '핵심' 칸 (v4 지시문, 5-8-5) → 구절 목록. 없거나 형식이 틀리면 빈 목록"""
    try:
        v = json.loads(txt).get("핵심", [])
    except (ValueError, AttributeError):
        return []
    return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []


def parse_full(txt, n):
    """→ (고른 번호, 결론 답변, {번호: 조문별 설명}). 조문별 설명은 v3 지시문에서만 옴 (없으면 빈 dict)"""
    j = json.loads(txt)
    if not isinstance(j, dict):
        raise ValueError("JSON 객체가 아님")
    picks = []
    for x in j.get("근거", []) or []:
        if str(x).strip().isdigit() and 1 <= int(x) <= n and int(x) not in picks:
            picks.append(int(x))
    explain = {}
    for it in j.get("조문별", []) or []:
        if isinstance(it, dict) and str(it.get("번호", "")).strip().isdigit():
            k = int(it["번호"])
            if k in picks and str(it.get("설명", "")).strip():
                explain[k] = str(it["설명"]).strip()
    return picks, str(j.get("답변", "") or "").strip(), explain


def compose(text, explain, cands, picks):
    """채점·한 덩어리 표시용 답변 글: 결론 + 조문별 설명 (조문별이 없으면 결론 그대로)"""
    lines = [text] if text else []
    for i in picks:
        if i in explain:
            lines.append(f"- {cands[i - 1]['title']}: {explain[i]}")
    return "\n".join(lines)


def law_url(a):
    """국가법령정보센터 원문 링크. 법령은 조문 단위, 행정규칙은 규칙 단위 (조문 주소가 없음)"""
    name = quote(a["law_name"])
    if a.get("law_level") == "행정규칙":
        return f"https://www.law.go.kr/행정규칙/{name}"
    label = a.get("article_label") or ""
    return f"https://www.law.go.kr/법령/{name}/{quote(label)}" if label.startswith("제") else f"https://www.law.go.kr/법령/{name}"


def source(a, no):
    return {"no": no, "doc_id": a["doc_id"], "title": a["title"], "cite": a.get("cite", ""), "law_name": a["law_name"],
            "law_level": a.get("law_level", ""), "article_label": a.get("article_label", ""),
            "effective_date": a.get("effective_date", ""), "score": round(a.get("score", 0.0), 4), "rank": a.get("rank"),
            "linked_from": a.get("linked_from"), "link_kind": a.get("link_kind"), "text_mode": a.get("text_mode", ""),
            "text": a["text"], "url": law_url(a)}


def warmup():
    """서버 시작 때 한 번: 임베딩 모델·DB 연결·연결 자료를 올린다 (답변 시간에서 빠지는 부분)"""
    t0 = time.perf_counter()
    search_vec(embed(["모델 미리 올리기"])[0], k=1)
    _neighbors()
    return round(time.perf_counter() - t0, 2)


def answer(question, effort=EFFORT, prompt=None, levels=None):
    """질문 하나 → 답변 + 근거 조문.
    {"question", "answer", "found", "sources": [{no, title, cite, law_name, effective_date, url, text, ...}],
     "candidates": [{no, title, score, rank}], "timing": {"search", "llm", "total"}, "usage", "error",
     "below_min_score": 검색 1위 점수가 MIN_SCORE 미만이라 LLM 을 부르지 않았는지}"""
    question = (question or "").strip()
    out = {"question": question, "answer": "", "found": False, "sources": [], "candidates": [],
           "timing": {"search": 0.0, "llm": 0.0, "total": 0.0}, "usage": {}, "error": "", "below_min_score": False}
    if not question:
        out["answer"], out["error"] = "질문을 입력해 주세요.", "빈 질문"
        return out
    t0 = time.perf_counter()
    cands = answer_candidates(question, levels=levels or None)  # levels: 법령 구분 필터 (데모 선택, 평가는 None)
    t1 = time.perf_counter()
    out["candidates"] = [{"no": i, "title": a["title"], "score": round(a["score"], 4), "rank": a.get("rank"),
                          "linked_from": a.get("linked_from"), "text_mode": a.get("text_mode", "")}
                         for i, a in enumerate(cands, 1)]
    if not cands or max(a["score"] for a in cands) < MIN_SCORE:  # 범위 밖으로 보고 LLM 을 부르지 않음
        out["answer"], out["below_min_score"] = NOT_FOUND, True
        out["timing"] = {"search": round(t1 - t0, 3), "llm": 0.0, "total": round(t1 - t0, 3)}
        return out
    try:
        txt, out["usage"] = call_llm(build(question, cands), effort, prompt=prompt)
        picks, text, explain = parse_full(txt, len(cands))
        out["sources"] = [dict(source(cands[i - 1], i), explain=explain.get(i, "")) for i in picks]
        out["found"] = bool(picks)
        out["answer"] = text if (text and picks) else (text or NOT_FOUND)
        out["answer_full"] = compose(out["answer"], explain, cands, picks)  # 결론 + 조문별 설명
        out["key_phrases"] = [k for k in key_phrases(txt) if k in out["answer"]] if picks else []  # 5-8-5: 답변에 실제로 있는 것만
    except (RuntimeError, ValueError, requests.RequestException) as e:  # 다시 부르지 않음 (LLM 최소 호출)
        out["answer"], out["error"] = FAILED, f"{type(e).__name__}: {str(e)[:300]}"
    t2 = time.perf_counter()
    out["timing"] = {"search": round(t1 - t0, 3), "llm": round(t2 - t1, 3), "total": round(t2 - t0, 3)}
    return out


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "예금자보호 한도는 1인당 얼마야?"
    print(f"준비 {warmup()}초 (모델 로드, 답변 시간에서 뺌)")
    r = answer(q)
    print(f"\n[질문] {r['question']}\n[답변] {r['answer']}\n")
    for s in r["sources"]:
        print(f"  근거 {s['no']}. {s['cite'] or s['title']}  (시행 {s['effective_date']}, 검색 {s['rank']}위"
              + (f", {s['link_kind']} 연결" if s["linked_from"] else "") + f")\n     {s['url']}")
    print(f"\n검색 {r['timing']['search']}초 · LLM {r['timing']['llm']}초 · 전체 {r['timing']['total']}초"
          + (f" · 오류: {r['error']}" if r["error"] else ""))
