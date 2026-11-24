"""법령 RAG 답변.

answer_candidates → 1위 점수가 MIN_SCORE 미만이면 LLM 없이 "관련 조문을 찾지 못했습니다"
→ LLM 한 번 (추론 low, JSON {"근거": [번호], "답변": "...", "핵심": [...]}) → 고른 번호 순서대로 근거 조문(출처·시행일·본문·원문 링크)
질문은 에이전트가 토큰·번호를 지운 뒤에 넘긴다 (공개 자료에는 개인정보가 없고, 개인정보를 보내지도 않음).
"""
import json
import re
import time
from urllib.parse import quote

import requests

from app.agents import progress
from app.core import answer_format, config

from . import search as S

NOT_FOUND = "관련 조문을 찾지 못했습니다."
FAILED = "답변을 만들지 못했습니다. 잠시 뒤 다시 시도해 주세요."
# 1위 점수 하한 = 답변 가능한 평가 질문 250개의 1위 점수 최솟값 0.5438 − 0.01 (다른 250개에서 막힌 문항 0)
MIN_SCORE = 0.5338
EFFORT = "low"
TIMEOUT = 30

SYSTEM = """너는 한국 금융 법령 질문에 답하는 도우미다. 아래 [후보 조문]만 근거로 쓴다.
1. 질문의 조건(대상, 금액, 기간, 예외 등)에 직접 답하는 조문을 근거로 고른다. 가장 직접적인 것부터 최대 3개.
2. 법률과 그 시행령·시행규칙·감독규정이 함께 있으면, 질문이 묻는 구체적 기준(숫자, 범위, 절차)을 정한 쪽을 먼저 둔다.
3. 후보에 답이 없으면 근거를 비우고 답변에 그렇게 쓴다.
4. 답변은 길이 제한 없이 질문이 물은 것에 빠짐없이 답한다.
   - 정확성: 숫자·기간·금액·비율과 '이상/초과/이하/미만', 주체·대상·조건·예외는 조문 표현을 그대로 쓴다. 조문에 목록이 있고 질문이 그 목록을 물으면 항목을 빠뜨리지 않는다. 후보 조문에 없는 내용은 쓰지 않는다.
   - 가독성: 아래 형식 규칙을 따른다. 문장 끝에 근거 조문을 (법령명 제N조) 형태로 밝힌다.
""" + answer_format.rules("(법령명 제N조)", NOT_FOUND) + """
출력은 JSON 하나만 ("답변" 값 안에 위 형식의 마크다운을 넣는다): {"근거": [후보 번호, ...], "답변": "위 4번에 맞춘 답변", "핵심": ["답변에 쓴 글 그대로의 핵심 단어·구절 1~3개"]}
핵심은 질문이 물은 값(금액, 기간, 비율, 대상, 기관, 해야 할 일 등)을 답변 글에서 그대로 옮긴 짧은 구절이다. 근거가 없으면 핵심도 비운다."""


def build(question, cands):
    lines = [f"[질문]\n{question}\n\n[후보 조문]"]
    for i, a in enumerate(cands, 1):
        lines.append(f"({i}) {a['title']}\n{a['text']}")
    return "\n\n".join(lines)


def call_llm(user):
    """→ (응답 글, usage). 429·5xx 만 두 번까지 다시 보냄"""
    # JSON 모드: 지시문만으로는 가끔 마크다운으로 답해 parse 가 'JSON 없음'으로 실패 → "답변을 만들지 못했습니다" (3번 중 2번)
    payload = {"model": config.LLM_MODEL, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
               "response_format": {"type": "json_object"}, **config.sampling_params(config.LLM_MODEL, effort=EFFORT)}
    url = config.LLM_API_BASE.rstrip("/") + "/chat/completions"
    for attempt in range(3):
        r = requests.post(url, headers={"Authorization": f"Bearer {config.LLM_API_KEY}"}, json=payload, timeout=TIMEOUT)
        if r.status_code == 429 or r.status_code >= 500:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
        if r.status_code != 200:
            raise RuntimeError(f"LLM {r.status_code}")
        body = r.json()
        return body["choices"][0]["message"]["content"] or "", body.get("usage") or {}
    raise RuntimeError("LLM 재시도 초과")


def parse(txt, n):
    """→ (고른 번호 목록, 답변, 핵심 구절). 범위 밖·중복 번호는 버림"""
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m:
        raise ValueError("JSON 없음")
    d = json.loads(m.group())
    picks = []
    for x in d.get("근거") or []:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= n and i not in picks:
            picks.append(i)
    keys = [k for k in d.get("핵심") or [] if isinstance(k, str) and k.strip()]
    return picks, str(d.get("답변") or "").strip(), keys


def law_url(a):
    """국가법령정보센터 원문 링크 (법령은 조문 단위, 행정규칙은 규칙 단위)"""
    name = quote(a["law_name"])
    if a.get("law_level") == "행정규칙":
        return f"https://www.law.go.kr/행정규칙/{name}"
    label = a.get("article_label") or ""
    return f"https://www.law.go.kr/법령/{name}/{quote(label)}" if label.startswith("제") else f"https://www.law.go.kr/법령/{name}"


def source(a, no):
    return {"no": no, "doc_id": a["doc_id"], "title": a["title"], "cite": a.get("cite", ""), "law_name": a["law_name"],
            "law_level": a.get("law_level", ""), "effective_date": a.get("effective_date", ""),
            "score": round(a.get("score", 0.0), 4), "rank": a.get("rank"), "linked_from": a.get("linked_from"),
            "text": a["text"], "url": law_url(a)}


def answer(question):
    """질문 하나 → {"answer", "found", "sources": [{no, title, text, url, ...}], "key_phrases", "timing", "error"}"""
    question = (question or "").strip()
    out = {"question": question, "answer": NOT_FOUND, "found": False, "sources": [], "key_phrases": [],
           "timing": {}, "error": ""}
    if not question:
        return out
    t0 = time.perf_counter()
    cands = S.answer_candidates(question)
    t1 = time.perf_counter()
    out["timing"]["search"] = round(t1 - t0, 3)
    if not cands or max(a["score"] for a in cands) < MIN_SCORE:  # 범위 밖으로 보고 LLM 을 부르지 않음
        return out
    progress.done("search", counts={"law": len(cands)})  # 스트리밍 단계 알림
    progress.step("answer")
    try:
        txt, _usage = call_llm(build(question, cands))
        picks, text, keys = parse(txt, len(cands))
        out["sources"] = [source(cands[i - 1], i) for i in picks]
        out["found"] = bool(picks)
        out["answer"] = text if (text and picks) else (text or NOT_FOUND)
        out["key_phrases"] = [k for k in keys if k in out["answer"]] if picks else []
    except (RuntimeError, ValueError, requests.RequestException) as e:  # 다시 부르지 않음
        out["answer"], out["error"] = FAILED, type(e).__name__
    out["timing"]["llm"] = round(time.perf_counter() - t1, 3)
    return out
