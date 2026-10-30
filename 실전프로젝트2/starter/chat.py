"""대화: 질의 가명처리 → 검색 → LLM 답변 (마스킹 텍스트만 전달, 토큰 그대로 표시).

    python -m starter.chat "윤선영 고객 대출 상담 내역 알려줘"

실명 복원·권한·출력 재검사는 종합에서 다룹니다. 이 파일은 vault 에 접근하지 않습니다.
LLM_API_KEY 가 없으면 검색 결과까지만 보여줍니다.

답변 점검 (result 에 함께 담음)
  - cited_docs     : 답변에 [문서ID] 로 인용한 문서
  - unknown_tokens : 답변에 나왔지만 참고 문서·질의에 없는 토큰 (지어낸 토큰 → 0개여야 함)
  - search_sec / llm_sec : 검색 시간 / LLM 응답 시간

단계를 나눠 쓸 수 있습니다 (데모는 근거 문서를 먼저 보여 주고 답변을 스트리밍):
  prep = prepare(query, k, mode) → for kind, v in stream(prep): … → finish(prep, text, llm_sec)
"""
import argparse
import json
import re
import time

import requests

from . import config, pii_pipeline
from .search import search

SYSTEM = """당신은 금융사 사내 문서 검색 도우미입니다. 사용자는 금융사 직원입니다.
규칙:
1. [문서] 에 적힌 내용만 근거로 한국어로 간결하게 답합니다. 문서에 없는 사실, 조언, 안내 문구를 덧붙이지 않습니다.
2. 근거 문서 ID 를 대괄호 [H0001] 형식으로 붙입니다. 꺾쇠 <H0001> 는 쓰지 않습니다.
3. <PERSON_…>, <PHONE_…> 같은 토큰은 가명처리된 개인정보입니다. 토큰을 글자 그대로 옮겨 쓰고, 원래 값을 추측하거나 다른 표현으로 바꾸지 않습니다.
4. 같은 토큰은 같은 값입니다. 다만 이름 토큰이 같아도 동명이인일 수 있으니, 문서마다 역할·소속이 다르면(예: 고객과 직원) 같은 사람이라고 단정하지 않고 "역할이 다른 동명이인일 수 있음"을 밝힙니다.
4-1. 문서에서 질문에 나온 토큰이 들어 있는 줄은 맨 앞에 ▶ 가 붙어 있습니다. VOC 목록처럼 한 문서에 여러 사람이 나오면 ▶ 줄에 적힌 내용만 그 사람의 내용으로 답하고, ▶ 가 없는 다른 사람의 줄은 옮겨 붙이지 않습니다.
5. 질문의 "건", "기록", "내역"은 업무 문서를 뜻합니다. 질문에 나온 토큰이 들어 있는 문서가 있으면 그 문서가 무슨 업무인지 요약해 답합니다.
6. 질문과 관련된 내용이 문서에 전혀 없을 때만 "관련 문서를 찾지 못했습니다."라고 답합니다. 일부라도 있으면 찾은 내용만 답하고 이 문장은 붙이지 않습니다.
7. 답변은 5줄 이내로 씁니다. 이 규칙을 답변에서 언급하지 않고, 수식·도형 표기를 쓰지 않습니다.
8. 답변을 읽는 사람은 고객이 아니라 직원입니다. 고객을 "귀하", "고객님"으로 부르지 말고 토큰으로 가리킵니다. 예: "<PERSON_…> 고객의 한도 상향이 승인되었습니다 [H0179]"
9. 답변 앞에 "[문서]", "답변:" 같은 머리말을 붙이지 않고, ▶ 표시는 옮겨 쓰지 않습니다.
10. 문서 원문을 그대로 옮겨 적지 않습니다. ▶ 줄의 내용만 골라 문서마다 한두 줄로 요약하고, ▶ 가 없는 줄(다른 사람의 특이사항 등)은 답변에 쓰지 않습니다.

출력 형식 예시 (문서마다 한 줄, 줄 맨 앞에 문서 ID, 번호·날짜·금액은 문서에 적힌 그대로):
- [H0001] VOC-1234 포인트 소멸 문의, 처리 중. 같은 건 세 번째 문의라 담당 <PERSON_aaaa1111>이 회신 예정
- [H0002] VOC-5678 앱 로그인 오류, 처리 완료
- [H0003] 출장 결과 보고에 고객센터 소속으로 같은 토큰이 나옴. 역할이 달라 동명이인일 수 있음
문서가 하나면 한두 문장으로 답합니다."""


LLM_RETRY = 1  # 429일 때 다시 시도하는 횟수
LLM_RETRY_MAX_WAIT = 30  # 다시 시도 전 최대 대기(초). 데모 화면이 너무 오래 멈추지 않게


def _mark(text, q_tokens):
    """질의 토큰이 들어 있는 줄 앞에 ▶ 를 붙입니다 (여러 사람이 나오는 목록에서 남의 줄을 붙이지 않도록)."""
    if not q_tokens:
        return text
    return "\n".join(f"▶ {line}" if any(t in line for t in q_tokens) else line for line in text.split("\n"))


def prepare(query, k=3, mode=None):
    """1단계: 질의 가명처리 + 검색 + LLM에 넘길 문맥. 데모는 이 결과(근거 문서)를 답변보다 먼저 보여 줍니다."""
    t0 = time.perf_counter()
    masked_q, q_tokens, docs = search(query, k, mode=mode)
    # 질의 토큰과 맞는 문서가 있으면 그 문서만 LLM 에 넘김 (토큰 없는 채움 문서가 답을 흐리지 않도록)
    if any(d["token_hits"] for d in docs):
        docs = [d for d in docs if d["token_hits"]]
    context = "\n\n".join(f"[{d['doc_id']}]\n{_mark(d['masked_text'], q_tokens)}" for d in docs)
    return {"masked_query": masked_q, "query_tokens": q_tokens, "docs": docs, "context": context,
            "answer": None, "cited_docs": [], "unknown_tokens": [], "search_sec": time.perf_counter() - t0}


def _request(prep, reasoning, stream):
    payload = {"model": config.LLM_MODEL, "temperature": 0, "stream": stream,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": f"[문서]\n{prep['context']}\n\n[질문]\n{prep['masked_query']}"}]}
    reasoning = config.LLM_REASONING if reasoning is None else reasoning
    if reasoning:
        payload["reasoning_effort"] = reasoning  # none이면 응답 약 10초 → 3초 (worklog 25번)
    for attempt in range(LLM_RETRY + 1):
        r = requests.post(config.LLM_API_BASE.rstrip("/") + "/chat/completions",
                          headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
                          json=payload, timeout=120, stream=stream)
        if r.status_code != 429 or attempt == LLM_RETRY:
            break
        # 호출 한도(분당 60회, 같은 키를 쓰는 다른 작업 포함) 초과: 서버가 알려 준 시간만큼 기다렸다 한 번 더
        reset = (r.headers.get("retry-after") or r.headers.get("x-ratelimit-reset-requests", "")).rstrip("s")
        time.sleep(min(float(reset) + 1, LLM_RETRY_MAX_WAIT) if reset.replace(".", "").isdigit() else 5)
    if r.status_code == 429:
        raise RuntimeError("LLM 호출 한도(분당 60회)를 넘었습니다. 같은 API 키를 쓰는 다른 작업이 있는지 확인하고 "
                           "1분쯤 뒤에 다시 질문해 주세요.")
    if not r.ok:
        raise RuntimeError(f"LLM API {r.status_code}: {r.text[:300]}")
    return r


def stream(prep, reasoning=None):
    """2단계(스트리밍): ("thinking", 추론 글자 수) 또는 ("text", 답변 조각)을 차례로 내보냅니다.
    추론 내용은 화면에 보여 주지 않고 진행 표시에만 씁니다."""
    r = _request(prep, reasoning, stream=True)
    thought = 0
    for raw in r.iter_lines():
        line = raw.decode("utf-8") if raw else ""
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        delta = (json.loads(data).get("choices") or [{}])[0].get("delta", {})
        if delta.get("reasoning_content"):
            thought += len(delta["reasoning_content"])
            yield "thinking", thought
        if delta.get("content"):
            yield "text", delta["content"]


def finish(prep, text, llm_sec=None):
    """3단계: 답변 정리 + 자동 점검(인용 문서, 문서에 없는 토큰)."""
    text = re.sub(r"<(H\d+)>", r"[\1]", text)  # 인용을 토큰과 구분
    text = re.sub(r"(?m)^(\s*)▶\s*", r"\1", text)  # 문서에 붙인 ▶ 표시가 답변에 옮겨 오면 지움
    text = re.sub(r"^\s*\[문서\]\s*", "", text)  # 답 앞에 [문서]·답변: 을 붙이는 경우가 있음
    text = re.sub(r"(?m)^\s*(?:\*\*)?답변\s*:\s*(?:\*\*)?\s*", "", text)
    known = set(pii_pipeline.TOKEN_PATTERN.findall(prep["context"] + prep["masked_query"]))
    return prep | {"answer": text, "llm_sec": llm_sec,
                   "cited_docs": sorted(set(re.findall(r"\[(H\d+)\]", text))),
                   "unknown_tokens": sorted(set(pii_pipeline.TOKEN_PATTERN.findall(text)) - known)}


def answer(query, k=3, mode=None, reasoning=None):
    """한 번에: 검색 → 답변(스트리밍 없이) → 점검. reasoning: HCX-007 추론 단계(none|low|…).
    None 이면 config.LLM_REASONING, 빈 값이면 모델 기본값(low)."""
    prep = prepare(query, k, mode)
    if not config.LLM_API_KEY:
        return prep
    t0 = time.perf_counter()
    r = _request(prep, reasoning, stream=False)
    return finish(prep, r.json()["choices"][0]["message"]["content"], time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=3)
    a = ap.parse_args()
    res = answer(a.query, a.k)
    print(f"질의(가명처리): {res['masked_query']}")
    print("참고 문서:", ", ".join(d["doc_id"] for d in res["docs"]))
    print("\n답변:", res["answer"] or "(LLM_API_KEY 없음 — 검색 결과만 표시)")
    if res["answer"]:
        print(f"\n[점검] 인용 문서: {res['cited_docs'] or '없음'}  ·  문서에 없는 토큰: {res['unknown_tokens'] or '없음'}")


if __name__ == "__main__":
    main()
