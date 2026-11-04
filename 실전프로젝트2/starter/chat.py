"""대화: 질의 가명처리 → 검색 → LLM 답변 (마스킹 텍스트만 전달) → 출력 검사 → 토큰 그대로 표시.

    python -m starter.chat "윤선영 고객 대출 상담 내역 알려줘"
    python -m starter.chat "보이스피싱 피해 접수 기록" --model HCX-007

출력 검사 (보여주기 전에)
  1. 토큰 검사: 답변 속 토큰이 LLM 에 준 문서·질의에 실제로 있던 토큰인지 확인.
     LLM 이 바꿔 쓴 토큰(<PERSON_4fefeAAA> 처럼 대소문자 변경)·지어낸 토큰은 [확인 안 된 토큰] 으로 바꿈
     → 나중에 실명 복원할 때 엉뚱한 사람으로 복원되는 것을 막음
  2. 원문 개인정보 검사: 답변을 탐지기에 한 번 더 통과시켜, 토큰이 아닌 원문 개인정보(LLM 이 지어낸 번호 등)가 있으면 가림
     적재 때와 같은 탐지기를 쓰면 LLM 답변의 마크다운·항목 이름('**위임인**')을 이름으로 잡아 정상 답변을 가림
     → 마크다운 기호를 지우고, 이름은 토큰 분류 모델만 쓰는 detect_output() 사용
실명 복원·권한은 종합에서 다룹니다. 이 파일은 vault 에 접근하지 않습니다.
LLM_API_KEY 가 없으면 검색 결과까지만 보여줍니다.
"""
import argparse
import json
import re
import time

import requests

from . import config, pii_pipeline
from .search import search

SYSTEM = (
    "당신은 금융사 사내 문서 검색 도우미입니다. 아래 [문서]의 내용만 근거로 한국어로 간결하게 답하세요.\n"
    "규칙:\n"
    "1. <PERSON_…>, <PHONE_…> 같은 꺾쇠 토큰은 가명처리된 개인정보입니다. 문서에 나온 그대로 한 글자도 바꾸지 말고 "
    "(대소문자 포함) 복사해 쓰세요. 원래 값을 추측하거나 새 토큰을 만들지 마세요.\n"
    "2. 근거 문서는 대괄호로 문서 ID 를 적으세요. [문서]에 있는 ID 만 쓰세요.\n"
    "3. 질문이 '내역·기록·건을 찾아줘'처럼 문서를 찾는 것이면, 질문과 관련된 문서마다 한 줄씩 "
    "'[문서 ID] 문서 종류 — 핵심 내용(무슨 일, 결과·상태, 날짜)' 형식으로 요약하세요. 번호나 ID 만 나열하지 마세요.\n"
    "4. 질문에 토큰(예: <PERSON_…>)이 있으면, 문서에서 그 토큰과 똑같은 토큰이 나오는 줄·문장을 찾아 "
    "그 줄의 내용(문의·업무 종류, 처리 상태, 특이사항)으로 답하세요. 같은 문서의 다른 사람 이야기는 섞지 마세요.\n"
    "5. 질문과 관련 없는 문서는 빼고, 뺀 문서의 ID 나 뺀 이유는 답변에 아예 쓰지 마세요. "
    "관련 문서가 하나도 없으면 '관련 문서에서 찾을 수 없습니다'라고 답하세요.\n"
    "6. 문서 머리의 (검색 근거: …)는 검색 시스템이 가명처리 전 값으로 확인한 사실입니다. 예를 들어 주소가 "
    "<ADDRESS_…> 로 가려져 있어도 '질문의 지역과 같은 지역의 주소가 있음'이면 그 지역 문서로 보고 답하세요."
)


def _context(docs):
    """LLM 에 넘길 문서. 검색 근거가 있으면 문서 머리에 붙인다 (질문에 있던 내용만, 새 개인정보 없음)."""
    parts = []
    for d in docs:
        why = f" (검색 근거: {'; '.join(d['reasons'])})" if d.get("reasons") else ""
        parts.append(f"[{d['doc_id']}]{why}\n{d['masked_text']}")
    return "\n\n".join(parts)
UNVERIFIED = "[확인 안 된 토큰]"
# 대소문자가 바뀐 토큰까지 잡으려고 검사용은 대소문자 무시
_ANY_TOKEN = re.compile(r"<[A-Za-z_]+_[0-9A-Za-z]+>")
_MARKDOWN = re.compile(r"[*#`>|]")


def check_output(answer_text, allowed_tokens, safe_words=()):
    """보여주기 전 출력 검사. (고친 답변, 검사 결과) 반환.
    safe_words: 개인정보가 아닌 걸 아는 말 (문서 종류 이름). '한도하향통지'를 모델이 이름으로 잡아 가리던 오탐 방지 (19장)"""
    bad = sorted({t for t in _ANY_TOKEN.findall(answer_text) if t not in allowed_tokens})
    fixed = answer_text
    for t in bad:
        fixed = fixed.replace(t, UNVERIFIED)
    # 토큰 자리와 마크다운 기호(** # ` > |)는 같은 길이의 공백으로 바꿔 두고 탐지 (위치는 그대로)
    # — '**신청일**:' 같은 굵은 글씨 머리말이 이름으로 잡히던 오탐의 주원인
    probe = _MARKDOWN.sub(" ", _ANY_TOKEN.sub(lambda m: " " * len(m.group()), fixed))
    for w in safe_words:
        probe = probe.replace(w, " " * len(w))
    # 적재용 detect() 대신 출력용 detect_output() (이름은 모델만 → '위임인'·'인사팀' 같은 오탐 방지)
    spans = [s for s in pii_pipeline.detect_output(probe) if probe[s["start"]:s["end"]].strip()]
    raw = [{"label": s["label"], "text": probe[s["start"]:s["end"]]} for s in spans]
    if spans:
        fixed, _ = pii_pipeline.pseudonymize(fixed, spans)
    return fixed, {"unverified_tokens": bad, "raw_pii": raw}


KEEP_MIN = 0.7  # 이 토큰 점수 이상은 남김 = 조건을 다 맞췄거나, 같은 이름의 '누구인지 미확정'(search.UNRESOLVED)


def select_docs(cands, k):
    """LLM 에 넘길 문서 고르기. 상위 k개를 무조건 넘기면 조건과 상관없는 문서(다른 사람 문서)까지 LLM 이 인용함 (인용 정밀도 0.80).
    - 검색 1위는 항상 넘김 (문서 종류·의미 점수로 1위가 된 문서를 토큰 점수만 보고 빼지 않게)
    - 질의 조건을 어느 정도 맞춘 문서가 있으면, 토큰 점수 KEEP_MIN 미만(다른 사람이 확실 0.3, 조건 무관 0)은 뺌
      미확정(0.7)은 같은 사람일 수 있으므로 남김 — 처음 규칙(1점만 남김)은 미확정 쪽 정답 문서를 버렸음 (worklog 13-4)"""
    if not cands or not config.SELECT_FULL_MATCH or max(d.get("token_score", 0) for d in cands) < KEEP_MIN:
        return cands[:k]
    return ([cands[0]] + [d for d in cands[1:] if d.get("token_score", 0) >= KEEP_MIN])[:k]


_ROLE_KO = {"customer": "고객", "staff": "직원", "external": "외부"}


def ask_back(amb, cands):
    """동명이인을 고르지 못했을 때 LLM 대신 보여줄 되묻기 답변 (토큰·역할·문서 ID 만, 실명 없음 — 실명 보기로 복원 가능)."""
    lines = []
    for a in amb:
        lines.append(f"질문 속 {a['query_token']} 와 이름이 같은 사람이 문서에 {len(a['candidates'])}명 있어 "
                     "누구인지 정할 수 없습니다.")
        for tok, role in a["candidates"]:
            ids = [f"[{d['doc_id']}] {d['doc_type']}" for d in cands if tok in d["tokens"]]
            lines.append(f"- {tok} ({_ROLE_KO.get(role, '역할 모름')}): {', '.join(ids) or '-'}")
    lines.append("\n어느 분인지 알려 주세요 (고객/직원, 소속 팀, 연락처, 문서 종류 등).")
    return "\n".join(lines)


_CITE_WORD = re.compile(r"\[\s*문서\s*([A-Z]\d{4})\s*\]")  # 추론을 끄면 '[문서 T0139]' 로 씀 → '[T0139]' 로 통일


def _complete(masked_q, context, model, on_delta=None):
    """LLM 호출. on_delta 를 주면 스트리밍: 글자가 올 때마다 지금까지의 글을 넘긴다 (출력 검사 전 글이므로 화면 표시용으로만)."""
    payload = {"model": model, "temperature": 0,
               "messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": f"[문서]\n{context}\n\n[질문]\n{masked_q}"}]}
    if config.LLM_REASONING_EFFORT and model == "HCX-007":
        payload["reasoning_effort"] = config.LLM_REASONING_EFFORT
    r = requests.post(config.LLM_API_BASE.rstrip("/") + "/chat/completions",
                      headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
                      json={**payload, "stream": on_delta is not None}, timeout=120, stream=on_delta is not None)
    r.raise_for_status()
    if on_delta is None:
        return r.json()["choices"][0]["message"]["content"]
    text = ""
    for line in r.iter_lines():
        if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
            continue
        choices = json.loads(line[5:]).get("choices") or [{}]
        piece = (choices[0].get("delta") or {}).get("content")
        if piece:
            text += piece
            on_delta(text)
    return text


def answer(query, k=3, model=None, mode="hybrid", embed_retries=None, rerank=None, on_delta=None):
    t0 = time.perf_counter()
    masked_q, q_tokens, cands = search(query, k * 2, mode=mode, embed_retries=embed_retries, rerank=rerank)
    docs = select_docs(cands, k)
    t_search = time.perf_counter() - t0
    context, g1 = pii_pipeline.guard(_context(docs))  # 적재 때 이미 가렸지만, LLM 로 보내기 직전에 한 번 더
    masked_q, g2 = pii_pipeline.guard(masked_q)
    result = {"masked_query": masked_q, "query_tokens": q_tokens, "docs": docs, "answer": None,
              "raw_answer": None, "check": None, "model": model or config.LLM_MODEL,
              "timing": {"search": t_search}, "guard": g1 + g2, "ask_back": False}
    amb = cands[0].get("ambiguous_people") if cands else None
    if amb and config.ASK_BACK:
        result["answer"], result["ask_back"] = ask_back(amb, cands), True
        result["check"] = {"unverified_tokens": [], "raw_pii": []}
        return result
    if not config.LLM_API_KEY:
        return result
    t1 = time.perf_counter()
    raw = _CITE_WORD.sub(r"[\1]", _complete(masked_q, context, result["model"], on_delta))
    result["timing"]["llm"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    allowed = set(pii_pipeline.TOKEN_PATTERN.findall(context + "\n" + masked_q))
    safe = sorted({d["doc_type"] for d in docs if d.get("doc_type")}, key=len, reverse=True)
    result["answer"], result["check"] = check_output(raw, allowed, safe)
    result["raw_answer"] = raw
    result["timing"]["check"] = time.perf_counter() - t2
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--model")
    a = ap.parse_args()
    res = answer(a.query, a.k, a.model)
    print(f"질의(가명처리): {res['masked_query']}")
    print("참고 문서:", ", ".join(d["doc_id"] for d in res["docs"]))
    print("\n답변:", res["answer"] or "(LLM_API_KEY 없음 — 검색 결과만 표시)")
    if res["check"] and (res["check"]["unverified_tokens"] or res["check"]["raw_pii"]):
        print("\n⚠ 출력 검사:", res["check"])
    print("\n시간:", {k: f"{v:.1f}s" for k, v in res["timing"].items()})


if __name__ == "__main__":
    main()
