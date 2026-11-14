"""① 질문 분류: 어떤 출처(RAG)가 필요한가. 가명처리된 질문만 받는다.

LLM 은 출처 목록만 고르고, 경로(route)는 코드가 출처에서 정한다 (LLM 이 둘을 따로 말하면 서로 어긋날 수 있음).
  internal = 사내만, public = 공개만, mixed = 둘 다, out_of_scope = 답할 범위 밖,
  clarify = 무엇을 묻는지 몰라 되물어야 함 ("그 건 괜찮아?")
"""
import re

from . import llm
from .tools import PUBLIC_SOURCES, SOURCES, TOOLS, allowed_sources

ROUTER_PROMPT = """너는 금융사 사내 챗봇의 질문 분류기다. 질문에 답하려면 어떤 자료가 필요한지 고른다.

자료
{sources}

판단 기준
1. internal 은 "특정 고객·직원·처리 건의 사내 기록 내용(날짜·금액·처리 상태·진행 단계 등)이 답에 필요할 때"만 고른다.
   질문에 사람 이름이나 <PERSON_…> 같은 토큰이 있어도, 실제로 묻는 것이 법·상품·사례의 일반 내용이면 internal 을 고르지 않는다.
   예: "<PERSON_ab12cd34> 고객 보이스피싱 건, 피해금 환급 절차가 법에 어떻게 돼?" → law 만 (환급 절차는 법의 일반 내용)
   예: "<PERSON_ab12cd34> 고객 보이스피싱 건, 법에서 정한 지급정지 조치가 다 됐어?" → internal + law (그 건의 처리 상태가 필요)
   말을 전한 사람("김 대리가 그러는데", "고객이 물어보는데")은 기록을 찾는 대상이 아니다.
   예: "옆자리 과장님이 그러던데 퇴직금은 1년 넘어야 나와?" → law 만
2. 이름이 없어도 사내 처리 기록을 찾는 질문은 internal: "지난달 반송 건", "이번 감사 지적 사항",
   그리고 그런 건이 있었는지 묻는 말("~건 있으면", "~있었어?", "~있나?", "~한 적 있는지", "들어온 거").
   예: "카드 불법 모집 건 있으면 관련 법도 같이 봐줘" → internal + law (건이 있는지는 사내 기록, 그다음 법)
3. 공개 자료는 질문이 묻는 내용에 맞는 것만 고른다. law = 법령 조문·의무·기준, product = 상품 조건·추천,
   noaction = 당국이 '이렇게 해도 되나'에 답한 사례. 두 가지 이상을 함께 물으면 모두 고른다.
4. 사내 기록과 공개 자료를 엮는 질문(기록이 법에 맞는지, 기록 속 사건과 비슷한 사례, 기록 기준으로 맞는 상품)은 둘 다 고른다.
   기록이 기한·기준을 지켰는지("기한 안에 처리됐어?", "늦게 한 건", "제대로 했어?")는 기준이 법이므로 internal + law.
5. 무엇을 가리키는지 알 수 없는 질문("그 건", "이거", "저번 거")은 clarify=true, sources=[] (되물어야 함).
   토큰(<PERSON_…> 등)이나 사건 종류가 있으면 clarify 가 아니다 (사내 기록에서 찾으면 됨).
6. 금융 업무와 관계없는 질문(잡담, 인사, 날씨, 코딩, 투자 종목 추천 등)은 out_of_scope=true, sources=[].

JSON 으로만 답한다:
{{"sources": ["internal" | "law" | "product" | "noaction", ...], "out_of_scope": true | false, "clarify": true | false, "reason": "한 문장"}}"""

ROUTES = ("internal", "public", "mixed", "out_of_scope", "clarify")
_TOKEN = re.compile(r"<[A-Za-z_]+_[0-9A-Za-z]+>")
_LEFTOVER = re.compile(r"(고객님|고객|님|씨|사원|주임|대리|과장|차장|부장|팀장|[\s,.?!])*")


def _listing(role):
    srcs = allowed_sources(role)
    return "\n".join(f"- {s}: {TOOLS[SOURCES[s]]['desc']}" for s in SOURCES if s in srcs)


def route_of(sources):
    s = set(sources)
    if not s:
        return "out_of_scope"
    if s == {"internal"}:
        return "internal"
    return "mixed" if "internal" in s else "public"


def validate(raw, role):
    """LLM 출력 검사. 허용되지 않은 출처는 버리고, 못 읽었으면 사내 검색 하나로 (기존 챗봇과 같은 동작)"""
    if raw is None:
        return {"sources": ["internal"], "route": "internal", "reason": "분류 실패 → 기본(사내)", "fallback": True}
    allowed = allowed_sources(role)
    sources = [s for s in SOURCES if s in (raw.get("sources") or []) and s in allowed]  # 순서 고정 + 중복 제거
    if raw.get("clarify") is True and not sources:
        return {"sources": [], "route": "clarify", "reason": str(raw.get("reason") or ""), "fallback": False}
    if raw.get("out_of_scope") is True and not sources:
        return {"sources": [], "route": "out_of_scope", "reason": str(raw.get("reason") or ""), "fallback": False}
    if not sources:
        sources = ["internal"]
    return {"sources": sources, "route": route_of(sources), "reason": str(raw.get("reason") or ""), "fallback": False}


def name_only(masked_q):
    """질문이 토큰(+ '고객'·직급)뿐인가. 이름만 넣은 질문 = 그 사람 기록 찾기 (hard_v5: 두 모델 다 범위 밖으로 보냄)"""
    return bool(_TOKEN.search(masked_q)) and _LEFTOVER.fullmatch(_TOKEN.sub("", masked_q)) is not None


def classify(masked_q, role="STAFF", model=None, provider=None):
    if name_only(masked_q) and "internal" in allowed_sources(role):
        meta = {"model": "rule", "latency": 0.0, "in_tokens": 0, "out_tokens": 0, "cost_usd": 0.0, "raw": None}
        return {"sources": ["internal"], "route": "internal", "reason": "이름만 있음 → 그 사람 사내 기록 (규칙)", "fallback": False}, meta
    raw, meta = llm.chat_json(ROUTER_PROMPT.format(sources=_listing(role)), masked_q, model, provider)
    r = validate(raw, role)
    if r["route"] == "clarify" and _TOKEN.search(masked_q) and "internal" in allowed_sources(role):
        # 대상(토큰)이 있으면 되묻지 않고 그 사람 기록부터 (hard_v6 mini: "<PERSON_…> 고객 건 괜찮아?" 를 되물음)
        r = {**r, "sources": ["internal"], "route": "internal", "reason": r["reason"] + " (토큰 있음 → 사내)"}
    return r, meta


__all__ = ["classify", "validate", "route_of", "ROUTES", "PUBLIC_SOURCES"]
