"""에이전트: ① 보호(가명처리) → ② 분류 → ③ 계획 → ④ 실행 (1라운드 병렬 → 다시 계획 → 추가 검색).
종합·출력 재검사·복원은 RAG 들이 완성된 뒤 (worklog 16장).

    python -m app.agents.agent "박현준 고객 보이스피싱 건, 피해금 환급 절차가 법에 어떻게 돼?" --no-db

외부 LLM(분류·계획)에는 가명처리된 질문만 간다. 가명처리가 계획보다 먼저인 이유.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from app.pii import pii_pipeline

from . import planner, router
from .tools import SEARCHERS, SOURCES


def protect(question, use_db=True):
    """(가명처리된 질문, 가린 원문 목록). use_db=False 면 DB 없이 탐지기만 (DB 의 기존 인물 잇기·이름 보강 없음)"""
    if use_db:
        from app.rag.search import mask_query
        masked, records = mask_query(question)
    else:
        spans = sorted(pii_pipeline.detect(question), key=lambda s: s["start"])
        masked, records = pii_pipeline.pseudonymize(question, spans)
        masked = pii_pipeline.guard(masked)[0]
    return masked, [r["original"] for r in records]


def plan_question(question, role="STAFF", use_db=True, model=None, provider=None, masked=None, plan_model=None):
    t0 = time.perf_counter()
    hidden = []
    if masked is None:
        masked, hidden = protect(question, use_db)
    t_protect = time.perf_counter() - t0
    route, m1 = router.classify(masked, role, model, provider)
    plan, m2 = planner.make_plan(masked, route["sources"], plan_model or model, provider)
    metas = [m for m in (m1, m2) if m]
    return {
        "masked_query": masked, "hidden": hidden, "route": route, "plan": plan,
        "timing": {"protect": round(t_protect, 3), "classify": m1["latency"], "plan": m2["latency"] if m2 else 0.0},
        "tokens": {"in": sum(m["in_tokens"] for m in metas), "out": sum(m["out_tokens"] for m in metas)},
        "cost_usd": sum(m["cost_usd"] for m in metas),
        "raw": {"classify": m1["raw"], "plan": m2["raw"] if m2 else None},
    }


def _run_steps(steps, searchers, k):
    """한 라운드의 검색을 병렬로. 도구가 없거나 실패하면 그 단계만 오류로 남기고 계속"""
    def one(s):
        fn = searchers.get(s["tool"])
        if fn is None:
            return s["id"], [], f"{s['tool']} 미등록"
        t0 = time.perf_counter()
        try:
            return s["id"], list(fn(s["query"], k) or []), None, round(time.perf_counter() - t0, 3)
        except Exception as e:  # 한 RAG 가 죽어도 나머지 결과로 답하게
            return s["id"], [], f"{type(e).__name__}: {e}", round(time.perf_counter() - t0, 3)
    if not steps:
        return {}, {}, {}
    with ThreadPoolExecutor(min(4, len(steps))) as ex:
        out = list(ex.map(one, steps))
    results = {o[0]: o[1] for o in out}
    errors = {o[0]: o[2] for o in out if o[2]}
    times = {o[0]: o[3] for o in out if len(o) > 3}
    return results, errors, times


def run_plan(masked_q, plan, searchers=None, k=5, model=None, provider=None):
    """④ 실행. 1라운드(지금 할 검색)를 병렬로 → later 가 있으면 사내 결과로 다시 계획(최대 1번) → 추가 검색 병렬.
    searchers: {도구: fn(질의, k)} (기본: tools.SEARCHERS). 결과는 모두 가명처리된 글."""
    searchers = SEARCHERS if searchers is None else searchers
    t0 = time.perf_counter()
    results, errors, times = _run_steps(plan["steps"], searchers, k)
    t_round1 = time.perf_counter() - t0
    rp, meta = planner.replan(masked_q, plan, results, model, provider)
    t1 = time.perf_counter()
    r2, e2, tm2 = _run_steps(rp["steps"], searchers, k)
    results.update(r2), errors.update(e2), times.update(tm2)
    return {
        "steps": plan["steps"] + rp["steps"], "results": results, "errors": errors, "replan": rp,
        "timing": {"round1": round(t_round1, 3), "replan": meta["latency"] if meta else 0.0,
                   "round2": round(time.perf_counter() - t1, 3), "steps": times},
        "cost_usd": meta["cost_usd"] if meta else 0.0,
    }


_SOURCE_KO = {"internal": "사내 기록", "law": "법령", "product": "금융상품", "noaction": "비조치의견서"}
OUT_OF_SCOPE_MSG = "사내 업무·금융 관련 질문에만 답할 수 있습니다. 찾으시는 고객·처리 건이나 업무 내용을 적어 주세요."
CLARIFY_MSG = "어떤 건을 말씀하시는지 알 수 없습니다. 고객 이름, 처리 건 종류(예: 보이스피싱 접수, 우편 반송), 날짜 등을 함께 적어 주세요."


def ask(question, user, k=3):
    """화면 질문 1건: ① 보호 → ② 분류 → 사내 RAG 답변. (worklog 25장)
    지금 등록된 RAG 는 사내(개인정보) 하나뿐이라 계획·다시 계획은 건너뜀 — 계획이 고를 도구가 사내 검색 하나라
    단계가 늘 [search_internal(질문)] 이기 때문. 공개 RAG 가 등록되면 run_plan + 종합으로 바꾼다.
    공개 자료가 필요하다고 분류됐는데 아직 없는 RAG 는 pending 으로 알려 주고, 사내 기록이 필요하면 그 부분만 답한다.
    사내 검색은 원문 질문으로 부른다 (이름 → 토큰·동명이인 구분이 원문 이름을 써야 해서. 원문은 이 서버 밖으로 안 나감).
    반환은 chat.answer 와 같은 꼴 + agent 정보."""
    from app.rag import chat as chat_mod

    t0 = time.perf_counter()
    masked, _ = protect(question)
    t_protect = time.perf_counter() - t0
    role = user.get("role", "STAFF")
    try:
        route, meta = router.classify(masked, role)
    except Exception as e:  # 분류 LLM 이 죽으면 예전 챗봇처럼 사내 검색 (router.validate 의 실패 처리와 같음)
        route = {"sources": ["internal"], "route": "internal", "reason": f"분류 실패({type(e).__name__}) → 기본(사내)",
                 "fallback": True}
        meta = {"model": None, "latency": 0.0, "cost_usd": 0.0}
    pending = [s for s in route["sources"] if SOURCES[s] not in SEARCHERS and s != "internal"]
    info = {"route": route["route"], "sources": route["sources"], "reason": route["reason"],
            "fallback": route.get("fallback", False), "pending": pending,
            "model": meta.get("model"), "cost_usd": round(meta.get("cost_usd") or 0.0, 6)}
    timing = {"protect": round(t_protect, 3), "classify": round(meta.get("latency") or 0.0, 3)}

    if "internal" in route["sources"]:
        note = None
        if pending:  # 혼합 질문: 공개 자료 부분까지 답하려다 '찾을 수 없습니다'로 끝나던 것 방지 (25장)
            names = ", ".join(_SOURCE_KO.get(s, s) for s in pending)
            note = (f"질문 중 {names} 자료가 필요한 부분은 아직 검색할 수 없어 [문서]에 없습니다. "
                    "그 부분은 빼고 사내 기록 부분만 답하세요.")
        r = chat_mod.answer(question, k=k, user=user, note=note)
        r["timing"] = {**timing, **r["timing"]}
        r["agent"] = info
        return r
    if route["route"] == "out_of_scope":
        text = OUT_OF_SCOPE_MSG
    elif route["route"] == "clarify":
        text = CLARIFY_MSG
    else:  # 공개 자료만 필요한데 그 RAG 가 아직 없음
        names = ", ".join(_SOURCE_KO.get(s, s) for s in pending)
        text = f"이 질문은 {names} 자료가 필요합니다. 해당 검색은 아직 준비 중이라 지금은 답할 수 없습니다."
    return {"masked_query": masked, "answer": text, "ask_back": False, "docs": [],
            "check": {"unverified_tokens": [], "raw_pii": []}, "model": None, "timing": timing, "agent": info}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--role", default="STAFF")
    ap.add_argument("--no-db", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--provider")
    a = ap.parse_args()
    out = plan_question(a.question, a.role, not a.no_db, a.model, a.provider)
    out.pop("hidden")  # 원문은 화면에 찍지 않음
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
