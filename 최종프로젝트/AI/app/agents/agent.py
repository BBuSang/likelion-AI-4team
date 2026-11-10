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
from .tools import SEARCHERS


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
