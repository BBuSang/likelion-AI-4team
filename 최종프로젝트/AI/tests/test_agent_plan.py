"""분류·계획·실행 검사 코드 (LLM 없이). LLM 이 어떤 JSON 을 줘도 안전한 계획이 나오는지."""
from app.agents import agent, llm, planner, router

Q = "<PERSON_ab12cd34> 고객 보이스피싱 건, 피해금 환급 절차가 법에 어떻게 돼?"
Q_NO_ISSUE = "<PERSON_ab12cd34> 고객 건 법적으로 문제 있어?"


def test_router_validate():
    assert router.validate(None, "STAFF")["route"] == "internal"  # 못 읽으면 기존 챗봇처럼 사내
    assert router.validate({"sources": [], "out_of_scope": True}, "STAFF")["route"] == "out_of_scope"
    v = router.validate({"sources": ["law", "internal", "law", "web"]}, "STAFF")
    assert v["sources"] == ["internal", "law"] and v["route"] == "mixed"
    assert router.validate({"sources": ["product", "noaction"]}, "STAFF")["route"] == "public"
    assert router.validate({"sources": [], "clarify": True}, "STAFF")["route"] == "clarify"
    assert router.validate({"sources": ["law"], "clarify": True}, "STAFF")["route"] == "public"  # 출처 고르면 되묻지 않음


def test_name_only_goes_internal_without_llm():
    assert router.name_only("<PERSON_ab12cd34>") and router.name_only("<PERSON_ab12cd34> 고객님?")
    assert not router.name_only("<PERSON_ab12cd34> 고객 연체") and not router.name_only("체크카드")
    r, meta = router.classify("<PERSON_ab12cd34> 대리")
    assert r["route"] == "internal" and meta["cost_usd"] == 0


def test_public_query_loses_tokens_and_numbers():
    raw = {"pattern": "single", "steps": [{"id": 1, "tool": "search_law", "query": Q + " 010-1234-5678"}]}
    p = planner.validate_plan(raw, ["law"], Q)
    q = p["steps"][0]["query"]
    assert "<" not in q and "1234" not in q and "고객" not in q and "환급" in q
    assert any("토큰" in f for f in p["fixes"])


def test_keeps_customer_word_without_token():
    assert "고객" in planner.strip_public("개인정보 유출 시 고객 통지 기한")


def test_drops_unchosen_tool_and_adds_missing():
    raw = {"pattern": "parallel", "steps": [{"id": 1, "tool": "search_internal", "query": Q},
                                             {"id": 2, "tool": "search_product", "query": "카드"}]}
    p = planner.validate_plan(raw, ["law"], Q)
    assert [s["tool"] for s in p["steps"]] == ["search_law"] and p["pattern"] == "single"


def test_later_makes_sequential_and_skips_now_search():
    raw = {"pattern": "parallel", "steps": [{"id": 1, "tool": "search_internal", "query": Q_NO_ISSUE},
                                             {"id": 2, "tool": "search_law", "query": "법"}],
           "later": ["search_law", "law", "search_internal", "web"]}
    p = planner.validate_plan(raw, ["internal", "law"], Q_NO_ISSUE)
    assert p["later"] == ["search_law"] and p["pattern"] == "sequential"
    assert [s["tool"] for s in p["steps"]] == ["search_internal"]  # later 도구의 지금 검색은 버림


def test_later_needs_internal_and_old_format_moves_to_later():
    p = planner.validate_plan({"steps": [{"id": 1, "tool": "search_law", "query": "x"}], "later": ["search_law"]},
                              ["law"], "x")
    assert p["later"] == [] and p["pattern"] == "single"
    old = {"steps": [{"id": 1, "tool": "search_internal", "query": Q_NO_ISSUE}, {"id": 2, "tool": "extract_issue", "input": 1},
                     {"id": 3, "tool": "search_law", "query_from": 2, "focus": "관련 법"}]}
    p = planner.validate_plan(old, ["internal", "law"], Q_NO_ISSUE)
    assert p["later"] == ["search_law"] and [s["tool"] for s in p["steps"]] == ["search_internal"]
    # 사내 단계를 빼먹고 later 만 줘도 사내 검색을 넣음
    p = planner.validate_plan({"steps": [], "later": ["search_law"]}, ["internal", "law"], Q_NO_ISSUE)
    assert [s["tool"] for s in p["steps"]] == ["search_internal"] and p["later"] == ["search_law"]


def test_garbage():
    raw = {"steps": [{"id": 1, "tool": "search_law", "query_from": 9}, "x", {"id": "a", "tool": "search_law"}]}
    p = planner.validate_plan(raw, ["law"], Q)
    assert len(p["steps"]) == 1 and p["steps"][0]["tool"] == "search_law" and "query" in p["steps"][0]
    assert planner.validate_plan(None, ["internal"], Q)["steps"][0]["query"] == Q


def _plan_with_later():
    return planner.validate_plan({"pattern": "parallel", "steps": [{"id": 1, "tool": "search_internal", "query": Q_NO_ISSUE}],
                                  "later": ["search_law"]}, ["internal", "law"], Q_NO_ISSUE)


def test_validate_replan():
    plan = _plan_with_later()
    raw = {"steps": [{"tool": "search_law", "query": "<PERSON_ab12cd34> 보이스피싱 지급정지 010-1234-5678"},
                     {"tool": "search_product", "query": "카드"}, {"tool": "search_law", "query": "환급"},
                     {"tool": "search_law", "query": "세번째"}, {"tool": "search_law", "query": "<PERSON_ab12cd34>"}]}
    r = planner.validate_replan(raw, plan)
    qs = [s["query"] for s in r["steps"]]
    assert qs == ["보이스피싱 지급정지", "환급"]  # 다른 도구·3번째·빈 검색어 버림, 토큰·번호 지움
    assert all(s["round"] == 2 and s["id"] > 1 for s in r["steps"])
    assert planner.validate_replan(None, plan)["steps"] == []


def test_replan_skips_llm_when_no_internal_hits(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LLM 을 부르면 안 됨")
    monkeypatch.setattr(llm, "chat_json", boom)
    plan = _plan_with_later()
    r, meta = planner.replan(Q_NO_ISSUE, plan, {1: []})
    assert r["steps"] == [] and meta is None
    simple = planner.validate_plan({"steps": [{"id": 1, "tool": "search_law", "query": "x"}]}, ["law"], "x")
    assert planner.replan("x", simple, {1: ["a"]})[1] is None  # later 없으면 다시 계획 안 함


def test_replan_input_masks_numbers_and_limits():
    plan = _plan_with_later()
    hits = [{"text": "보이스피싱 접수 010-1234-5678"}] + [{"text": f"기록{i}"} for i in range(10)]
    text, n = planner.replan_input(Q_NO_ISSUE, plan, {1: hits})
    assert n == planner.REPLAN_HITS and "1234-5678" not in text


def test_run_plan_two_rounds(monkeypatch):
    calls = []

    def fake_chat(system, user, model=None, provider=None):
        assert "보이스피싱 지급정지 지연" in user  # 사내 결과가 다시 계획에 들어감
        return {"steps": [{"tool": "search_law", "query": "보이스피싱 지급정지 지연"}]}, {
            "latency": 0.1, "cost_usd": 0.001, "in_tokens": 1, "out_tokens": 1, "model": "fake", "raw": ""}
    monkeypatch.setattr(llm, "chat_json", fake_chat)
    searchers = {
        "search_internal": lambda q, k: calls.append(("internal", q)) or [{"text": "<PERSON_ab12cd34> 보이스피싱 지급정지 지연"}],
        "search_law": lambda q, k: calls.append(("law", q)) or [{"text": "통신사기피해환급법 제4조"}],
    }
    out = agent.run_plan(Q_NO_ISSUE, _plan_with_later(), searchers)
    assert calls == [("internal", Q_NO_ISSUE), ("law", "보이스피싱 지급정지 지연")]
    assert [s["tool"] for s in out["steps"]] == ["search_internal", "search_law"] and not out["errors"]
    assert out["results"][2][0]["text"].startswith("통신사기")


def test_run_plan_survives_missing_or_broken_tool():
    plan = planner.validate_plan({"pattern": "parallel", "steps": [{"id": 1, "tool": "search_law", "query": "a"},
                                                                    {"id": 2, "tool": "search_product", "query": "b"}]},
                                 ["law", "product"], "a b")
    out = agent.run_plan("a b", plan, {"search_law": lambda q, k: 1 / 0})
    assert set(out["errors"]) == {1, 2} and out["results"] == {1: [], 2: []}
