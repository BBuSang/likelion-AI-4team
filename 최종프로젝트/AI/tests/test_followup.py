"""멀티턴: 분류의 followup 판단, 다시 쓰기 검사, 토큰 → 원문 되돌리기, 앞 대화 있을 때만 다시 쓰기"""
from app.agents import followup, router

H1 = {"query": "황시온 고객 우편물 반송 처리 어디까지 됐어?",
      "masked_query": "<PERSON_ca80d53c> 고객 우편물 반송 처리 어디까지 됐어?",
      "answer": "<PERSON_ca80d53c> 고객의 우편물은 반송 후 주소 확인 중입니다 [H0071]. 담당 <PERSON_11112222> 대리가 처리했습니다."}


def test_router_followup_flag():
    r = router.validate({"sources": [], "clarify": True, "followup": True, "reason": "x"}, "STAFF")
    assert r["route"] == "clarify" and r["followup"] is True
    r = router.validate({"sources": ["law"], "followup": True}, "STAFF")
    assert r["route"] == "public" and r["followup"] is True
    assert router.validate({"sources": ["internal"]}, "STAFF")["followup"] is False
    assert router.validate(None, "STAFF")["followup"] is False       # 분류 실패 → 다시 쓰기 안 함


def test_recent_keeps_last_turns_only():
    hist = [{"masked_query": f"q{i}", "answer": "a"} for i in range(5)] + [{"masked_query": "  "}, "x"]
    assert [h["masked_query"] for h in followup.recent(hist)] == ["q3", "q4"]
    assert followup.recent(None) == [] and followup.recent([]) == []


def test_rewrite_sends_short_context_and_rejects_made_up_tokens(monkeypatch):
    sent = {}

    def fake(system, user, *a, **k):
        sent["user"] = user
        return {"question": fake.q}, {"cost_usd": 0.0001}
    monkeypatch.setattr(followup.llm, "chat_json", fake)

    fake.q = "<PERSON_ca80d53c> 고객 다른 민원 기록 있어?"
    q, _ = followup.rewrite("그 고객 다른 민원도 있어?", [H1])
    assert q == fake.q
    assert "[앞 질문 1]" in sent["user"] and "[지금 질문] 그 고객 다른 민원도 있어?" in sent["user"]
    assert "황시온" not in sent["user"]                                # 원문은 LLM 에 안 감

    long = {**H1, "answer": "가" * 1000}
    followup.rewrite("그럼?", [long])
    assert "가" * followup.ANSWER_CHARS + "…" in sent["user"] and "가" * (followup.ANSWER_CHARS + 1) not in sent["user"]

    fake.q = "<PERSON_99999999> 고객 민원"                            # 앞 대화에 없는 토큰 → 버림
    q, meta = followup.rewrite("그 고객 민원?", [H1])
    assert q is None and meta["rejected"] == "unknown_token"
    fake.q = ""                                                      # 대상을 못 정함
    assert followup.rewrite("그거?", [H1])[0] is None
    assert followup.rewrite("그거?", [])[0] is None                  # 앞 대화 없으면 LLM 도 안 부름


def test_to_raw_uses_question_pairs_then_vault(monkeypatch):
    import app.rag.search as search
    from app.pii import restore

    def fake_mask(text):
        recs = [{"token": "<PERSON_ca80d53c>", "original": "황시온"}] if "황시온" in text else []
        return text, recs
    monkeypatch.setattr(search, "mask_query", fake_mask)
    monkeypatch.setattr(restore, "lookup", lambda toks: {t: {"display": "김담당", "label": "person", "pid": "P7"}
                                                         for t in toks if t == "<PERSON_11112222>"})
    out, pinned = followup.to_raw("<PERSON_ca80d53c> 고객 건 <PERSON_11112222> 대리가 처리했어? <PERSON_00000000>",
                                  "그 고객 건 담당 대리가 처리했어?", [H1])
    assert out == "황시온 고객 건 김담당 대리가 처리했어? <PERSON_00000000>"   # 모르는 토큰은 그대로
    assert pinned == {"김담당": "<PERSON_11112222>"}   # 앞 답변 속 확정된 사람은 검색에서 그 사람으로 고정
