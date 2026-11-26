"""공개 RAG 연결 — DB·LLM 없이 도는 부분"""
from app.agents import agent, planner


def test_public_query_has_no_names_or_numbers():
    q = planner.strip_public("<PERSON_261a4fef> 고객 보이스피싱 건, 010-1234-5678 피해금 환급 절차가 법에 어떻게 돼?")
    assert "<PERSON" not in q and "고객" not in q and "010" not in q
    assert "보이스피싱" in q and "환급" in q


def test_renumber_per_source_and_dedupe():
    ev = [{"source": "law", "cite_id": "x", "title": "A법 제1조", "text": "a"},
          {"source": "law", "cite_id": "y", "title": "A법 제1조", "text": "a"},
          {"source": "product", "cite_id": "z", "title": "적금 설명서", "text": "b"},
          {"source": "law", "cite_id": "w", "title": "B법 제2조", "text": "c"}]
    out = agent._renumber(ev)
    assert [e["cite_id"] for e in out] == ["LAW-1", "PRD-1", "LAW-2"]


def test_public_docs_shape_for_api():
    docs = agent._public_docs([{"source": "law", "cite_id": "LAW-1", "title": "A법 제1조", "text": "본문", "score": 0.7}])
    d = docs[0]
    assert d["doc_id"] == "LAW-1" and d["masked_text"] == "본문" and d["doc_type"].startswith("법령")


def test_public_docs_case_type_is_its_own_gubun():
    """회신 사례는 '비조치의견서 · ' 를 붙이지 않고 제목 앞 구분(법령해석·현장건의 등) 그대로"""
    ev = [{"source": "noaction", "cite_id": "L5453", "title": "법령해석 · 금융투자업인가", "text": "본문"},
          {"source": "noaction", "cite_id": "F7", "title": "현장건의 · 금융교육", "text": "본문"}]
    assert [d["doc_type"] for d in agent._public_docs(ev)] == ["법령해석 · 금융투자업인가", "현장건의 · 금융교육"]


def test_law_adapter_unavailable_without_password(monkeypatch):
    from app.core import config
    from app.rag_public import law
    monkeypatch.setattr(config, "LAW_PASSWORD", "")
    assert law.available() is False


def test_noaction_unavailable_without_password(monkeypatch):
    from app.core import config
    from app.rag_public import noaction
    monkeypatch.setattr(config, "NOACTION_PASSWORD", "")
    assert noaction.available() is False


def test_product_unavailable_without_vectors(monkeypatch, tmp_path):
    from app.rag_public import product
    monkeypatch.setattr(product, "VECTORS", tmp_path / "vectors.npz")
    assert product.available() is False


def test_manual_law_aliases_point_to_official_names():
    from app.rag_public.noaction.aliases import ALIASES
    assert len(ALIASES) >= 100
    assert ALIASES["자본시장법"] == "자본시장과 금융투자업에 관한 법률"
    assert all(isinstance(v, str) and v for v in ALIASES.values())


def test_noaction_period_split():
    from datetime import date
    from app.rag_public.noaction.respond import split_period
    topic, period = split_period("2023년 이후 회신 중에 적금 캐시백이 특별이익인지 본 것 있어?")
    assert period and period["kind"] == "from" and period["from"] == date(2023, 1, 1) and "2023" not in topic
    assert split_period("적금 캐시백이 특별이익 제공에 해당하나요?")[1] is None


def test_public_answer_keeps_words_misread_as_names(monkeypatch):
    """'후보 조문만으로는' 의 '조문'을 이름 모델이 사람 이름으로 잡아 가리던 것: 공개 자료만 쓴 답은 이름을 안 가림"""
    from app.pii import pii_pipeline
    from app.rag import chat
    text = "후보 조문만으로는 판단할 수 없습니다. 문의 010-1234-5678"
    at = text.index("조문")
    ph = text.index("010")
    monkeypatch.setattr(pii_pipeline, "detect_output", lambda t: [
        {"label": "person", "start": at, "end": at + 2}, {"label": "phone_number", "start": ph, "end": len(text)}])
    kept, _ = chat.check_output(text, set(), skip_labels=("person",))
    assert "조문" in kept and "010-1234-5678" not in kept  # 이름은 그대로, 번호 모양은 여전히 가림
    masked, _ = chat.check_output(text, set())
    assert "조문" not in masked  # 사내 문서가 있는 답은 이름도 그대로 검사


def test_checked_stream_sends_only_checked_sentences(monkeypatch):
    """답변 스트리밍: 문장·줄이 끝나야 내보내고, 내보내기 전에 출력 검사를 거침"""
    from app.pii import pii_pipeline
    from app.rag import chat

    def fake_detect(t):  # '홍길동' 을 이름으로 잡는 가짜 탐지기
        i = t.find("홍길동")
        return [{"label": "person", "start": i, "end": i + 3}] if i >= 0 else []
    monkeypatch.setattr(pii_pipeline, "detect_output", fake_detect)
    out = []
    s = chat.CheckedStream(out.append, allowed={"<PERSON_ab12cd34>"})
    full = "담당은 <PERSON_ab12cd34> 입니다. 확인자는 홍길동입니다.\n끝"
    for i in range(1, len(full) + 1):  # 한 글자씩 오는 것처럼
        s.feed(full[:i])
        assert all("<PERSON_ab" not in o or "<PERSON_ab12cd34>" in o for o in out)  # 토큰이 반쯤 잘려 나가지 않음
    assert out == ["담당은 <PERSON_ab12cd34> 입니다. "] + out[1:]  # 첫 문장이 끝나자마자 나감
    assert "끝" not in "".join(out)  # 끝나지 않은 줄은 아직 안 나감
    s.finish(full)
    joined = "".join(out)
    assert "홍길동" not in joined and "<PERSON_" in joined.split("확인자는")[1]  # 검사로 가린 뒤에만 나감
    assert joined.endswith("끝")


def test_ensure_registered_rechecks_missing_rag(monkeypatch):
    """DB 가 늦게 켜져도 서버를 다시 켜지 않고 RECHECK_SEC 뒤에 등록됨"""
    import app.rag_public as rp
    from app.agents import tools

    monkeypatch.setattr(tools, "SEARCHERS", {})
    monkeypatch.setattr(rp, "SEARCHERS", tools.SEARCHERS)
    monkeypatch.setattr(rp, "_checked_at", None)
    up = {"law": False}
    monkeypatch.setattr(rp.law, "available", lambda: up["law"])
    monkeypatch.setattr(rp.noaction, "available", lambda: False)
    monkeypatch.setattr(rp.product, "available", lambda: False)

    assert rp.ensure_registered(now=0) == set()          # 처음: DB 꺼짐
    up["law"] = True
    assert rp.ensure_registered(now=10) == set()         # 간격 전엔 다시 안 봄
    assert rp.ensure_registered(now=61) == {"search_law"}  # 간격 지나면 등록
    calls = []
    monkeypatch.setattr(rp.law, "available", lambda: calls.append(1) or True)
    rp.ensure_registered(now=200)
    assert calls == []                                   # 이미 등록된 건 다시 확인 안 함


def test_weak_query_name_only_drops_ordinary_words():
    """질문 속 이름 오탐: 규칙만 이름으로 본 구간 중 보통 말로 쪼개지는 것만 풀고, 낯선 이름·호칭 붙은 이름은 그대로"""
    from app.rag.search import _weak_query_name

    def span(q, w, flag="disagree", src="rule"):
        i = q.index(w)
        return {"label": "person", "start": i, "end": i + len(w), "src": src, "flag": flag}

    q1 = "지주 자회사 법인카드 한도 크게 주려면 이사회 승인 필요해요?"
    assert _weak_query_name(q1, span(q1, "주려면"))            # 주/VV + 려면/EC, '이사회'는 호칭 아님
    q2 = "신한 골드리슈 금적립 상품 핵심만 정리해줘"
    assert _weak_query_name(q2, span(q2, "금적립"))            # 금/NNG + 적립/NNG
    q3 = "육가은 포인트이관신청 내역 보여줘"
    assert not _weak_query_name(q3, span(q3, "육가은"))        # 낯선 이름은 한 덩어리 → 계속 가림
    q4 = "금적립 고객 민원"
    assert not _weak_query_name(q4, span(q4, "금적립"))        # 뒤에 호칭 → 이름으로 둠
    assert not _weak_query_name(q1, span(q1, "주려면", flag=None))  # 모델도 이름으로 본 건 그대로
    assert not _weak_query_name(q1, span(q1, "주려면", src="model"))


def test_law_legal_query_adds_statute_terms():
    """흔한 말 → 조문 용어 덧붙이기"""
    from importlib import import_module
    S = import_module("app.rag_public.law.search")
    assert S.legal_query("보이스피싱 관련 법") == "보이스피싱 관련 법 전기통신금융사기"
    assert S.legal_query("스미싱·보이스피싱 피해") == "스미싱·보이스피싱 피해 전기통신금융사기"   # 중복 없이
    assert S.legal_query("전기통신금융사기 보이스피싱") == "전기통신금융사기 보이스피싱"            # 이미 있으면 그대로
    assert S.legal_query("예금자보호 한도") == "예금자보호 한도"


def test_query_sido_only_when_region_meant():
    """시·도만 있는 지역 질문: 긴 이름은 그대로, 줄임말은 '사는·거주·쪽·지역' 이 붙을 때만"""
    from app.rag.search import query_sido
    assert query_sido("충청남도에 사는 고객 찾아줘") == ["충남"]
    assert query_sido("부산에 사는 고객 문서 있어?") == ["부산"]
    assert query_sido("충남 지역 고객") == ["충남"]
    assert query_sido("서울 쪽 거주 고객") == ["서울"]
    assert query_sido("부산 지점 공지 있어?") == []          # 사는 곳이 아님
    assert query_sido("서울보증 관련 문서") == []
    assert query_sido("대출 상담 기록") == []
