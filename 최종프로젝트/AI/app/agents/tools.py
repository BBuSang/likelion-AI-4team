"""도구(=RAG) 목록. 설명(desc)이 곧 분류·계획 기준이라 '무엇이 있고 무엇이 없는지'를 구체적으로 쓴다.

사내 RAG 1개(개인정보, 가명처리) + 공개 RAG 3개(법령·금융상품·비조치의견서, 팀원 담당).
쟁점 추출 도구는 없앰 — 사내 결과를 보고 공개 검색어를 정하는 일은 planner.replan (worklog 16-9).
공개 RAG 설명은 담당자 확인 전 초안 (worklog 16-2).
"""

# 출처 이름 → 도구 이름. 분류기는 출처를, 계획기는 도구를 다룬다
SOURCES = {
    "internal": "search_internal",
    "law": "search_law",
    "product": "search_product",
    "noaction": "search_noaction",
}
PUBLIC_SOURCES = {"law", "product", "noaction"}

TOOLS = {
    "search_internal": {
        "source": "internal", "roles": {"STAFF", "LEAD", "AUDITOR"},
        "desc": ("사내 업무 기록(가명처리됨): 고객 상담·민원·VOC·보이스피싱 피해 접수·우편 반송 처리·해외송금 접수·"
                 "카드 발급 심사·한도 심사·대리인 위임 확인·대출 상담·연체 안내·법인카드 소명·감사 지적·사내 메신저·회의록·"
                 "인사 발령 등. 특정 고객·직원·처리 건의 사실(날짜·금액·처리 상태·진행 단계) 확인. "
                 "권한 밖 문서는 검색 전에 걸러짐."),
    },
    "search_law": {
        "source": "law", "roles": {"STAFF", "LEAD", "AUDITOR"},
        "desc": ("법령·시행령·감독규정 조문: 개인정보 보호법, 신용정보법, 여신전문금융업법, 전자금융거래법, "
                 "외국환거래법, 통신사기피해환급법, 금융소비자보호법, 근로기준법 등. 의무·금지·책임·기한·금액 기준. "
                 "특정 고객 정보는 없음."),
    },
    "search_product": {
        "source": "product", "roles": {"STAFF", "LEAD", "AUDITOR"},
        "desc": ("금융상품 설명서·약관: 신용·체크카드, 대출, 예금·적금 상품의 연회비·금리·한도·수수료·혜택·가입 조건. "
                 "상품 비교·추천. 특정 고객 정보는 없음."),
    },
    "search_noaction": {
        "source": "noaction", "roles": {"STAFF", "LEAD", "AUDITOR"},
        "desc": ("금융위원회·금융감독원 비조치의견서·법령해석 회신: 새 서비스·업무 방식이 규제 위반인지 당국이 미리 "
                 "판단한 사례(질의 요지, 회신, 이유). 제휴·데이터 활용·비대면 절차 같은 '이렇게 해도 되나' 질문. "
                 "특정 고객 정보는 없음."),
    },
}
PUBLIC_TOOLS = {SOURCES[s] for s in PUBLIC_SOURCES}
SEARCHERS = {}  # 도구 이름 → fn(가명처리 질의, k) → [{'text': 가명처리 글, ...}]. 각 RAG 가 완성되면 register 로 등록


def register(tool_name, fn):
    assert tool_name in TOOLS, tool_name
    SEARCHERS[tool_name] = fn


def allowed_tools(role):
    """역할이 못 쓰는 도구는 LLM 에 보여 주지도 않는다 (프롬프트로 막는 것보다 목록에서 빼는 게 안전)"""
    return {name: t for name, t in TOOLS.items() if role in t["roles"]}


def allowed_sources(role):
    tools = allowed_tools(role)
    return {s for s, name in SOURCES.items() if name in tools}
