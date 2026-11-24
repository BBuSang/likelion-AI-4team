"""공개 자료 RAG 3개 — 개인정보가 없는 자료라 HMAC 토큰화 없이, 사내 RAG 와 저장소·계정을 나눠 둔다.

| 패키지   | 자료                                                   | 저장                          | 설계 (팀원)          |
|----------|--------------------------------------------------------|-------------------------------|----------------------|
| law      | 금융 법령 380개 (법률·시행령·시행규칙·감독규정)        | DB lawrag, law_reader         | 백송이 (baek_rag)    |
| product  | 신한은행 판매 중 상품 391개의 설명서·약관 247개        | 파일 색인 data/public_rag/product | 이한별 (hh)       |
| noaction | 금융당국 법령해석·비조치의견서 회신, 현장건의           | DB interp, interp_reader      | 박지훈 (kloid_rag)   |

공통 결과 모양 (Evidence): {"source", "cite_id", "title", "text", "score", "meta"}. cite_id 앞머리 LAW- / PRD- / CASE-
에이전트는 공개 RAG 로 보내는 검색어에서 토큰·이름·번호를 지운 뒤에 부른다 (agents/planner.strip_public).
"""
import threading
import time

from app.agents.tools import SEARCHERS, register

from . import law, noaction, product

PACKAGES = {"search_law": law, "search_product": product, "search_noaction": noaction}
RECHECK_SEC = 60  # 준비 안 된 RAG 를 다시 확인하는 간격
_checked_at = None
_lock = threading.Lock()


def ensure_registered(now=None):
    """데이터가 있는 RAG 만 에이전트 도구로 등록 (없는 건 에이전트가 '준비 중'으로 안내).
    한 번 등록된 건 그대로 두고, 아직 없는 건 RECHECK_SEC 마다 다시 확인한다 (AI 서버가 DB 보다 먼저 켜지면 법령·비조치의견서가 서버를 다시 켤 때까지 '준비 중'으로 남던 문제)"""
    global _checked_at
    now = time.monotonic() if now is None else now
    with _lock:
        missing = [t for t in PACKAGES if t not in SEARCHERS]
        if missing and (_checked_at is None or now - _checked_at >= RECHECK_SEC):
            for tool in missing:
                if PACKAGES[tool].available():
                    register(tool, PACKAGES[tool].search)
            _checked_at = now
    return set(SEARCHERS)


def module_for(source):
    return {"law": law, "product": product, "noaction": noaction}[source]
