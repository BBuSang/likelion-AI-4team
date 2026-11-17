"""6-1 답변 평가 — 검색 상위 k개로 답을 만들고, 정답과 맞는지 채점

사용: python analysis/s15_answer_eval.py dev [k ...] [--version=v1|v2|v3|v4] [--retrieval=...]
      python analysis/s15_answer_eval.py test 5 --retrieval=rerank_score --final   (최종 방식 확정 뒤 한 번만)     (기본 k = 1 5, 기본 v2)
  v1: 근거 본문만 (6-1) / v2: 근거마다 문서 층·질문 상품 연결 여부 표시 + 과한 유보 금지 + 상품 특약 우선 (6-2)
  v3: v2 + 특약 우선은 모순일 때만, 특약이 넘긴 값은 설명서의 구체 값을 함께
  v4: v3 + 우선순위는 같은 상황을 다룰 때만 (기본)
  v5: v4 + 없는 값 안내(이미 본 문서 다시 보라고 하지 않음, 상품명 못 찾으면 이름 확인 요청) (8-3)
  --retrieval=rerank(목록형 재순위, 기본) | rerank_score(점수형 재순위) | routed(재순위 없음)
- 질문: 검색 평가셋 dev 105개 (정답 = 질문을 만든 섹션의 짧은 답 + 근거 원문). test는 잠가 둠
- 검색: 확정 방식 (s12 rerank: .env 임베딩 + 상품명 좁히기 + 결합 + 재순위, 재순위는 캐시 사용)
- 답변: CHAT_MODEL에 상위 k개를 [1]~[k] 번호와 함께 주고, docs/답변원칙.md의 원칙 중 검색 답변에 해당하는 것만 넣음
  (근거에 있는 것만, 출처 번호 표시, 숫자에는 근거에 적힌 기준일, 근거에 없으면 '보유 문서에서 확인되지 않음')
- 채점: CHAT_MODEL이 '정답 + 근거 원문'과 비교해 4가지로 판정 — correct / partial / wrong / no_answer
  · 같은 모델이 답하고 채점하므로 후하게 줄 수 있음 → 결과 일부를 사람이 직접 검수 (6-1 문서)
- 검색에 정답 섹션이 있었는지(gold_in_context)도 함께 기록 → '다른 문서로 맞게 답함'을 셀 수 있음
결과: data/eval/answer_eval_dev.jsonl (질문·k·답변·판정·이유), 이미 한 것은 다시 하지 않음
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import s12_retrieval as R  # noqa: E402
from s8_extract import load_env  # noqa: E402

EVAL = ROOT / "data" / "eval"
VERSION = "v2"
RETR = "rerank"

ANSWER_PROMPT_V1 = """너는 신한은행 영업점 직원을 돕는 상품 안내 도우미다. 아래 [근거]만 보고 직원의 질문에 한국어로 답한다.
원칙
1. 근거에 있는 내용만 답한다. 근거에 없으면 추측하지 말고 "보유 문서에서 확인되지 않습니다"라고 하고, 확인할 곳(상품설명서·영업점)을 안내한다.
2. 문장마다 근거 번호를 [1]처럼 붙인다.
3. 금리·금액 같은 숫자를 말할 때 근거에 기준일(예: "2026.10.16 현재, 세전")이 있으면 함께 적는다.
4. 질문한 상품과 다른 상품의 근거는 쓰지 않는다. 근거가 상품 공통 약관이면 그렇다고 밝힌다.
5. 대출 공통 설명서의 예시 숫자를 그 상품의 실제 값처럼 말하지 않는다.
6. 2~4문장으로 짧게."""

# v4 (6-3): v3 + 우선순위는 같은 상황을 다룰 때만 (다른 상황의 특약 조항을 앞세우지 않음)
# v3 (6-2): v2 + '특약 우선'은 모순일 때만, 특약이 위임한 값은 설명서의 구체 값을 함께
# v2 (6-2): 근거마다 '문서 층'과 '질문 상품에 연결된 문서인지'를 붙이고, 과한 유보를 막고, 문서끼리 다르면 상품 특약 우선
ANSWER_PROMPT = """너는 신한은행 영업점 직원을 돕는 상품 안내 도우미다. 아래 [근거]만 보고 직원의 질문에 한국어로 답한다.
근거마다 머리에 문서 정보가 있다: 문서 층(설명서 / 상품 특약 / 공통 약관 / 약정서 / 공통 조항 / 용어 안내)과, 질문한 상품에 연결된 문서인지.
원칙
1. 근거에 있는 내용만 답한다. 근거에 없으면 추측하지 말고 "보유 문서에서 확인되지 않습니다"라고 하고, 확인할 곳(상품설명서·영업점)을 안내한다.
2. 문장마다 근거 번호를 [1]처럼 붙인다.
3. 금리·금액 같은 숫자를 말할 때 근거에 기준일(예: "2026.10.16 현재, 세전")이 있으면 함께 적는다.
4. **'질문 상품에 연결된 문서'로 표시된 근거는 그 상품에 적용되는 문서다.** 공통 약관·약정서·대출 공통 설명서여도 "이 상품에 적용되는지 확인되지 않는다"고 하지 말고, "○○ 기준"처럼 어느 문서 기준인지만 밝혀 답한다.
   '다른 상품 문서'로 표시된 근거는 쓰지 않는다.
5. 문서끼리 **서로 모순될 때** 상품 특약을 우선한다 — 각 상품 특약 제1조가 "이 특약에서 정하지 않은 사항은 예금거래기본약관 등을 적용한다"고 정하고,
   예금거래기본약관 제23조(약관적용의 순서)는 예금별 약관을 기본약관보다 먼저 적용한다고 정한다. 설명서의 요약 문장과 특약 규정이 모순되면 특약 규정을 따른다.
   모순이 아니라 **특약이 "고시한 이율"처럼 값을 다른 곳에 넘기고 설명서가 구체적인 값(이율·산식·기준일)을 적었으면, 설명서의 구체 값을 함께 쓴다.**
   우선순위는 **같은 상황을 다룰 때만** 따진다. 특약 조항이 다른 상황을 다루면(예: '재예치 후 만기 전 해지'와 '계약기간 연장 후 당초 만기일이 지난 뒤 해지')
   특약을 앞세우지 말고, 질문한 상황을 직접 다루는 조항(공통 약관이라도)으로 답한다.
   시행일처럼 문서마다 값이 따로 있으면 어느 문서의 값인지 밝히고, 필요하면 문서별 값을 함께 알린다.
6. 대출 공통 설명서의 예시 숫자를 그 상품의 실제 값처럼 말하지 않는다.
7. 2~4문장으로 짧게."""

# v5 (8-3): v4 + '없는 값' 안내 개선 — ① 이미 근거로 본 문서를 다시 보라고 하지 않고 실제 확인할 곳 안내
#            ② 상품명 찾기가 질문 속 상품을 못 찾았으면 "보유 상품 목록에서 찾지 못함 → 정확한 이름 확인" (없는 상품이라고 단정하지 않음 — 줄임말일 수 있음)
V5_RULE1 = """1. 근거에 있는 내용만 답한다. 근거에 없으면 추측하지 말고 "보유 문서에서 확인되지 않습니다"라고 한다.
   확인할 곳을 안내할 때 **이미 근거로 받은 문서를 다시 보라고 하지 않는다** (그 상품의 설명서가 근거에 있는데 답이 없으면 "상품설명서에서 확인"이라고 하지 않음).
   대출의 상품별 한도·기간·금리·수수료율은 "여신 시스템 또는 영업점 여신 담당", 예금·적금·기타 상품은 "영업점 또는 신한은행 홈페이지 상품 안내"처럼 실제로 확인할 수 있는 곳을 안내한다."""
V5_RULE8 = """
8. 입력 맨 앞 [상품명 찾기]가 '찾지 못함'인데 질문이 **특정 상품 이름**을 말하면:
   - 근거에 그 이름과 같은 상품으로 보이는 문서가 있으면 그 문서로 답하되, 어느 상품 문서 기준인지 밝힌다 (줄임말로 부른 실제 상품일 수 있음).
   - 그런 문서가 없으면 먼저 "보유 상품 목록에서 '(질문의 상품 이름)' 상품을 찾지 못했습니다"라고 말하고 정확한 상품명(정식 이름) 확인을 부탁한다.
     이름이 맞다면 새 상품이거나 보유 문서에 없는 상품일 수 있으니 "영업점 또는 신한은행 홈페이지 상품 안내"에서 확인하도록 함께 안내한다.
     없는 상품이라고 단정하지 않고, 다른 상품의 값·조건으로 대신 답하지 않는다. 이 안내 문장들에는 근거 번호를 붙이지 않는다.
   - 질문이 특정 상품이 아니라 공통 규정·용어를 묻는 것이면 이 규칙은 적용하지 않는다.
9. [상품명 찾기]에 '(추정)'이 붙어 있으면 질문의 호칭을 그 상품으로 추정한 것이다. 답 첫머리에 "질문하신 '(호칭)'을 '(상품 이름)'으로 보고 답합니다"라고 밝힌다 (9-1)."""


b_v3 = ANSWER_PROMPT[ANSWER_PROMPT.index("5. 문서끼리"):ANSWER_PROMPT.index("6. 대출")]
b_v4_extra = ANSWER_PROMPT[ANSWER_PROMPT.index("   우선순위는 **같은 상황"):ANSWER_PROMPT.index("   시행일처럼")]
b_v2 = """5. 같은 주제를 문서마다 다르게 적었으면 **상품 특약을 우선**한다 — 각 상품 특약 제1조가 "이 특약에서 정하지 않은 사항은 예금거래기본약관 등을 적용한다"고 정하고,
   예금거래기본약관 제23조(약관적용의 순서)는 예금별 약관을 기본약관보다 먼저 적용한다고 정한다. 설명서의 요약 문장과 특약 규정이 다르면 특약 규정을 따른다.
   시행일처럼 문서마다 값이 따로 있으면 어느 문서의 값인지 밝히고, 필요하면 문서별 값을 함께 알린다.
"""

LAYER = {"상품설명서": "설명서(안내·요약)", "간이투자설명서": "설명서(안내·요약)", "투자설명서": "설명서(안내·요약)",
         "특약": "상품 특약(이 상품 전용 규정)", "약관": "공통 약관(여러 상품 공통 규정)", "약정서": "약정서"}


def context_header(idx, i, docs, linked_docs, linked_clauses, pnames):
    """근거 머리: 문서 층 + 질문 상품에 연결된 문서인지 (상품은 질문에서 찾은 것 — 실제 서비스와 같은 방식)"""
    it = idx.items[i]
    if it["kind"] == "term":
        return "[용어 안내]"
    if it["kind"] == "clause":
        tag = f"질문 상품({pnames})에 연결된 문서에 들어 있는 공통 조항" if it["clause_id"] in linked_clauses else "여러 상품 공통 조항"
        return f"[공통 조항 · {tag}]"
    d = docs[it["doc_id"]]
    layer = LAYER.get(d["doc_type"], d["doc_type"])
    if not linked_docs:
        link = "상품을 특정하지 못한 질문"
    else:
        link = f"질문 상품({pnames})에 연결된 문서" if it["doc_id"] in linked_docs else "다른 상품 문서"
    return f"[{layer} · {link}]"


V6_RULE10 = """
10. 입력 맨 앞 [판매 상태]를 따른다 (상품공시 목록 기준: 판매 중 목록 2026-11-06, 판매 중지 목록 2026-11-09).
   - 판매 중지: 답 첫머리에 "'(상품 이름)'은 판매 중지된 상품이에요. 신규 가입은 안 돼요"라고 말한다. 판매중지 날짜는 말하지 않는다.
     판매 중지 상품의 설명서는 보유하지 않으므로 금리·한도 같은 세부 조건은 지어내지 말고, 기존 가입 고객 문의는 영업점에서 확인하도록 안내한다.
     근거에 다른 상품의 문서만 있으면 그 내용으로 답하지 않는다. 판매 중지로 나온 상품에는 8번('찾지 못함' 안내)을 쓰지 않는다.
   - 판매 중·판매 중지 양쪽에 같은 이름: 판매 중지라고 단정하지 말고, 판매 중인 같은 이름의 상품이 있다고 알린 뒤 판매 중 상품 기준으로 답한다.
   - 판매 중: 판매 여부·신규 가입 여부를 물으면 "판매 중인 상품이에요(상품공시 목록 기준)"라고 답하고 근거로 안내한다."""


def status_header(q, idx):
    """[판매 상태] 머리 (11-2): 판매 중지·양쪽 이름은 목록 대조, 판매 중은 상품명 찾기로 찾은 상품"""
    lines = []
    for s in idx.sale_status(q):
        if s.get("inferred"):   # 11-5: 줄임말을 판매 중지 목록의 상품으로 추정
            lines.append(f"{s['name']}: 판매 중지 (질문 속 호칭을 판매 중지 목록의 이 상품으로 추정 — 답할 때 추정임을 밝힐 것)")
            continue
        lines.append(f"{s['name']}: " + ("판매 중지 (판매 중지 목록에만 있음)" if s["status"] == "discontinued" else "판매 중 목록과 판매 중지 목록에 같은 이름이 모두 있음"))
    names = {s["name"] for s in idx.sale_status(q)}
    for pid in sorted(set().union(*idx.detect_groups(q)) if idx.detect_groups(q) else set()):
        n = idx._pid_name.get(pid) if getattr(idx, "_pid_name", None) else None
        if n and n not in names:
            lines.append(f"{n}: 판매 중 (판매 중 목록)")
            names.add(n)
    return "[판매 상태] " + ("; ".join(lines) if lines else "질문에서 목록의 상품 이름을 찾지 못함")


def source_lines(idx, q, answer, ctx_ids, min_score=1):
    """15-1 답변 아래 근거 문서 — 코드로 만듦 (LLM 없음, 답변 내용은 그대로)
    근거 문서: 답변 속 [n]으로 실제 인용한 근거 / 함께 볼 문서: 인용하지 않은 근거 중 재순위 점수 min_score 이상(0점 = 무관·다른 상품은 뺌)
    같은 조항이 원문 문서 조각과 공통 조항으로 두 번 오면 하나만 (문서 이름·쪽이 있는 원문 쪽을 남김)"""
    import re as _re
    scores = getattr(idx, "_q_scores", {}).get(q, {})

    def label(i):
        h = idx.items[i]["text"].split("\n", 1)[0].replace("\x07", "")
        m = _re.match(r"\[([^\]]+)\]\s*(.*)", h)
        return f"{m.group(1)} > {m.group(2)}" if m and m.group(2) else h

    def topic(i):   # 중복 판단용: 문서 이름을 뺀 섹션 제목
        h = idx.items[i]["text"].split("\n", 1)[0]
        return _re.sub(r"\s*\(p\.\d+\)$", "", _re.sub(r"^\[[^\]]+\]\s*", "", h)).replace(" ", "")

    cited = []
    for n in _re.findall(r"\[(\d+)\]", answer):
        k = int(n) - 1
        if 0 <= k < len(ctx_ids) and k not in cited:
            cited.append(k)
    is_cl = lambda k: idx.items[ctx_ids[k]]["kind"] == "clause"
    seen = [(topic(ctx_ids[k]), is_cl(k)) for k in cited]
    # 판매 중인 대상 상품을 찾지 못했으면(판매 중지·목록에 없는 이름) 근거는 그 상품의 자료가 아니므로 '함께 볼 문서'를 보이지 않음
    groups = idx.detect_groups(q)
    has_target = bool(set().union(*groups) if len(groups) >= 2 else idx.detect_products(q))
    rest = [k for k in range(len(ctx_ids)) if has_target and k not in cited and scores.get(ctx_ids[k], 0) >= min_score]
    rest.sort(key=is_cl)   # 원문 조각을 먼저 보고 같은 공통 조항은 건너뜀
    extra = []
    for k in rest:
        t, c = topic(ctx_ids[k]), is_cl(k)
        # 중복은 '공통 조항 ↔ 원문 조각' 사이에서만 봄 (원문 조각끼리는 제목이 비슷해도 다른 문서)
        if t and any((c or sc) and c != sc and (t in s or s in t) for s, sc in seen):
            continue
        seen.append((t, c))
        extra.append(k)
    extra.sort()
    lines = ["근거 문서: " + ("인용한 근거 없음" if not cited else "")]
    lines += [f"  [{k + 1}] {label(ctx_ids[k])}" for k in sorted(cited)]
    if extra:
        lines.append("함께 볼 문서:")
        lines += [f"  - {label(ctx_ids[k])}" for k in extra]
    return "\n".join(lines).replace("근거 문서: \n", "근거 문서:\n")


def product_line(idx, q, max_names=3):
    """15-1 답변 맨 위 '대상 상품' 한 줄 — 어떤 상품으로 답했는지와 판매 상태를 늘 같은 자리에 (LLM 없이 코드로, 답변 내용은 그대로)
    판매 중: 상품명 찾기 결과 · 판매 중지/양쪽 이름: 판매 상태 대조 결과 · 추정: 별칭이 애매해 LLM이 확인했거나(verified) 별칭 없이 골랐을 때(fallback)"""
    parts, seen = [], set()
    for s in idx.sale_status(q):
        if s["name"] in seen:
            continue
        seen.add(s["name"])
        if s["status"] == "discontinued":
            parts.append(f"{s['name']} (판매 중지{' · 추정' if s.get('inferred') else ''})")
        else:
            parts.append(f"{s['name']} (판매 중·판매 중지 목록에 같은 이름)")
    groups = idx.detect_groups(q)
    pids = set().union(*groups) if len(groups) >= 2 else idx.detect_products(q)
    inferred = len(groups) < 2 and idx.detect_how(q) in ("verified", "fallback")
    if getattr(idx, "_pid_name", None) is None:
        idx.alias_ambiguous([])
    names = []
    for g in (groups if len(groups) >= 2 else [pids]):
        for p in sorted(g):
            n = idx._pid_name.get(p)
            if n and n not in seen and n not in names:
                names.append(n)
    shown = names[:max_names]
    parts += [f"{n} (판매 중{' · 추정' if inferred else ''})" for n in shown]
    if len(names) > max_names:
        parts.append(f"외 {len(names) - max_names}개")
    return "대상 상품: " + (" / ".join(parts) if parts else "찾지 못함 (질문에서 보유 상품 목록의 이름을 찾지 못함)")


V7_RULE11 = """
11. 질문이 두 개 이상의 상품을 비교하면, 상품마다 **그 상품에 연결된 근거에서만** 값을 가져와 "상품 A: …, 상품 B: …"처럼 밝힌 뒤 비교 결론을 말한다.
   한 상품의 근거에 값이 없으면 그 상품만 "보유 문서에서 확인되지 않습니다"라고 하고, 다른 상품의 값으로 대신하지 않는다. 기준일이 다르면 함께 적는다."""


SPLIT_QUESTIONS = __import__("os").environ.get("SPLIT_QUESTIONS", "1") == "1"   # 16-1: 한 메시지에 질문이 여러 개면 나눠서 질문마다 근거 (기본 켬, 끄기: SPLIT_QUESTIONS=0)


def split_question(q):
    """16-1 질문 나누기 (코드만, LLM 없음): 물음표 뒤에서 나누고, 없으면 '그리고' 앞뒤로. 각 부분이 6글자 이상일 때만, 최대 3개"""
    import re as _re
    parts = [p.strip() for p in _re.split(r"(?<=[?？])\s+", q.strip()) if p.strip()]
    if len(parts) < 2:
        parts = [p.strip() for p in _re.split(r"\s+그리고\s+", q.strip()) if p.strip()]
    parts = [p for p in parts if len(_re.sub(r"\s", "", p)) >= 6]
    return parts[:3] if len(parts) >= 2 else [q]


def context_ids(idx, q, qv, version, k=5, per=3):
    """근거 고르기 (11-3): v7부터 질문에 상품이 둘 이상이면 상품마다 점수형 재순위 상위 per개씩 (최대 3개 상품), 아니면 상위 k개
    16-1 (SPLIT_QUESTIONS=1): 상품이 하나 이하이고 질문이 여러 개면, 질문마다(전체 질문에서 찾은 상품으로 범위를 좁혀) 상위 per개씩 — 재순위는 동시에"""
    groups = idx.detect_groups(q) if version >= "v7" else []
    if SPLIT_QUESTIONS and len(groups) < 2:
        parts = split_question(q)
        if len(parts) >= 2:
            from concurrent.futures import ThreadPoolExecutor
            pids = idx.detect_products(q) or None
            if not hasattr(idx, "_client"):
                from openai import OpenAI
                from s8_extract import load_env
                env = load_env()
                idx._client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
            with ThreadPoolExecutor(3) as ex:
                ranked = list(ex.map(lambda p: idx.rank("rerank_score", p, idx.text_vec(p), pids=pids)[:per], parts))
            qs_ = idx.__dict__.setdefault("_q_scores", {})
            for p in parts:   # 근거 문서·함께 볼 문서(15-1)가 전체 질문으로 점수를 찾을 수 있게
                qs_.setdefault(q, {}).update(qs_.get(p, {}))
            out = []
            for r in ranked:
                for i in r:
                    if int(i) not in out:
                        out.append(int(i))
            return out, groups
    if len(groups) >= 2:
        # 14-1: 상품별 재순위(LLM)를 차례로 부르지 않고 동시에 — 결과·순서는 같음
        from concurrent.futures import ThreadPoolExecutor
        if not hasattr(idx, "_client"):   # 클라이언트를 미리 만들어 두어 스레드끼리 동시에 만들지 않게
            from openai import OpenAI
            from s8_extract import load_env
            env = load_env()
            idx._client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
        with ThreadPoolExecutor(3) as ex:
            ranked = list(ex.map(lambda g: idx.rank("rerank_score", q, qv, pids=g)[:per], groups[:3]))
        out = []
        for r in ranked:
            for i in r:
                if int(i) not in out:
                    out.append(int(i))
        return out, groups
    return [int(i) for i in idx.rank("rerank_score", q, qv)[:k]], groups


def multi_header(idx, i, docs, groups, pid_name):
    """여러 상품일 때 근거 머리: 어느 상품에 연결된 문서인지"""
    it = idx.items[i]
    if it["kind"] == "term":
        return "[용어 안내]"
    own = []
    for g in groups:
        ds = set().union(*(idx.links[p] for p in g))
        cl = set().union(*(idx.doc_clause[d] for d in ds))
        if (it["kind"] == "chunk" and it["doc_id"] in ds) or (it["kind"] == "clause" and it["clause_id"] in cl):
            own.append(", ".join(sorted(pid_name[p] for p in g))[:40])
    if it["kind"] == "clause":
        return f"[공통 조항 · {' / '.join(own) + '에 연결' if own else '여러 상품 공통'}]"
    layer = LAYER.get(docs[it["doc_id"]]["doc_type"], docs[it["doc_id"]]["doc_type"])
    return f"[{layer} · {(' / '.join(own) + '에 연결된 문서') if own else '질문 상품과 연결 없음'}]"


def extra_context(q, idx):
    return status_header(q, idx)


V8_RULE10B = """
   - (11-5) [판매 상태]에 **추정한 판매 중지 상품과 판매 중 상품이 함께** 있으면, 어느 한쪽으로 정해 답하지 말고 "판매 중지된 ○○과 판매 중인 △△이 있어요"라고 둘 다 알린 뒤
     어느 상품인지 확인을 부탁한다. 판매 중 상품의 정보는 그 상품이 맞다면이라는 전제로만 덧붙인다."""


def finalize_answer(version, ans, how, pnames):
    """v9 (11-6): 상품명을 추정으로 찾았는데(verified·fallback) 답변이 그 사실을 밝히지 않았으면 코드가 맨 앞에 붙임 — 모델이 규칙을 빼먹어도 늘 표시"""
    if version < "v9" or how not in ("verified", "fallback") or not pnames:
        return ans
    head = ans[:160]
    if "보고 답" in head or "추정" in head or "뜻하는 것으로" in head:
        return ans
    return f"(질문 속 상품 이름을 '{pnames}'(으)로 보고 답합니다.) " + ans


LIST_RULE = __import__("os").environ.get("LIST_RULE", "0") == "1"   # 17-1 시험: 목록형 답은 빠짐없이 (켜기: LIST_RULE=1)
V12_LIST = """
12. 질문이 대상·조건·서류·방법·제한처럼 여러 항목으로 답해야 하는 것을 묻고 근거에 항목이 나열돼 있으면, 근거의 항목을 하나도 빼지 말고 모두 적는다.
   7번(짧게)보다 이 규칙이 먼저다 — 항목은 쉼표로 이어 쓰고, 항목이 아닌 설명을 줄인다."""


def prompt_of(version):
    if version == "v9":
        return prompt_of("v8") + (V12_LIST if LIST_RULE else "")
    if version == "v8":
        return prompt_of("v7") + V8_RULE10B
    if version == "v7":
        return prompt_of("v6") + V7_RULE11
    if version == "v6":
        return prompt_of("v5") + V6_RULE10
    if version == "v5":
        p = ANSWER_PROMPT
        r1 = p[p.index("1. 근거에 있는"):p.index("2. 문장마다")]
        return p.replace(r1, V5_RULE1 + "\n") + V5_RULE8
    return {"v4": ANSWER_PROMPT, "v3": ANSWER_PROMPT.replace(b_v4_extra, ""),
            "v2": ANSWER_PROMPT.replace(b_v3, b_v2)}[version]


def answer_input(version, ctx, question, pnames, inferred=False):
    """v5부터 입력 맨 앞에 상품명 찾기 결과를 알려 줌 (서비스에서도 알 수 있는 정보). inferred: 2단계(줄임말 추정)로 찾음 — 9-1"""
    head = (f"[상품명 찾기] {pnames}{' (추정)' if inferred else ''}\n\n" if pnames else "[상품명 찾기] 찾지 못함 (질문에서 보유 상품 목록의 이름을 찾지 못함)\n\n") if version >= "v5" else ""
    return f"{head}[근거]\n{ctx}\n\n[질문] {question}"


JUDGE_PROMPT = """너는 은행 상품 안내 답변의 채점자다. [질문], [정답](짧은 답과 근거 원문), [답변]을 비교해 판정한다.
- correct: 질문의 핵심을 정답과 같은 내용으로 답함 (표현이 달라도 됨, 정답에 없는 내용을 더 말해도 정답과 모순되지 않으면 correct)
- partial: 방향은 맞지만 핵심 조건·숫자 일부가 빠지거나 덜 정확함
- wrong: 정답과 다른 숫자·조건을 말하거나 다른 상품 내용으로 답함
- no_answer: 확인되지 않는다고 하거나 답하지 않음
reason에 한두 문장으로 이유를 적는다."""


class Judge(BaseModel):
    verdict: Literal["correct", "partial", "wrong", "no_answer"]
    reason: str


def main():
    global VERSION, RETR
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    VERSION = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--version=")), "v4")
    split = args[0] if args else "dev"
    if split != "dev" and "--final" not in sys.argv:
        sys.exit("test는 잠가 둔 시험 세트 — 방식을 다 정한 뒤 --final을 붙여 한 번만 실행")
    ks = [int(x) for x in args[1:]] or [1, 5]
    import csv
    docs = {r["doc_id"]: r for r in csv.DictReader(open(ROOT / "data" / "meta" / "documents.csv"))}
    prod_names = {r["product_id"]: r["name"] for r in csv.DictReader(open(ROOT / "data" / "meta" / "products.csv"))}
    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
    qs = [json.loads(l) for l in open(EVAL / f"retrieval_{split}.jsonl")]
    idx = R.Index()
    qv = R.embed_questions(qs, idx)
    RETR = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--retrieval=")), "rerank")
    orders = [idx.rank(RETR, q["question"], v)[:max(ks)] for q, v in zip(qs, qv)]
    qv_by = {q["qid"]: v for q, v in zip(qs, qv)}

    out_path = EVAL / f"answer_eval_{split}.jsonl"
    done = {}
    if out_path.exists():
        for l in open(out_path):
            r = json.loads(l)
            done[r["key"]] = {**r, "version": r.get("version", "v1")}

    def job(args):
        q, order, k = args
        groups = []
        if VERSION >= "v7" and RETR == "rerank_score":   # 11-3: 상품이 둘 이상이면 상품마다 근거
            ctx_ids, groups = context_ids(idx, q["question"], qv_by[q["qid"]], VERSION, k=k)
        else:
            ctx_ids = [int(i) for i in order[:k]]
        key = hashlib.sha256(f"{env['EMBED_MODEL']}|{env['CHAT_MODEL']}|{q['qid']}|{k}|{ctx_ids}|{VERSION}{'|' + hashlib.sha256(prompt_of(VERSION).encode()).hexdigest()[:12] if VERSION >= 'v5' else ''}".encode()).hexdigest()
        # 검색 방식이 달라도 근거(ctx_ids)가 같으면 같은 답변을 재사용 (key에 근거가 들어 있음)
        if key in done:
            return done[key], False
        if VERSION == "v1":
            ctx = "\n\n".join(f"[{n}] {idx.items[i]['text'][:2000]}" for n, i in enumerate(ctx_ids, 1))
            prompt = ANSWER_PROMPT_V1
        else:
            pids = set().union(*groups) if len(groups) >= 2 else idx.detect_products(q["question"])
            linked = set().union(*(idx.links[p] for p in pids)) if pids else set()
            lclauses = set().union(*(idx.doc_clause[d] for d in linked)) if linked else set()
            pnames = ", ".join(sorted(prod_names[p] for p in pids))[:120 if len(groups) >= 2 else 60]
            hdr = (lambda i: multi_header(idx, i, docs, groups, prod_names)) if len(groups) >= 2 else (lambda i: context_header(idx, i, docs, linked, lclauses, pnames))
            ctx = "\n\n".join(f"[{n}] {hdr(i)}\n{idx.items[i]['text'][:2000]}" for n, i in enumerate(ctx_ids, 1))
            prompt = prompt_of(VERSION)
        status = (status_header(q["question"], idx) + "\n\n") if VERSION >= "v6" else ""
        ans = client.responses.create(model=env["CHAT_MODEL"], instructions=prompt,
                                      input=status + answer_input(VERSION, ctx, q["question"], pnames if VERSION != "v1" else "",
                                                         inferred=VERSION != "v1" and bool(pids) and idx.detect_how(q["question"]) != "exact")).output_text
        if VERSION != "v1":
            ans = finalize_answer(VERSION, ans, idx.detect_how(q["question"]), pnames)
        j = client.responses.parse(model=env["CHAT_MODEL"], instructions=JUDGE_PROMPT, text_format=Judge,
                                   input=f"[질문] {q['question']}\n[정답] {q['answer']}\n[정답 근거 원문] {q['evidence']}\n\n[답변] {ans}").output_parsed
        rec = {"key": key, "version": VERSION, "retrieval": RETR, "qid": q["qid"], "type": q["type"], "k": k, "question": q["question"], "gold_answer": q["answer"],
               "gold_in_context": any(idx.is_gold(i, q["gold"]) for i in ctx_ids), "answer": ans,
               "verdict": j.verdict, "reason": j.reason, "context": [idx.items[i]["text"].split("\n", 1)[0] for i in ctx_ids]}
        return rec, True

    only = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--only=")), None)
    jobs = [(q, o, k) for k in ks for q, o in zip(qs, orders) if not only or q["qid"] in only]
    print(f"답변 {len(jobs)}개 (새로 {sum(1 for q, o, k in jobs if True)}개 이하)", flush=True)
    recs = []
    with ThreadPoolExecutor(6) as ex, open(out_path, "a") as f:
        for rec, new in ex.map(job, jobs):
            if new:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            recs.append(rec)

    if only:
        for r in recs:
            print(f"\n[{r['qid']} k={r['k']} {r['verdict']}] {r['question']}\n  답변: {r['answer']}\n  채점: {r['reason']}")
        return
    for k in ks:
        rs = [r for r in recs if r["k"] == k]
        c = Counter(r["verdict"] for r in rs)
        n = len(rs)
        print(f"\n[k={k}] 질문 {n} | correct {c['correct']} ({c['correct'] / n:.0%}) · partial {c['partial']} · wrong {c['wrong']} · no_answer {c['no_answer']}")
        g = defaultdict(Counter)
        for r in rs:
            g["정답 섹션이 근거에 있음" if r["gold_in_context"] else "정답 섹션이 근거에 없음"][r["verdict"]] += 1
        for name, cc in g.items():
            m = sum(cc.values())
            print(f"   {name} {m}개: correct {cc['correct']} · partial {cc['partial']} · wrong {cc['wrong']} · no_answer {cc['no_answer']}")
        t = defaultdict(Counter)
        for r in rs:
            t[r["type"]][r["verdict"]] += 1
        print("   유형별 correct: " + " · ".join(f"{ty} {cc['correct']}/{sum(cc.values())}" for ty, cc in t.items()))


if __name__ == "__main__":
    main()
