"""금융상품 RAG 답변 (worklog 53장).

근거 고르기 → 근거마다 머리(문서 층 · 질문 상품에 연결된 문서인지) → [판매 상태]·[상품명 찾기] 머리 + LLM 한 번
→ 상품을 추정으로 찾았는데 답변이 밝히지 않았으면 코드가 맨 앞에 붙임 → '대상 상품' 줄 + 근거 문서 (코드, LLM 없음)
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor

from app.agents import progress
from app.core import config

PROMPT = """너는 신한은행 영업점 직원을 돕는 상품 안내 도우미다. 아래 [근거]만 보고 직원의 질문에 한국어로 답한다.
근거마다 머리에 문서 정보가 있다: 문서 층(설명서 / 상품 특약 / 공통 약관 / 약정서 / 공통 조항 / 용어 안내)과, 질문한 상품에 연결된 문서인지.
원칙
1. 근거에 있는 내용만 답한다. 근거에 없으면 추측하지 말고 "보유 문서에서 확인되지 않습니다"라고 한다.
   확인할 곳을 안내할 때 **이미 근거로 받은 문서를 다시 보라고 하지 않는다** (그 상품의 설명서가 근거에 있는데 답이 없으면 "상품설명서에서 확인"이라고 하지 않음).
   대출의 상품별 한도·기간·금리·수수료율은 "여신 시스템 또는 영업점 여신 담당", 예금·적금·기타 상품은 "영업점 또는 신한은행 홈페이지 상품 안내"처럼 실제로 확인할 수 있는 곳을 안내한다.
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
7. 2~4문장으로 짧게.
8. 입력 맨 앞 [상품명 찾기]가 '찾지 못함'인데 질문이 **특정 상품 이름**을 말하면:
   - 근거에 그 이름과 같은 상품으로 보이는 문서가 있으면 그 문서로 답하되, 어느 상품 문서 기준인지 밝힌다 (줄임말로 부른 실제 상품일 수 있음).
   - 그런 문서가 없으면 먼저 "보유 상품 목록에서 '(질문의 상품 이름)' 상품을 찾지 못했습니다"라고 말하고 정확한 상품명(정식 이름) 확인을 부탁한다.
     이름이 맞다면 새 상품이거나 보유 문서에 없는 상품일 수 있으니 "영업점 또는 신한은행 홈페이지 상품 안내"에서 확인하도록 함께 안내한다.
     없는 상품이라고 단정하지 않고, 다른 상품의 값·조건으로 대신 답하지 않는다. 이 안내 문장들에는 근거 번호를 붙이지 않는다.
   - 질문이 특정 상품이 아니라 공통 규정·용어를 묻는 것이면 이 규칙은 적용하지 않는다.
9. [상품명 찾기]에 '(추정)'이 붙어 있으면 질문의 호칭을 그 상품으로 추정한 것이다. 답 첫머리에 "질문하신 '(호칭)'을 '(상품 이름)'으로 보고 답합니다"라고 밝힌다.
10. 입력 맨 앞 [판매 상태]를 따른다 (상품공시 목록 기준: 판매 중 목록 2026-11-06, 판매 중지 목록 2026-11-09).
   - 판매 중지: 답 첫머리에 "'(상품 이름)'은 판매 중지된 상품이에요. 신규 가입은 안 돼요"라고 말한다. 판매중지 날짜는 말하지 않는다.
     판매 중지 상품의 설명서는 보유하지 않으므로 금리·한도 같은 세부 조건은 지어내지 말고, 기존 가입 고객 문의는 영업점에서 확인하도록 안내한다.
     근거에 다른 상품의 문서만 있으면 그 내용으로 답하지 않는다. 판매 중지로 나온 상품에는 8번('찾지 못함' 안내)을 쓰지 않는다.
   - 판매 중·판매 중지 양쪽에 같은 이름: 판매 중지라고 단정하지 말고, 판매 중인 같은 이름의 상품이 있다고 알린 뒤 판매 중 상품 기준으로 답한다.
   - 판매 중: 판매 여부·신규 가입 여부를 물으면 "판매 중인 상품이에요(상품공시 목록 기준)"라고 답하고 근거로 안내한다.
   - [판매 상태]에 **추정한 판매 중지 상품과 판매 중 상품이 함께** 있으면, 어느 한쪽으로 정해 답하지 말고 "판매 중지된 ○○과 판매 중인 △△이 있어요"라고 둘 다 알린 뒤
     어느 상품인지 확인을 부탁한다. 판매 중 상품의 정보는 그 상품이 맞다면이라는 전제로만 덧붙인다.
11. 질문이 두 개 이상의 상품을 비교하면, 상품마다 **그 상품에 연결된 근거에서만** 값을 가져와 "상품 A: …, 상품 B: …"처럼 밝힌 뒤 비교 결론을 말한다.
   한 상품의 근거에 값이 없으면 그 상품만 "보유 문서에서 확인되지 않습니다"라고 하고, 다른 상품의 값으로 대신하지 않는다. 기준일이 다르면 함께 적는다."""

LAYER = {"상품설명서": "설명서(안내·요약)", "간이투자설명서": "설명서(안내·요약)", "투자설명서": "설명서(안내·요약)",
         "특약": "상품 특약(이 상품 전용 규정)", "약관": "공통 약관(여러 상품 공통 규정)", "약정서": "약정서"}
K, PER, CTX_CHARS = 5, 3, 2000   # 근거 수, 상품·질문마다 근거 수, 근거마다 글자 수


def split_question(q):
    """한 메시지에 질문이 여럿이면 나눔 (물음표 뒤, 없으면 '그리고' 앞뒤). 부분마다 6글자 이상, 최대 3개"""
    parts = [p.strip() for p in re.split(r"(?<=[?？])\s+", q.strip()) if p.strip()]
    if len(parts) < 2:
        parts = [p.strip() for p in re.split(r"\s+그리고\s+", q.strip()) if p.strip()]
    parts = [p for p in parts if len(re.sub(r"\s", "", p)) >= 6]
    return parts[:3] if len(parts) >= 2 else [q]


def context_ids(idx, q):
    """근거 고르기 → (항목 번호들, 상품 묶음들)
    상품이 둘 이상(비교) → 상품마다 상위 3개 / 질문이 여럿 → 질문마다 상위 3개 (전체 질문의 상품으로 좁힘) / 그 밖 → 상위 5개"""
    groups = idx.detect_groups(q)
    if len(groups) >= 2:
        qv = idx.text_vec(q)
        with ThreadPoolExecutor(3) as ex:
            ranked = list(ex.map(lambda g: idx.ranked(q, qv, pids=g)[:PER], groups[:3]))
    else:
        parts = split_question(q)
        if len(parts) < 2:
            return idx.ranked(q, idx.text_vec(q))[:K], groups
        pids = idx.detect_products(q) or None
        with ThreadPoolExecutor(3) as ex:
            ranked = list(ex.map(lambda p: idx.ranked(p, idx.text_vec(p), pids=pids)[:PER], parts))
        for p in parts:  # 근거 문서 고르기가 전체 질문으로 점수를 찾게
            idx.q_scores.setdefault(q, {}).update(idx.q_scores.get(p, {}))
    out = []
    for r in ranked:
        for i in r:
            if i not in out:
                out.append(i)
    return out, groups


def context_header(idx, i, linked_docs, linked_clauses, pnames):
    it = idx.items[i]
    if it["kind"] == "term":
        return "[용어 안내]"
    if it["kind"] == "clause":
        tag = f"질문 상품({pnames})에 연결된 문서에 들어 있는 공통 조항" if it["clause_id"] in linked_clauses else "여러 상품 공통 조항"
        return f"[공통 조항 · {tag}]"
    d = idx.docs[it["doc_id"]]
    layer = LAYER.get(d["doc_type"], d["doc_type"])
    if not linked_docs:
        link = "상품을 특정하지 못한 질문"
    else:
        link = f"질문 상품({pnames})에 연결된 문서" if it["doc_id"] in linked_docs else "다른 상품 문서"
    return f"[{layer} · {link}]"


def multi_header(idx, i, groups):
    """여러 상품일 때: 어느 상품에 연결된 문서인지"""
    it = idx.items[i]
    if it["kind"] == "term":
        return "[용어 안내]"
    own = []
    for g in groups:
        ds = set().union(*(idx.links[p] for p in g))
        cl = set().union(*(idx.doc_clause[d] for d in ds))
        if (it["kind"] == "chunk" and it["doc_id"] in ds) or (it["kind"] == "clause" and it["clause_id"] in cl):
            own.append(", ".join(sorted(idx.pid_name[p] for p in g))[:40])
    if it["kind"] == "clause":
        return f"[공통 조항 · {' / '.join(own) + '에 연결' if own else '여러 상품 공통'}]"
    layer = LAYER.get(idx.docs[it["doc_id"]]["doc_type"], idx.docs[it["doc_id"]]["doc_type"])
    return f"[{layer} · {(' / '.join(own) + '에 연결된 문서') if own else '질문 상품과 연결 없음'}]"


def status_header(idx, q, groups):
    lines, names = [], set()
    for s in idx.sale_status(q):
        if s.get("inferred"):
            lines.append(f"{s['name']}: 판매 중지 (질문 속 호칭을 판매 중지 목록의 이 상품으로 추정 — 답할 때 추정임을 밝힐 것)")
        else:
            lines.append(f"{s['name']}: " + ("판매 중지 (판매 중지 목록에만 있음)" if s["status"] == "discontinued"
                                             else "판매 중 목록과 판매 중지 목록에 같은 이름이 모두 있음"))
        names.add(s["name"])
    for pid in sorted(set().union(*groups) if groups else set()):
        n = idx.pid_name.get(pid)
        if n and n not in names:
            lines.append(f"{n}: 판매 중 (판매 중 목록)")
            names.add(n)
    return "[판매 상태] " + ("; ".join(lines) if lines else "질문에서 목록의 상품 이름을 찾지 못함")


def product_line(idx, q, groups, how, max_names=3):
    """'대상 상품' 한 줄 — 어떤 상품으로 답했는지와 판매 상태 (코드)"""
    parts, seen = [], set()
    for s in idx.sale_status(q):
        if s["name"] in seen:
            continue
        seen.add(s["name"])
        parts.append(f"{s['name']} (판매 중지{' · 추정' if s.get('inferred') else ''})" if s["status"] == "discontinued"
                     else f"{s['name']} (판매 중·판매 중지 목록에 같은 이름)")
    inferred = len(groups) < 2 and how in ("verified", "fallback")
    names = []
    for g in (groups if len(groups) >= 2 else [set().union(*groups) if groups else set()]):
        for p in sorted(g):
            n = idx.pid_name.get(p)
            if n and n not in seen and n not in names:
                names.append(n)
    parts += [f"{n} (판매 중{' · 추정' if inferred else ''})" for n in names[:max_names]]
    if len(names) > max_names:
        parts.append(f"외 {len(names) - max_names}개")
    return "대상 상품: " + (" / ".join(parts) if parts else "찾지 못함 (질문에서 보유 상품 목록의 이름을 찾지 못함)")


def finalize(ans, how, pnames):
    """상품을 추정으로 찾았는데 답변이 밝히지 않았으면 맨 앞에 붙임 (모델이 9번 규칙을 빼먹어도 늘 표시)"""
    if how not in ("verified", "fallback") or not pnames:
        return ans
    if any(w in ans[:160] for w in ("보고 답", "추정", "뜻하는 것으로")):
        return ans
    return f"(질문 속 상품 이름을 '{pnames}'(으)로 보고 답합니다.) " + ans


def cited_and_related(idx, q, ans, ctx_ids, groups, min_score=1):
    """답변이 [n]으로 인용한 근거 + 함께 볼 문서(인용 안 했지만 재순위 점수 1점 이상).
    같은 조항이 원문 조각과 공통 조항으로 두 번 오면 원문 쪽만"""
    scores = idx.q_scores.get(q, {})

    def topic(i):
        h = idx.items[i]["text"].split("\n", 1)[0]
        return re.sub(r"\s*\(p\.\d+\)$", "", re.sub(r"^\[[^\]]+\]\s*", "", h)).replace(" ", "")

    cited = []
    for n in re.findall(r"\[(\d+)\]", ans):
        k = int(n) - 1
        if 0 <= k < len(ctx_ids) and k not in cited:
            cited.append(k)
    is_cl = lambda k: idx.items[ctx_ids[k]]["kind"] == "clause"
    seen = [(topic(ctx_ids[k]), is_cl(k)) for k in cited]
    has_target = bool(set().union(*groups)) if groups else False  # 판매 중 대상 상품이 없으면 함께 볼 문서 없음
    rest = sorted((k for k in range(len(ctx_ids)) if has_target and k not in cited and scores.get(ctx_ids[k], 0) >= min_score), key=is_cl)
    extra = []
    for k in rest:
        t, c = topic(ctx_ids[k]), is_cl(k)
        if t and any((c or sc) and c != sc and (t in s or s in t) for s, sc in seen):
            continue
        seen.append((t, c))
        extra.append(k)
    return sorted(cited), sorted(extra)


def evidence_text(idx, q, ctx_ids, groups):
    """LLM 에 넘길 [근거] (번호 = 목록 순서) + 상품 이름"""
    pids = set().union(*groups) if groups else set()
    if len(groups) >= 2:
        pnames = ", ".join(sorted(idx.pid_name[p] for p in pids))[:120]
        hdr = lambda i: multi_header(idx, i, groups)
    else:
        linked = set().union(*(idx.links[p] for p in pids)) if pids else set()
        lclauses = set().union(*(idx.doc_clause[d] for d in linked)) if linked else set()
        pnames = ", ".join(sorted(idx.pid_name[p] for p in pids))[:120]
        hdr = lambda i: context_header(idx, i, linked, lclauses, pnames)
    ctx = "\n\n".join(f"[{n}] {hdr(i)}\n{idx.items[i]['text'][:CTX_CHARS]}" for n, i in enumerate(ctx_ids, 1))
    return ctx, pnames


def answer(idx, question):
    """질문 하나 → {"answer"(대상 상품 줄 + 답), "sources": [{no, title, text, cited}], "timing"}"""
    t0 = time.perf_counter()
    ctx_ids, groups = context_ids(idx, question)
    if len(groups) < 2:
        p = idx.detect_products(question)
        groups = [p] if p else []
    how = idx.detect_how(question)
    ctx, pnames = evidence_text(idx, question, ctx_ids, groups)
    head = (f"[상품명 찾기] {pnames}{' (추정)' if pnames and how != 'exact' else ''}" if pnames
            else "[상품명 찾기] 찾지 못함 (질문에서 보유 상품 목록의 이름을 찾지 못함)")
    inp = f"{status_header(idx, question, groups)}\n\n{head}\n\n[근거]\n{ctx}\n\n[질문] {question}"
    progress.done("search", counts={"product": len(ctx_ids)})  # 스트리밍 단계 알림 (worklog 60장)
    progress.step("answer")
    t1 = time.perf_counter()
    ans = idx.client().responses.create(model=config.LLM_MODEL, instructions=PROMPT, input=inp).output_text
    if len(groups) < 2:
        ans = finalize(ans, how, pnames)
    cited, extra = cited_and_related(idx, question, ans, ctx_ids, groups)
    sources = [{"no": k + 1, "title": idx.label(ctx_ids[k]), "text": idx.items[ctx_ids[k]]["text"], "display": idx.display(ctx_ids[k]),
                "cited": k in cited}
               for k in cited + extra]
    return {"answer": product_line(idx, question, groups, how) + "\n" + ans, "found": bool(cited), "sources": sources,
            "timing": {"search": round(t1 - t0, 3), "llm": round(time.perf_counter() - t1, 3)}}
