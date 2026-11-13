"""데모 처리 로직 (화면과 분리). `demo.py`(92번 이전 데모)의 처리 부분을 동작 그대로 옮긴 것이다. Streamlit을 들여오지 않으므로 화면(`app.py`)과 다른 앱에서 함께 쓸 수 있다.

흐름 (89번에서 dev7·test7로 잰 설정 그대로)
1. 질문의 개인정보를 가린다
2. 기존 회신: 벡터 후보 20 → LLM 리랭킹 5 (`rerank.py`, 실패하면 벡터 상위 5)
3. 현장건의: 별도 후보군에서 점수 ≥ 0.65 이면서 회신 1위 벡터 점수보다 높은 묶음만 최대 2건 (`exmnt_search.py`)
4. 답변: `APP_SYSTEM_V2`(현장건의가 붙은 질문에만 현장건의 규칙 추가) → 출력 검사
5. 첫 답변이 "찾지 못했습니다"(또는 인용 없음)일 때만 키워드 순차 완화로 최대 3건을 더해 한 번 더 답한다 (`keyword_relax.py`)
6. (선택, 기본 끔) 답변 검증(`verify.check`, 99번): 인용 사례가 질문의 대상·조건을 직접 다루지 않으면 "찾지 못했습니다"로 바꾸고, 인용 사례에 없다고 본 줄은 지우지 않고 표시만 한다
`rag.py`는 평가 지문에 들어 있어 고치지 않는다.
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag  # noqa: E402
import exmnt_search as es  # noqa: E402
import keyword_relax as kr  # noqa: E402
import rerank as rr  # noqa: E402
import verify as vf  # noqa: E402
from prompts import APP_SYSTEM_V2 as APP_SYSTEM, REPLY_DATE  # noqa: E402

CITE = re.compile(r"\[([LOF]\d+)\]")          # F = 현장건의
ASIDE = re.compile(r"\[(\d{4}-\d{2}-\d{2} 회신에서[^\[\]]*?)\]")          # 문장 전체를 대괄호로 감싼 경우


def warm():
    """임베딩 모델을 불러온다 (처음 30~80초)."""
    rag.embed("준비")
    return True


def _answer(masked, context, ids, system=APP_SYSTEM):
    raw, usage = rag.llm([{"role": "system", "content": system}, {"role": "user", "content": f"[질문]\n{masked}\n\n[사례]\n{context}"}])
    raw = raw.strip()
    check = rag.check_output(raw, ids)
    cited = [k for k in dict.fromkeys(CITE.findall(raw)) if k in ids]
    return raw, usage, check, cited


def ask(question, relax=True, use_rerank=True, use_exmnt=True, use_verify=False):
    """질문 하나를 처리해 화면에 필요한 값(답변, 인용 순서, 사례 카드)을 돌려준다."""
    t0 = time.time()
    masked = rag.mask_pii(question)
    relax_info = None
    with rag.connect() as conn:
        reranked = False
        if use_rerank:
            cands = rr.candidates(conn, masked)
            try:
                order, _, _ = rr.rerank(masked, [c["doc_id"] for c in cands], rr.load_brief(conn, [c["doc_id"] for c in cands]))
                by_id = {c["doc_id"]: c for c in cands}
                hits, reranked = [by_id[i] for i in order], order != [c["doc_id"] for c in cands[:5]]
            except Exception:                      # 리랭킹 호출이 실패하면 벡터 순서 상위 5개로 답한다
                hits = cands[:5]
        else:
            hits = rag.search(masked, 5, conn)
        ids = [h["doc_id"] for h in hits]
        cases = rag.load_cases(ids, conn)
        ref = max(c["score"] for c in cands) if use_rerank else (hits[0]["score"] if hits else None)     # 기존 회신 1위 벡터 점수 (리랭킹 전)
        fields, f_top1 = es.search(conn, rag.embed(masked), ref) if use_exmnt else ([], None)
    t_search = time.time() - t0
    f_ids = [f["case_key"] for f in fields]
    f_ctx = "".join("\n\n" + es.block(f) for f in fields)
    system = APP_SYSTEM + es.EXMNT_RULES if fields else APP_SYSTEM          # 현장건의가 붙은 질문에만 규칙을 더한다
    context = "\n\n".join(rag.case_block(cases[i]) for i in ids)
    raw, usage, check, cited = _answer(masked, context + f_ctx, ids + f_ids, system)
    used_ctx = context + f_ctx
    extra = []
    if relax and (check["not_found"] or not cited):
        with rag.connect() as conn:
            concepts, _ = kr.extract_concepts(masked)
            groups = {r["group_id"] for r in conn.execute("select group_id from interp.case_groups where representative_case_key = any(%s)", (ids,)).fetchall()}
            found = kr.relax_search(conn, concepts, rag.embed(masked), groups)
            extra = found["extra"]
            if extra:
                cases.update(rag.load_cases([e["doc_id"] for e in extra], conn))
        relax_info = {"concepts": concepts, "trail": found["trail"], "extra": [e["doc_id"] for e in extra], "retried": bool(extra)}
        if extra:
            context2 = context + "".join("\n\n" + kr.extra_block(cases[e["doc_id"]], e) for e in extra) + f_ctx
            raw, u2, check, cited = _answer(masked, context2, ids + [e["doc_id"] for e in extra] + f_ids, system)
            usage = {k: usage[k] + u2[k] for k in usage}
            used_ctx = context2
    verify_info = None
    if use_verify:                                 # 답변 검증: 직접 다루지 않으면 거절, 없는 줄은 표시만 (답변 글은 거절할 때만 바뀐다)
        new_raw, verify_info = vf.check(masked, raw, used_ctx)
        usage = {**usage, "input": usage["input"] + verify_info["usage"]["input"], "output": usage["output"] + verify_info["usage"]["output"]}
        if verify_info["abstained"]:
            raw, cited = new_raw, []
            check = rag.check_output(raw, ids + [e["doc_id"] for e in extra] + f_ids)
    by_extra = {e["doc_id"]: e for e in extra}
    all_ids = ids + [e["doc_id"] for e in extra]
    by_hit = {h["doc_id"]: h for h in hits}
    cards = {}
    for i in all_ids:
        c = cases[i]
        src = by_hit.get(i) or by_extra[i]
        cards[i] = {"case_key": i, "title": (c["title"] or "").strip(), "gubun": c["gubun"], "reply_date": str(c["reply_date"] or "미상"), "answer_type": c["answer_type"], "doc_class": c["doc_class"],
                    "question": (c["question"] or "").strip(), "answer": (c["answer"] or "").strip(), "url": c["url"], "time_mark": c["time_mark"], "time_label": c["time_label"],
                    "citations": c["citations"], "n_same": src["n_same"], "score": src["score"], "chunk_type": by_hit[i]["chunk_type"] if i in by_hit else "키워드 완화",
                    "relaxed": {"matched": by_extra[i]["matched"], "of": by_extra[i]["of"], "missing": by_extra[i]["missing_terms"]} if i in by_extra else None}
    for f in fields:                               # 현장건의 카드 (시점 판정은 아직 없다)
        cards[f["case_key"]] = {"case_key": f["case_key"], "title": (f["title"] or "").strip(), "gubun": "현장건의", "reply_date": str(f["reply_date"] or "미상"), "answer_type": f["answer_type"],
                                "doc_class": "field_proposal", "question": (f["question"] or "").strip(), "answer": (f["answer"] or "").strip(), "url": f["url"], "time_mark": None,
                                "time_label": "", "citations": [], "n_same": f["n_same"], "score": f["score"], "chunk_type": "현장건의", "relaxed": None,
                                "changed_after": f["changed_after"], "already_in_place": f["already_in_place"]}
    return {"question": question, "masked": masked, "raw": raw, "cited": cited, "cards": cards, "order": all_ids + f_ids, "check": check, "usage": usage, "relax": relax_info, "reranked": reranked, "verify": verify_info,
            "exmnt": {"top1": f_top1, "ref_score": ref, "attached": f_ids, "threshold": es.THRESH} if use_exmnt else None,
            "seconds": {"search": round(t_search, 1), "total": round(time.time() - t0, 1)}}


def trail_text(info):
    """키워드 완화 과정: '적금·캐시백·특별이익 3개 모두 0건 → 2개 이상 2건'. 처음으로 결과가 나온 단계까지만."""
    if not info:
        return None
    if not info["concepts"]:
        return "범위를 넓혀 다시 찾으려 했으나 질문에서 회신 검색용 핵심어를 뽑지 못했습니다."
    names = "·".join(c[0] for c in info["concepts"])
    steps = []
    for t in info["trail"]:
        label = f"{t['need']}개 모두" if t["need"] == len(info["concepts"]) else f"{t['need']}개 이상"
        steps.append(f"{label} 일치 {t['groups']}건")
        if t["groups"]:
            break
    return f"상위 5건에서 찾지 못해 키워드로 넓혀 다시 찾았습니다. 키워드 검색({names}): " + " → ".join(steps)


def answer_text(raw):
    """고정 안내 문구·회신 날짜 머리말·굵은 글씨를 떼어 낸 답변 글. 근거 번호 `[L123]`는 그대로 둔다(화면이 배지로 바꾼다)."""
    t = raw.replace(rag.CLOSING, "").strip()
    t = ASIDE.sub(r"\1", t)
    for rx in REPLY_DATE:
        t = rx.sub("", t)
    return re.sub(r"[ \t]{2,}", " ", t.replace("**", ""))


def partly_not_found(res):
    """답변 앞부분에서 '직접 일치하는 회신은 찾지 못했다'고 하면서 사례를 인용한 경우."""
    head = res["raw"][:140]
    return bool(res["cited"]) and "찾지 못" in head and not res["check"]["not_found"]
