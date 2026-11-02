"""RAG 전체(검색 → LLM 답변 → 출력 검사)를 holdout 질의 88개로 테스트하고 채점한다 (팀 추가).

    python eval/eval_rag_e2e.py --model HCX-007 --out outputs/rag_e2e_hcx007.jsonl      # 실행 (끊겨도 이어서)
    python eval/eval_rag_e2e.py --out outputs/rag_e2e_hcx007.jsonl --score-only         # 채점만

정답 파일(holdout_retrieval_queries_gold.jsonl, holdout_labeled.jsonl 의 entities)은 채점에만 쓴다.

채점 항목
  - 근거 적중   : LLM 에 넘긴 문서(k개) 안에 정답 문서가 있나 (Hit@k)
  - 인용 적중   : 답변이 [H0141] / <H0141> 처럼 인용한 문서 중 정답 문서가 있나
  - 인용 정밀도 : 인용한 문서 중 정답 문서 비율 (혼동 문서 인용 비율도 따로)
  - 못 찾음     : '찾을 수 없습니다' 답변 비율
  - 사람 토큰   : 질의에 이름이 있으면, 답변에 그 사람의 토큰(질의 토큰)이 나오나
  - 출력 검사   : 확인 안 된 토큰 / 원문 개인정보가 걸린 답변 수
  - 유출 검사   : 정답 라벨의 원문 값(이름 3자 이상·전화·이메일·주민·계좌·카드·주소)이
                 LLM 에 보낸 내용(문서+질의) / LLM 원 답변 / 최종 답변에 그대로 나온 횟수 — 0 이어야 함
  - 시간        : 검색·LLM·검사 평균
  - 되물음      : 동명이인을 못 골라 LLM 대신 되물은 비율 (worklog 16장). 되물은 질의는 인용·못 찾음 항목에서 빼고,
                 '정당한 되물음' = 그 질의에 실제로 혼동 문서(같은 이름의 다른 사람)가 있었던 비율
"""
import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GOLD = ROOT / "data" / "holdout_retrieval_queries_gold.jsonl"
CORPUS = ROOT / "data" / "holdout_labeled.jsonl"
_CITE = re.compile(r"[\[<]([A-Z]\d{4})[\]>]")  # holdout H0141, synth2 S0141
_NOT_FOUND = re.compile(r"찾을 수 없")


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def run(a):
    from starter import chat  # 모델 로딩은 실행할 때만
    done = {r["qid"] for r in load(a.out)} if Path(a.out).exists() else set()
    queries = [q for q in load(GOLD) if q["qid"] not in done
               and (not a.types or q["type"] in a.types or (a.homonym and q.get("confusable_doc_ids")))
               and (not a.qids or q["qid"] in a.qids)]
    print(f"남은 질의 {len(queries)}개 (이미 {len(done)}개 완료)", flush=True)
    with open(a.out, "a", encoding="utf-8") as f:
        for i, q in enumerate(queries, 1):
            for attempt in range(3):
                try:
                    res = chat.answer(q["query"], k=a.k, model=a.model, embed_retries=3)
                    break
                except Exception as e:  # LLM 일시 오류 등
                    err = f"{type(e).__name__}: {e}"
                    time.sleep(5 * (attempt + 1))
            else:
                res = {"error": err}
            ctx = chat._context(res.get("docs", []))  # 검색 근거 포함, LLM 에 실제로 보낸 것과 같게
            f.write(json.dumps({
                "qid": q["qid"], "query": q["query"], "model": a.model, "k": a.k,
                "masked_query": res.get("masked_query"), "query_tokens": res.get("query_tokens"),
                "doc_ids": [d["doc_id"] for d in res.get("docs", [])],
                "fallback": bool(res.get("docs") and res["docs"][0].get("fallback")),
                "llm_input": ctx + "\n\n" + (res.get("masked_query") or ""),
                "raw_answer": res.get("raw_answer"), "answer": res.get("answer"), "check": res.get("check"),
                "timing": res.get("timing"), "error": res.get("error"),
                "ask_back": res.get("ask_back", False), "guard": res.get("guard", 0)}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(queries)}] {q['qid']} {res.get('timing', {})}", flush=True)


def pii_values():
    """정답 라벨의 원문 값 (유출 검사용). 짧은 이름(2자 이하)은 일반 단어와 겹쳐 제외."""
    vals = set()
    for d in load(CORPUS):
        for e in d["entities"]:
            t = e["text"].strip()
            if e["label"] == "person" and len(t.replace(" ", "")) < 3:
                continue
            if len(t) >= 3:
                vals.add((e["label"], t))
    return vals


def score(a):
    gold = {q["qid"]: q for q in load(GOLD)}
    rows = load(a.out)
    vals = pii_values()
    by_type = collections.defaultdict(lambda: collections.defaultdict(list))
    leaks = collections.Counter()
    leak_cases = []
    times = collections.defaultdict(list)
    errors = 0
    for r in rows:
        q = gold[r["qid"]]
        if r.get("error"):
            errors += 1
            continue
        rel, conf = set(q["relevant_doc_ids"]), set(q.get("confusable_doc_ids", []))
        cited = set(_CITE.findall(r["answer"] or ""))
        m = {"근거 적중": 1.0 if rel & set(r["doc_ids"]) else 0.0, "되물음": 1.0 if r.get("ask_back") else 0.0}
        if r.get("ask_back"):
            m["정당한 되물음"] = 1.0 if conf else 0.0
            cited = set()
        else:
            m["인용 적중"] = 1.0 if rel & cited else 0.0
            m["못 찾음"] = 1.0 if _NOT_FOUND.search(r["answer"] or "") else 0.0
        if cited:
            m["인용 정밀도"] = len(cited & rel) / len(cited)
            if conf:
                m["혼동 인용"] = len(cited & conf) / len(cited)
        # 동명이인 구분을 켜면 질의 토큰이 문서 속 사람 토큰으로 바뀌어 LLM 에 감 → LLM 에 간 질의 기준으로 셈
        person_tokens = sorted(set(re.findall(r"<PERSON_[0-9a-f]{8}>", r.get("masked_query") or ""))
                               | {t for t in (r["query_tokens"] or []) if t.startswith("<PERSON_")})
        if not r.get("ask_back") and any(e["label"] == "person" for e in q.get("query_entities", [])) and person_tokens:
            m["사람 토큰"] = 1.0 if any(t in (r["answer"] or "") for t in person_tokens) else 0.0
        chk = r.get("check") or {}
        m["확인 안 된 토큰"] = 1.0 if chk.get("unverified_tokens") else 0.0
        m["원문 개인정보 걸림"] = 1.0 if chk.get("raw_pii") else 0.0
        m["토큰만 검색"] = 1.0 if r.get("fallback") else 0.0
        for g in ("ALL", q["type"]):
            for k, v in m.items():
                by_type[g][k].append(v)
            by_type[g]["n"].append(1)
        for where in ("llm_input", "raw_answer", "answer"):
            text = r.get(where) or ""
            for lab, v in vals:
                if v in text:
                    leaks[where] += 1
                    leak_cases.append((r["qid"], where, lab, v))
        for k, v in (r.get("timing") or {}).items():
            times[k].append(v)

    cols = ["근거 적중", "인용 적중", "인용 정밀도", "혼동 인용", "못 찾음", "되물음", "정당한 되물음", "사람 토큰",
            "확인 안 된 토큰", "원문 개인정보 걸림"]
    print(f"질의 {len(rows)}개 (오류 {errors}), 모델 {rows[0]['model'] if rows else '-'}, k={rows[0]['k'] if rows else '-'}")
    print(f"{'type':<26}{'n':>4}" + "".join(f"{c:>10}" for c in cols))
    for g in ["ALL"] + sorted(x for x in by_type if x != "ALL"):
        m = by_type[g]
        print(f"{g:<26}{len(m['n']):>4}" + "".join(
            f"{sum(m[c]) / len(m[c]):>10.2f}" if m[c] else f"{'-':>10}" for c in cols))
    fb = by_type["ALL"]["토큰만 검색"]
    print(f"\n임베딩 한도로 토큰만 검색한 질의: {int(sum(fb))}개")
    print("유출 검사 (정답 원문 값이 그대로 나온 횟수, 0 이어야 함):",
          {w: leaks.get(w, 0) for w in ("llm_input", "raw_answer", "answer")})
    for c in leak_cases[:10]:
        print("  ", c)
    print("평균 시간(초):", {k: round(sum(v) / len(v), 1) for k, v in times.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HCX-007")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--types", nargs="*", help="이 질의 유형만 (예: homonym_role hybrid_homonym_semantic)")
    ap.add_argument("--homonym", action="store_true", help="혼동 문서(동명이인)가 있는 질의만")
    ap.add_argument("--qids", nargs="*", help="이 질의 ID 만 (예: Q052 Q053)")
    ap.add_argument("--gold", default=str(GOLD), help="질의 정답 파일 (기본: holdout)")
    ap.add_argument("--corpus", default=str(CORPUS), help="코퍼스 (유출 검사용 정답 라벨, 기본: holdout)")
    a = ap.parse_args()
    globals().update(GOLD=Path(a.gold), CORPUS=Path(a.corpus))  # synth2 등 다른 데이터로 돌릴 때
    if a.homonym and not a.types:
        a.types = ["__none__"]  # 유형 조건은 없이 혼동 문서 조건만
    if not a.score_only:
        run(a)
    score(a)


if __name__ == "__main__":
    main()
