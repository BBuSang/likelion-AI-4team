"""동작 보존/지연 검증. 시험셋은 읽지 않는다.

기본: 합성 의존성으로 분기별 기존/수정 service 결과와 LLM 입력 비교.
--live: 실제 DB·임베딩·LLM으로 예시 질문을 실행하고 응답을 요청 안에 기록한 뒤,
        같은 LLM 응답을 재생해 기존/수정 경로를 교차 비교한다.
재생 시간은 네트워크 대기가 없는 로컬 처리 시간이며 실제 전체 응답시간과 구분한다.
--rounds N: 질문마다 기존/수정 재생 비교 횟수 (기본 3).
--local: 실제 DB·임베딩만 사용하고 LLM은 고정 응답으로 대체한다. 외부 전송 없음.
기존/후보 파일은 data/performance/20261112/에 보관한다. 기본 앱은 기존 버전이다.
"""
import argparse
import copy
import hashlib
import importlib.util
import itertools
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
import rag
import keyword_relax as kr
import exmnt_search as es
import verify as vf

OUT = ROOT / "data/performance/20261112"


def load_snapshot(name):
    spec = importlib.util.spec_from_file_location(name, OUT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rr = load_snapshot("rerank_candidate")
current = load_snapshot("service_candidate")
current.rr = rr


def load_before():
    module = load_snapshot("service_before")
    # qvec를 생략하면 보관된 원본과 동일한 SQL을 실행한다.
    module.rr = rr
    return module


def stable(result):
    return {k: v for k, v in result.items() if k != "seconds"}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


class Connection:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, *args):
        return SimpleNamespace(fetchall=lambda: [{"group_id": 1}])


def offline(before):
    results = []
    # 모든 토글, 첫 답변 실패, 리랭킹 실패를 조합한다.
    for rerank, field, relax, verify, miss, fail in itertools.product([False, True], repeat=6):
        traces = []
        outputs = []
        embed_counts = []
        for module in (before, current):
            trace, calls, embeds = [], [], []

            def embed(text):
                embeds.append(text)
                return [0.25, 0.75]

            def candidates(conn, text, n=20, *, qvec=None):
                vector = rag.embed(rag.mask_pii(text)) if qvec is None else qvec
                trace.append(("candidates", text, list(vector)))
                return [{"doc_id": "L1", "score": .8, "n_same": 0, "chunk_type": "question"}]

            def reranking(*args):
                trace.append(("rerank", args))
                if fail:
                    raise ValueError("synthetic failure")
                return ["L1"], {}, ["L1"]

            def llm(messages):
                calls.append(copy.deepcopy(messages))
                answer = rag.NOT_FOUND if miss and len(calls) == 1 else "당국은 조건을 확인했습니다. [L1]"
                return answer, {"input": 10, "output": 5, "reasoning": 0}

            def cases(ids, conn):
                return {key: dict(case_key=key, title="사례", gubun="법령해석", reply_date="2020-01-01", answer_type="",
                                 doc_class="general", question="질의", answer="회답", reason="이유", url="https://example.com",
                                 time_mark=None, time_label="", citations=[]) for key in ids}

            def fields(conn, vec, ref):
                trace.append(("fields", list(vec), ref))
                return [dict(case_key="F1", title="건의", reply_date="2020-01-01", answer_type="수용", question="건의", answer="검토",
                             url="https://example.com", n_same=0, score=.9, changed_after=False, already_in_place=False)], .9

            def relaxed(conn, concepts, vec, groups):
                trace.append(("relax", concepts, list(vec), sorted(groups)))
                return {"extra": [dict(doc_id="L2", score=.7, n_same=0, matched=1, of=2, missing_terms=["조건"])],
                        "trail": [{"need": 1, "groups": 1}]}

            def verification(question, raw, context):
                trace.append(("verify", question, raw, context))
                abstain = fail  # 성공/거절 양쪽을 확인한다.
                return (rag.NOT_FOUND if abstain else raw), {"checked": True, "abstained": abstain,
                         "flagged": [] if abstain else [raw], "usage": {"input": 3, "output": 1}}

            with patch.object(rag, "connect", return_value=Connection()), patch.object(rag, "embed", side_effect=embed), \
                 patch.object(rr, "candidates", side_effect=candidates), patch.object(rr, "load_brief", return_value={}), \
                 patch.object(rr, "rerank", side_effect=reranking), patch.object(rag, "load_cases", side_effect=cases), \
                 patch.object(rag, "search", side_effect=lambda q, k, c: candidates(c, q)), \
                 patch.object(es, "search", side_effect=fields), patch.object(rag, "llm", side_effect=llm), \
                 patch.object(kr, "extract_concepts", return_value=([["상품"], ["조건"]], {})), \
                 patch.object(kr, "relax_search", side_effect=relaxed), patch.object(vf, "check", side_effect=verification):
                outputs.append(stable(module.ask("test person@example.com", relax, rerank, field, verify)))
            traces.append((trace, calls))
            embed_counts.append(len(embeds))
        assert outputs[0] == outputs[1], (rerank, field, relax, verify, miss, fail)
        assert traces[0] == traces[1], "retrieval or model input changed"
        assert embed_counts[1] <= embed_counts[0]
        results.append({"flags": [rerank, field, relax, verify, miss, fail], "embeddings": embed_counts})

    # 실제 후보 조회 함수에서도 qvec 생략/지정 시 SQL과 결과가 같은지 확인한다.
    class CandidateConnection:
        def __init__(self):
            self.trace = []

        def execute(self, sql, params=None):
            self.trace.append((sql, params))
            rows = []
            if "from interp.chunks" in sql:
                rows = [dict(group_id=1, case_key="L1", chunk_type="question", text="x", score=.8)]
            elif "from interp.case_groups" in sql:
                rows = [dict(group_id=1, n_cases=1, representative_case_key="L1", title="x")]
            return SimpleNamespace(fetchall=lambda: rows)

    left, right = CandidateConnection(), CandidateConnection()
    with patch.object(rag, "embed", return_value=[.25, .75]):
        a = rr.candidates(left, "question")
        b = rr.candidates(right, "question", qvec=[.25, .75])
    assert a == b and left.trace == right.trace
    return {"scenarios": len(results), "exact_result_and_input_match": True, "candidate_sql_match": True,
            "embedding_counts": sorted({tuple(r["embeddings"]) for r in results})}


def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p
    lo = int(pos)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)


def local_response(*args, **kwargs):
    """속도/입력 보존 검사용 대역. 답변 정확도 평가에 사용하지 않는다."""
    messages = kwargs["messages"]
    system, user = messages[0]["content"], messages[-1]["content"]
    if system == rr.RERANK_SYSTEM:
        content = json.dumps({"order": re.findall(r"\[후보 ([LO]\d+)\]", user)[:5]})
    elif system == kr.EXTRACT_SYSTEM:
        content = json.dumps({"concepts": [["적금"], ["캐시백"]] if "적금" in user else []})
    else:
        question = user.split("\n\n[사례]", 1)[0]
        ids = re.findall(r"\[사례 ([LOF]\d+)\]", user)
        missing = "점심" in question or ("적금" in question and "키워드로 넓혀 찾은 사례" not in user)
        content = rag.NOT_FOUND if missing or not ids else f"검증용 고정 응답입니다. [{ids[0]}]"
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                           usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, completion_tokens_details=None))


def live(before, rounds, local=False):
    # 공개된 데모 예시와 무관한 질문만 사용한다. 시험셋은 열지 않는다.
    questions = ["저축은행 대주주에 대한 신용공여 한도 관련 회신 알려주세요",
                 "보험회사가 방카슈랑스 판매 비중 규제를 받는지 알려주세요",
                 "적금 가입 고객에게 제휴 카드사 캐시백을 주는 이벤트를 하려는데, 특별이익 제공으로 문제된 회신이 있나요?",
                 "점심 메뉴 추천해 주세요"]
    start = time.perf_counter()
    rag.embed("준비")
    cold = time.perf_counter() - start
    create = local_response if local else rag.client.chat.completions.create
    rows = []
    for index, question in enumerate(questions):
        tape = []

        def record(*args, **kwargs):
            response = create(*args, **kwargs)
            tape.append((copy.deepcopy((args, kwargs)), copy.deepcopy(response)))
            return response

        with patch.object(rag.client.chat.completions, "create", side_effect=record):
            initial = before.ask(question)
        runs = {"before": [], "after": []}
        after_stages = []
        for iteration in range(rounds):
            order = [("before", before), ("after", current)]
            if iteration % 2:
                order.reverse()
            for name, module in order:
                cursor = 0

                def replay(*args, **kwargs):
                    nonlocal cursor
                    assert cursor < len(tape), "extra model call"
                    expected, response = tape[cursor]
                    assert (args, kwargs) == expected, "model input changed"
                    cursor += 1
                    return copy.deepcopy(response)

                with patch.object(rag.client.chat.completions, "create", side_effect=replay):
                    begin = time.perf_counter()
                    result = module.ask(question)
                    elapsed = time.perf_counter() - begin
                assert cursor == len(tape), "missing model call"
                assert stable(initial) == stable(result), "result changed"
                runs[name].append(elapsed)
                if name == "after":
                    after_stages.append(result["seconds"]["stages"])
        rows.append({"example": index + 1, "original_live_seconds": initial["seconds"]["total"], "llm_calls": len(tape),
                     "result_sha256": digest(stable(initial)), "exact_match": True,
                     "replay_seconds": runs, "after_stages": after_stages})
        if not local:
            # 실제 호출은 출력이 달라질 수 있다. 동일성 검증(위 재생)과 별도로 기록한다.
            actual = {"before": [initial], "after": []}
            for name, module in [("after", current), ("after", current), ("before", before)]:
                actual[name].append(module.ask(question))
            rows[-1]["e2e"] = {name: [{"seconds": res["seconds"], "answer": res["raw"],
                                      "cited": res["cited"], "check": res["check"], "order": res["order"],
                                      "relax": res["relax"]} for res in items] for name, items in actual.items()}
        print(json.dumps({"example": index + 1, "exact_match": True, "llm_calls": len(tape),
                          "replay_median": {k: round(percentile(v, .5), 4) for k, v in runs.items()}}, ensure_ascii=False), flush=True)
        # 실행이 중단돼도 완료한 예시의 결과를 남긴다.
        if not local:
            (OUT / "live_partial.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {name: {"p50": percentile([v for row in rows for v in row["replay_seconds"][name]], .5),
                      "p95": percentile([v for row in rows for v in row["replay_seconds"][name]], .95)} for name in ("before", "after")}
    e2e = None if local else {name: {"p50": percentile([r["seconds"]["total"] for row in rows for r in row["e2e"][name]], .5),
                                   "p95": percentile([r["seconds"]["total"] for row in rows for r in row["e2e"][name]], .95)} for name in ("before", "after")}
    return {"local_only": local, "warmup_seconds": cold, "replay_only_summary": summary, "e2e_summary": e2e, "rows": rows,
            "limitation": "Replay excludes network latency; exact equality tests preservation, not correctness of original answers."}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--live", action="store_true")
    modes.add_argument("--local", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("rounds must be positive")
    before = load_before()
    report = {"offline": offline(before), "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [OUT / "service_before.py", OUT / "service_candidate.py", OUT / "rerank_candidate.py", ROOT / "pipeline/rag.py"]}}
    print(json.dumps(report["offline"], ensure_ascii=False), flush=True)
    if args.live or args.local:
        report["live"] = live(before, args.rounds, local=args.local)
    dest = OUT / ("local_result.json" if args.local else "live_result.json" if args.live else "offline_result.json")
    dest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(dest), flush=True)
