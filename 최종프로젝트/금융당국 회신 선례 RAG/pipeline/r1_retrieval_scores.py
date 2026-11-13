"""R1 검색(리트리버) 점수 내보내기 (사용자 요청 2026-11-11 "리트리버로 탐지했을 때 데이터를 점수를 매겨서 csv로", worklog 94번).
질문마다 벡터 검색이 찾은 사례와 유사도 점수를 CSV로 남긴다. LLM은 부르지 않는다(리랭킹·답변 없음). 시스템·설정은 바꾸지 않는다.
- 회신 후보군: `rerank.candidates`와 같은 검색(조각 200개 → 묶음 20개). 점수 = 그 묶음에서 질문과 가장 가까운 조각의 코사인 유사도
- 현장건의 후보군: `exmnt_search.search`와 같은 검색(조각 50개 → 묶음). 붙임 여부는 지금 규칙(점수 ≥ 0.65 이면서 회신 1위 점수 초과, 최대 2건)
대상: 개발 데이터(합성 150, dev1~7, devF, 답변불가 20)와 test7(90번에서 잠금 해제). 잠긴 시험 세트(test1~6, final_test)는 읽지 않는다.
출력 (data/eval, 저장소에 올리지 않음)
- r1_retrieval_by_question.csv: 질문당 한 줄 (정답 순위·점수, 1위 사례·점수, Hit@1/3/5/10/20, 현장건의 1위 점수·붙임 여부)
- r1_retrieval_candidates.csv: 질문 × 후보 한 줄 (회신 상위 5 + 현장건의 상위 5: 순위, 사례번호, 제목, 점수, 정답 여부)
"""
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import exmnt_search as es  # noqa: E402
import rag  # noqa: E402
import rerank as rr  # noqa: E402
from common import DATA  # noqa: E402

EVAL = DATA / "eval"
SETS = ("rewrite_150", "dev1", "dev2", "dev3", "dev4", "dev5", "dev6", "dev7", "devF", "test7", "unanswerable_20")
TOP = 5
r4 = lambda x: "" if x is None else round(x, 4)


def corpus_of(name, q):
    return "답변불가" if name == "unanswerable_20" else "현장건의" if name == "devF" or q.get("corpus") == "exmnt" else "기존 회신"


def main():
    qs = [(n, json.loads(l)) for n in SETS for l in open(EVAL / f"{n}.jsonl", encoding="utf-8") if l.strip()]
    t0 = time.time()
    rag.embed("준비")
    masked = [rag.mask_pii(q["question"]) for _, q in qs]
    vecs = rag._local_model.encode(masked, normalize_embeddings=True, batch_size=32)
    print(f"[임베딩] {len(qs)}문항 {time.time() - t0:.0f}초", flush=True)
    cache = {m: v.tolist() for m, v in zip(masked, vecs)}
    one = rag.embed
    rag.embed = lambda t: cache[t] if t in cache else one(t)          # rr.candidates가 다시 임베딩하지 않게
    by_q, by_c = [], []
    with rag.connect() as conn:
        grp_rep = {r["group_id"]: r["representative_case_key"] for r in conn.execute("select group_id, representative_case_key from interp.case_groups").fetchall()}
        ex = {r["group_id"]: r for r in conn.execute("select g.group_id, c.case_key, c.title, c.answer_type from interp.exmnt_groups g join interp.exmnt_cases c on c.case_key = g.representative_case_key").fetchall()}
        for (name, q), m in zip(qs, masked):
            corpus, gold = corpus_of(name, q), q.get("gold_group_id")
            cands = rr.candidates(conn, m)                                         # 회신 묶음 20개 (점수순)
            v = "[" + ",".join(map(str, cache[m])) + "]"
            conn.execute("SET hnsw.ef_search = 200")
            rows = conn.execute("select group_id, 1 - (embedding <=> %(v)s::vector) s from interp.exmnt_chunks order by embedding <=> %(v)s::vector limit %(n)s", {"v": v, "n": es.FETCH}).fetchall()
            best = {}
            for r in rows:
                best.setdefault(r["group_id"], float(r["s"]))
            fx = list(best.items())                                                # 현장건의 묶음 (점수순)
            ref = cands[0]["score"] if cands else None
            attach = [g for g, s in fx[:es.MAX_N] if ref is not None and s >= max(es.THRESH, ref + 1e-9)]
            if corpus == "기존 회신":
                pool = [(c["group_id"], c["score"]) for c in cands]
                gold_key = grp_rep.get(gold)
            elif corpus == "현장건의":
                pool, gold_key = fx, ex[gold]["case_key"]
            else:
                pool, gold_key = [], ""
            rank = next((i + 1 for i, (g, _) in enumerate(pool) if g == gold), None)
            gscore = pool[rank - 1][1] if rank else None
            top1 = pool[0] if pool else (None, None)
            hit = lambda k: "" if corpus == "답변불가" else int(bool(rank and rank <= k))
            by_q.append([name, q["qid"], corpus, q.get("tag", ""), q.get("gubun", ""), q["question"], gold_key, rank or ("" if corpus == "답변불가" else f"{len(pool)}위 밖"), r4(gscore),
                         r4(top1[1]), r4(top1[1] - gscore) if rank else "", hit(1), hit(3), hit(5), hit(10), hit(20),
                         cands[0]["doc_id"], r4(ref), r4(cands[4]["score"]), ex[fx[0][0]]["case_key"] if fx else "", r4(fx[0][1]) if fx else "",
                         len(attach), ", ".join(ex[g]["case_key"] for g in attach), "" if corpus != "현장건의" else int(gold in attach)])
            for i, c in enumerate(cands[:TOP], 1):
                by_c.append([name, q["qid"], corpus, "기존 회신", i, c["doc_id"], c["title"], r4(c["score"]), c["chunk_type"], int(corpus == "기존 회신" and c["group_id"] == gold), ""])
            for i, (g, s) in enumerate(fx[:TOP], 1):
                by_c.append([name, q["qid"], corpus, "현장건의", i, ex[g]["case_key"], ex[g]["title"], r4(s), ex[g]["answer_type"], int(corpus == "현장건의" and g == gold), int(g in attach)])
    rag.embed = one
    with open(EVAL / "r1_retrieval_by_question.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["세트", "qid", "정답 종류", "질문 유형", "구분", "질문", "정답 사례", "정답 순위", "정답 점수", "정답 후보군 1위 점수", "1위와 정답의 점수 차",
                    "Hit@1", "Hit@3", "Hit@5", "Hit@10", "Hit@20", "회신 1위 사례", "회신 1위 점수", "회신 5위 점수", "현장건의 1위 사례", "현장건의 1위 점수",
                    "붙는 현장건의 수", "붙는 현장건의", "정답 현장건의가 붙음"])
        w.writerows(by_q)
    with open(EVAL / "r1_retrieval_candidates.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["세트", "qid", "정답 종류", "후보군", "순위", "사례번호", "제목", "유사도 점수", "조각 유형(현장건의는 검토결과)", "정답 여부", "문맥에 붙음(현장건의)"])
        w.writerows(by_c)
    print(f"[완료] 질문 {len(by_q)}줄, 후보 {len(by_c)}줄, {time.time() - t0:.0f}초")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
