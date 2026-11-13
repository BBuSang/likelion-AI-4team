"""X6 현장건의 별도 테이블 적재 (사용자 결정 2026-11-11): interp.exmnt_groups / exmnt_cases / exmnt_chunks. 기존 테이블은 건드리지 않는다.
- 트랜잭션 하나: 실패하면 아무것도 남지 않는다. 테이블이 이미 있으면 멈춘다(다시 만들려면 --rebuild, 현장건의 테이블 3개만 지우고 다시 만든다)
- 적재 전 점검: 벡터 개수·1024차원·NaN·정규화, Colab 입력 글(exmnt_colab_input.json)과 지금 청크의 글·순서 일치
- 적재 전후 기존 설정 지문(f2.fingerprint, interp.chunks 기준)이 같아야 한다. 다르면 되돌린다
- 적재 후: 개수 확인, DB 벡터 검색과 numpy 정확 검색의 상위 묶음 비교(질문 몇 개)
입력: processed/exmnt_cases.jsonl, exmnt_groups.jsonl, exmnt_chunks.jsonl, embeddings_exmnt_kure-v1.npy   접속 값은 .env의 RAG_DSN을 읽기만 하고 출력하지 않는다
"""
import json
import os
import re
import sys

import numpy as np
import psycopg
from psycopg.rows import dict_row

import f2_test_eval as f2
import rag
from common import OUT, ROOT, manifest, pg_connect, read_jsonl
from s14_load import copy, to_date

MODEL = "nlpai-lab/KURE-v1"
TABLES = ("exmnt_chunks", "exmnt_cases", "exmnt_groups")
PROBES = ["대출 실행 전후로 고객에게 적금 가입을 권유하면 구속성 영업에 걸리나요?", "보험 가입 고객에게 사은품을 줄 때 특별이익 한도 금액은 어떻게 계산하나요?",
          "신용카드 포인트를 현금으로 바꿔 주는 절차를 간소화할 수 있나요?", "증권사가 고객 정보를 계열 은행과 공유해 복합점포를 운영해도 되나요?"]


def check_vectors(chunks, X):
    if X.shape != (len(chunks), 1024):
        sys.exit(f"벡터 크기 불일치: {X.shape} / 청크 {len(chunks)}")
    if not np.isfinite(X).all():
        sys.exit("벡터에 NaN/Inf가 있습니다.")
    n = np.linalg.norm(X, axis=1)
    if n.min() < 0.99 or n.max() > 1.01:
        sys.exit(f"정규화되지 않은 벡터: {n.min():.4f}~{n.max():.4f}")
    src = json.load(open(OUT / "exmnt_colab_input.json", encoding="utf-8"))
    if src["model"] != MODEL or src["chunks"] != [c["embed_text"] for c in chunks]:
        sys.exit("벡터를 만든 글(exmnt_colab_input.json)과 지금 청크의 글·순서·모델이 다릅니다.")


def main():
    cases = read_jsonl("exmnt_cases.jsonl")
    groups = read_jsonl("exmnt_groups.jsonl")
    chunks = read_jsonl("exmnt_chunks.jsonl")
    X = np.load(OUT / "embeddings_exmnt_kure-v1.npy")
    check_vectors(chunks, X)
    gid_of = {k: g["group_id"] for g in groups for k in g["members"]}
    with pg_connect(os.environ["RAG_DSN"], row_factory=dict_row) as c, c.cursor() as cur:
        before = f2.fingerprint(c)["id"]
        have = [t for t in TABLES if cur.execute("select to_regclass(%s) r", (f"interp.{t}",)).fetchone()["r"]]
        if have:
            if "--rebuild" not in sys.argv:
                sys.exit(f"현장건의 테이블이 이미 있습니다({have}). 다시 만들려면 --rebuild를 붙이세요.")
            for t in TABLES:
                cur.execute(f"DROP TABLE IF EXISTS interp.{t} CASCADE")
        cur.execute((ROOT / "db" / "exmnt_schema.sql").read_text(encoding="utf-8"))
        counts = {}
        counts["exmnt_groups"] = copy(cur, "exmnt_groups", ("group_id", "group_type", "representative_case_key", "n_cases"),
                                      ((g["group_id"], g["group_type"], g["representative_case_key"], g["n_cases"]) for g in groups))
        counts["exmnt_cases"] = copy(cur, "exmnt_cases", ("case_key", "idx", "title", "reply_date", "category", "category_top", "dept", "visit", "answer_type", "question", "answer",
                                                          "plan", "close_reason", "closed", "url", "changed_after", "already_in_place", "articles", "norm_kinds", "indexable", "skip_reason", "group_id"),
                                     ((r["case_key"], r["idx"], r["title"], to_date(r["reply_date"]), r["category"], r["category_top"], r["dept"], r["visit"], r["answer_type"], r["question"],
                                       r["answer"], r["plan"], r["close_reason"], to_date(r["closed"]), r["url"], r["changed_after"], r["already_in_place"], r["articles"], r["norm_kinds"],
                                       r["skip_reason"] is None, r["skip_reason"], gid_of.get(r["case_key"])) for r in cases))
        counts["exmnt_chunks"] = copy(cur, "exmnt_chunks", ("case_key", "group_id", "chunk_type", "chunk_no", "text", "embed_text", "bm25_tokens", "category", "reply_date", "answer_type",
                                                            "embed_model", "embedding"),
                                      ((x["case_key"], x["group_id"], x["chunk_type"], x["chunk_no"], x["text"], x["embed_text"], x["bm25_tokens"], x["category_final"], to_date(x["reply_date"]),
                                        x["answer_type"], MODEL, "[" + ",".join(map(str, X[i].tolist())) + "]") for i, x in enumerate(chunks)))
        cur.execute("ANALYZE interp.exmnt_groups; ANALYZE interp.exmnt_cases; ANALYZE interp.exmnt_chunks")
        after = f2.fingerprint(c)["id"]
        if after != before:
            raise SystemExit(f"기존 설정 지문이 바뀌었습니다({before} → {after}). 되돌립니다.")      # with 블록을 예외로 빠져나가 롤백된다
        db_n = {t: cur.execute(f"select count(*) n from interp.{t}").fetchone()["n"] for t in TABLES}
        if db_n != {"exmnt_chunks": len(chunks), "exmnt_cases": len(cases), "exmnt_groups": len(groups)}:
            raise SystemExit(f"개수 불일치: {db_n}")

        # DB 벡터 검색 vs numpy 정확 검색 (상위 5 묶음)
        cg = np.array([x["group_id"] for x in chunks])
        uniq, inv = np.unique(cg, return_inverse=True)
        same = 0
        cur.execute("SET LOCAL hnsw.ef_search = 200")
        for q in PROBES:
            qv = np.array(rag.embed(rag.mask_pii(q)), dtype=np.float32)
            best = np.full(len(uniq), -1.0)
            np.maximum.at(best, inv, X @ qv)
            exact = [int(uniq[i]) for i in np.argsort(-best)[:5]]
            rows = cur.execute("select group_id from interp.exmnt_chunks order by embedding <=> %s::vector limit 50", ("[" + ",".join(map(str, qv.tolist())) + "]",)).fetchall()
            got = list(dict.fromkeys(r["group_id"] for r in rows))[:5]
            same += got == exact
            print(f"  {q[:30]}… DB {got} / 정확 {exact} {'같음' if got == exact else '다름'}")
    manifest("x6_exmnt_load", counts=counts, fingerprint=before, fingerprint_unchanged=True, probes_same=f"{same}/{len(PROBES)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except psycopg.Error as e:
        pw = re.search(r"://[^:/@]+:([^@]+)@", os.getenv("RAG_DSN", ""))
        msg = str(e).replace(pw.group(1), "****") if pw else str(e)
        sys.exit(f"DB 오류, 적재를 모두 되돌렸습니다: {type(e).__name__}: {msg[:400]}")
