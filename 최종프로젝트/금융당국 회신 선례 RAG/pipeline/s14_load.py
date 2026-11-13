"""14 적재: 스키마를 만들고 처리 결과와 벡터를 DB에 넣는다. 전체가 트랜잭션 하나라서 실패하면 아무것도 남지 않는다.
기존 스키마가 있으면 멈춘다. 다시 만들려면: python s14_load.py --rebuild
청크 세트 선택: --set baseline-openai (기본, .env의 OpenAI 모델·차원) / --set d3-kure (질의 카드 + 판단 본문, 로컬 KURE-v1 1024차원. .env 값을 쓰지 않는다)
입력: processed/ 의 사례·묶음·법령·조문·판정·청크, embeddings.npy   접속 값은 .env의 RAG_DSN을 읽기만 하고 출력하지 않는다
"""
import hashlib
import os
import re
import sqlite3
import sys
from collections import Counter
from datetime import date

import numpy as np
import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from common import OUT, ROOT, manifest, pg_connect, read_jsonl
from law_rules import ALIASES, key

load_dotenv(ROOT / ".env", override=True)
ORGS = {"금융위원회", "금융감독원", "금융정보분석원"}


def to_date(s):
    """'2024-01-31'·'20240131' → date. 달력에 없는 값은 None."""
    d = re.sub(r"\D", "", s or "")
    if len(d) != 8:
        return None
    try:
        return date(int(d[:4]), int(d[4:6]), int(d[6:]))
    except ValueError:
        return None


def copy(cur, table, cols, rows):
    n = 0
    with cur.copy(f"COPY interp.{table} ({', '.join(cols)}) FROM STDIN") as cp:
        for r in rows:
            cp.write_row(r)
            n += 1
    return n


def check_vectors(chunks, X, model, dim):
    """벡터가 지금 청크 글에서 만들어진 것인지 캐시와 맞춰 본다."""
    if X.shape != (len(chunks), dim):
        sys.exit(f"벡터 크기 불일치: {X.shape} / 청크 {len(chunks)}, 차원 {dim}")
    db = sqlite3.connect(OUT / "embedding_cache.sqlite")
    got = dict(db.execute("select hash, vec from cache where model = ?", (model,)))
    bad = sum(got.get(hashlib.sha256(f"{model}\n{c['embed_text']}".encode()).hexdigest()) != X[i].tobytes() for i, c in enumerate(chunks))
    if bad:
        sys.exit(f"청크와 벡터가 맞지 않는 행 {bad}개. s13을 다시 실행하세요.")


def check_local_vectors(chunks, X, model):
    """로컬 모델(Colab에서 만든) 벡터 점검: 개수, 유한성, 정규화, Colab 입력 글과의 순서 일치, 로컬 fp32 캐시와의 일치."""
    import json
    if X.shape[0] != len(chunks):
        sys.exit(f"벡터 개수 불일치: {X.shape} / 청크 {len(chunks)}")
    if not np.isfinite(X).all():
        sys.exit("벡터에 NaN/Inf가 있습니다.")
    norms = np.linalg.norm(X, axis=1)
    if norms.min() < 0.99 or norms.max() > 1.01:
        sys.exit(f"정규화되지 않은 벡터: {norms.min():.3f}~{norms.max():.3f}")
    src = json.load(open(OUT / "colab_input.json", encoding="utf-8"))["chunks"]
    if src != [c["embed_text"] for c in chunks]:
        sys.exit("벡터를 만든 글(colab_input.json)과 지금 청크의 임베딩 글·순서가 다릅니다.")
    db = sqlite3.connect(OUT / "embedding_cache_local.sqlite")
    got = dict(db.execute("select hash, vec from cache where model = ?", (model,)))
    cos = [float(np.frombuffer(got[h], dtype=np.float32) @ X[i]) for i, c in enumerate(chunks)
           if (h := hashlib.sha256(f"{model}\n{c['embed_text']}".encode()).hexdigest()) in got]
    if cos and min(cos) < 0.999:
        sys.exit(f"로컬 fp32 벡터와 다른 행이 있습니다 (최소 코사인 {min(cos):.4f}, {len(cos)}개 비교).")
    print(f"벡터 점검 통과: {X.shape}, 로컬 fp32와 비교 {len(cos)}개 최소 코사인 {min(cos) if cos else float('nan'):.5f}")


SETS = {                                                   # 청크 세트: (청크 파일, 벡터 파일, 모델명 또는 None=.env)
    "baseline-openai": ("chunks.jsonl", "embeddings.npy", None),
    "d3-kure": ("chunks_d3.jsonl", "embeddings_d3_kure-v1.npy", "nlpai-lab/KURE-v1"),
}


def main():
    name = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "baseline-openai"
    if name not in SETS:
        sys.exit(f"알 수 없는 청크 세트 {name}: {list(SETS)}")
    chunk_file, emb_file, local_model = SETS[name]
    dsn = os.getenv("RAG_DSN", "")
    chunks = read_jsonl(chunk_file)
    X = np.load(OUT / emb_file)
    if local_model:
        model, dim = local_model, int(X.shape[1])
        check_local_vectors(chunks, X, model)
    else:
        model, dim = os.getenv("EMBED_MODEL", "").strip(), int(os.getenv("EMBED_DIM", "0"))
        check_vectors(chunks, X, model, dim)

    cases = read_jsonl("cases_enriched.jsonl")
    gmap = {m["case_key"]: m for m in read_jsonl("case_group_map.jsonl")}
    tc = {t["case_key"]: t for t in read_jsonl("case_time_check.jsonl")}
    ids = {x["law_name"]: x for x in read_jsonl("law_ids.jsonl")}
    laws = sorted(read_jsonl("laws.jsonl"), key=lambda x: x["law_name"])
    lid = {x["law_name"]: i for i, x in enumerate(laws, 1)}
    articles = read_jsonl("case_articles.jsonl")
    status = read_jsonl("case_article_status.jsonl")
    history = [h for h in read_jsonl("law_article_history.jsonl") if h["change_date"]]   # 이력 없음 표시 행은 넣지 않는다

    aliases, conflicts = {}, 0
    for x in ids.values():
        a = (x.get("abbrev") or "").strip()
        if a and key(a) != key(x["law_name"]):
            if aliases.setdefault(a, (lid[x["law_name"]], "official"))[0] != lid[x["law_name"]]:
                conflicts += 1
    for a, name in ALIASES.items():
        if name in lid and key(a) != key(name):
            if aliases.setdefault(a, (lid[name], "manual"))[0] != lid[name]:
                conflicts += 1

    with pg_connect(dsn) as c, c.cursor() as cur:
        exists = cur.execute("select 1 from information_schema.schemata where schema_name = 'interp'").fetchone()
        if exists:
            if "--rebuild" not in sys.argv:
                sys.exit("interp 스키마가 이미 있습니다. 다시 만들려면 --rebuild를 붙이세요.")
            cur.execute("DROP SCHEMA interp CASCADE")
        cur.execute((ROOT / "db" / "schema.sql").read_text(encoding="utf-8").replace("{EMBED_DIM}", str(dim)))

        counts = {}
        counts["case_groups"] = copy(cur, "case_groups", ("group_id", "group_type", "representative_case_key", "n_cases"),
                                     ((g["group_id"], g["group_type"], g["representative_case_key"], g["n_cases"]) for g in read_jsonl("case_groups.jsonl")))

        def case_row(r):
            k, m, t = r["case_key"], gmap.get(r["case_key"]), tc.get(r["case_key"])
            return (k, r["gubun"], r["idx"], r["serial_no"], r["title"], r["title_source"], r["reply_title"], r["doc_class"],
                    m["group_id"] if m else None, bool(m and m["body_suspect"]), r["category"], r["category_inferred"], r["category_final"],
                    r["category_final_source"], to_date(r["reg_date"]), to_date(r["reply_date"]), r["reply_date_source"], r["status"],
                    r["answer_type"], r["answer_type_raw"], r["registrant"], r["question"], r["answer"], r["reason"], r["body_source"],
                    r["body_chars"], r["indexable"], r["skip_reason"], r["url"], t["time_check"] if t else None,
                    Jsonb(t["counts"]) if t else None, to_date(t["checked_on"]) if t else None)

        counts["cases"] = copy(cur, "cases", ("case_key", "gubun", "idx", "serial_no", "title", "title_source", "reply_title", "doc_class", "group_id",
                                              "body_suspect", "category", "category_inferred", "category_final", "category_final_source", "reg_date",
                                              "reply_date", "reply_date_source", "status", "answer_type", "answer_type_raw", "registrant", "question",
                                              "answer", "reason", "body_source", "body_chars", "indexable", "skip_reason", "url", "time_check",
                                              "time_check_counts", "time_checked_on"), map(case_row, cases))
        counts["case_depts"] = copy(cur, "case_depts", ("case_key", "seq", "dept_name", "is_org"),
                                    ((r["case_key"], i, d, d in ORGS) for r in cases for i, d in enumerate(r["depts"] or [], 1)))
        counts["attachments"] = copy(cur, "attachments", ("case_key", "org_file_name", "ext", "text_used"),
                                     ((r["case_key"], a["org_file_name"], a["ext"], r["body_source"] == "attachment" and (a["ext"] or "").lower() in ("hwp", "hwpx"))
                                      for r in cases for a in r["attachments"] or []))

        def source_type(n):
            return "law" if ids[n].get("law_id") else ("admrul" if ids[n].get("admrul_id") else "unknown")

        counts["laws"] = copy(cur, "laws", ("law_id", "law_name", "law_kind", "source_type", "category", "is_current"),
                              ((lid[x["law_name"]], x["law_name"], x["law_kind"], source_type(x["law_name"]), x["category"], ids[x["law_name"]].get("current")) for x in laws))
        counts["law_aliases"] = copy(cur, "law_aliases", ("alias", "law_id", "alias_source"), ((a, v[0], v[1]) for a, v in sorted(aliases.items())))
        counts["law_source_ids"] = copy(cur, "law_source_ids", ("law_id", "source_kind", "source_id"),
                                        ((lid[x["law_name"]], x["source_kind"], x["source_id"]) for x in read_jsonl("law_source_ids.jsonl")))
        counts["case_laws"] = copy(cur, "case_laws", ("case_key", "law_id", "mentions"), ((x["case_key"], lid[x["law_name"]], x["mentions"]) for x in read_jsonl("case_laws.jsonl")))
        counts["case_articles"] = copy(cur, "case_articles", ("case_key", "law_id", "article", "article_sub", "paragraph", "item", "item_sub", "supplementary", "section", "pos", "raw", "resolve_method", "law_guess"),
                                       ((a["case_key"], lid.get(a["law_name"]), a["article"], a["article_sub"], a["paragraph"], a["item"], a["item_sub"], a["supplementary"], a["section"], a["pos"], a["raw"], a["resolve_method"], a["law_guess"]) for a in articles))
        art_id = {(r[1], r[2], r[3], r[4]): r[0] for r in cur.execute("select id, case_key, section, pos, raw from interp.case_articles").fetchall()}
        counts["case_article_status"] = copy(cur, "case_article_status", ("case_article_id", "checked_on", "status", "why", "n_later", "later_changes", "cited_part_same"),
                                             ((art_id[(s["case_key"], s["section"], s["pos"], s["raw"])], to_date(s["checked_on"]), s["status"], s["why"], s["n_later"], Jsonb(s["later_changes"]), s["cited_part_same"]) for s in status))
        counts["law_article_history"] = copy(cur, "law_article_history", ("law_id", "source_id", "article", "article_sub", "change_date", "change_reason", "effective_date", "promulgation_date", "revision_type"),
                                             ((lid[h["law_name"]], h["source_id"], h["article"], h["article_sub"], to_date(h["change_date"]), h["change_reason"], to_date(h["effective_date"]), to_date(h["promulgation_date"]), h["revision_type"]) for h in history))
        counts["admrul_versions"] = copy(cur, "admrul_versions", ("serial", "law_id", "admrul_id", "name_at_time", "status", "issue_date", "effective_date", "revision_type"),
                                         ((v["serial"], lid[v["law_name"]], v["admrul_id"], v["name_at_time"], v["status"], to_date(v["issue_date"]), to_date(v["effective_date"]), v["revision_type"]) for v in read_jsonl("admrul_versions.jsonl")))
        counts["chunks"] = copy(cur, "chunks", ("case_key", "group_id", "chunk_type", "chunk_no", "text", "embed_text", "bm25_tokens", "doc_class", "gubun", "category_final", "reply_date", "embed_model", "embedding"),
                                ((x["case_key"], x["group_id"], x["chunk_type"], x["chunk_no"], x["text"], x["embed_text"], x["bm25_tokens"], x["doc_class"], x["gubun"], x["category_final"], to_date(x["reply_date"]), model, "[" + ",".join(map(str, X[i].tolist())) + "]") for i, x in enumerate(chunks)))

        cur.execute("SET LOCAL maintenance_work_mem = '512MB'")
        cur.execute((ROOT / "db" / "indexes.sql").read_text(encoding="utf-8"))
        cur.execute("ANALYZE")

    manifest("14_load", counts=counts, alias_conflicts=conflicts, statuses=dict(Counter(s["status"] for s in status)))


if __name__ == "__main__":
    try:
        main()
    except psycopg.Error as e:
        pw = re.search(r"://[^:/@]+:([^@]+)@", os.getenv("RAG_DSN", ""))
        msg = str(e).replace(pw.group(1), "****") if pw else str(e)
        sys.exit(f"DB 오류, 적재를 모두 되돌렸습니다: {type(e).__name__}: {msg[:400]}")
