"""금융당국 회신 선례 RAG 를 사공삼 DB 에 넣기 (worklog 53장) — interp 스키마 + 읽기 전용 계정 interp_reader.

    python scripts/load_noaction.py            # 이미 있으면 멈춤
    python scripts/load_noaction.py --rebuild  # interp 스키마를 지우고 다시

입력 data/public_rag/noaction/processed/ (회신 원문·조각·KURE-v1 벡터, 포털 저작권정책상 git 제외) + schema.sql·exmnt_schema.sql·indexes.sql
관리자 계정(infra/.env 의 POSTGRES_USER·PASSWORD)으로 트랜잭션 하나 — 실패하면 아무것도 남지 않음.
"""
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import psycopg
from dotenv import dotenv_values
from psycopg import sql
from psycopg.types.json import Jsonb

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))
from app.core import config  # noqa: E402
from app.rag_public.noaction.aliases import ALIASES as MANUAL_ALIASES  # noqa: E402

BASE = AI / "data" / "public_rag" / "noaction"
P = BASE / "processed"
MODEL = "nlpai-lab/KURE-v1"
ORGS = {"금융위원회", "금융감독원", "금융정보분석원"}


def rows(name):
    with open(P / name, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_date(s):
    d = re.sub(r"\D", "", s or "")
    if len(d) != 8:
        return None
    try:
        return date(int(d[:4]), int(d[4:6]), int(d[6:]))
    except ValueError:
        return None


def copy(cur, table, cols, data):
    n = 0
    with cur.copy(f"COPY interp.{table} ({', '.join(cols)}) FROM STDIN") as cp:
        for r in data:
            cp.write_row(r)
            n += 1
    return n


def vectors(name, n):
    X = np.load(P / name)
    if X.shape != (n, 1024) or not np.isfinite(X).all():
        sys.exit(f"{name}: 벡터 크기·값 이상 {X.shape} / 조각 {n}")
    norms = np.linalg.norm(X, axis=1)
    if norms.min() < 0.99 or norms.max() > 1.01:
        sys.exit(f"{name}: 정규화되지 않은 벡터")
    return X


def vec(v):
    return "[" + ",".join(map(str, v.tolist())) + "]"


def main():
    if not config.NOACTION_PASSWORD:
        sys.exit("AI/.env 에 NOACTION_PASSWORD (읽기 전용 계정 interp_reader 비밀번호)가 없습니다")
    infra = dotenv_values(AI.parent / "infra" / ".env")
    chunks, ex_chunks = rows("chunks_d3.jsonl"), rows("exmnt_chunks.jsonl")
    X, XE = vectors("embeddings_d3_kure-v1.npy", len(chunks)), vectors("embeddings_exmnt_kure-v1.npy", len(ex_chunks))
    cases = rows("cases_enriched.jsonl")
    gmap = {m["case_key"]: m for m in rows("case_group_map.jsonl")}
    tc = {t["case_key"]: t for t in rows("case_time_check.jsonl")}
    ids = {x["law_name"]: x for x in rows("law_ids.jsonl")}
    laws = sorted(rows("laws.jsonl"), key=lambda x: x["law_name"])
    lid = {x["law_name"]: i for i, x in enumerate(laws, 1)}
    ex_groups = rows("exmnt_groups.jsonl")
    gid_of = {k: g["group_id"] for g in ex_groups for k in g["members"]}

    with psycopg.connect(host=config.DB_HOST, port=config.DB_PORT, dbname=config.DB_NAME,
                         user=infra.get("POSTGRES_USER", "admin"), password=infra["POSTGRES_PASSWORD"]) as c, c.cursor() as cur:
        if cur.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name = 'interp'").fetchone():
            if "--rebuild" not in sys.argv:
                sys.exit("interp 스키마가 이미 있습니다. 다시 만들려면 --rebuild")
            cur.execute("DROP SCHEMA interp CASCADE")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        cur.execute((BASE / "schema.sql").read_text(encoding="utf-8").replace("{EMBED_DIM}", "1024"))
        cur.execute((BASE / "exmnt_schema.sql").read_text(encoding="utf-8"))
        n = {}
        n["case_groups"] = copy(cur, "case_groups", ("group_id", "group_type", "representative_case_key", "n_cases"),
                                ((g["group_id"], g["group_type"], g["representative_case_key"], g["n_cases"]) for g in rows("case_groups.jsonl")))

        def case_row(r):
            k, m, t = r["case_key"], gmap.get(r["case_key"]), tc.get(r["case_key"])
            return (k, r["gubun"], r["idx"], r["serial_no"], r["title"], r["title_source"], r["reply_title"], r["doc_class"],
                    m["group_id"] if m else None, bool(m and m["body_suspect"]), r["category"], r["category_inferred"], r["category_final"],
                    r["category_final_source"], to_date(r["reg_date"]), to_date(r["reply_date"]), r["reply_date_source"], r["status"],
                    r["answer_type"], r["answer_type_raw"], r["registrant"], r["question"], r["answer"], r["reason"], r["body_source"],
                    r["body_chars"], r["indexable"], r["skip_reason"], r["url"], t["time_check"] if t else None,
                    Jsonb(t["counts"]) if t else None, to_date(t["checked_on"]) if t else None)

        n["cases"] = copy(cur, "cases", ("case_key", "gubun", "idx", "serial_no", "title", "title_source", "reply_title", "doc_class", "group_id",
                                         "body_suspect", "category", "category_inferred", "category_final", "category_final_source", "reg_date",
                                         "reply_date", "reply_date_source", "status", "answer_type", "answer_type_raw", "registrant", "question",
                                         "answer", "reason", "body_source", "body_chars", "indexable", "skip_reason", "url", "time_check",
                                         "time_check_counts", "time_checked_on"), map(case_row, cases))
        n["case_depts"] = copy(cur, "case_depts", ("case_key", "seq", "dept_name", "is_org"),
                               ((r["case_key"], i, d, d in ORGS) for r in cases for i, d in enumerate(r["depts"] or [], 1)))
        n["attachments"] = copy(cur, "attachments", ("case_key", "org_file_name", "ext", "text_used"),
                                ((r["case_key"], a["org_file_name"], a["ext"], r["body_source"] == "attachment" and (a["ext"] or "").lower() in ("hwp", "hwpx"))
                                 for r in cases for a in r["attachments"] or []))

        def source_type(name):
            return "law" if ids[name].get("law_id") else ("admrul" if ids[name].get("admrul_id") else "unknown")

        n["laws"] = copy(cur, "laws", ("law_id", "law_name", "law_kind", "source_type", "category", "is_current"),
                         ((lid[x["law_name"]], x["law_name"], x["law_kind"], source_type(x["law_name"]), x["category"], ids[x["law_name"]].get("current"))
                          for x in laws))
        aliases = {}
        for x in ids.values():
            a = (x.get("abbrev") or "").strip()
            if a and re.sub(r"\s", "", a) != re.sub(r"\s", "", x["law_name"]):
                aliases.setdefault(a, (lid[x["law_name"]], "official"))
        for a, name in MANUAL_ALIASES.items():   # 공식 약칭에 없는 약칭·오기 (팀원 수작업 사전)
            if name in lid and re.sub(r"\s", "", a) != re.sub(r"\s", "", name):
                aliases.setdefault(a, (lid[name], "manual"))
        n["law_aliases"] = copy(cur, "law_aliases", ("alias", "law_id", "alias_source"), ((a, v, s) for a, (v, s) in sorted(aliases.items())))
        n["law_source_ids"] = copy(cur, "law_source_ids", ("law_id", "source_kind", "source_id"),
                                   ((lid[x["law_name"]], x["source_kind"], x["source_id"]) for x in rows("law_source_ids.jsonl")))
        n["case_laws"] = copy(cur, "case_laws", ("case_key", "law_id", "mentions"),
                              ((x["case_key"], lid[x["law_name"]], x["mentions"]) for x in rows("case_laws.jsonl")))
        n["case_articles"] = copy(cur, "case_articles", ("case_key", "law_id", "article", "article_sub", "paragraph", "item", "item_sub",
                                                         "supplementary", "section", "pos", "raw", "resolve_method", "law_guess"),
                                  ((a["case_key"], lid.get(a["law_name"]), a["article"], a["article_sub"], a["paragraph"], a["item"], a["item_sub"],
                                    a["supplementary"], a["section"], a["pos"], a["raw"], a["resolve_method"], a["law_guess"]) for a in rows("case_articles.jsonl")))
        art_id = {(r[1], r[2], r[3], r[4]): r[0] for r in cur.execute("SELECT id, case_key, section, pos, raw FROM interp.case_articles").fetchall()}
        n["case_article_status"] = copy(cur, "case_article_status", ("case_article_id", "checked_on", "status", "why", "n_later", "later_changes", "cited_part_same"),
                                        ((art_id[(s["case_key"], s["section"], s["pos"], s["raw"])], to_date(s["checked_on"]), s["status"], s["why"],
                                          s["n_later"], Jsonb(s["later_changes"]), s["cited_part_same"]) for s in rows("case_article_status.jsonl")))
        n["law_article_history"] = copy(cur, "law_article_history", ("law_id", "source_id", "article", "article_sub", "change_date", "change_reason",
                                                                     "effective_date", "promulgation_date", "revision_type"),
                                        ((lid[h["law_name"]], h["source_id"], h["article"], h["article_sub"], to_date(h["change_date"]), h["change_reason"],
                                          to_date(h["effective_date"]), to_date(h["promulgation_date"]), h["revision_type"])
                                         for h in rows("law_article_history.jsonl") if h["change_date"]))
        n["admrul_versions"] = copy(cur, "admrul_versions", ("serial", "law_id", "admrul_id", "name_at_time", "status", "issue_date", "effective_date", "revision_type"),
                                    ((v["serial"], lid[v["law_name"]], v["admrul_id"], v["name_at_time"], v["status"], to_date(v["issue_date"]),
                                      to_date(v["effective_date"]), v["revision_type"]) for v in rows("admrul_versions.jsonl")))
        n["chunks"] = copy(cur, "chunks", ("case_key", "group_id", "chunk_type", "chunk_no", "text", "embed_text", "bm25_tokens", "doc_class", "gubun",
                                           "category_final", "reply_date", "embed_model", "embedding"),
                           ((x["case_key"], x["group_id"], x["chunk_type"], x["chunk_no"], x["text"], x["embed_text"], x["bm25_tokens"], x["doc_class"],
                             x["gubun"], x["category_final"], to_date(x["reply_date"]), MODEL, vec(X[i])) for i, x in enumerate(chunks)))
        # 현장건의
        n["exmnt_groups"] = copy(cur, "exmnt_groups", ("group_id", "group_type", "representative_case_key", "n_cases"),
                                 ((g["group_id"], g["group_type"], g["representative_case_key"], g["n_cases"]) for g in ex_groups))
        n["exmnt_cases"] = copy(cur, "exmnt_cases", ("case_key", "idx", "title", "reply_date", "category", "category_top", "dept", "visit", "answer_type",
                                                     "question", "answer", "plan", "close_reason", "closed", "url", "changed_after", "already_in_place",
                                                     "articles", "norm_kinds", "indexable", "skip_reason", "group_id"),
                                ((r["case_key"], r["idx"], r["title"], to_date(r["reply_date"]), r["category"], r["category_top"], r["dept"], r["visit"],
                                  r["answer_type"], r["question"], r["answer"], r["plan"], r["close_reason"], to_date(r["closed"]), r["url"],
                                  r["changed_after"], r["already_in_place"], r["articles"], r["norm_kinds"], r["skip_reason"] is None,
                                  r["skip_reason"], gid_of.get(r["case_key"])) for r in rows("exmnt_cases.jsonl")))
        n["exmnt_chunks"] = copy(cur, "exmnt_chunks", ("case_key", "group_id", "chunk_type", "chunk_no", "text", "embed_text", "bm25_tokens", "category",
                                                       "reply_date", "answer_type", "embed_model", "embedding"),
                                 ((x["case_key"], x["group_id"], x["chunk_type"], x["chunk_no"], x["text"], x["embed_text"], x["bm25_tokens"],
                                   x["category_final"], to_date(x["reply_date"]), x["answer_type"], MODEL, vec(XE[i])) for i, x in enumerate(ex_chunks)))
        # 벡터 색인을 병렬로 만들면 공유 메모리(/dev/shm)를 써서 Docker 기본 64MB 에서 DiskFull → 병렬 끔
        cur.execute("SET LOCAL maintenance_work_mem = '512MB'")
        cur.execute("SET LOCAL max_parallel_maintenance_workers = 0")
        cur.execute((BASE / "indexes.sql").read_text(encoding="utf-8"))
        # 읽기 전용 계정 (사내 RAG 계정과 서로 못 읽음)
        if not cur.execute("SELECT 1 FROM pg_roles WHERE rolname = 'interp_reader'").fetchone():
            cur.execute("CREATE ROLE interp_reader LOGIN")
        cur.execute(sql.SQL("ALTER ROLE interp_reader PASSWORD {}").format(sql.Literal(config.NOACTION_PASSWORD)))
        cur.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO interp_reader").format(sql.Identifier(config.DB_NAME)))
        cur.execute("REVOKE ALL ON SCHEMA interp FROM PUBLIC")
        cur.execute("GRANT USAGE ON SCHEMA interp TO interp_reader")
        cur.execute("GRANT SELECT ON ALL TABLES IN SCHEMA interp TO interp_reader")
        for (t,) in cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'interp'").fetchall():
            cur.execute(sql.SQL("ANALYZE interp.{}").format(sql.Identifier(t)))   # DB 전체 ANALYZE 는 lawrag 까지 돌아 느림
    print(json.dumps(n, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except psycopg.Error as e:  # 비밀번호가 담길 수 있어 종류만
        sys.exit(f"DB 오류, 적재를 모두 되돌렸습니다: {type(e).__name__}")
