"""기획안 5-5 적재: PostgreSQL + pgvector (team4-lawrag2-db, 127.0.0.1:5435)

DB(`POSTGRES_DB`, law_rag)를 **지우고 새로 만든 뒤** ERD(4-4) 테이블에 넣는다. 몇 번을 돌려도 같은 결과.
  laws(380) · law_aliases · articles(조문 전부, 검색 제외 포함) · article_lines(항·호·목 줄) · chunks(조각 + 1024차원 벡터)
  · article_links(위임·준용 연결, v0.31)

입력: data/work/01_fetch/laws.csv, selected.json · 02_clean/articles.csv, article_lines.csv · 04_chunks/chunks.csv
      05_embed/kure_vectors.npz (chunk_id·input_hash 가 chunks.csv 와 다르면 멈춤) · 06_links/article_links.csv

    python pipeline/load_05.py
"""
import csv
import json
import time
from datetime import date

import numpy as np
import psycopg
from psycopg import sql

from common import ENV, TODAY, WORK

csv.field_size_limit(2**31 - 1)
DB = ENV["POSTGRES_DB"]
RULE_VERSION = "5-5 v2 (기획안 v0.31)"


def connect(dbname, user=None, pw=None):
    return psycopg.connect(host=ENV["DB_HOST"], port=ENV["DB_PORT"], dbname=dbname,
                           user=user or ENV["POSTGRES_USER"], password=pw or ENV["POSTGRES_PASSWORD"], autocommit=True)


def d8(s):
    s = (s or "").strip()
    return date(int(s[:4]), int(s[4:6]), int(s[6:8])) if len(s) == 8 and s.isdigit() else None


def b(s):
    return s == "True"


def i(s):
    return int(s) if str(s).strip().lstrip("-").isdigit() else None


DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA lawrag;

CREATE TABLE lawrag.laws (
    law_uid        text PRIMARY KEY,          -- law:MST / admrul:행정규칙일련번호
    law_id         text,                      -- 법령ID / 행정규칙ID (판이 바뀌어도 같음)
    mst            text,                      -- 판 번호
    law_name       text NOT NULL,
    abbrev         text,
    law_level      text NOT NULL,             -- 법률 | 시행령 | 시행규칙 | 행정규칙
    kind_raw       text,                      -- 대통령령 | 총리령 | 고시 | 세칙 …
    parent_law_uid text REFERENCES lawrag.laws(law_uid),
    scope_group    text,                      -- A | B | C | D
    sector         text,                      -- 업권별 법률 | 공통 규제 | 일반법·인접 법률
    n_cases        int,                       -- 팀 목록 회신 사례 수
    org            text,
    promulgation_date date,
    effective_date date NOT NULL,
    n_articles_src int,
    source_file    text,
    fetched_on     date
);
CREATE TABLE lawrag.law_aliases (
    alias        text NOT NULL,
    law_uid      text NOT NULL REFERENCES lawrag.laws(law_uid),
    alias_source text NOT NULL,               -- official | team
    PRIMARY KEY (alias, law_uid)
);
CREATE TABLE lawrag.articles (
    article_uid   text PRIMARY KEY,
    law_uid       text NOT NULL REFERENCES lawrag.laws(law_uid),
    seq           int  NOT NULL,
    article_no    int, article_dash int, article_sub int,
    article_label text NOT NULL,              -- 제34조의2 / 제1-4조 / 1.(지침)
    title         text,
    chapter       text,
    raw_text      text,                       -- 원문 그대로
    text          text,                       -- 표시용
    embed_body    text,                       -- 임베딩용 (한자 괄호 뺀 것)
    article_effective_date date,
    searchable    boolean NOT NULL,
    skip_reason   text,                       -- 삭제 | 본문 없음
    is_deleted    boolean NOT NULL,
    has_image boolean, has_table boolean, has_delegation boolean, is_reference_list boolean,
    parse_method  text,                       -- api | admrul_list | admrul_split | guideline_items | whole_body
    n_chars int, n_tokens int
);
CREATE TABLE lawrag.article_lines (
    line_uid        text PRIMARY KEY,
    article_uid     text NOT NULL REFERENCES lawrag.articles(article_uid),
    seq             int  NOT NULL,
    level           text NOT NULL,            -- 본문 | 항 | 호 | 목 | 세목
    marker          text, para_no text, item_no text, sub_no text,
    parent_line_uid text,                     -- 상위 줄 (호 → 항, 목 → 호)
    text            text, embed_text text,
    is_deleted      boolean, not_in_force boolean
);
CREATE TABLE lawrag.chunks (
    chunk_id      text PRIMARY KEY,           -- article_uid#조각번호
    article_uid   text NOT NULL REFERENCES lawrag.articles(article_uid),
    law_uid       text NOT NULL REFERENCES lawrag.laws(law_uid),
    chunk_no      int NOT NULL, n_chunks int NOT NULL,
    item_from text, item_to text,
    cite          text NOT NULL,              -- 출처: 법령명 제O조(제목) 제O항
    text          text NOT NULL,              -- 표시용 조각 본문
    embed_text    text NOT NULL,              -- 머리말 + 본문 (임베딩 입력)
    n_chars int, n_tokens int,
    law_name      text NOT NULL, law_level text NOT NULL, scope_group text,
    article_label text, title text, effective_date date,
    has_ctx boolean, by_sentence boolean, has_image boolean, has_table boolean,
    has_delegation boolean, is_reference_list boolean,
    embed_model   text NOT NULL,
    input_hash    text NOT NULL,
    embedding     vector(1024) NOT NULL
);
CREATE TABLE lawrag.article_links (            -- v0.31: 위임·준용 연결 (5-6-6, pipeline/links_06.py)
    src_uid  text NOT NULL REFERENCES lawrag.articles(article_uid),   -- 참조하는 조문
    dst_uid  text NOT NULL REFERENCES lawrag.articles(article_uid),   -- 참조되는 조문
    kind     text NOT NULL CHECK (kind IN ('위임', '준용')),
    PRIMARY KEY (src_uid, dst_uid)
);
"""

INDEX = """
CREATE INDEX chunks_embedding_hnsw ON lawrag.chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX chunks_law_level ON lawrag.chunks (law_level);
CREATE INDEX chunks_article ON lawrag.chunks (article_uid);
CREATE INDEX chunks_law ON lawrag.chunks (law_uid);
CREATE INDEX articles_law ON lawrag.articles (law_uid);
CREATE INDEX lines_article ON lawrag.article_lines (article_uid);
CREATE INDEX links_dst ON lawrag.article_links (dst_uid);
"""


def recreate_db():
    with connect("postgres") as c:
        c.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()", (DB,))
        c.execute(f'DROP DATABASE IF EXISTS "{DB}"')
        c.execute(f'CREATE DATABASE "{DB}"')
        if not c.execute("SELECT 1 FROM pg_roles WHERE rolname='law_reader'").fetchone():
            c.execute(sql.SQL("CREATE ROLE law_reader LOGIN PASSWORD {}").format(sql.Literal(ENV["LAW_PASSWORD"])))
    with connect(DB) as c:
        c.execute(DDL)
        c.execute(f'GRANT CONNECT ON DATABASE "{DB}" TO law_reader')
        c.execute("GRANT USAGE ON SCHEMA lawrag TO law_reader")
        c.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA lawrag GRANT SELECT ON TABLES TO law_reader")


def copy(c, table, cols, rows):
    with c.cursor() as cur, cur.copy(f"COPY lawrag.{table} ({', '.join(cols)}) FROM STDIN") as cp:
        n = 0
        for r in rows:
            cp.write_row(r)
            n += 1
    return n


def main():
    t0 = time.time()
    sel = json.loads((WORK / "01_fetch" / "selected.json").read_text(encoding="utf-8"))
    laws = list(csv.DictReader(open(WORK / "01_fetch" / "laws.csv", encoding="utf-8-sig")))
    arts = list(csv.DictReader(open(WORK / "02_clean" / "articles.csv", encoding="utf-8-sig")))
    lines = list(csv.DictReader(open(WORK / "02_clean" / "article_lines.csv", encoding="utf-8-sig")))
    chunks = list(csv.DictReader(open(WORK / "04_chunks" / "chunks.csv", encoding="utf-8-sig")))
    links = list(csv.DictReader(open(WORK / "06_links" / "article_links.csv", encoding="utf-8-sig")))
    z = np.load(WORK / "05_embed" / "kure_vectors.npz")
    ids, hashes, vecs = z["chunk_id"].tolist(), z["input_hash"].tolist(), z["vectors"]
    # 벡터 짝 확인 (5-4 기준 1·2): 다르면 적재하지 않음
    assert ids == [c["chunk_id"] for c in chunks], "벡터의 chunk_id 가 chunks.csv 와 다릅니다 (5-3을 다시 돌렸다면 5-4 임베딩도 다시)"
    assert hashes == [c["input_hash"] for c in chunks], "벡터의 input_hash 가 조각 글과 다릅니다"
    print(f"읽기 {time.time() - t0:.0f}초 · 법령 {len(laws)} · 조문 {len(arts):,} · 줄 {len(lines):,} · 조각 {len(chunks):,} · 연결 {len(links):,}")

    recreate_db()
    print(f"DB `{DB}` 새로 만듦 (확장 vector, 스키마 lawrag, law_reader 읽기 권한)")
    fetched = d8(sel["checked_on"])
    with connect(DB) as c:
        # 상위 법률이 먼저 들어가야 FK 가 맞으므로 상위 없는 것부터
        order = sorted(laws, key=lambda r: bool(r["parent_uid"]))
        n = copy(c, "laws", ["law_uid", "law_id", "mst", "law_name", "abbrev", "law_level", "kind_raw", "parent_law_uid",
                             "scope_group", "sector", "n_cases", "org", "promulgation_date", "effective_date",
                             "n_articles_src", "source_file", "fetched_on"],
                 ([r["uid"], r["law_id"], r["mst"], r["law_name"], r["abbrev"] or None, r["level"], r["kind_raw"],
                   r["parent_uid"] or None, r["scope"], r["sector"] or None, i(r["n_cases"]), r["org"],
                   d8(r["promulgation"]), d8(r["effective"]), i(r["n_articles"]), r["file"], fetched] for r in order))
        print("  laws", n)
        tgt = json.loads((WORK.parents[1] / "data" / "source" / "target_laws.json").read_text(encoding="utf-8"))
        by_name = {r["law_name"]: r["uid"] for r in laws}
        alias = {}  # (약칭, 법령) → 출처. 공식 약칭(API)이 팀 목록 짧은 이름보다 우선
        for g, d in tgt.items():
            if not g.startswith("_"):
                for short, (full, _) in d.items():
                    if full in by_name and short != full:
                        alias[(short, by_name[full])] = "team"
        for r in laws:
            if r["abbrev"]:
                alias[(r["abbrev"], r["uid"])] = "official"
        n = copy(c, "law_aliases", ["alias", "law_uid", "alias_source"], sorted((a, u, s) for (a, u), s in alias.items()))
        print("  law_aliases", n)
        n = copy(c, "articles", ["article_uid", "law_uid", "seq", "article_no", "article_dash", "article_sub", "article_label",
                                 "title", "chapter", "raw_text", "text", "embed_body", "article_effective_date", "searchable",
                                 "skip_reason", "is_deleted", "has_image", "has_table", "has_delegation", "is_reference_list",
                                 "parse_method", "n_chars", "n_tokens"],
                 ([a["article_uid"], a["law_uid"], i(a["seq"]), i(a["article_no"]), i(a["article_dash"]), i(a["article_sub"]),
                   a["article_label"], a["title"] or None, a["chapter"] or None, a["raw_text"], a["text"], a["embed_body"],
                   d8(a["article_effective_date"]), b(a["searchable"]), a["skip_reason"] or None, b(a["is_deleted"]),
                   b(a["has_image"]), b(a["has_table"]), b(a["has_delegation"]), b(a["is_reference_list"]),
                   a["parse_method"], i(a["n_chars"]), i(a["n_tokens"])] for a in arts))
        print("  articles", n)
        n = copy(c, "article_lines", ["line_uid", "article_uid", "seq", "level", "marker", "para_no", "item_no", "sub_no",
                                      "parent_line_uid", "text", "embed_text", "is_deleted", "not_in_force"],
                 ([ln["line_uid"], ln["article_uid"], i(ln["seq"]), ln["level"], ln["marker"] or None, ln["para_no"] or None,
                   ln["item_no"] or None, ln["sub_no"] or None, ln["parent_line_uid"] or None, ln["text"], ln["embed_text"],
                   b(ln["is_deleted"]), b(ln["not_in_force"])] for ln in lines))
        print("  article_lines", n)
        t1 = time.time()
        n = copy(c, "chunks", ["chunk_id", "article_uid", "law_uid", "chunk_no", "n_chunks", "item_from", "item_to", "cite",
                               "text", "embed_text", "n_chars", "n_tokens", "law_name", "law_level", "scope_group",
                               "article_label", "title", "effective_date", "has_ctx", "by_sentence", "has_image", "has_table",
                               "has_delegation", "is_reference_list", "embed_model", "input_hash", "embedding"],
                 ([ch["chunk_id"], ch["article_uid"], ch["law_uid"], i(ch["chunk_no"]), i(ch["n_chunks"]),
                   ch["item_from"] or None, ch["item_to"] or None, ch["cite"], ch["text"], ch["embed_text"],
                   i(ch["n_chars"]), i(ch["n_tokens"]), ch["law_name"], ch["law_level"], ch["scope"],
                   ch["article_label"], ch["title"] or None, d8(ch["effective_date"]), b(ch["has_ctx"]),
                   b(ch["by_sentence"]), b(ch["has_image"]), b(ch["has_table"]), b(ch["has_delegation"]),
                   b(ch["is_reference_list"]), ch["embed_model"], ch["input_hash"],
                   "[" + ",".join(f"{x:.7g}" for x in v) + "]"] for ch, v in zip(chunks, vecs)))
        print(f"  chunks {n} ({time.time() - t1:.0f}초)")
        n = copy(c, "article_links", ["src_uid", "dst_uid", "kind"], ([r["src_uid"], r["dst_uid"], r["kind"]] for r in links))
        print("  article_links", n)
        t1 = time.time()
        # 도커 컨테이너의 공유 메모리(/dev/shm)는 기본 64MB 라 병렬 색인 빌드가 실패함 → 병렬 없이, 일반 메모리로
        c.execute("SET max_parallel_maintenance_workers = 0")
        c.execute("SET maintenance_work_mem = '512MB'")
        c.execute(INDEX)
        c.execute("ANALYZE")
        print(f"  색인 + ANALYZE {time.time() - t1:.0f}초")
        c.execute("CREATE TABLE lawrag.load_log (loaded_at timestamptz DEFAULT now(), rule text, vectors_meta jsonb, counts jsonb)")
        counts = {t: c.execute(f"SELECT count(*) FROM lawrag.{t}").fetchone()[0]
                  for t in ("laws", "law_aliases", "articles", "article_lines", "chunks", "article_links")}
        c.execute("INSERT INTO lawrag.load_log (rule, vectors_meta, counts) VALUES (%s, %s, %s)",
                  (RULE_VERSION, str(z["meta"]), json.dumps(counts, ensure_ascii=False)))
        # 검색 전용 계정: 이미 만든 테이블에도 SELECT (ALTER DEFAULT PRIVILEGES 는 앞으로 만들 테이블에만 적용됨)
        c.execute("GRANT SELECT ON ALL TABLES IN SCHEMA lawrag TO law_reader")
    print(f"끝 {time.time() - t0:.0f}초 · {counts}")


if __name__ == "__main__":
    main()
