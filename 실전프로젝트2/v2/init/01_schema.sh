#!/bin/bash
# v2 스키마 (plan_v2.md 1장 ERD). 컨테이너 최초 실행 시 한 번만 실행됩니다.
# 다시 만들려면: docker compose --env-file ../.env down -v 후 up
#
# rag   : 가명처리된 텍스트와 비개인정보 속성만   (쓰기 ingest_user, 읽기 rag_user)
# vault : 토큰 ↔ 원문, 원문 기준 위치            (vault_user 만)
# ops   : 질의·검색·답변 기록 (가명 텍스트만)       (rag_user)
# rag ↔ vault 는 FK 없이 token, mention_id 로만 논리 연결 (스키마 분리 유지)
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<SQL
CREATE EXTENSION IF NOT EXISTS vector;

CREATE ROLE rag_user    LOGIN PASSWORD '${RAG_PASSWORD}';
CREATE ROLE ingest_user LOGIN PASSWORD '${INGEST_PASSWORD}';
CREATE ROLE vault_user  LOGIN PASSWORD '${VAULT_PASSWORD}';
GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO rag_user, ingest_user, vault_user;

-- ───────────────────────── rag: 검색용 (원문 저장 금지) ─────────────────────────
CREATE SCHEMA rag AUTHORIZATION ingest_user;

CREATE TABLE rag.documents (
    doc_id        TEXT PRIMARY KEY,                -- 예: H0001
    split         TEXT,                            -- dev | test | holdout
    doc_type      TEXT,                            -- 코퍼스 메타데이터 (VOC요약 등)
    title_masked  TEXT,                            -- 첫 줄 (가명처리)
    doc_date      DATE,                            -- 본문에서 추출한 날짜
    masked_text   TEXT NOT NULL,
    char_len      INT,                             -- 원문 길이
    pipeline_ver  TEXT NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE rag.chunks (
    chunk_id     TEXT PRIMARY KEY,                 -- 예: H0023#d, H0023#r2
    doc_id       TEXT NOT NULL REFERENCES rag.documents(doc_id) ON DELETE CASCADE,
    parent_id    TEXT REFERENCES rag.chunks(chunk_id) ON DELETE CASCADE,  -- row 청크의 doc 청크
    kind         TEXT NOT NULL CHECK (kind IN ('doc', 'section', 'row')),
    line_no      INT,                              -- row 청크의 줄 번호 (0부터)
    header       TEXT,                             -- [doc_type] 제목 (임베딩 앞에 붙임)
    masked_text  TEXT NOT NULL,
    tokens       TEXT[] NOT NULL DEFAULT '{}',
    embedding    vector(${EMBED_DIM}) NOT NULL
);

CREATE TABLE rag.tokens (
    token     TEXT PRIMARY KEY,                    -- 예: <PERSON_03f0d420>
    label     TEXT NOT NULL,
    doc_freq  INT  NOT NULL DEFAULT 0,             -- 토큰이 나온 문서 수 (IDF)
    key_ver   INT  NOT NULL DEFAULT 1              -- HMAC 키 버전
);

-- 같은 이름 토큰 안에서 가명 토큰 공기로 나눈 인물 단위 (plan_v2 S6, 선택 확장)
CREATE TABLE rag.entities (
    entity_id     TEXT PRIMARY KEY,                -- 예: E0001
    person_token  TEXT NOT NULL REFERENCES rag.tokens(token),
    role          TEXT,
    org_unit      TEXT,
    confidence    REAL,
    method        TEXT CHECK (method IN ('cooccur', 'role', 'single', 'llm'))  -- llm = 가명 본문 LLM 판단으로 합침
);

CREATE TABLE rag.mentions (
    mention_id  BIGINT PRIMARY KEY,                -- vault.mention_raw 와 같은 번호
    doc_id      TEXT NOT NULL REFERENCES rag.documents(doc_id) ON DELETE CASCADE,
    chunk_id    TEXT REFERENCES rag.chunks(chunk_id) ON DELETE CASCADE,  -- 가장 작은 청크 (row 우선)
    token       TEXT NOT NULL REFERENCES rag.tokens(token),
    label       TEXT NOT NULL,
    m_start     INT  NOT NULL,                     -- 가명처리 본문 기준 위치
    m_end       INT  NOT NULL,
    line_no     INT,                               -- 문서 안 줄 번호 (줄 청크 없이 같은 줄 판단)
    form        TEXT CHECK (form IN ('full', 'spaced', 'given')),        -- person만
    role        TEXT CHECK (role IN ('customer', 'staff', 'agent', 'external')),  -- 문서 안 같은 토큰끼리 공유
    own_role    TEXT,                              -- 그 언급 자체의 단서로 정한 역할 (문서 안 동명이인 확인용)
    title       TEXT,                              -- 고객·과장·대리 등 호칭·직함
    org_unit    TEXT,                              -- 정보보호팀 등 (비개인정보)
    entity_id   TEXT REFERENCES rag.entities(entity_id) ON DELETE SET NULL
);

CREATE TABLE rag.entity_links (
    entity_id       TEXT NOT NULL REFERENCES rag.entities(entity_id) ON DELETE CASCADE,
    token           TEXT NOT NULL REFERENCES rag.tokens(token),           -- 같은 줄의 전화·주민·이메일 등
    evidence_chunk  TEXT NOT NULL REFERENCES rag.chunks(chunk_id) ON DELETE CASCADE,
    PRIMARY KEY (entity_id, token, evidence_chunk)
);

-- 주소 일반화: 시·도 / 시·군·구까지만 (읍·면·동 이하 저장 안 함)
CREATE TABLE rag.address_regions (
    token    TEXT PRIMARY KEY REFERENCES rag.tokens(token),
    sido     TEXT,
    sigungu  TEXT
);

-- 비개인정보 속성: 처리 상태, 접수 번호(VOC-1234), 금액, 기한 등
CREATE TABLE rag.doc_fields (
    field_id  BIGSERIAL PRIMARY KEY,
    doc_id    TEXT NOT NULL REFERENCES rag.documents(doc_id) ON DELETE CASCADE,
    chunk_id  TEXT REFERENCES rag.chunks(chunk_id) ON DELETE CASCADE,
    line_no   INT,
    field     TEXT NOT NULL,
    value     TEXT NOT NULL
);

CREATE INDEX chunks_doc_id          ON rag.chunks (doc_id);
CREATE INDEX chunks_kind            ON rag.chunks (kind);
CREATE INDEX chunks_tokens_gin      ON rag.chunks USING GIN (tokens);
CREATE INDEX chunks_embedding_hnsw  ON rag.chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX documents_doc_type     ON rag.documents (doc_type);
CREATE INDEX mentions_token         ON rag.mentions (token);
CREATE INDEX mentions_doc_id        ON rag.mentions (doc_id);
CREATE INDEX mentions_entity_id     ON rag.mentions (entity_id);
CREATE INDEX entity_links_token     ON rag.entity_links (token);
CREATE INDEX address_regions_region ON rag.address_regions (sido, sigungu);
CREATE INDEX doc_fields_field_value ON rag.doc_fields (field, value);

ALTER TABLE rag.documents       OWNER TO ingest_user;
ALTER TABLE rag.chunks          OWNER TO ingest_user;
ALTER TABLE rag.tokens          OWNER TO ingest_user;
ALTER TABLE rag.entities        OWNER TO ingest_user;
ALTER TABLE rag.mentions        OWNER TO ingest_user;
ALTER TABLE rag.entity_links    OWNER TO ingest_user;
ALTER TABLE rag.address_regions OWNER TO ingest_user;
ALTER TABLE rag.doc_fields      OWNER TO ingest_user;

-- 챗봇(rag_user)은 읽기만
GRANT USAGE ON SCHEMA rag TO rag_user;
GRANT SELECT ON ALL TABLES IN SCHEMA rag TO rag_user;
ALTER DEFAULT PRIVILEGES FOR ROLE ingest_user IN SCHEMA rag GRANT SELECT ON TABLES TO rag_user;

-- ───────────────────────── vault: 매핑용 (원문) ─────────────────────────
CREATE SCHEMA vault AUTHORIZATION vault_user;

CREATE TABLE vault.token_map (
    token       TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    normalized  TEXT NOT NULL,                     -- 토큰을 만든 정규화 값
    key_ver     INT  NOT NULL DEFAULT 1,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE vault.token_surface (
    token      TEXT NOT NULL REFERENCES vault.token_map(token) ON DELETE CASCADE,
    surface    TEXT NOT NULL,                      -- 실제 표기 (예: 윤 선영)
    doc_count  INT  NOT NULL DEFAULT 0,
    PRIMARY KEY (token, surface)
);

CREATE TABLE vault.mention_raw (
    mention_id  BIGINT PRIMARY KEY,                -- rag.mentions 와 같은 번호
    doc_id      TEXT NOT NULL,
    o_start     INT  NOT NULL,                     -- 원문 기준 위치 (채점 형식)
    o_end       INT  NOT NULL,
    raw_text    TEXT NOT NULL
);
CREATE INDEX mention_raw_doc_id ON vault.mention_raw (doc_id);

ALTER TABLE vault.token_map     OWNER TO vault_user;
ALTER TABLE vault.token_surface OWNER TO vault_user;
ALTER TABLE vault.mention_raw   OWNER TO vault_user;

-- ───────────────────────── ops: 질의·검색·답변 기록 (가명 텍스트만) ─────────────────────────
CREATE SCHEMA ops AUTHORIZATION rag_user;

CREATE TABLE ops.query_log (
    query_id      TEXT PRIMARY KEY,
    masked_query  TEXT NOT NULL,                   -- 원문 질의 저장 금지
    query_tokens  TEXT[] NOT NULL DEFAULT '{}',
    parsed        JSONB,                           -- doc_type·지역·상태·역할 의도
    mode          TEXT,
    k             INT,
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- chunk_id 는 FK 없음: 재적재해도 기록은 남김
CREATE TABLE ops.retrieval_log (
    query_id    TEXT NOT NULL REFERENCES ops.query_log(query_id) ON DELETE CASCADE,
    rank        INT  NOT NULL,
    chunk_id    TEXT NOT NULL,
    doc_id      TEXT NOT NULL,
    s_token     REAL,
    s_semantic  REAL,
    s_final     REAL,
    PRIMARY KEY (query_id, rank)
);

CREATE TABLE ops.answer_log (
    query_id        TEXT PRIMARY KEY REFERENCES ops.query_log(query_id) ON DELETE CASCADE,
    answer_masked   TEXT,
    cited_docs      TEXT[] NOT NULL DEFAULT '{}',
    unknown_tokens  TEXT[] NOT NULL DEFAULT '{}',  -- 지어낸 토큰
    row_mismatch    INT,                           -- 줄 대응 점검 실패 수
    llm_sec         REAL,
    reasoning       TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE ops.query_log     OWNER TO rag_user;
ALTER TABLE ops.retrieval_log OWNER TO rag_user;
ALTER TABLE ops.answer_log    OWNER TO rag_user;

-- ───────────────────────── 접근 차단 ─────────────────────────
REVOKE ALL ON SCHEMA vault FROM PUBLIC, rag_user, ingest_user;
REVOKE ALL ON SCHEMA rag   FROM PUBLIC, vault_user;
REVOKE ALL ON SCHEMA ops   FROM PUBLIC, vault_user, ingest_user;
SQL
