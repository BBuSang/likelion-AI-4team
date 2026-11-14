#!/bin/bash
# 컨테이너 최초 실행 시 한 번만 실행됩니다. (다시 만들려면: docker compose down -v 후 up)
# 실전 2 init/01_schema.sh 를 바탕으로, 최종프로젝트 스키마를 더했습니다 (worklog 6-1, 3-7).
# app   : 회원·직책·권한 (Spring, app_user)
# rag   : 가명처리된 청크 + 토큰 + 임베딩, 추정 인물(토큰만)  (AI 서버, rag_user)
# docs  : 일반 문서 RAG 3개 (공개 자료, 개인정보 없음)          (docs_user, 테이블은 6-5-3 정한 뒤)
# vault : 토큰 ↔ 원문, 검수 대기 원문, 검수·복원 기록            (AI 서버만, vault_user)
set -euo pipefail
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<SQL
CREATE EXTENSION IF NOT EXISTS vector;

CREATE ROLE app_user   LOGIN PASSWORD '${APP_PASSWORD}';
CREATE ROLE rag_user   LOGIN PASSWORD '${RAG_PASSWORD}';
CREATE ROLE docs_user  LOGIN PASSWORD '${DOCS_PASSWORD}';
CREATE ROLE vault_user LOGIN PASSWORD '${VAULT_PASSWORD}';
GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO app_user, rag_user, docs_user, vault_user;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;

-- ---------------------------------------------------------------- app (Spring JPA 가 테이블을 만듦)
CREATE SCHEMA app AUTHORIZATION app_user;

-- ---------------------------------------------------------------- docs (테이블은 나중에)
CREATE SCHEMA docs AUTHORIZATION docs_user;

-- ---------------------------------------------------------------- rag (실전 2 와 같음)
CREATE SCHEMA rag AUTHORIZATION rag_user;

CREATE TABLE rag.chunks (
    chunk_id     TEXT PRIMARY KEY,               -- 예: J3-H0001#0
    doc_id       TEXT NOT NULL,
    chunk_idx    INT  NOT NULL,
    doc_type     TEXT,
    masked_text  TEXT NOT NULL,                  -- 원문 저장 금지
    tokens       TEXT[] NOT NULL DEFAULT '{}',
    embedding    vector(${EMBED_DIM}) NOT NULL,
    -- (최종) 문서 권한 (worklog 3-4-1, 10장). 검색 SQL 이 이 값으로 먼저 거름
    owner_emp_id TEXT,                           -- 담당 직원 (모르면 NULL)
    team         TEXT,                           -- 담당 팀 (모르면 NULL)
    visibility   TEXT NOT NULL DEFAULT 'restricted', -- public / team / restricted(감사만)
    created_at   TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX chunks_acl            ON rag.chunks (visibility, team);
CREATE INDEX chunks_owner          ON rag.chunks (owner_emp_id);
CREATE INDEX chunks_tokens_gin     ON rag.chunks USING GIN (tokens);
CREATE INDEX chunks_embedding_hnsw ON rag.chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_doc_id         ON rag.chunks (doc_id);

-- 추정 인물. 실명 없이 토큰만
CREATE TABLE rag.persons (
    pid           TEXT PRIMARY KEY,
    person_token  TEXT NOT NULL UNIQUE,
    name_token    TEXT NOT NULL,
    given_token   TEXT NOT NULL,
    role          TEXT,                          -- customer / staff / external / NULL
    affiliations  TEXT[] NOT NULL DEFAULT '{}',
    status        TEXT NOT NULL,                 -- resolved / unresolved
    mentions      INT  NOT NULL DEFAULT 0
);
CREATE INDEX persons_name_token  ON rag.persons (name_token);
CREATE INDEX persons_given_token ON rag.persons (given_token);

CREATE TABLE rag.person_attrs (
    pid    TEXT NOT NULL REFERENCES rag.persons(pid) ON DELETE CASCADE,
    label  TEXT NOT NULL,
    token  TEXT NOT NULL,
    PRIMARY KEY (pid, token)
);
CREATE INDEX person_attrs_token ON rag.person_attrs (token);

-- (최종) 문서 메타데이터 (문서 단위, 청크에도 같은 값을 복사해 둠)
CREATE TABLE rag.documents (
    doc_id        TEXT PRIMARY KEY,
    doc_type      TEXT,
    owner_emp_id  TEXT,
    team          TEXT,
    visibility    TEXT NOT NULL,
    basis         JSONB NOT NULL DEFAULT '{}',   -- 담당자·팀을 어떻게 정했나 (worklog 10-2)
    created_at    TIMESTAMPTZ DEFAULT now()
);

-- (최종) 직원 목록 — 이름 없이 (이름은 vault.employees). 로그인 계정이 emp_id 로 연결됨
CREATE TABLE rag.employees (
    emp_id      TEXT PRIMARY KEY,
    team        TEXT,
    position    TEXT,
    role        TEXT NOT NULL,                   -- STAFF / LEAD / AUDITOR
    n_docs      INT NOT NULL DEFAULT 0
);

ALTER TABLE rag.chunks OWNER TO rag_user;
ALTER TABLE rag.documents OWNER TO rag_user;
ALTER TABLE rag.employees OWNER TO rag_user;
ALTER TABLE rag.persons OWNER TO rag_user;
ALTER TABLE rag.person_attrs OWNER TO rag_user;

-- ---------------------------------------------------------------- vault
CREATE SCHEMA vault AUTHORIZATION vault_user;

CREATE TABLE vault.token_map (
    token       TEXT PRIMARY KEY,
    label       TEXT NOT NULL,
    normalized  TEXT NOT NULL,
    originals   TEXT[] NOT NULL DEFAULT '{}',
    pid         TEXT,
    display     TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE vault.person_mentions (
    doc_id      TEXT NOT NULL,
    start_pos   INT  NOT NULL,
    end_pos     INT  NOT NULL,
    token       TEXT NOT NULL,
    pid         TEXT NOT NULL,
    basis       TEXT NOT NULL,                   -- anchor/aff/... + 최종: anchor_link/aff_link/unique_link/new
    role        TEXT,
    PRIMARY KEY (doc_id, start_pos)
);

-- (최종) 추정 인물의 묶음 key. 같은 key → 같은 사람 토큰. key 에 원문 단서가 들어갈 수 있어 vault 쪽
CREATE TABLE vault.person_keys (
    pid         TEXT PRIMARY KEY,
    name        TEXT NOT NULL,                   -- 정규화한 이름 (원문)
    person_key  TEXT,                            -- NULL = 이름만으로 만든 미확정 인물
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE vault.restore_log (
    id          BIGSERIAL PRIMARY KEY,
    actor       TEXT NOT NULL,
    purpose     TEXT,
    tokens      TEXT[] NOT NULL,
    role        TEXT,                            -- (최종) 복원한 사람 역할
    doc_ids     TEXT[] NOT NULL DEFAULT '{}',    -- (최종) 어느 문서의 토큰을 복원했나
    denied      TEXT[] NOT NULL DEFAULT '{}',    -- (최종) 권한이 없어 토큰으로 남긴 것
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- (최종) 직원 목록 — 이름 포함 (문서에서 추론, worklog 10장)
CREATE TABLE vault.employees (
    emp_id      TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    team        TEXT,
    position    TEXT,
    role        TEXT NOT NULL,
    n_mentions  INT NOT NULL DEFAULT 0,
    n_docs      INT NOT NULL DEFAULT 0
);

-- (최종) 적재: 업로드 1번 = job, 문서 1건 = staging_doc (worklog 3-6, 3-7)
CREATE TABLE vault.ingest_job (
    id          BIGSERIAL PRIMARY KEY,
    uploader    TEXT NOT NULL,                   -- 올린 사람 (Spring 이 넘긴 이메일)
    source      TEXT NOT NULL,                   -- jsonl / text
    filename    TEXT,
    total       INT  NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'processing', -- processing(탐지 중) / ready / failed
    error       TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE vault.staging_doc (
    id            BIGSERIAL PRIMARY KEY,
    job_id        BIGINT NOT NULL REFERENCES vault.ingest_job(id) ON DELETE CASCADE,
    seq           INT  NOT NULL,                 -- 업로드 안 순서
    ext_id        TEXT,                          -- 파일에 있던 id (예: H0001)
    doc_id        TEXT NOT NULL UNIQUE,          -- 저장할 때 쓰는 문서 id (J{job}-{ext_id 또는 seq})
    doc_type      TEXT,
    text          TEXT NOT NULL,                 -- 원문 (검수 대기)
    meta          JSONB NOT NULL DEFAULT '{}',
    detected      JSONB NOT NULL,                -- 탐지기 결과 그대로 [{label,start,end}]
    gold          JSONB,                         -- 파일에 정답 라벨이 있으면 (자동 승인 비교용만)
    review        JSONB,                         -- 검수 판단 {needs_review, flags, hints} (값 없이 위치·점수만, worklog 29장)
    final_spans   JSONB,                         -- 승인된 구간
    status        TEXT NOT NULL DEFAULT 'pending', -- pending / stored / rejected / failed
    approved_by   TEXT,                          -- 검수자 이메일 / 'gold'(정답과 같음) / 'auto'(검수 신호 없음)
    reviewed_at   TIMESTAMPTZ,
    reject_reason TEXT,
    error         TEXT,
    created_at    TIMESTAMPTZ DEFAULT now(),
    UNIQUE (job_id, seq)
);
CREATE INDEX staging_doc_job_status ON vault.staging_doc (job_id, status);

-- 검수 기록: 값은 남기지 않고 라벨·위치·개수만
CREATE TABLE vault.review_log (
    id              BIGSERIAL PRIMARY KEY,
    staging_doc_id  BIGINT NOT NULL REFERENCES vault.staging_doc(id) ON DELETE CASCADE,
    actor           TEXT NOT NULL,
    action          TEXT NOT NULL,               -- view / approve / auto_approve / reject / store_failed
    added           INT NOT NULL DEFAULT 0,      -- 검수자가 추가한 구간 수 (= 탐지기가 놓친 수)
    removed         INT NOT NULL DEFAULT 0,      -- 지운 구간 수 (= 오탐)
    relabeled       INT NOT NULL DEFAULT 0,
    detail          JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- (최종) 저장된 문서를 관리자가 지운 기록 (worklog 15장). 값 없이 개수만
CREATE TABLE vault.deletion_log (
    id          BIGSERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    actor       TEXT NOT NULL,
    reason      TEXT NOT NULL,
    counts      JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE vault.token_map OWNER TO vault_user;
ALTER TABLE vault.person_mentions OWNER TO vault_user;
ALTER TABLE vault.person_keys OWNER TO vault_user;
ALTER TABLE vault.restore_log OWNER TO vault_user;
ALTER TABLE vault.employees OWNER TO vault_user;
ALTER TABLE vault.ingest_job OWNER TO vault_user;
ALTER TABLE vault.staging_doc OWNER TO vault_user;
ALTER TABLE vault.review_log OWNER TO vault_user;
ALTER SEQUENCE vault.ingest_job_id_seq OWNER TO vault_user;
ALTER SEQUENCE vault.staging_doc_id_seq OWNER TO vault_user;
ALTER SEQUENCE vault.review_log_id_seq OWNER TO vault_user;
ALTER SEQUENCE vault.restore_log_id_seq OWNER TO vault_user;
ALTER TABLE vault.deletion_log OWNER TO vault_user;
ALTER SEQUENCE vault.deletion_log_id_seq OWNER TO vault_user;

-- 서로의 스키마에는 접근 불가
REVOKE ALL ON SCHEMA vault FROM PUBLIC, app_user, rag_user, docs_user;
REVOKE ALL ON SCHEMA rag   FROM PUBLIC, app_user, docs_user, vault_user;
REVOKE ALL ON SCHEMA app   FROM PUBLIC, rag_user, docs_user, vault_user;
REVOKE ALL ON SCHEMA docs  FROM PUBLIC, app_user, vault_user;
-- 일반 문서는 AI 서버(rag_user)가 읽기만 (테이블이 생기면 GRANT SELECT 추가)
GRANT USAGE ON SCHEMA docs TO rag_user;
SQL
