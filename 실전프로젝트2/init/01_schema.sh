#!/bin/bash
# 컨테이너 최초 실행 시 한 번만 실행됩니다. (다시 만들려면: docker compose down -v 후 up)
# rag   : 마스킹된 청크 + 토큰 목록 + 임베딩, 추정 인물(토큰만)   (rag_user 만 접근)
# vault : 토큰 ↔ 원문 매핑 테이블, 이름 언급별 판단 근거           (vault_user 만 접근)
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<SQL
CREATE EXTENSION IF NOT EXISTS vector;

CREATE ROLE rag_user   LOGIN PASSWORD '${RAG_PASSWORD}';
CREATE ROLE vault_user LOGIN PASSWORD '${VAULT_PASSWORD}';
GRANT CONNECT ON DATABASE ${POSTGRES_DB} TO rag_user, vault_user;

-- 검색용 스키마: 가명처리된 텍스트만 저장
CREATE SCHEMA rag AUTHORIZATION rag_user;
CREATE TABLE rag.chunks (
    chunk_id     TEXT PRIMARY KEY,               -- 예: H0001#0
    doc_id       TEXT NOT NULL,
    chunk_idx    INT  NOT NULL,
    doc_type     TEXT,
    masked_text  TEXT NOT NULL,                  -- 원문 저장 금지
    tokens       TEXT[] NOT NULL DEFAULT '{}',   -- 청크에 들어 있는 토큰 목록
    embedding    vector(${EMBED_DIM}) NOT NULL,
    created_at   TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX chunks_tokens_gin     ON rag.chunks USING GIN (tokens);
CREATE INDEX chunks_embedding_hnsw ON rag.chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_doc_id         ON rag.chunks (doc_id);
ALTER TABLE rag.chunks OWNER TO rag_user;

-- 추정 인물 (팀 추가). 실명 없이 토큰만 — 검색이 vault 없이 동명이인을 고를 수 있게
-- pid 는 정답 인물 ID 가 아니라 person_resolve.py 가 원문 단서(닻·소속·역할)로 추정한 사람
CREATE TABLE rag.persons (
    pid           TEXT PRIMARY KEY,              -- 예: P_1a2b3c4d
    person_token  TEXT NOT NULL UNIQUE,          -- 본문에 들어가는 사람 토큰 <PERSON_…>
    name_token    TEXT NOT NULL,                 -- 이름 토큰 <NAME_…> (동명이인이 공유)
    given_token   TEXT NOT NULL,                 -- 이름만 토큰 <GIVEN_…>
    role          TEXT,                          -- customer / staff / external / NULL
    affiliations  TEXT[] NOT NULL DEFAULT '{}',  -- 소속 조직명 (개인정보 아님)
    status        TEXT NOT NULL,                 -- resolved(단서로 확정) / unresolved(이름만, 누구인지 모름)
    mentions      INT  NOT NULL DEFAULT 0
);
CREATE INDEX persons_name_token  ON rag.persons (name_token);
CREATE INDEX persons_given_token ON rag.persons (given_token);
ALTER TABLE rag.persons OWNER TO rag_user;

-- 추정 인물별 가명처리된 개인정보 (전화·이메일·계좌 등의 토큰)
CREATE TABLE rag.person_attrs (
    pid    TEXT NOT NULL REFERENCES rag.persons(pid) ON DELETE CASCADE,
    label  TEXT NOT NULL,
    token  TEXT NOT NULL,
    PRIMARY KEY (pid, token)
);
CREATE INDEX person_attrs_token ON rag.person_attrs (token);
ALTER TABLE rag.person_attrs OWNER TO rag_user;

-- 매핑용 스키마: 토큰 ↔ 원문
CREATE SCHEMA vault AUTHORIZATION vault_user;
CREATE TABLE vault.token_map (
    token       TEXT PRIMARY KEY,                -- 예: <PERSON_03f0d420>
    label       TEXT NOT NULL,
    normalized  TEXT NOT NULL,                   -- 토큰을 만든 정규화 값
    originals   TEXT[] NOT NULL DEFAULT '{}',    -- 실제로 나온 표기들 (예: 윤 선영, 윤선영)
    pid         TEXT,                            -- (팀 추가) 사람 토큰이면 추정 인물 ID
    display     TEXT,                            -- (팀 추가) 복원할 때 보여줄 대표 표기
    created_at  TIMESTAMPTZ DEFAULT now()
);
ALTER TABLE vault.token_map OWNER TO vault_user;

-- (팀 추가) 이름 언급별 판단 근거. 원문 위치가 들어가므로 vault 쪽. 새 문서가 들어와 동명이인이 생기면
-- basis 가 unique/role/same_doc 인 언급만 다시 판단하면 된다 (anchor 는 유지)
CREATE TABLE vault.person_mentions (
    doc_id      TEXT NOT NULL,
    start_pos   INT  NOT NULL,
    end_pos     INT  NOT NULL,
    token       TEXT NOT NULL,
    pid         TEXT NOT NULL,
    basis       TEXT NOT NULL,                   -- anchor/aff/same_doc/doc_anchor/role/role_new/unique/unresolved/name_only
    role        TEXT,
    PRIMARY KEY (doc_id, start_pos)
);
ALTER TABLE vault.person_mentions OWNER TO vault_user;

-- (팀 추가) 실명 복원 기록. 권한 확인은 종합에서 붙이고, 지금은 누가·언제·무엇을 복원했는지만 남긴다
CREATE TABLE vault.restore_log (
    id          BIGSERIAL PRIMARY KEY,
    actor       TEXT NOT NULL,                   -- 복원한 사용자 (데모는 'demo')
    purpose     TEXT,                            -- 질문 ID 등
    tokens      TEXT[] NOT NULL,                 -- 복원한 토큰들 (값은 남기지 않음)
    created_at  TIMESTAMPTZ DEFAULT now()
);
ALTER TABLE vault.restore_log OWNER TO vault_user;

-- 서로의 스키마에는 접근 불가
REVOKE ALL ON SCHEMA vault FROM PUBLIC, rag_user;
REVOKE ALL ON SCHEMA rag   FROM PUBLIC, vault_user;
SQL
