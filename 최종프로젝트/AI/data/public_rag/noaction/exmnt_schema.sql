-- 현장건의 과제: 기존 회신(interp.cases/case_groups/chunks)과 분리한 별도 테이블 (2026-11-11 사용자 결정, worklog 85번)
-- 기존 rag.search()와 설정 지문은 interp.chunks만 보므로 이 테이블을 추가해도 바뀌지 않는다.

CREATE TABLE interp.exmnt_groups (
    group_id bigint PRIMARY KEY,                         -- 1,000,000부터 (기존 묶음 번호와 겹치지 않게)
    group_type text NOT NULL CHECK (group_type IN ('single', 'same_title_answer')),
    representative_case_key text NOT NULL,
    n_cases int NOT NULL
);

CREATE TABLE interp.exmnt_cases (
    case_key text PRIMARY KEY,                           -- 'F' + 포털 checkplaceNo
    idx bigint NOT NULL,
    title text NOT NULL,
    reply_date date,
    category text,                                       -- 과제분류 전체 (예: 4. 영업·업무규제 >구속성 예금·보험등 금지)
    category_top text,
    dept text,
    visit text,
    answer_type text,                                    -- 검토결과: 불수용 / 수용 / 일부수용 / 기조치(수용)
    question text,                                       -- 건의내용 (개인정보 가림)
    answer text,                                         -- 검토 사유 (개인정보 가림)
    plan text,                                           -- 향후계획 (원본 보관용, 청크에는 넣지 않음)
    close_reason text,
    closed date,
    url text NOT NULL,
    changed_after boolean NOT NULL,                      -- 수용·일부수용: 이후 제도가 바뀌었을 수 있음
    already_in_place boolean NOT NULL,                   -- 기조치: 회신 당시 이미 운영 중
    articles text[] NOT NULL DEFAULT '{}',
    norm_kinds text[] NOT NULL DEFAULT '{}',
    indexable boolean NOT NULL,
    skip_reason text,
    group_id bigint REFERENCES interp.exmnt_groups,
    CHECK (indexable = (skip_reason IS NULL))
);

CREATE TABLE interp.exmnt_chunks (
    chunk_id bigserial PRIMARY KEY,
    case_key text NOT NULL REFERENCES interp.exmnt_cases,
    group_id bigint NOT NULL REFERENCES interp.exmnt_groups,
    chunk_type text NOT NULL CHECK (chunk_type IN ('question_card', 'judgment')),
    chunk_no int NOT NULL,
    text text NOT NULL,
    embed_text text NOT NULL,
    bm25_tokens text NOT NULL,
    category text,
    reply_date date,
    answer_type text,
    embed_model text NOT NULL,
    embedding vector(1024) NOT NULL,
    UNIQUE (case_key, chunk_type, chunk_no)
);

CREATE INDEX ON interp.exmnt_cases (group_id);
CREATE INDEX ON interp.exmnt_chunks (case_key);
CREATE INDEX ON interp.exmnt_chunks (group_id);
CREATE INDEX exmnt_chunks_embedding_hnsw ON interp.exmnt_chunks USING hnsw (embedding vector_cosine_ops);
