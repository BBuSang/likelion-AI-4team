-- 금융당국 회신 선례 RAG 스키마. {EMBED_DIM}은 pipeline/s14_load.py가 .env 값으로 바꾼다.
CREATE SCHEMA interp;

CREATE TABLE interp.case_groups (
    group_id bigint PRIMARY KEY,
    group_type text NOT NULL CHECK (group_type IN ('single', 'same_body', 'same_question')),
    representative_case_key text NOT NULL,
    n_cases int NOT NULL CHECK (n_cases > 0)
);

CREATE TABLE interp.cases (
    case_key text PRIMARY KEY,
    gubun text NOT NULL,
    idx int NOT NULL,
    serial_no text,
    title text,
    title_source text,
    reply_title text,
    doc_class text NOT NULL CHECK (doc_class IN ('general', 'rule_review_2016', 'extension')),
    group_id bigint REFERENCES interp.case_groups,
    body_suspect boolean NOT NULL DEFAULT false,
    category text,
    category_inferred text,
    category_final text,
    category_final_source text,
    reg_date date,
    reply_date date,
    reply_date_source text,
    status text,
    answer_type text,
    answer_type_raw text,
    registrant text,
    question text,
    answer text,
    reason text,
    body_source text,
    body_chars int,
    indexable boolean NOT NULL,
    skip_reason text,
    url text,
    time_check text,
    time_check_counts jsonb,
    time_checked_on date
);

CREATE TABLE interp.case_depts (
    case_key text NOT NULL REFERENCES interp.cases,
    seq int NOT NULL,
    dept_name text NOT NULL,
    is_org boolean NOT NULL,
    PRIMARY KEY (case_key, seq)
);

CREATE TABLE interp.attachments (
    att_id bigserial PRIMARY KEY,
    case_key text NOT NULL REFERENCES interp.cases,
    org_file_name text,
    ext text,
    text_used boolean NOT NULL
);

CREATE TABLE interp.laws (
    law_id int PRIMARY KEY,
    law_name text NOT NULL UNIQUE,
    law_kind text NOT NULL CHECK (law_kind IN ('법률', '시행령', '시행규칙', '행정규칙')),
    source_type text NOT NULL CHECK (source_type IN ('law', 'admrul', 'association', 'unknown')),
    category text,
    is_current boolean
);

CREATE TABLE interp.law_aliases (
    alias text PRIMARY KEY,
    law_id int NOT NULL REFERENCES interp.laws,
    alias_source text NOT NULL CHECK (alias_source IN ('official', 'manual'))
);

CREATE TABLE interp.law_source_ids (
    law_id int NOT NULL REFERENCES interp.laws,
    source_kind text NOT NULL CHECK (source_kind IN ('lawgo_law', 'lawgo_admrul')),
    source_id text NOT NULL,
    PRIMARY KEY (law_id, source_kind, source_id)
);

CREATE TABLE interp.case_laws (
    case_key text NOT NULL REFERENCES interp.cases,
    law_id int NOT NULL REFERENCES interp.laws,
    mentions int NOT NULL,
    PRIMARY KEY (case_key, law_id)
);

CREATE TABLE interp.case_articles (
    id bigserial PRIMARY KEY,
    case_key text NOT NULL REFERENCES interp.cases,
    law_id int REFERENCES interp.laws,
    article text NOT NULL,
    article_sub int,
    paragraph int,
    item int,
    item_sub int,
    supplementary boolean NOT NULL,
    section text NOT NULL,
    pos int NOT NULL,
    raw text NOT NULL,
    resolve_method text NOT NULL CHECK (resolve_method IN ('explicit', 'context', 'default', 'unresolved')),
    law_guess text,
    UNIQUE (case_key, section, pos, raw)
);

CREATE TABLE interp.law_article_history (
    id bigserial PRIMARY KEY,
    law_id int NOT NULL REFERENCES interp.laws,
    source_id text,
    article text NOT NULL,
    article_sub int,
    change_date date,
    change_reason text,
    effective_date date,
    promulgation_date date,
    revision_type text
);

CREATE TABLE interp.admrul_versions (
    serial text PRIMARY KEY,
    law_id int NOT NULL REFERENCES interp.laws,
    admrul_id text,
    name_at_time text,
    status text,
    issue_date date,
    effective_date date,
    revision_type text
);

CREATE TABLE interp.case_article_status (
    case_article_id bigint PRIMARY KEY REFERENCES interp.case_articles,
    checked_on date NOT NULL,
    status text NOT NULL CHECK (status IN ('changed', 'unchanged', 'reg_changed', 'reg_unchanged', 'scheduled', 'unknown')),
    why text,
    n_later int NOT NULL,
    later_changes jsonb NOT NULL,
    cited_part_same boolean
);

CREATE TABLE interp.chunks (
    chunk_id bigserial PRIMARY KEY,
    case_key text NOT NULL REFERENCES interp.cases,
    group_id bigint NOT NULL REFERENCES interp.case_groups,
    chunk_type text NOT NULL CHECK (chunk_type IN ('question', 'answer', 'reason', 'question_card', 'judgment', 'rule_card')),
    chunk_no int NOT NULL,
    text text NOT NULL,
    embed_text text NOT NULL,
    bm25_tokens text NOT NULL,
    doc_class text NOT NULL,
    gubun text NOT NULL,
    category_final text,
    reply_date date,
    embed_model text NOT NULL,
    embedding vector({EMBED_DIM}) NOT NULL,
    UNIQUE (case_key, chunk_type, chunk_no)
);
