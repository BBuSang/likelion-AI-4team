# 사공삼(403) 전체 ERD

사내 RAG(개인정보) + 공개 자료 RAG 3개(법령·금융상품·비조치의견서)를 한 서비스에서 쓰는 데이터 구조입니다.
**스키마마다 다른 DB 계정**을 써서 서로 읽을 수 없게 나눴습니다.

## 1. 한눈에 보기

```mermaid
flowchart LR
    subgraph DB["PostgreSQL 16 + pgvector (saebom)"]
        APP[("app<br/>계정·권한")]
        RAG[("rag<br/>가명처리 청크·벡터")]
        VAULT[("vault<br/>토큰↔원문·복원 기록")]
        LAW[("lawrag<br/>법령 조문·벡터")]
        INTERP[("interp<br/>비조치의견서·회신")]
    end
    PRD[["금융상품 RAG<br/>파일 색인<br/>(AI/data/public_rag/product)"]]
    BE[Spring Boot] -- app_user --> APP
    AI[AI 서버] -- rag_user --> RAG
    AI -- vault_user --> VAULT
    AI -- "law_reader (읽기만)" --> LAW
    AI -- "interp_reader (읽기만)" --> INTERP
    AI -- 파일 읽기 --> PRD
```

| 스키마 | 계정 | 내용 | 개인정보 |
|---|---|---|---|
| `app` | app_user | 계정·팀·직급·관리자 기록·이메일 인증 | 계정 이메일·이름 |
| `rag` | rag_user | **가명처리된** 청크·토큰·벡터, 문서 권한, 추정 인물(토큰만), 직원(이름 없음) | 없음 (토큰만) |
| `vault` | vault_user | 토큰↔원문, 이름 포함 직원, 실명 복원 기록, 업로드·검수 원문 | **있음** — AI 서버만 |
| `lawrag` | law_reader (SELECT 만) | 법령 380개 · 조문 28,510 · 조각 35,321 · 위임·준용 연결 11,477 | 없음 |
| `interp` | interp_reader (SELECT 만) | 금융당국 회신 선례·현장건의 과제 | 없음 (적재 때 가림) |
| (파일) | — | 금융상품 391 · 문서 247 · 청크 2,218 | 없음 |

- 사내 RAG 계정은 공개 자료를 못 읽고, 공개 자료 계정은 `rag`·`vault` 를 못 읽습니다 (확인: `law_reader` → `rag` 거부, `rag_user` → `lawrag` 거부, `law_reader` 쓰기 거부).
- 공개 자료는 특정 고객 정보가 없어 HMAC 토큰화를 하지 않습니다. 에이전트가 공개 RAG 로 보내는 검색어에서 토큰·번호를 지웁니다.

---

## 2. app — 계정·권한 (Spring JPA)

```mermaid
erDiagram
    teams ||--o{ users : "소속"
    positions ||--o{ users : "직급"
    users ||--o{ admin_audit_log : "변경 대상"
    users {
        BIGINT id PK
        TEXT email UK
        TEXT pw "BCrypt"
        TEXT name
        BIGINT team_id FK
        BIGINT position_id FK
        TEXT role "STAFF·LEAD·AUDITOR (데이터 접근)"
        BOOL is_admin
        TEXT status "PENDING·ACTIVE·SUSPENDED"
        TEXT emp_id UK "문서 속 직원 연결 (rag.employees)"
        BIGINT approved_by
        TIMESTAMP approved_at
    }
    teams {
        BIGINT id PK
        TEXT name UK "문서 메타데이터 team 과 같은 글자"
        BOOL active
    }
    positions {
        BIGINT id PK
        TEXT name UK
        INT sort_order
        TEXT default_role
    }
    admin_audit_log {
        BIGINT id PK
        BIGINT admin_id
        BIGINT target_user_id
        TEXT action "approve·update·suspend"
        JSONB changes "전후 값"
    }
    email_verifications {
        BIGINT id PK
        TEXT email
        TEXT purpose "SIGNUP·RESET"
        TEXT code_hash "HMAC, 원문 없음"
        TIMESTAMP expires_at
        INT attempts
        TIMESTAMP verified_at
        TIMESTAMP used_at
    }
```

---

## 3. rag · vault — 사내 RAG (개인정보)

`rag` 와 `vault` 는 계정이 달라 DB 외래 키로 잇지 않고, **토큰·pid·doc_id 값으로** 이어집니다 (점선 = 값으로 연결).

```mermaid
erDiagram
    documents ||--|{ chunks : "doc_id"
    persons ||--o{ person_attrs : "pid"
    employees ||--o{ documents : "owner_emp_id"
    chunks {
        TEXT chunk_id PK
        TEXT doc_id
        TEXT masked_text "가명처리된 글만"
        TEXT_ARRAY tokens "GIN 색인"
        VECTOR embedding "1024, HNSW"
        TEXT owner_emp_id
        TEXT team
        TEXT visibility "public·team·restricted"
    }
    documents {
        TEXT doc_id PK
        TEXT doc_type
        TEXT owner_emp_id
        TEXT team
        TEXT visibility
        JSONB basis "담당자·팀 판단 근거"
    }
    persons {
        TEXT pid PK
        TEXT person_token UK
        TEXT name_token "동명이인 판단"
        TEXT role "customer·staff·external"
        TEXT_ARRAY affiliations
        TEXT status "resolved·unresolved"
    }
    person_attrs {
        TEXT pid FK
        TEXT label
        TEXT token "그 사람의 전화·계좌 등 토큰"
    }
    employees {
        TEXT emp_id PK "E001(문서 추론)·U0001(계정용)"
        TEXT team
        TEXT position
        TEXT role
    }
```

```mermaid
erDiagram
    ingest_job ||--|{ staging_doc : "업로드 1번"
    staging_doc ||--o{ review_log : "검수 기록"
    token_map {
        TEXT token PK "<PERSON_…>"
        TEXT label
        TEXT normalized
        TEXT_ARRAY originals "원문 표기"
        TEXT pid
    }
    person_mentions {
        TEXT doc_id PK
        INT start_pos PK
        TEXT token
        TEXT pid
        TEXT basis "사람 잇기 근거"
    }
    person_keys {
        TEXT pid PK
        TEXT name
        TEXT person_key
    }
    employees_v {
        TEXT emp_id PK
        TEXT name "직원 이름 — vault 에만"
        TEXT team
        TEXT role
    }
    restore_log {
        BIGINT id PK
        TEXT actor
        TEXT role
        TEXT emp_id
        TEXT purpose
        TEXT_ARRAY tokens "복원한 토큰"
        TEXT_ARRAY doc_ids
        TEXT_ARRAY denied "권한 없어 남긴 토큰"
        TEXT refused "거절 이유"
    }
    ingest_job {
        BIGINT id PK
        TEXT uploader
        TEXT status
    }
    staging_doc {
        BIGINT id PK
        BIGINT job_id FK
        TEXT doc_id UK
        TEXT text "원문 (검수 대기)"
        JSONB detected
        JSONB review "검수 신호"
        TEXT status "pending·stored·rejected"
        TEXT approved_by "검수자·gold·auto"
    }
    review_log {
        BIGINT id PK
        BIGINT staging_doc_id FK
        TEXT action
        INT added
        INT removed
    }
    deletion_log {
        BIGINT id PK
        TEXT doc_id
        TEXT actor
        TEXT reason
    }
```
(`employees_v` = `vault.employees`)

---

## 4. lawrag — 법령 RAG (`AI/app/rag_public/law/`)

국가법령정보센터 Open API 로 받은 법률·시행령·시행규칙·감독규정. 덤프 `AI/data/public_rag/law/law_rag.dump` (Git LFS) → `infra/load_public_rag.sh` 로 넣음.

```mermaid
erDiagram
    laws ||--o{ laws : "parent_law_uid (시행령→법률)"
    laws ||--o{ law_aliases : "별칭"
    laws ||--|{ articles : "조문"
    articles ||--o{ article_lines : "항·호·목"
    articles ||--|{ chunks : "검색 조각"
    articles ||--o{ article_links : "위임·준용 (src)"
    articles ||--o{ article_links : "(dst)"
    laws {
        TEXT law_uid PK "law:MST · admrul:일련번호"
        TEXT law_name
        TEXT law_level "법률·시행령·시행규칙·행정규칙"
        TEXT parent_law_uid FK
        DATE effective_date
    }
    law_aliases {
        TEXT alias PK
        TEXT law_uid PK
        TEXT alias_source "official·team"
    }
    articles {
        TEXT article_uid PK
        TEXT law_uid FK
        TEXT article_label "제34조의2"
        TEXT text
        DATE article_effective_date
    }
    article_lines {
        TEXT line_uid PK
        TEXT article_uid FK
        TEXT parent_line_uid
        TEXT text
    }
    chunks {
        TEXT chunk_id PK "article_uid#조각번호"
        TEXT article_uid FK
        TEXT law_uid FK
        TEXT text
        VECTOR embedding "KURE-v1 1024, HNSW"
    }
    article_links {
        TEXT src_uid PK
        TEXT dst_uid PK
        TEXT kind "위임·준용"
    }
```

---

## 5. interp — 금융당국 회신 선례(비조치의견서) RAG (`AI/app/rag_public/noaction/`)

금융위원회·금융감독원 법령해석·비조치의견서 회신과 현장건의 과제. 스키마 파일: `AI/data/public_rag/noaction/schema.sql`·`exmnt_schema.sql`·`indexes.sql`. 데이터(`processed/`, 포털 저작권정책상 git 제외) → `AI/scripts/load_noaction.py` 로 넣음 (회신 4,366 · 조각 8,973 · 현장건의 조각 12,445).

```mermaid
erDiagram
    case_groups ||--|{ cases : "같은 내용 묶음"
    cases ||--o{ chunks : "검색 조각"
    cases ||--o{ case_laws : "인용 법령"
    laws ||--o{ case_laws : ""
    laws ||--o{ law_aliases : "별칭"
    cases ||--o{ case_articles : "인용 조문"
    case_articles ||--o| case_article_status : "이후 개정 여부"
    exmnt_groups ||--|{ exmnt_cases : "현장건의 묶음"
    exmnt_cases ||--o{ exmnt_chunks : "검색 조각"
    cases {
        TEXT case_key PK
        TEXT gubun "법령해석·비조치의견서"
        TEXT title
        DATE reply_date
        TEXT answer_type
        TEXT question
        TEXT answer
        TEXT reason
        BIGINT group_id FK
        TEXT url
    }
    case_groups {
        BIGINT group_id PK
        TEXT group_type "single·same_body·same_question"
        TEXT representative_case_key
    }
    chunks {
        BIGINT chunk_id PK
        TEXT case_key FK
        TEXT chunk_type "question_card·judgment 등"
        TEXT text
        VECTOR embedding "KURE-v1 1024"
    }
    laws {
        INT law_id PK
        TEXT law_name UK
        TEXT law_kind
    }
    case_articles {
        BIGINT id PK
        TEXT case_key FK
        INT law_id FK
        TEXT article
    }
    case_article_status {
        BIGINT case_article_id PK
        TEXT status "changed·unchanged…"
    }
    exmnt_cases {
        TEXT case_key PK
        TEXT title
        TEXT answer_type "수용·불수용…"
        TEXT question
        TEXT answer
    }
    exmnt_chunks {
        BIGINT chunk_id PK
        TEXT case_key FK
        VECTOR embedding "1024"
    }
```
(그 밖에 `case_depts`, `attachments`, `law_source_ids`, `law_article_history`, `admrul_versions`, `exmnt_groups` — 전체는 스키마 파일)

---

## 6. 금융상품 RAG (`AI/app/rag_public/product/`)

신한은행 상품공시의 판매 중 상품 391개와 설명서·약관·특약 247개. 실행 코드는 DB 가 아니라 `AI/data/public_rag/product/` 의 `meta/*.csv`(git 포함)·`chunks/*.jsonl`(은행 문서 원문, git 제외)·`vectors.npz`(git 제외, `scripts/build_product_vectors.py`)를 메모리로 읽습니다. 아래는 같은 내용을 DB 로 옮길 때의 설계 ERD (금리·수수료·혜택 추출 표는 답변 경로에서 아직 안 씀).

```mermaid
erDiagram
    product ||--o{ product_document : ""
    document ||--o{ product_document : ""
    document ||--o{ chunk : "검색 청크"
    document ||--o{ doc_rate : "금리"
    document ||--o{ doc_fee : "수수료"
    document ||--o{ doc_benefit : "혜택"
    document ||--o{ document_clause : ""
    common_clause ||--o{ document_clause : "공통 조항"
    product {
        TEXT product_id PK
        TEXT product_name
        TEXT category
    }
    product_document {
        TEXT product_id PK
        TEXT doc_id PK
        TEXT role "explainer·terms"
    }
    document {
        TEXT doc_id PK "D001~D247"
        TEXT title
        TEXT doc_type
        INT n_products "1 전용, 2+ 공유"
        DATE info_base_date
    }
    chunk {
        BIGINT chunk_id PK
        TEXT doc_id FK
        INT page
        TEXT content
        VECTOR embedding "KURE-v1 1024"
    }
    doc_rate {
        BIGINT rate_id PK
        TEXT doc_id FK
        TEXT rate_kind
        TEXT source_text "근거 원문"
    }
    doc_fee {
        BIGINT fee_id PK
        TEXT doc_id FK
    }
    doc_benefit {
        BIGINT benefit_id PK
        TEXT doc_id FK
    }
    common_clause {
        BIGINT clause_id PK
        TEXT content
    }
    document_clause {
        TEXT doc_id PK
        BIGINT clause_id PK
    }
```

---

## 7. 에이전트가 이 저장소들을 쓰는 방식

| 도구 | 저장소 | 계정 | 검색어 | 결과 번호 |
|---|---|---|---|---|
| `search_internal` | rag (+ vault 는 복원 때만) | rag_user | 원문 질문 (서버 안에서만, 권한 밖 문서 검색 전 제외) | 문서 ID `H0071` |
| `search_law` | lawrag | law_reader | 토큰·번호 지운 질문 | `LAW-1` |
| `search_product` | 파일 색인 | — | 토큰·번호 지운 질문 | `PRD-1` |
| `search_noaction` | interp | interp_reader | 토큰·번호 지운 질문 | `CASE-1` (혼합 질문) · `L123`/`O45` (비조치만) |
