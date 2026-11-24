-- 신한은행 상품 RAG schema (PostgreSQL + pgvector)
-- 원칙: 공통 core 컬럼 + 유형별 detail(JSONB) + 상품 간 비교가 필요한 것(금리/수수료/혜택)만 정규화
-- 관계: 상품 ↔ 문서는 N:M (공유 설명서·약관이 많음: 가계대출 설명서 85개, 기업용 약관 139개 상품 등)
--       추출 결과(금리·수수료·혜택·청크)는 모두 '문서' 기준으로 저장하고, 상품은 product_document로 연결한다.
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. 상품 core: data/meta/products.csv 에서 적재 (LLM이 만들지 않음)
CREATE TABLE product (
  product_id        TEXT PRIMARY KEY,   -- products.csv: '분류-구분-사이트번호' (예: SAV-적립식예금-27)
  product_name      TEXT NOT NULL,      -- UNIQUE 아님: 사이트 목록에 같은 이름이 여러 번 있음(기업대출 등)
  category          TEXT NOT NULL,      -- 사이트 분류: 저축상품 / 대출상품 / 기타상품 / 복합금융상품
  sub_category      TEXT NOT NULL,      -- 사이트 구분: 적립식예금, 거치식예금, 입출금자유예금, 주택청약관련예금, 외화예금,
                                        --   주택담보대출, 기타담보대출, 신용대출, 기업대출, 계좌상품, 계좌외상품, 복합금융상품
  site_no           INT,                -- 상품공시 목록 번호
  dup_name          BOOLEAN NOT NULL DEFAULT false,
  snapshot_date     DATE NOT NULL,      -- 상품공시 목록 기준일 (예: 2026-11-06 판매분)
  target_tags       TEXT[],             -- 정규화 태그: {군인, 청년, 개인사업자, 직장인 ...} 필터용 (나중에 채움)
  created_at        TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON product (category, sub_category);
CREATE INDEX ON product (product_name);
CREATE INDEX ON product USING gin (target_tags);

-- 2. 원본 문서: data/meta/documents.csv 에서 적재 + 추출 결과로 메타 보강
CREATE TABLE document (
  doc_id            TEXT PRIMARY KEY,   -- documents.csv: D001 ...
  title             TEXT NOT NULL,
  file_name         TEXT NOT NULL,
  file_hash         TEXT NOT NULL UNIQUE,   -- 재적재 idempotent
  doc_type          TEXT NOT NULL,      -- 상품설명서 / 특약 / 약관 / 약정서 / 투자설명서 / 간이투자설명서
  n_products        INT NOT NULL,       -- 연결된 상품 수 (1이면 상품 전용, 2 이상이면 공유 문서)
  -- 아래는 문서 추출 결과(DocumentExtraction)로 채움
  summary           TEXT,               -- 상품특징 1~2문장
  target_customer   TEXT,               -- 가입/신청 대상 원문
  join_channels     TEXT[],             -- {영업점, 슈퍼SOL, 인터넷뱅킹 ...}
  deposit_protected BOOLEAN,            -- 예금자보호
  risk_grade        SMALLINT,           -- 투자설명서의 위험등급 (원문 기준)
  detail            JSONB NOT NULL DEFAULT '{}',  -- 유형별 상세 (models.py의 DepositDetail / LoanDetail / InvestDetail)
  review_no         TEXT,               -- 준법감시인 심사필 번호 (2-4 코드 추출: doc_meta.csv)
  valid_from        DATE,
  valid_to          DATE,
  info_base_date    DATE,               -- "2026.07.02 현재" 같은 기준일
  page_count        INT,
  extract_status    TEXT DEFAULT 'pending', -- pending / ok / needs_review
  extract_notes     TEXT[],             -- 추출 때 확정 못 한 곳 (DocumentExtraction.notes)
  ingested_at       TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON document (doc_type);
CREATE INDEX ON document USING gin (detail jsonb_path_ops);

-- 3. 상품 ↔ 문서 연결 (N:M): data/meta/product_docs.csv
CREATE TABLE product_document (
  product_id  TEXT REFERENCES product ON DELETE CASCADE,
  doc_id      TEXT REFERENCES document ON DELETE CASCADE,
  role        TEXT NOT NULL CHECK (role IN ('explainer', 'terms')),  -- 사이트의 설명서 열 / 약관·특약 열
  PRIMARY KEY (product_id, doc_id)
);
CREATE INDEX ON product_document (doc_id);

-- 4. 금리 (예금/적금/대출 공통). "금리 2% 이상", "가장 금리 높은 적금" 같은 SQL 질의용
--    문서 기준으로 저장. 상품별 금리는 product_document(role='explainer')로 조인해서 구한다.
CREATE TABLE doc_rate (
  rate_id      BIGSERIAL PRIMARY KEY,
  doc_id       TEXT REFERENCES document ON DELETE CASCADE,
  rate_kind    TEXT NOT NULL,   -- base(기본) / preferential(우대) / max(최고) / early_termination(중도해지) / after_maturity(만기후)
                                --   recruit_period(ELD 모집기간) / index_linked(ELD 지수연동) / loan_base(기준) / loan_spread(가산)
                                --   loan_preferential(감면) / loan_final(최종) / overdue(연체)
  rate_min     NUMERIC(6,3),    -- 단일값이면 min=max
  rate_max     NUMERIC(6,3),
  term_min_m   INT,             -- 적용 기간(개월) 구간, 없으면 null
  term_max_m   INT,
  amount_min   NUMERIC,         -- 적용 금액 구간 (예: 0 ~ 1,000,000)
  amount_max   NUMERIC,
  currency     TEXT,            -- 외화예금 통화 (원화는 null)
  rate_type    TEXT,            -- fixed / variable (대출)
  condition    TEXT,            -- 적용 조건 원문 요약
  pre_tax      BOOLEAN,         -- 원문의 세전(true)/세후(false), 없으면 null
  section_seq  INT,             -- 근거 섹션 (data/sections의 seq)
  source_text  TEXT             -- 근거 원문 (환각 검증용: 그 섹션에 있는 문장인지 코드로 확인)
);
CREATE INDEX ON doc_rate (doc_id);
CREATE INDEX ON doc_rate (rate_kind, rate_max);

-- 5. 수수료/비용 (대출 중도상환수수료, 이체수수료 등)
CREATE TABLE doc_fee (
  fee_id      BIGSERIAL PRIMARY KEY,
  doc_id      TEXT REFERENCES document ON DELETE CASCADE,
  fee_kind    TEXT NOT NULL,   -- early_repayment(중도상환해약금) / early_termination(중도해지수수료) / spread(금·은 스프레드)
                               --   physical_withdrawal(실물 인출) / cash_handling(외화 현찰) / stamp_tax(인지세) / transfer / limit(한도 수수료) / etc
  amount      NUMERIC,         -- 정액
  rate_pct    NUMERIC(6,3),    -- 정률
  condition   TEXT,
  section_seq INT,
  source_text TEXT
);
CREATE INDEX ON doc_fee (doc_id);

-- 6. 우대조건/혜택 (우대금리 요건, 수수료 면제 등)
CREATE TABLE doc_benefit (
  benefit_id   BIGSERIAL PRIMARY KEY,
  doc_id       TEXT REFERENCES document ON DELETE CASCADE,
  benefit_kind TEXT NOT NULL,  -- rate_bonus / fee_waiver / discount / point / cashback / service
  requirement  TEXT,           -- 충족 조건 (예: 군 급여 이체, 전월실적 30만원)
  benefit      TEXT,           -- 받는 혜택 (예: 연 1.9%p, ATM 수수료 면제 월 5회)
  value_num    NUMERIC,        -- 숫자로 표현 가능하면 (1.9, 5)
  value_unit   TEXT,           -- %p / 원 / 회 / %
  section_seq  INT,
  source_text  TEXT
);
CREATE INDEX ON doc_benefit (doc_id);

-- 7. RAG chunk: 원문 전체를 여기 넣음 → 정형 추출에서 빠진 정보도 검색으로 커버
--    청크는 문서에만 속한다. 상품으로 좁힐 때는 product_document로 doc_id 목록을 구한 뒤 검색한다.
CREATE TABLE chunk (
  chunk_id     BIGSERIAL PRIMARY KEY,
  doc_id       TEXT REFERENCES document ON DELETE CASCADE,
  section      TEXT,           -- 표의 '구분' 또는 목차 제목: 가입대상, 우대이자율, 상환방법 ... (짧은 섹션을 묶으면 '큰 제목 > 이름 · 이름')
  section_seqs INT[],          -- 이 청크에 든 섹션 번호 (data/sections의 seq, 출처 표시용)
  page         INT,
  page_end     INT,
  content      TEXT NOT NULL,  -- 원문 (citation)
  embed_text   TEXT NOT NULL,  -- "[문서 제목 > section] " + content (공유 문서는 상품명 대신 문서 제목)
  embedding    vector(1536),
  tsv          tsvector,       -- kiwipiepy 토큰화 결과를 'simple'로
  content_hash TEXT
);
CREATE INDEX ON chunk USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON chunk USING gin (tsv);
CREATE INDEX ON chunk (doc_id);

-- 8. 공통 보일러플레이트 (위법계약해지권, 자료열람요구권, 민원안내 등) 1회만 저장
CREATE TABLE common_clause (
  clause_id    BIGSERIAL PRIMARY KEY,
  title        TEXT,
  content      TEXT NOT NULL,
  content_hash TEXT NOT NULL UNIQUE,
  embedding    vector(1536)
);
CREATE TABLE document_clause (
  doc_id    TEXT REFERENCES document ON DELETE CASCADE,
  clause_id BIGINT REFERENCES common_clause,
  PRIMARY KEY (doc_id, clause_id)
);

-- 9. 용어집 (부록 용어 안내 - 압류, 질권설정 등)
CREATE TABLE glossary_term (
  term       TEXT PRIMARY KEY,
  definition TEXT NOT NULL,
  embedding  vector(1536)
);

-- 10. 고시 금리 (shinhan.com 금리안내 팝업, 날짜별 스냅샷) — data/rates/rate_notice.csv
--     설명서 금리는 '그 문서의 기준일' 값, 이 테이블은 '수집일(as_of)'의 고시 값 → 숫자 답변은 이 테이블 우선, 날짜 표시
--     예금: 기간·금액구간별 기본이율/온라인이율, 대출: 기준금리 종류·기준·가산·우대·최저·최고 + 산출 가정(예시 금리)
CREATE TABLE rate_notice (
  notice_id     BIGSERIAL PRIMARY KEY,
  as_of         DATE NOT NULL,          -- 수집 기준일 (페이지의 '기준일시')
  kind          TEXT NOT NULL CHECK (kind IN ('deposit', 'loan')),
  page_group    TEXT NOT NULL,          -- 목돈만들기/목돈굴리기/입출금자유 · 신용대출/부동산담보대출/MY CAR 대출/기타대출
  page_product  TEXT NOT NULL,          -- 금리 페이지의 상품명 (상품공시 목록에 없는 상품도 있음)
  pay_method    TEXT,                   -- 예금: 만기일시지급-단리식 등
  term          TEXT,                   -- 예금: '12개월제', '6개월이상', '30일이상' (원문 그대로)
  amount_from   NUMERIC,                -- 입출금: 금액구간 시작(원). 같은 이율이 이어지는 구간은 첫 구간만 저장
  base_index    TEXT,                   -- 대출: 금융채(6개월), 신 잔액기준 COFIX, CD금리 …
  base_rate     NUMERIC(6,3),           -- 예금: 기본이율 / 대출: 기준금리
  online_rate   NUMERIC(6,3),           -- 예금: 온라인가입시 이율
  spread        NUMERIC(6,3),           -- 대출: 가산금리
  pref          NUMERIC(6,3),           -- 대출: 우대금리
  min_rate      NUMERIC(6,3),           -- 대출: 최저금리 (= 기준 + 가산 − 우대, 이차보전 등 예외 있음)
  max_rate      NUMERIC(6,3),           -- 대출: 최고금리
  assumption    TEXT,                   -- 대출: '대출금액 1억5천만원 / 12개월 / 신용 3등급 가정 예시' 등 산출 가정
  sale_status   TEXT,                   -- 금리 페이지 상품의 판매 상태: on_sale / discontinued / not_found (rate_product_map.csv)
  source        TEXT NOT NULL
);
CREATE INDEX ON rate_notice (as_of, kind);
CREATE INDEX ON rate_notice (page_product);
-- 금리 페이지 상품 ↔ 상품공시 상품 (이름 정규화로 연결, 이름 중복 시 여러 개) — data/rates/rate_product_map.csv
CREATE TABLE rate_notice_product (
  page_group    TEXT NOT NULL,
  page_product  TEXT NOT NULL,
  product_id    TEXT REFERENCES product ON DELETE CASCADE,
  match_method  TEXT NOT NULL,          -- exact / strip_paren / contains
  PRIMARY KEY (page_group, page_product, product_id)
);

-- 10-2. 판매 중지 상품 (상품공시 '판매 중지된 상품' 목록) — data/meta/discontinued.csv
--      답변 원칙: "판매 중지된 상품"이라고만 안내, 날짜 없음. also_on_sale=true(같은 이름이 판매 중 목록에도 있음)이면 단정하지 않음
CREATE TABLE discontinued_product (
  discontinued_id BIGSERIAL PRIMARY KEY,
  category        TEXT NOT NULL,
  sub_category    TEXT NOT NULL,
  site_no         INT,
  product_name    TEXT NOT NULL,
  also_on_sale    BOOLEAN NOT NULL DEFAULT false,
  snapshot_date   DATE NOT NULL
);
CREATE INDEX ON discontinued_product (product_name);

-- 11. 상품 기준으로 보기 (질의 편의용)
--  상품 → 설명서(explainer)의 금리
CREATE VIEW product_rate AS
SELECT pd.product_id, r.*
FROM product_document pd
JOIN doc_rate r ON r.doc_id = pd.doc_id
WHERE pd.role = 'explainer';

--  상품 → 검색 대상 문서 목록 (설명서 + 약관·특약)
CREATE VIEW product_chunk AS
SELECT pd.product_id, pd.role, c.*
FROM product_document pd
JOIN chunk c ON c.doc_id = pd.doc_id;
