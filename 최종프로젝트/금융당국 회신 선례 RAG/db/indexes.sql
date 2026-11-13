-- 적재가 끝난 뒤 만든다.
ALTER TABLE interp.case_groups ADD FOREIGN KEY (representative_case_key) REFERENCES interp.cases;

CREATE INDEX ON interp.cases (group_id);
CREATE INDEX ON interp.cases (reply_date);
CREATE INDEX ON interp.cases USING gin (title gin_trgm_ops);
CREATE INDEX ON interp.case_laws (law_id);
CREATE INDEX ON interp.case_articles (case_key);
CREATE INDEX ON interp.case_articles (law_id, article, article_sub);
CREATE INDEX ON interp.law_article_history (law_id, article, article_sub);
CREATE INDEX ON interp.admrul_versions (law_id);
CREATE INDEX ON interp.chunks (case_key);
CREATE INDEX ON interp.chunks (group_id);
CREATE INDEX ON interp.chunks (gubun, category_final, reply_date);
CREATE INDEX ON interp.chunks (doc_class);
CREATE INDEX ON interp.chunks USING gin (text gin_trgm_ops);
CREATE INDEX chunks_embedding_hnsw ON interp.chunks USING hnsw (embedding vector_cosine_ops);
