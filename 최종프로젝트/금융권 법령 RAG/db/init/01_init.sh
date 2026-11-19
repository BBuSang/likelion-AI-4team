#!/bin/bash
# 처음 볼륨을 만들 때 한 번만 실행된다 (기획안 5-5)
# - 확장: vector (벡터 검색)
# - 스키마: lawrag
# - 검색 전용 계정 law_reader: SELECT 권한만 (비밀번호 = .env LAW_PASSWORD)
set -e
psql -v ON_ERROR_STOP=1 -v pw="$LAW_PASSWORD" -v db="$POSTGRES_DB" \
     --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS lawrag;
CREATE ROLE law_reader LOGIN PASSWORD :'pw';
GRANT CONNECT ON DATABASE :"db" TO law_reader;
GRANT USAGE ON SCHEMA lawrag TO law_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA lawrag GRANT SELECT ON TABLES TO law_reader;
SQL
