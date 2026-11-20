#!/bin/bash
# 공개 자료 RAG 스키마·읽기 전용 계정 (worklog 53장). 컨테이너 최초 실행 때 01_schema.sh 다음에 한 번.
# 데이터는 각 RAG 의 DB 덤프로 넣음: infra/load_public_rag.sh
#   lawrag : 금융권 법령 RAG (팀원 백송이, baek_rag 브랜치)          — 읽기 전용 law_reader
#   interp : 금융당국 회신 선례(비조치의견서) RAG (팀원 박지훈)       — 읽기 전용 interp_reader (AI/scripts/load_noaction.py 가 만듦)
# 개인정보가 없는 자료라 HMAC 토큰화는 하지 않고, 사내 RAG 계정(rag_user·vault_user)과는 서로 못 읽게 나눔
set -euo pipefail
if [ -z "${LAW_PASSWORD:-}" ]; then
  echo "LAW_PASSWORD 가 없어 공개 RAG 계정을 만들지 않음 (infra/.env 에 넣고 다시 만들거나 load_public_rag.sh 사용)"
  exit 0
fi
psql -v ON_ERROR_STOP=1 -v lawpw="$LAW_PASSWORD" -v db="$POSTGRES_DB" --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
CREATE ROLE law_reader LOGIN PASSWORD :'lawpw';
GRANT CONNECT ON DATABASE :"db" TO law_reader;
CREATE SCHEMA lawrag;
REVOKE ALL ON SCHEMA lawrag FROM PUBLIC;
GRANT USAGE ON SCHEMA lawrag TO law_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA lawrag GRANT SELECT ON TABLES TO law_reader;
SQL
