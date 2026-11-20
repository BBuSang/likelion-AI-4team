#!/bin/bash
# 공개 자료 RAG 를 사공삼 DB 에 넣기 (worklog 53장). 처음 한 번, 데이터가 바뀌면 다시.
#   bash infra/load_public_rag.sh            # 개발 DB (컨테이너 saebom_db)
#   CONTAINER=saebom_eval_db DB_PORT=5435 bash infra/load_public_rag.sh
# 법령: AI/data/public_rag/law/law_rag.dump (Git LFS, 193MB) → lawrag 스키마, 읽기 전용 law_reader (02_public_rag.sh 가 만듦)
# 비조치의견서: AI/data/public_rag/noaction/processed/ (git 제외, 담당자에게 받기) → interp 스키마, 읽기 전용 interp_reader
# 금융상품: DB 없음 — python AI/scripts/build_product_vectors.py 로 벡터만 만듦
set -euo pipefail
export MSYS_NO_PATHCONV=1
HERE="$(cd "$(dirname "$0")" && pwd)"
AI="$HERE/../AI"
C="${CONTAINER:-saebom_db}"
LAW_DUMP="$AI/data/public_rag/law/law_rag.dump"

if [ -f "$LAW_DUMP" ] && [ "$(wc -c < "$LAW_DUMP")" -gt 100000 ]; then
  echo "[법령] 덤프 넣는 중 (색인 다시 만드느라 몇 분)"
  docker cp "$LAW_DUMP" "$C:/tmp/law_rag.dump"
  docker exec "$C" sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -qc "DROP SCHEMA IF EXISTS lawrag CASCADE; CREATE SCHEMA lawrag; REVOKE ALL ON SCHEMA lawrag FROM PUBLIC;" \
    && pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" -n lawrag --no-owner --no-acl /tmp/law_rag.dump \
    && psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -qc "GRANT USAGE ON SCHEMA lawrag TO law_reader; GRANT SELECT ON ALL TABLES IN SCHEMA lawrag TO law_reader;" \
    && rm /tmp/law_rag.dump'
else
  echo "[법령] 덤프가 없음 ($LAW_DUMP) — git lfs pull 필요"
fi

if [ -f "$AI/data/public_rag/noaction/processed/chunks_d3.jsonl" ]; then
  echo "[비조치의견서] 적재 중 (벡터 색인 만드느라 10분 남짓)"
  (cd "$AI" && python -X utf8 scripts/load_noaction.py --rebuild)
else
  echo "[비조치의견서] 데이터가 없음 (AI/data/public_rag/noaction/processed/) — 담당자에게 받기"
fi

if [ -f "$AI/data/public_rag/product/chunks/chunks.jsonl" ]; then
  echo "[금융상품] 벡터 만들기 (이미 있는 건 건너뜀)"
  (cd "$AI" && python -X utf8 scripts/build_product_vectors.py)
else
  echo "[금융상품] 원문 청크가 없음 (AI/data/public_rag/product/chunks/) — 담당자에게 받기"
fi
