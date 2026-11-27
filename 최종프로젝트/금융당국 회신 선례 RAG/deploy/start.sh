#!/bin/sh
# DB를 켠 뒤 Streamlit을 띄운다. 포트는 PORT(기본 8080). 서버에서는 -p 80:8080 으로 연결한다.
set -e
# 서버가 갑자기 꺼졌다 켜지면 남는 잠금 파일을 지운다 (이 컨테이너만 쓰는 DB라 안전)
rm -f "$PGDATA/postmaster.pid"
pg_ctl -D "$PGDATA" -o "-c listen_addresses=127.0.0.1 -k /tmp" -l /tmp/postgres.log -w start
exec python -m streamlit run app/app.py \
  --server.port "${PORT:-8080}" --server.address 0.0.0.0 --server.headless true \
  --server.enableCORS false --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
