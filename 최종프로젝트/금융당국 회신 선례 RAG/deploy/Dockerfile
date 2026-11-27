# AWS EC2 배포용 데모 이미지: PostgreSQL 17 + pgvector + Streamlit 앱을 한 컨테이너에 넣는다.
# DB는 빌드할 때 rag_db.dump를 복원해 이미지에 담고, KURE-v1 모델도 빌드할 때 받아 둔다(시작 시 다운로드 없음).
# 빌드(프로젝트 폴더에서): docker build -f deploy/Dockerfile -t interp-rag .
#   넣는 파일은 프로젝트 폴더의 .dockerignore가 정한다 (data/, .env, 평가 스크립트는 빠짐)
FROM pgvector/pgvector:pg17

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*

# PostgreSQL은 root로 실행할 수 없으므로 일반 사용자로 DB와 앱을 돌린다
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/venv/bin:/usr/lib/postgresql/17/bin:$PATH \
    PGDATA=/home/user/pgdata \
    HF_HOME=/home/user/.cache/huggingface \
    RAG_DSN=postgresql://rag@localhost:5432/rag \
    PYTHONUNBUFFERED=1 \
    PORT=8080
WORKDIR /home/user/app

RUN python3 -m venv /home/user/venv
COPY --chown=user deploy/requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nlpai-lab/KURE-v1', device='cpu')"

COPY --chown=user rag_db.dump /tmp/rag_db.dump
RUN initdb -D "$PGDATA" -U rag --auth=trust -E UTF8 --locale=C.UTF-8 \
 && pg_ctl -D "$PGDATA" -o "-c listen_addresses=127.0.0.1 -k /tmp" -w start \
 && createdb -h 127.0.0.1 -U rag rag \
 && (pg_restore -h 127.0.0.1 -U rag -d rag --no-owner --no-privileges /tmp/rag_db.dump || true) \
 && test "$(psql -h 127.0.0.1 -U rag -d rag -tAc 'select count(*) from interp.chunks')" -gt 0 \
 && pg_ctl -D "$PGDATA" -w stop \
 && rm /tmp/rag_db.dump

COPY --chown=user app ./app
COPY --chown=user pipeline ./pipeline
COPY --chown=user deploy/start.sh .
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

CMD ["./start.sh"]
