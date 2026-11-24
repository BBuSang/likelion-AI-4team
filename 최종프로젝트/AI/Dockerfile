# AI 서버 (FastAPI) — 배포용
# 데이터·모델은 이미지에 넣지 않고 서버 폴더를 붙임: /app/data (공개 RAG 금융상품 색인), /app/outputs (이름 보태기 모델)
# 허깅페이스 모델(KURE-v1, atonlee)은 첫 실행 때 받아 /hf-cache 볼륨에 둠
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf-cache TZ=Asia/Seoul

WORKDIR /app

# torch 는 CPU 전용 휠 (GPU 판은 수 GB) → 나머지 패키지
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.0 \
 && grep -v '^torch==' requirements.txt > /tmp/req.txt \
 && pip install -r /tmp/req.txt

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
