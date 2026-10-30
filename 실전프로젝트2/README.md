# 실전 2 — 가명처리 기반 검색 챗봇

실전 1 결과물 중 실전 2에 필요한 파일만 복사해 둔 폴더예요. 원본은 `../실전프로젝트1/`에 그대로 있어요.
연구 계획(실험 E1~E4, 일정)은 프로젝트 맨 위 `README.md` 참고.

## 폴더 구성

```
실전프로젝트2/
├── data/
│   ├── pii_synth_finance_dev.jsonl    제공 데이터 dev 100건 (개선용)
│   └── pii_synth_finance_test.jsonl   제공 데이터 test 200건 (수치 보고용)
│                                      → 개체마다 person_id 있음 (토큰 일관성 채점 기준)
├── preds/
│   ├── rules+atonlee_dev.jsonl        실전 1 운영 탐지기의 탐지 결과 (적재 시 재사용 가능)
│   └── rules+atonlee_test.jsonl
├── pii_detector.py                    운영 탐지기 입구: PIIDetector().detect(text)  ← 질의 처리는 반드시 이걸로
├── common.py / make_union.py / postprocess.py   탐지기 공통 기능
├── detectors/                         rule_based(Kiwi 규칙) · token_cls(atonlee) · hf_ner · analyzers
├── requirements.txt
│
│   ── 아래는 강사 키트(pgvector_starter)에서 복사 (안내: README_starter.md) ──
├── docker-compose.yml · init/         PostgreSQL + pgvector, rag·vault 스키마와 계정 분리
├── .env.example                       .env 로 복사해서 사용 (이 폴더에 둠, 커밋 금지)
├── starter/                           pii_pipeline.py ★ · search.py ★ · ingest · run_queries · chat · embed · db · config
├── eval/                              check_token_consistency.py · eval_holdout_retrieval.py
└── data/holdout_*.jsonl               적재 코퍼스 187건 · 검색 질의 88개 · 질의 정답 · 인물 정답
```

## 탐지기 사용

```python
from pii_detector import PIIDetector
det = PIIDetector()   # 모델 로딩 몇 초 (atonlee는 첫 실행 때 Hugging Face에서 내려받음)
det.detect("윤 선영 고객(01003492063)이 추가 한도를 문의함.")
```

## 지켜야 할 것
- 실행은 `python -X utf8 ...` (윈도우 인코딩 오류 방지), starter는 이 폴더에서 `python -X utf8 -m starter.ingest` 형태로
- API 키·HMAC 키는 이 폴더의 `.env`에 (`starter/config.py`와 `docker compose`가 여기서 읽음, `.gitignore`로 제외됨)
- 임베딩·LLM API에는 마스킹된 텍스트만 보냄
- `preds/`는 탐지기 v10 결과인지 몇 건 다시 돌려 확인한 뒤 사용 (적재·질의 토큰이 어긋나지 않도록)
