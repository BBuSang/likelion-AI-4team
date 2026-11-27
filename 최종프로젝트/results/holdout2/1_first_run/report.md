# 무인 파이프라인 결과 — holdout2_first

- 실행: 2026-11-26 15:07 · DB localhost:5435/saebom
- 문서 756건 · 질의 155개 · 임베딩 hf/nlpai-lab/KURE-v1 · 재순위 True
- 입력: docs=../../holdout2/holdout2_public.jsonl queries=../../holdout2/holdout2_retrieval_queries_gold.jsonl gold=../../holdout2/holdout2_retrieval_queries_gold.jsonl
- 사람 손: 없음 (검수 없이 탐지 결과 그대로 적재)

## 단계별 시간 (초)

| 단계 | 초 |
|---|---|
| detect | 91.5 |
| load | 319.2 |
| search | 237.7 |
| answer | 658.8 |
| total | 1307.4 |

## 적재

```
{"docs": 756, "chunks": 921, "tokens": 2041, "persons": 731, "employees": 170, "guard_masked": 66}
```

## 검색

| 유형 | n | Hit@1 | Hit@5 | Recall@10 | MRR | Confuse@10 |
|---|---|---|---|---|---|---|
| ALL | 155 | 0.9097 | 0.9613 | 0.8615 | 0.9312 | 0.2875 |
| address_exact | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | - |
| address_partial | 10 | 0.9000 | 1.0000 | 0.9078 | 0.9250 | - |
| homonym_role | 8 | 0.3750 | 0.8750 | 0.7028 | 0.5938 | 0.2875 |
| hybrid_paraphrase | 15 | 0.8000 | 0.9333 | 0.9333 | 0.8556 | - |
| long_doc_person | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | - |
| name_topic | 29 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | - |
| name_variant | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | - |
| semantic_no_pii | 22 | 0.9545 | 0.9545 | 0.3186 | 0.9545 | - |
| staff_owner | 15 | 0.8000 | 0.8667 | 0.9273 | 0.8333 | - |
| value_exact | 20 | 1.0000 | 1.0000 | 0.9955 | 1.0000 | - |
| value_format_variant | 12 | 0.9167 | 0.9167 | 0.9167 | 0.9167 | - |

검색 시간 평균 1.514s · p95 2.081s

## 답변 (모델 gpt-6-luna)

| 항목 | 값 |
|---|---|
| 근거 적중 | 0.9419 |
| 인용 적중 | 0.0000 |
| 인용 정밀도 | - |
| 혼동 인용 | - |
| 못 찾음 | 0.0350 |
| 되물음 | 0.0774 |
| 오류 | 0/155 |
| 출력 재검사 (확인 안 된 토큰 / 원문 개인정보 걸린 답변) | 0 / 5 |
| **유출** (LLM 입력 / LLM 원 답변 / 최종 답변, 0 이어야 함) | 0 / 0 / 0 |
| 평균 시간 | search 1.3s, llm 2.9s, check 0.2s |
