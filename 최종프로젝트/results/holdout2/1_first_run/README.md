# Holdout2 첫 실행 (수정 없이, 최종 성능)

- 실행: 2026-11-26 14:45~15:07 (1,307초), 로컬 채점 DB `saebom_eval_db`(포트 5435) 를 `--reset` 으로 비우고 시작
- 코드: 커밋 `001d685` — Holdout2 를 받은 뒤 파이프라인·탐지·검색 코드는 고치지 않음 (그 커밋의 변경은 화면 업로드 빈 문서 건너뛰기뿐이고, 무인 파이프라인은 그 경로를 안 씀)
- 입력: `holdout2_public.jsonl` 문서 756건 / `holdout2_retrieval_queries_gold.jsonl` 질의 155개. 사람 손 없음
- 명령:
  ```
  DB_PORT=5435 python scripts/run_pipeline.py --docs holdout2_public.jsonl --queries holdout2_retrieval_queries_gold.jsonl --gold holdout2_retrieval_queries_gold.jsonl --reset --out outputs/holdout2_first
  python scripts/to_compare_pred.py outputs/holdout2_first/detect.jsonl outputs/holdout2_first/pred_compare.jsonl
  python ../../실전프로젝트1/run_compare.py --gold holdout2_labeled.jsonl --pred pipeline=outputs/holdout2_first/pred_compare.jsonl --out outputs/holdout2_first/detect_score
  ```
- 실행 결과: **멈춤 없이 끝까지 돎** (빈 문서 2건 H20345·H20437 포함 756건 적재)

## 점수

| 영역 | 지표 | 값 |
|---|---|---|
| 탐지 (`run_compare.py`) | strict P / R / F1 | 0.937 / 0.986 / 0.961 |
| | partial P / R / F1 | 0.939 / 0.989 / 0.963 |
| | 가리기 재현율 (mask_rec) | 0.992 |
| 검색 (재순위 켬) | Hit@1 / Hit@5 / Hit@10 | 0.910 / 0.961 / 0.968 |
| | MRR / Recall@10 / Confuse@10 | 0.931 / 0.861 / 0.288 |
| 답변 (gpt-6-luna) | 근거 적중 / 못 찾음 / 되물음 | 0.942 / 0.035 / 0.077 |
| | 인용 적중 / 인용 정밀도 / 혼동 인용 | **0.937 / 0.955 / 0.238** (아래 채점 주의 1) |
| 안전 | 원문 유출 LLM 입력 / 원 답변 / 최종 답변 | **8 / 0 / 0** (아래 채점 주의 2) |
| | 출력 재검사에 걸린 답변 (원문 개인정보 모양) | 5 |
| 시간 | 탐지 92 · 적재 319 · 검색 238 · 답변 659 초 | 합계 1,307초 |
| 적재 | 문서 / 청크 / 사람 / 직원 | 756 / 921 / 731 / 170 |

검색 유형별 (`report.md`): 약한 곳은 homonym_role Hit@1 0.375 (8개), hybrid_paraphrase 0.800, staff_owner 0.800. name_topic·name_variant·value_exact·address_exact·long_doc_person 은 1.000.

## 채점 주의 (파이프라인 자체가 아니라 채점 쪽)

1. **인용 적중이 `report.md` 에는 0.000 으로 나옴**: 채점 정규식이 문서 ID 를 4자리(`H0001`)로만 받는데 Holdout2 ID 는 5자리(`H20139`). 답변에는 `[H20139]` 처럼 제대로 인용돼 있음. 같은 답변 파일을 5자리도 받는 정규식으로 다시 채점한 값이 위 표 (답변 자체는 다시 만들지 않음)
2. **유출이 `report.md` 에는 0/0/0 으로 나옴**: 유출 검사는 탐지 정답의 원문 값과 비교하는데, 입력을 정답 없는 `public` 으로 줘서 비교할 값이 없었음. `labeled` 의 정답 값으로 같은 답변 파일을 다시 검사하면 **LLM 입력에 8건** (원 답변·최종 답변 0). 모두 탐지 미탐이 원인 — 로마자 이름(영문고객메일), 한글로 읽은 전화번호(챗봇상담로그) 등이 문서에서 안 가려진 채 LLM 에 들어감

## 막힌 곳·원인 후보 (분석은 다음 단계)

- 탐지 오탐 person 307건이 정밀도를 깎음 — 포인트이관신청 86·결재보고서 80·이의제기회신 41·해외결제차단해제 40 에 몰림. 그 외 주민번호 오탐 24·계좌 오탐 13 (decoy_numbers 문서 오탐 146)
- 탐지 미탐 person 46·phone 15 — 채무조정상담 19·챗봇상담로그 15·영문고객메일 12. 이름만 쓴 경우(form=given) 재현율 0.892
- **문서 ID 형식 의존 (실행 오류 성격)**: 재순위(`app/rag/search.py` `_rerank`)가 LLM 답에서 ID 를 `[A-Z]\d{4}` 로 뽑아 `H20139` → `H2013` 이 되어 **재순위가 조용히 효과 없음**. `app/rag/chat.py` 의 인용 정규식(`_CITE`, `_CITE_WORD`)도 4자리 기준 → '못 찾음' 판정·'[문서 X]' 표기 통일이 5자리 ID 에서 안 됨

## 파일

`report.md`·`scores.json`·`run.log`·`console.log` (파이프라인 그대로), `detect.jsonl`·`search_run.jsonl`·`answers.jsonl`·`employees.jsonl`·`doc_meta.jsonl`·`load_summary.json`, `pred_compare.jsonl`(채점기 형식), `detect_score/`(run_compare 결과·오류 CSV)
