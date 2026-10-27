# 실전 1 — 개인정보 탐지 방식 비교

작업 과정과 모든 결과는 [worklog.md](worklog.md)에 있어요. 이 문서는 폴더 안내와 자주 쓰는 명령만 담아요.

## 폴더 구성

```
실전프로젝트1/
├── report/               실전 1 보고서 (장마다 .md 원본 + .pdf 제출용)
│   ├── 0_요약_및_결론
│   ├── 1_데이터_관찰_리포트            ①
│   ├── 2_방식별_비교표                ②
│   ├── 3_오류_유형_분석                ③
│   ├── 4_탐지기_선정_근거              ④   (1부: dev·test, v8 최신 수치)
│   ├── 5_Holdout_데이터_관찰           ⑤   (2부: 멘토 holdout)
│   ├── 6_Holdout_성능                  ⑥   개선 전 점수
│   ├── 7_Holdout_오류_분석             ⑦
│   ├── 8_Holdout_개선_내역             ⑧   개선 전후 + dev·test 재확인
│   └── 9_부록_개선_과정과_데이터_사용_기록
│       PDF 다시 만들기: python -X utf8 tools/build_report.py (PowerShell에서 실행)
├── presentation/           발표 자료 (발표자료.html 원본 → 발표자료.pdf, 16:9 12장)
│       PDF 다시 만들기: python -X utf8 tools/build_slides.py (PowerShell에서 실행)
├── worklog.md              작업 기록 (결과·판단 근거 전부)
├── run_detect.py           탐지기 실행 → preds/ 에 예측 파일 저장
├── run_compare.py          채점기 (멘토 제공 holdout 버전, 수정하지 않음)
├── make_union.py           두 탐지기 결과 합치기 (합집합)
├── pii_detector.py         운영 탐지기(규칙 + atonlee)를 문서 1건 단위로 호출 — 실전 2 가명처리용
├── common.py               데이터 읽기·경로(ROOT, DATA)·공통 기능
├── postprocess.py          모델 경계 보정, 대표번호·장소 주소 제외
├── detectors/              탐지기
│   ├── rule_based.py         규칙 기반 (정규식 + Kiwi)
│   ├── token_cls.py          토큰 분류 모델 (KoELECTRA, PIILOT)
│   ├── llm.py                LLM (OpenAI)
│   ├── analyzers.py          형태소 분석기 교체용
│   ├── hf_ner.py             토큰 모델 래퍼 (박지훈 팀원)
│   └── llm_hcx.py, llm_openai.py  LLM 7종 탐지기 (baek 팀원, 비교 실험용)
├── holdout/                멘토 배포 holdout 187건 (holdout_labeled.jsonl, 받은 그대로)
├── data/                   정답 데이터
│   ├── pii_synth_finance_dev.jsonl / _test.jsonl   제공 데이터
│   ├── edge_cases.jsonl      엣지 케이스 64건
│   ├── generated/            일상 문서 1,983건 (synth_dev / synth_holdout, CSV 2개, 생성 원문 raw/)
│   │   + synth_fresh 1,775건(평가 전용), synth_hard_dev 379건(개선용) / synth_hard_eval 403건(평가용) (worklog 26~28장)
│   │   + generated_documents.csv / generated_entities.csv: 5세트를 합친 엑셀용 CSV, _work/: 생성 원문·중간 파일 (안내: data/generated/README.md)
│   ├── gold_by_label/        라벨별 정답 CSV (팀 분석용)
│   └── gold_by_label_holdout/  멘토 holdout 라벨별 정답 CSV
├── experiments/            보고서용 실험 (필요할 때 한 번씩)
│   ├── bootstrap_ci.py       점수 오차 범위
│   ├── template_experiment.py  처음 보는 문서 하락 폭 추정
│   ├── compare_analyzers.py  형태소 분석기 5종 비교
│   └── edge_report.py        엣지 케이스 종류별 채점
├── docs/team_baek/         baek 팀원 LLM 7종 비교 문서 (참고용)
├── tools/                  데이터 만들 때만 + LLM 비교 도구(merge_rules_llm, make_compare_table, eval_customer — baek 팀원)
│   ├── make_gold_by_label.py   data/gold_by_label/
│   ├── make_edge_cases.py      data/edge_cases.jsonl
│   ├── generate_synthetic.py   일상 문서 생성 (OpenAI, 비용 발생)
│   ├── parse_synthetic.py      생성 결과 → 정답 파일, 절반 분할
│   ├── qc_synthetic.py         생성 정답 검수 후보 목록
│   ├── repair_marks.py         생성 문서 표시 보정 (hard 세트)
│   ├── qc_fresh.py             평가 전용 세트 정답 검수 (문제 있는 문서 제외)
│   ├── split_hard.py           synth_hard 를 개선용/평가용 반으로
│   └── export_generated_csv.py 생성 데이터 5세트 → CSV 2개
├── preds/                  예측 파일 (<탐지기>_<데이터>.jsonl)
│   ├── rules_*, atonlee_*    규칙, 토큰 모델(atonlee) 단독
│   ├── combined/rules+atonlee_*   운영 탐지기 (규칙 + atonlee 전체 합집합)
│   ├── llm_dev/test_r1~3     LLM 3회 실행 (참고용, gpt-4.1-mini)
│   └── holdout_before/, holdout_after/  멘토 holdout 개선 전(v7)·후(v8) 예측
└── results/                채점 결과
    ├── final_<데이터>/       규칙·atonlee·규칙+atonlee 채점 (오류 목록 errors_*.csv 포함, dev·test는 LLM 포함)
    ├── holdout_before/, holdout_after/  멘토 holdout 개선 전·후 채점
    ├── version_history/v8·v9·v10/  버전별 세트 점수 (summary, 회귀 확인 근거)
    ├── system_comparison.csv  정확도·시간 비교표
    ├── analyzer_comparison.*  형태소 분석기 5종 비교
    └── synth_qc_candidates.csv 생성 데이터 정답 검수 후보
```

## 설치

```bash
pip install -r requirements.txt
```

API 키(LLM 실험용)는 프로젝트 맨 위 `.env`에 둬요 (깃에 올라가지 않음).

## 자주 쓰는 명령 (이 폴더에서 실행)

```bash
python -X utf8 run_detect.py --detector rules --split dev
```
`--detector`: `rules`, `koelectra_pii`, `piilot`, `llm` / `--split`: `dev`, `test`, `edge`, `synth_dev`, `synth_holdout`, `synth_fresh`, `synth_hard_dev`, `synth_hard_eval`, `holdout`(멘토)

```bash
python -X utf8 make_union.py --out preds/union_rules_koelectra_dev.jsonl preds/rules_dev.jsonl preds/koelectra_pii_dev.jsonl
```

```bash
python -X utf8 run_compare.py --gold data/pii_synth_finance_dev.jsonl --pred rules=preds/rules_dev.jsonl --pred union=preds/union_rules_koelectra_dev.jsonl --out results/dev_check
```

```bash
python -X utf8 experiments/edge_report.py --pred rules=preds/rules_edge.jsonl --pred model=preds/atonlee_edge.jsonl
```

규칙 / 토큰(atonlee) / 규칙+토큰 정확도·시간 비교표 (예측 파일이 먼저 있어야 함)
```bash
python -X utf8 experiments/compare_systems.py
```

**최종 결정: 탐지기 = 규칙(Kiwi) + atonlee 전체 합집합, LLM은 RAG 답변 생성에서만 사용** (worklog 15·17장). 현재 탐지기 버전 v10 (worklog 30장)

LLM 7종 비교표 (baek 팀원):
```bash
python -X utf8 tools/make_compare_table.py test --rank
```

## 지켜야 할 것
- `-X utf8`을 붙여 실행 (윈도우에서 결과를 파일로 넘길 때 인코딩 오류 방지)
- 규칙 개선은 `dev`, `edge`, `synth_dev`, `synth_hard_dev`만 보고. `test`·`synth_holdout`·`synth_fresh`·`synth_hard_eval`은 점수만 확인
- 멘토 holdout은 개선 전 점수를 먼저 기록(`preds/holdout_before/`)한 뒤 분석·개선. 개선 후 점수는 일반화 성능이 아님 (worklog 21~24장, 보고서 2부)
- API 키는 프로젝트 맨 위 `.env`에 (깃 업로드 제외됨)
