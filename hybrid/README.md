# hybrid — 결합 규칙기반(rules_best) + atonlee 하이브리드(hybrid_best)

기존 규칙기반(`detectors/rules.py`, kloid99 브랜치)과 hosung 규칙기반(`detectors/hosung_rules.py`, 팀장님 `rule_based.py` 사본)을 합쳐 개선한 규칙기반입니다. 이를 atonlee 토큰 모델과 결합한 하이브리드도 함께 들어 있습니다. 작성일 2026-10-26.

**Holdout 평가 보고서(⑤~⑧): [holdout_report.md](holdout_report.md)**

## 실행

```
cd hybrid
python -X utf8 run_detect.py --detector rules_best --split dev
python -X utf8 run_detect.py --detector hybrid_best --split test
python -X utf8 run_detect.py --detector hybrid_best --data data/holdout_labeled.jsonl
python -X utf8 run_compare.py --gold data/holdout_labeled.jsonl --pred hybrid_best=preds/hybrid_best_holdout_labeled.jsonl --out results/check
```

필요 패키지: `kiwipiepy==0.24.1`, `transformers`, `torch` (atonlee 모델은 처음 실행할 때 Hugging Face에서 내려받음).

| 환경변수 | 기본 | 설명 |
|---|---|---|
| `PII_HOLDOUT_FIXES` | `1` | `0`이면 holdout 오류를 보고 넣은 수정을 모두 끔 (holdout 공정 채점용) |
| `PII_HARD_FIXES` | `1` | `0`이면 synth_hard_dev 오류를 보고 넣은 수정(이름 뒤 호칭·조사 자르기, 괄호 없는 우편번호)을 끔 |
| `PII_HYBRID_MODE` | `union` | `names`면 모델에서 두 글자 이상 이름만 가져옴 |
| `PII_MODEL_NAME_FILTER` | `1` | `0`이면 모델 이름 거르기를 끔 |

## 구성

| 파일 | 내용 |
|---|---|
| `detectors/rules_best.py` | **결합 규칙기반**. hosung 규칙을 그대로 돌린 뒤, 겹치지 않는 곳에만 규칙을 더하고 오탐을 거름 (파일 맨 위 설명 참고) |
| `detectors/hybrid_best.py` | **하이브리드**. rules_best 결과를 우선으로 두고, 겹치지 않는 atonlee 결과(모든 라벨)를 거른 뒤 더함 |
| `detectors/hosung_rules.py` | hosung `rule_based.py` 사본 (import 2줄만 바꿈, 로직은 원본 그대로) |
| `detectors/analyzers.py` | hosung 형태소 분석기 어댑터 사본 |
| `detectors/rules.py`, `ner.py`, `base.py` | 기존 저장소 파일 사본 (rules_best·hybrid_best가 사용) |
| `run_detect.py`, `run_compare.py` | 실행기·채점기 (채점기는 holdout 태그를 지원하는 `holdout/run_compare.py`) |
| `data/` | dev·test·holdout + hosung 추가 데이터(edge_cases, synth_dev, synth_holdout) |
| `preds/` | 선정 탐지기(`selected_hosung+atonlee_*`), `*_nofix_*`(holdout 수정 끔), 최종(`rules_best_*`, `hybrid_best_*`) 예측 |
| `results/` | 데이터별 채점 결과 (`compare_<데이터>/summary.csv`, `per_label.csv`, `hard_cases.csv`, `errors_*.csv`, `report.txt`), holdout 데이터 관찰 |

## 결과 (strict F1 · mask_rec)

| 시스템 | dev | test | holdout | synth_holdout |
|---|---|---|---|---|
| 기존 rules | 1.000 · 1.000 | 1.000 · 1.000 | 0.833 · 0.855 | 0.866 · 0.914 |
| hosung rules | 0.998 · 0.999 | 0.993 · 0.997 | 0.923 · 0.961 | 0.954 · 0.972 |
| rules_best_nofix | 1.000 · 1.000 | 0.998 · 1.000 | 0.932 · 0.964 | 0.964 · 0.976 |
| **rules_best** | 1.000 · 1.000 | 0.998 · 1.000 | 0.961 · 0.983 | 0.965 · 0.976 |
| 선정 탐지기 (hosung rules + atonlee) | 0.994 · 1.000 | 0.982 · 1.000 | 0.944 · 0.993 | 0.943 · 0.988 |
| hybrid_best_nofix | 0.997 · 1.000 | 0.995 · 1.000 | 0.952 · 0.991 | 0.950 · 0.988 |
| **hybrid_best** | 0.997 · 1.000 | 0.997 · 1.000 | 0.994 · 0.999 | 0.950 · 0.988 |

- holdout 점수에서 `_nofix`가 아닌 행은 holdout 오류를 보고 고친 결과입니다. 처음 보는 데이터에서 기대할 성능은 `_nofix` 행과 synth_holdout 쪽이 더 정확합니다.
- synth_holdout도 마지막 회귀 확인 때 오류를 일부 봤습니다. 자세한 데이터 사용 기록은 보고서 ⑧에 있습니다.
- 처리 시간은 hybrid_best가 holdout에서 문서당 약 28ms입니다(같은 조건 연속 측정). 컴퓨터 부하에 따라 크게 흔들려서 `summary.csv`의 ms 값은 참고용입니다.
