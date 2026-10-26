# hybrid — final_model (규칙 + atonlee 하이브리드)

금융 사내 문서에서 개인정보 7종을 찾는 최종 탐지기입니다. 규칙 기반 결과를 우선으로 두고, 겹치지 않는 atonlee 토큰 모델 결과를 걸러서 더합니다. 작성일 2026-10-26.

**검사 수치(보고서 ⑤~⑧ 표 대응): [final_model_수치.md](final_model_수치.md)**

## 실행

```
cd hybrid
python -X utf8 run_detect.py --detector final_model --split dev
python -X utf8 run_detect.py --detector final_model --data data/holdout_labeled.jsonl
python -X utf8 run_compare.py --gold data/holdout_labeled.jsonl --pred final_model=preds/final_model_holdout_labeled.jsonl --out results/check
```

필요 패키지: `kiwipiepy==0.24.1`, `transformers`, `torch` (atonlee 모델은 처음 실행할 때 Hugging Face에서 내려받음).

| 환경변수 | 기본 | 설명 |
|---|---|---|
| `PII_HOLDOUT_FIXES` | `1` | `0`이면 holdout 오류를 보고 넣은 수정(hosung v8 규칙 재구현 포함)을 모두 끔 |
| `PII_HARD_FIXES` | `1` | `0`이면 synth_hard_dev 오류를 보고 넣은 수정(이름 뒤 호칭·조사 자르기, 괄호 없는 우편번호)을 끔 |
| `PII_HYBRID_MODE` | `union` | `names`면 모델에서 두 글자 이상 이름만 가져옴 |
| `PII_MODEL_NAME_FILTER` | `1` | `0`이면 모델 이름 거르기를 끔 |

## 구성

`final_model`은 아래 순서로 다른 탐지기 파일을 불러 씁니다. 하나라도 지우면 실행되지 않습니다.

| 파일 | 내용 |
|---|---|
| `detectors/final_model.py` | **최종 탐지기**. hybrid_best + hosung v8 규칙 중 빠져 있던 것(worklog 24-1 설명대로 다시 구현. v8 코드는 저장소에 없음) |
| `detectors/hybrid_best.py` | 규칙 + atonlee 결합, 모델 결과 후처리(hosung `postprocess.py` 사본)와 모델 이름 거르기 |
| `detectors/rules_best.py` | 결합 규칙기반: hosung 규칙 + 기존 kloid99 rules 일부 + 이름·형식 보강 |
| `detectors/hosung_rules.py` | hosung `rule_based.py` v7 사본 (import 2줄만 바꿈) |
| `detectors/analyzers.py` | hosung 형태소 분석기 어댑터 사본 |
| `detectors/rules.py` | 기존 kloid99 규칙기반 (rules_best가 일부 사용) |
| `detectors/ner.py` | 토큰 분류 모델 래퍼 (atonlee) |
| `detectors/base.py`, `__init__.py` | 공통 후처리 (run_detect가 사용) |
| `run_detect.py`, `run_compare.py` | 실행기·채점기 (holdout 태그 지원판) |
| `data/` | dev · test · holdout · edge_cases · synth_dev · synth_holdout · synth_hard_dev · synth_hard_eval · synth_fresh |
| `preds/final_model_*.jsonl` | final_model 예측 (9개 세트) |
| `preds/final_rules_*.jsonl`, `preds/final_token_*.jsonl` | final_model의 규칙 부분·토큰 부분 예측 (dev·test·holdout) |
| `preds/selected_hosung+atonlee_*.jsonl` | 개선 전 비교 기준: 팀 선정 탐지기 v7 (hosung 규칙 v7 + atonlee) |
| `results/final_model/` | 채점 결과: `compare_<데이터>/`(v7 · hybrid_best · final_model), `components_holdout/`(구성 요소별), `make_components.py` |

## 결과 요약 (strict F1 · mask_rec)

| 데이터 | 개선 전 (v7) | **final_model** | 참고: hosung v8 |
|---|---|---|---|
| dev | 0.994 · 1.000 | **0.997 · 1.000** | 0.994 · 1.000 |
| test | 0.982 · 1.000 | **0.999 · 1.000** | 0.986 · 1.000 |
| 멘토 holdout | 0.944 · 0.993 | **0.995 · 1.000** | 0.995 · 1.000 |
| 일상 문서 평가용 (synth_holdout) | 0.943 · 0.988 | **0.961 · 0.988** | 0.956 · 0.988 |
| synth_fresh | 0.852 · 0.977 | **0.920 · 0.983** | 0.855 · 0.976 |
| synth_hard_eval | 0.767 · 0.941 | **0.816 · 0.934** | 0.781 · 0.933 |

- **holdout 점수는 holdout 오류를 보고 고친 값입니다.** 일반화 성능으로 보기 어렵고, 처음 보는 데이터에 가까운 것은 synth_fresh와 synth_hard_eval입니다.
- **synth_hard_eval에서는 mask_rec가 개선 전보다 낮습니다**(0.941 → 0.934). 일반 단어와 같은 이름을 놓친 탓입니다.
- **hosung v9는 hard_eval이 더 높습니다**(F1 0.861). 자세한 내용은 [final_model_수치.md](final_model_수치.md)에 있습니다.
- **처리 시간**: holdout에서 문서당 약 29~35ms입니다(PC 부하로 흔들림).
