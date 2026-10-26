# 생성 데이터 (OpenAI gpt-4.1로 만든 가상 문서)

엑셀로 볼 때는 **CSV 2개**만 열면 돼요. 채점은 `.jsonl` 파일로 해요.

| 파일 | 내용 |
|---|---|
| `generated_documents.csv` | 5세트 문서 전부 (1줄 = 문서 1건, `set` 열로 세트 구분, 표시 포함 본문) |
| `generated_entities.csv` | 5세트 정답 전부 (1줄 = 개인정보 1개, 라벨·값·위치·앞뒤 문맥) |

| 세트 (jsonl) | 문서 | 정답 | 용도 | worklog |
|---|---|---|---|---|
| `synth_dev` | 989 | 1,643 | 1차 일상 문서 · **개선용** (오류를 보고 고침) | 10장 |
| `synth_holdout` | 994 | 1,732 | 1차 일상 문서 · **평가용** (점수만 봄) | 10장 |
| `synth_fresh` | 1,775 | 5,918 | 2차 금융 업무 문서 25종 · **평가 전용** (고치는 데 안 씀) | 26장 |
| `synth_hard_dev` | 379 | 1,892 | 어려운 표기 · **개선용** | 27·28장 |
| `synth_hard_eval` | 403 | 1,958 | 어려운 표기 · **평가용** | 27·28장 |

`_work/` 는 다시 만들 때만 필요한 중간 파일이에요 (생성 원문 `raw*/`, 검수 전 `*_all.jsonl`, 생성·검수·보정 보고서, 토큰 사용량).
만드는 순서: `generate_synthetic.py` → (hard만) `repair_marks.py` → `parse_synthetic.py` → `qc_fresh.py` → (hard만) `split_hard.py` → `export_generated_csv.py`
