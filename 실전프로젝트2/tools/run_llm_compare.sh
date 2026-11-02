#!/bin/bash
# 17장: LLM 제공자 비교 (같은 코드·같은 검색). 사용: bash tools/run_llm_compare.sh openai|clova [holdout|synth3]
# 해당 데이터가 DB 에 적재돼 있어야 함
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 RERANK=1
F='oneDNN|tensorflow|deprecated|WARNING'
SET=${2:-holdout}
if [ "$SET" = holdout ]; then G=data/holdout_retrieval_queries_gold.jsonl; C=data/holdout_labeled.jsonl
else G=data/$SET/${SET}_retrieval_queries_gold.jsonl; C=data/$SET/${SET}_labeled.jsonl; fi
run () {  # 모델 이름, 제공자
  LLM_PROVIDER=$2 OPENAI_LLM_MODEL=$1 python eval/eval_rag_e2e.py --model $1 --gold $G --corpus $C \
    --out outputs/llm/rag_e2e_${SET}_$1.jsonl 2>&1 | grep -v -E "$F" | grep -v -E '^\[[0-9]+/' > outputs/llm/score_${SET}_$1.txt
}
if [ "$1" = openai ]; then run gpt-4.1-mini openai; run gpt-4.1 openai; else run HCX-007 clova; fi
