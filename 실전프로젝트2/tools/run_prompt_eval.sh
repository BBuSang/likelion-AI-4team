#!/bin/bash
# 19장: HCX-007 프롬프트 수정 + 추론 끔/켬 비교 (RAG 전체). 사용: bash tools/run_prompt_eval.sh synth3|holdout
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 RERANK=1
SET=$1
if [ "$SET" = holdout ]; then G=data/holdout_retrieval_queries_gold.jsonl; C=data/holdout_labeled.jsonl
else G=data/$SET/${SET}_retrieval_queries_gold.jsonl; C=data/$SET/${SET}_labeled.jsonl; fi
F='oneDNN|tensorflow|deprecated|WARNING'
run () { LLM_REASONING_EFFORT=$2 python eval/eval_rag_e2e.py --model HCX-007 --gold $G --corpus $C --out outputs/llm/p19_${SET}_$1.jsonl 2>&1 \
  | grep -v -E "$F" | grep -v -E '^\[[0-9]+/' > outputs/llm/p19_${SET}_$1.txt; }
run none none
run default ""
