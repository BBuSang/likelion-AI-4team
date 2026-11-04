#!/bin/bash
# synth2(새 평가 데이터) 전체 평가. 한 번만 돌리고 결과를 보고 고치지 않는다 (worklog 14장).
#   bash tools/run_synth2_eval.sh
# 끝나면 데모용으로 holdout 을 다시 적재한다 (임베딩 캐시가 있어 빠름).
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
F='oneDNN|tensorflow|deprecated'
D=data/synth2
DATA=$D/synth2_labeled.jsonl
Q=$D/synth2_retrieval_queries.jsonl
G=$D/synth2_retrieval_queries_gold.jsonl
O=outputs/synth2
mkdir -p $O

echo "===== 1. 토큰 일관성 (DB·API 없음)"
python tools/eval_resolution.py synth2 --off 2>&1 | grep -E '판단|쪼개짐|합쳐짐|미탐'
python tools/eval_resolution.py synth2 2>&1 | grep -E '판단|쪼개짐|합쳐짐|미탐'

retrieval () {  # 이름, 추가 옵션들
  name=$1; shift
  python -m starter.run_queries --queries $Q --out $O/run_$name.jsonl "$@" 2>&1 | grep -v -E "$F"
  python eval/eval_holdout_retrieval.py --gold $G --run $O/run_$name.jsonl > $O/eval_$name.txt
  echo "[$name]"; head -2 $O/eval_$name.txt | tail -1
}

echo "===== 2. 동명이인 구분 켬으로 적재 + 검색"
python -m starter.ingest --data $DATA --reset --detect-cache 2>&1 | grep -v -E "$F" | grep -E '문서|충돌|Error'
python eval/check_token_consistency.py --gold $D/synth2_persons_gold.jsonl --pred outputs/pseudo_synth2_labeled.jsonl \
  --out $O/token_consistency_resolve.csv | grep -E '쪼개짐|합쳐짐|미탐'
retrieval v1_semantic --mode semantic
retrieval v3_full
retrieval v3_full_rerank --rerank
SEARCH_DOCTYPE=0 retrieval v3_nodoctype_rerank --rerank

echo "===== 3. 동명이인 구분 끔으로 적재 + 검색"
python -m starter.ingest --data $DATA --reset --detect-cache --no-resolve 2>&1 | grep -v -E "$F" | grep -E '문서|충돌|Error'
retrieval v2_full_rerank --rerank

echo "===== 4. 구분 켬으로 다시 적재 + RAG 전체 테스트"
python -m starter.ingest --data $DATA --reset --detect-cache 2>&1 | grep -v -E "$F" | grep -E '문서|충돌|Error'
RERANK=1 python eval/eval_rag_e2e.py --model HCX-007 --gold $G --corpus $DATA --out $O/rag_e2e_v3_full.jsonl 2>&1 \
  | grep -v -E "$F" | grep -v -E '^\[[0-9]+/'

echo "===== 5. 데모용 holdout 다시 적재"
python -m starter.ingest --reset --detect-cache 2>&1 | grep -v -E "$F" | grep -E '문서|충돌|Error'
echo "===== 끝"
