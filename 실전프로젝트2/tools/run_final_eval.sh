#!/bin/bash
# 20장: 팀 비교용 최종 수치 (현재 코드, 추론 끔). holdout 은 outputs/final 에 이미 있음. 끝나면 데모용 holdout 다시 적재
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 RERANK=1
F='oneDNN|tensorflow|deprecated|WARNING'
O=outputs/final
for S in synth3 synth2; do
  D=data/$S
  python -m starter.ingest --data $D/${S}_labeled.jsonl --reset --detect-cache 2>&1 | grep -E '문서 |guard'
  echo "== $S 토큰 일관성"; python tools/eval_resolution.py $S 2>&1 | grep -E '미탐|쪼개짐|합쳐짐'
  python -m starter.run_queries --queries $D/${S}_retrieval_queries.jsonl --out $O/run_$S.jsonl --rerank 2>&1 | grep -v -E "$F"
  python eval/eval_holdout_retrieval.py --gold $D/${S}_retrieval_queries_gold.jsonl --run $O/run_$S.jsonl > $O/eval_$S.txt
  echo "== $S 검색"; sed -n 1,2p $O/eval_$S.txt
  echo "== $S RAG"; python eval/eval_rag_e2e.py --model HCX-007 --gold $D/${S}_retrieval_queries_gold.jsonl --corpus $D/${S}_labeled.jsonl --out $O/rag_e2e_$S.jsonl 2>&1 \
    | grep -v -E "$F" | grep -E '^질의|^type|^ALL|유출|평균'
done
python -m starter.ingest --reset --detect-cache 2>&1 | grep -E '문서 '
