#!/bin/bash
# 16장 최종 평가. synth3(잠금 데이터)는 여기서 처음이자 한 번만 본다. 결과를 보고 고치지 않는다.
#   bash tools/run_synth3_eval.sh > outputs/synth3_eval.log
# 0) 개발 세트(holdout·synth2)로 최종 설정 확인 → 1~4) synth3 → 5) 데모용 holdout 다시 적재
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
F='oneDNN|tensorflow|deprecated|WARNING'
O=outputs/synth3
mkdir -p $O

retrieval () {  # 출력 폴더, 질의 파일, 이름, 추가 옵션들 (환경변수는 앞에)
  out=$1; q=$2; name=$3; shift 3
  python -m starter.run_queries --queries $q --out $out/run_$name.jsonl "$@" 2>&1 | grep -v -E "$F" > /dev/null
  python eval/eval_holdout_retrieval.py --gold ${q%.jsonl}_gold.jsonl --run $out/run_$name.jsonl > $out/eval_$name.txt
  echo "[$name]"; sed -n 2p $out/eval_$name.txt
}
ingest () { python -m starter.ingest "$@" --reset --detect-cache 2>&1 | grep -v -E "$F" | grep -E '문서 |guard|Error'; }

echo "===== 0. 개발 세트 최종 설정 확인 (재순위 켬)"
ingest
retrieval $O data/holdout_retrieval_queries.jsonl dev_holdout_final --rerank
ingest --data data/synth2/synth2_labeled.jsonl
retrieval $O data/synth2/synth2_retrieval_queries.jsonl dev_synth2_final --rerank

D=data/synth3
Q=$D/synth3_retrieval_queries.jsonl
echo "===== 1. synth3 탐지 + 토큰 일관성"
python tools/detect_cache.py $D/synth3_labeled.jsonl 2>&1 | grep '→'
python - <<'PY'
import json, collections
docs = [json.loads(l) for l in open("data/synth3/synth3_labeled.jsonl", encoding="utf-8")]
cache = {json.loads(l)["id"]: json.loads(l)["spans"] for l in open("outputs/detect_synth3_labeled.jsonl", encoding="utf-8")}
tot, miss, fp = collections.Counter(), collections.Counter(), collections.Counter()
for d in docs:
    sp = cache[d["id"]]
    for e in d["entities"]:
        tot[e["label"]] += 1
        miss[e["label"]] += not any(s["start"] < e["end"] and e["start"] < s["end"] for s in sp)
    for s in sp:
        fp[s["label"]] += not any(s["start"] < e["end"] and e["start"] < s["end"] for e in d["entities"])
print("탐지 누락(라벨별):", {k: f"{miss[k]}/{tot[k]}" for k in tot})
print("정답과 안 겹치는 예측(오탐 후보):", dict(fp))
PY
python tools/eval_resolution.py synth3 --off 2>&1 | grep -E '판단|쪼개짐|합쳐짐|미탐'
python tools/eval_resolution.py synth3 2>&1 | grep -E '판단|쪼개짐|합쳐짐|미탐'
python tools/homonym_cases.py synth3 2>&1 | tail -1

echo "===== 2. synth3 적재 + 유출 검사"
ingest --data $D/synth3_labeled.jsonl
RERANK=1 python tools/leak_check.py --gold $D/synth3_retrieval_queries_gold.jsonl --corpus $D/synth3_labeled.jsonl \
  --out $O/leak_check.jsonl 2>&1 | grep -v -E "$F" | grep -v '^\['

echo "===== 3. synth3 검색"
retrieval $O $Q v1_semantic --mode semantic
retrieval $O $Q final_rerank --rerank
DOC_TYPE_MODE=dict retrieval $O $Q doctype_dict_rerank --rerank
SEARCH_DOCTYPE=0 retrieval $O $Q doctype_off_rerank --rerank
retrieval $O $Q final_norerank

echo "===== 4. synth3 RAG 전체 (HCX-007, 되묻기 켬)"
RERANK=1 python eval/eval_rag_e2e.py --model HCX-007 --gold $D/synth3_retrieval_queries_gold.jsonl --corpus $D/synth3_labeled.jsonl \
  --out $O/rag_e2e_final.jsonl 2>&1 | grep -v -E "$F" | grep -v -E '^\[[0-9]+/'

echo "===== 5. 데모용 holdout 다시 적재"
ingest
echo "===== 끝"
