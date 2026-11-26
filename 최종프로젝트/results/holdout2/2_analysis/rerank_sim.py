"""분석용: 재순위 ID 정규식만 5자리도 받게 바꿔(메모리 안에서만) 같은 DB 로 검색을 다시 돌려 점수 비교. 파일은 안 고침"""
import json, re, sys, types
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from app.rag import search as S
shim = types.SimpleNamespace(**{k: getattr(re, k) for k in dir(re) if not k.startswith('__')})
shim.findall = lambda p, s, *a: re.findall(r"[A-Z]\d{4,}" if p == r"[A-Z]\d{4}" else p, s, *a)
S.re = shim
import run_pipeline as rp
q = rp.read('../../holdout2/holdout2_retrieval_queries_gold.jsonl')
run, lat = rp.stage_search(q, 10, print)
rp.write_jsonl('outputs/holdout2_first/search_run_rerankfix.jsonl', run)
s = rp.score_search(q, run)
for k in ('ALL', 'homonym_role', 'hybrid_paraphrase', 'staff_owner', 'semantic_no_pii', 'address_partial'):
    print(k, {c: round(v, 4) for c, v in s[k].items() if c in ('Hit@1', 'Hit@5', 'MRR', 'Confuse@10')})
