import csv, json, sys
from collections import Counter, defaultdict
sys.path.insert(0, '.')
from app.pii import pii_pipeline
R = '../results/holdout2/1_first_run/detect_score/errors_pipeline.csv'
rows = list(csv.DictReader(open(R, encoding='utf-8-sig')))
docs = {d['id']: d for d in map(json.loads, open('../../holdout2/holdout2_public.jsonl', encoding='utf-8'))}
err = defaultdict(list)
for x in rows:
    if x['error_type'] in ('오탐', '경계 오류'):
        err[x['doc_id']].append(x)
src = Counter()
for did in list(err):
    sc = pii_pipeline.detect_scored(docs[did]['text'])['spans']
    for x in err[did]:
        for s in sc:
            if docs[did]['text'][s['start']:s['end']] == x['pred_text'] and s['label'] == x['label']:
                src[(x['label'], x['pred_text'], s['src'], 'flag' if 'flag' in s else 'noflag')] += 1
                break
for k, v in src.most_common(60):
    print(v, k)
