"""final_model 의 구성 요소별 예측: 규칙 부분(final_rules), 토큰 부분(final_token). 결합(final_model)은 run_detect 로 만든 것을 씀."""
import json, sys, time
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from detectors import final_model as FM, hybrid_best as HB, rules_best as R
from detectors.base import postprocess
FM.load()

def rules_part(text):
    ents = R.detect(text)
    if R.HOLDOUT_FIXES:
        ents = FM.v8_drop_speaker_items(text, ents)
        ents += FM.v8_rule_names(text, ents)
    return ents

def token_part(text):
    out = []
    for m in HB.model_detect(text):
        m = HB.trim_model_name(text, m)
        if m["label"] == "person" and HB.FILTER and not HB.model_name_ok(text, m):
            continue
        if m["label"] in HB.NUMBER_LABELS and not HB.model_number_ok(text, m):
            continue
        if R.HOLDOUT_FIXES and not FM.v8_model_ok(text, m):
            continue
        if R.HARD_FIXES:
            m = R.trim_name_tail(text, m)
        out.append(m)
    return out

gold, outdir, tag = sys.argv[1], sys.argv[2], sys.argv[3]
docs = [json.loads(l) for l in open(gold, encoding="utf-8") if l.strip()]
for name, fn in [("final_rules", rules_part), ("final_token", token_part)]:
    fn(docs[0]["text"])
    with open(f"{outdir}/{name}_{tag}.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            t0 = time.perf_counter(); e = postprocess(d["text"], fn(d["text"])); ms = (time.perf_counter() - t0) * 1000
            f.write(json.dumps({"id": d["id"], "entities": e, "ms": round(ms, 2)}, ensure_ascii=False) + "\n")
    print(name, tag, "done")
