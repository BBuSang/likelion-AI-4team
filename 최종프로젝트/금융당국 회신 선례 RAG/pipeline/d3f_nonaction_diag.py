"""D3 후속 진단 (읽기 전용): KURE-v1이 비조치의견서에서 법령해석보다 낮은 이유. dev1·dev2·dev3만 사용(테스트셋은 읽지 않는다).
사용: python d3f_nonaction_diag.py"""
import re
import sys
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

from common import read_jsonl
from d2_baseline import load
from d3_eval import load_arm
from d3b_eval import qvec_fn

arm = load_arm("chunks_d3.jsonl", "embeddings_d3_kure-v1.npy")
chunks = arm["chunks"]
cases = {r["case_key"]: r for r in read_jsonl("cases_enriched.jsonl") if r["indexable"]}
rep = {g["group_id"]: g["representative_case_key"] for g in read_jsonl("case_groups.jsonl")}
gcase = {c["group_id"]: c["case_key"] for c in chunks}              # 묶음 → 사례 하나
norm = lambda t: re.sub(r"[\s()\[\]·,.\-「」『』\"']", "", re.sub(r"\((?:연장|재연장|\d차)[^)]*\)", "", t or ""))

# A. 코퍼스 구조: 구분별
by_case = defaultdict(set)
for c in chunks:
    by_case[c["case_key"]].add(c["chunk_type"])
title_groups = defaultdict(set)
for ck, r in cases.items():
    if ck in by_case:
        title_groups[norm(r["title"])].add(r["case_key"])
stat = defaultdict(Counter)
for ck, r in cases.items():
    if ck not in by_case:
        continue
    g = r["gubun"]
    stat[g]["사례"] += 1
    stat[g]["질의 카드 있음"] += "question_card" in by_case[ck]
    stat[g]["규제 정비 카드(2016)"] += "rule_card" in by_case[ck]
    stat[g]["같은 제목(연장 표기 뺀) 사례 2개 이상"] += len(title_groups[norm(r["title"])]) >= 2
    stat[g]["제목에 '연장'"] += "연장" in (r["title"] or "")
    stat[g]["회답 20자 미만"] += len((r["answer"] or "")) < 20
print("[A] 구분별 구조 (D3 청크가 있는 사례)")
for g, s in stat.items():
    n = s["사례"]
    print(f"  {g}: {n}건 | " + " | ".join(f"{k} {v} ({v / n:.0%})" for k, v in s.items() if k != "사례"))

# B. dev1·dev2·dev3 비조치 vs 법령해석 (KURE-v1)
qv = qvec_fn("kure-v1")
rows = []
for name in ("dev1", "dev2", "dev3"):
    for r in load(name):
        sim = arm["E"] @ qv(r["question"])
        best = np.full(len(arm["uniq"]), -1e9, dtype=np.float32)
        np.maximum.at(best, arm["gi"], sim)
        gi = int(np.flatnonzero(arm["uniq"] == r["gold_group_id"])[0])
        rank = int((best > best[gi]).sum()) + 1
        top = int(arm["uniq"][int(np.argmax(best))])
        gold_ck = gcase[r["gold_group_id"]]
        gold_types = by_case[gold_ck]
        same_topic = top != r["gold_group_id"] and norm(cases[gold_ck]["title"]) == norm(cases[gcase[top]]["title"])
        near_title = len(title_groups[norm(cases[gold_ck]["title"])]) >= 2
        rows.append({"set": name, "qid": r["qid"], "gubun": r["gubun"], "tag": r["tag"], "rank": rank, "doc_class": r["doc_class"] if "doc_class" in r else cases[gold_ck]["doc_class"],
                     "has_card": "question_card" in gold_types, "rule": "rule_card" in gold_types, "same_title_top": same_topic, "dup_title_in_corpus": near_title,
                     "answer_type": cases[gold_ck]["answer_type"], "gold_title": cases[gold_ck]["title"], "top_title": cases[gcase[top]]["title"]})
print("\n[B] dev1+dev2+dev3 구분별 Hit@1 (n)")
for g in ("법령해석", "비조치의견서"):
    rr = [x for x in rows if x["gubun"] == g]
    print(f"  {g}: n={len(rr)} Hit@1 {sum(x['rank'] == 1 for x in rr) / len(rr):.3f} Hit@5 {sum(x['rank'] <= 5 for x in rr) / len(rr):.3f}")
nb = [x for x in rows if x["gubun"] == "비조치의견서"]
print("  비조치 안에서:")
for label, f in (("정답에 질의 카드 있음", lambda x: x["has_card"] and not x["rule"]), ("질의 카드 없음(판단 본문뿐)", lambda x: not x["has_card"] and not x["rule"]), ("2016 규제 정비 카드", lambda x: x["rule"]),
                 ("코퍼스에 같은 제목 사례가 또 있음", lambda x: x["dup_title_in_corpus"]), ("같은 제목 사례 없음", lambda x: not x["dup_title_in_corpus"])):
    rr = [x for x in nb if f(x)]
    if rr:
        print(f"    {label}: n={len(rr)} Hit@1 {sum(x['rank'] == 1 for x in rr) / len(rr):.2f} Hit@5 {sum(x['rank'] <= 5 for x in rr) / len(rr):.2f}")
lg = [x for x in rows if x["gubun"] == "법령해석"]
for label, f in (("법령해석: 같은 제목 사례가 또 있음", lambda x: x["dup_title_in_corpus"]), ("법령해석: 같은 제목 사례 없음", lambda x: not x["dup_title_in_corpus"])):
    rr = [x for x in lg if f(x)]
    print(f"    {label}: n={len(rr)} Hit@1 {sum(x['rank'] == 1 for x in rr) / len(rr):.2f}")
miss_nb = [x for x in nb if x["rank"] > 1]
print(f"\n[C] 비조치 1위 아님 {len(miss_nb)}/{len(nb)}: 1위가 정답과 제목이 같음(다른 묶음) {sum(x['same_title_top'] for x in miss_nb)}, 정답 제목이 코퍼스에 중복 {sum(x['dup_title_in_corpus'] for x in miss_nb)}, 질의 카드 없음 {sum((not x['has_card']) and not x['rule'] for x in miss_nb)}, 규제 정비 {sum(x['rule'] for x in miss_nb)}")
print(f"    법령해석 1위 아님 {len([x for x in lg if x['rank'] > 1])}/{len(lg)}: 정답 제목 중복 {sum(x['dup_title_in_corpus'] for x in lg if x['rank'] > 1)}")
for x in miss_nb:
    print(f"  [{x['qid']}] {x['rank']}위 카드{'O' if x['has_card'] else 'X'}{' 규제정비' if x['rule'] else ''}{' 동일제목중복' if x['dup_title_in_corpus'] else ''} | 정답: {x['gold_title'][:40]} | 1위: {x['top_title'][:40]}")
