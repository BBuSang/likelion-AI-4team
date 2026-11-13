"""X5 현장건의 별도 후보군 메모리 검색 시험 (사용자 결정 11-11: 섞지 않고 별도로 둔다). DB·rag.py는 바꾸지 않는다.
방식: 기존 회신 상위 5(rag.search, 지금과 동일)는 그대로 두고, 현장건의 임베딩(embeddings_exmnt_kure-v1.npy)에서 따로 묶음 상위 k를 찾아 붙인다.
보고 항목 (결과를 보기 전에 정함). 답변(LLM) 수준 평가는 이번에 하지 않는다
1. 현장건의가 정답인 새 개발 질문 devF 40개(합성, 개발 전용)에서 현장건의 후보군 Hit@1·2·3·5
2. 기존 개발셋(합성 150, dev1~5 = 460문항)과 답변불가 20에서 현장건의 1위 점수 분포, 기존 회신 1위보다 높은 비율(섞었다면 위를 차지했을 비율)
3. 붙이는 조건 비교: 항상 k=1·2 / 현장건의 1위 점수가 기준값 이상일 때만 / 기존 상위 5의 5위 점수 이상일 때만
   → devF에서 정답을 붙이는 비율, 답변불가·기존 개발셋에서 붙는 비율
devF 생성: d1_dev5_test5의 생성·검수 함수를 쓰되 정답 후보는 현장건의(검색 대상, 묶음 대표, 사유 80자 이상). 유형 4가지 × 10. 검수(답이 되는가, 결론 노출)를 통과한 것만
출력: data/eval/devF.jsonl (캐시 _devF_raw.jsonl), data/eval/x5_exmnt_search_test.json
"""
import json
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import d1_dev5_test5 as gen
import rag
from common import DATA, OUT, manifest, read_jsonl

EVAL = DATA / "eval"
SEED = 20261111
RAW = EVAL / "_devF_raw.jsonl"
DEV = ("rewrite_150", "dev1", "dev2", "dev3", "dev4", "dev5")
TYPES = {"실무 상황형": "새 상품·업무를 준비하는 담당자가, 이 자료가 다루는 규제가 그 업무에 어떻게 적용되는지 묻는 질문으로 쓰세요. 상황(대상·방식·조건)을 구체적으로 쓰세요.",
         "우회 서술형": "법령·제도 이름이나 전문 용어를 거의 쓰지 말고, 실제 업무에서 막힌 상황을 평범한 말로 설명하며 묻는 형태로 쓰세요.",
         "구어체형": "영업점 직원이 메신저에 급히 치듯 짧은 구어체로 쓰세요. 줄임말, 불완전한 문장을 써도 됩니다.",
         "검색어형": "문장이 아니라 검색창에 넣는 핵심어 3~6개를 띄어쓰기로 나열하세요. 조사·서술어는 쓰지 마세요."}
LEN = {"실무 상황형": (60, 220), "우회 서술형": (80, 220), "구어체형": (30, 120), "검색어형": (10, 60)}
N_PER, EXTRA = 10, 1.6
NOTE = "\n- 아래 자료는 업계 건의에 대한 금융당국 검토 결과입니다. '건의가 있었는지'를 묻지 말고, 직원이 그 규제가 지금 업무에 어떻게 적용되는지(해도 되는지, 완화됐는지)를 묻는 질문으로 쓰세요."


def build_items():
    cases = {r["case_key"]: r for r in read_jsonl("exmnt_cases.jsonl")}
    groups = read_jsonl("exmnt_groups.jsonl")
    pool = [cases[g["representative_case_key"]] for g in groups]
    pool = [r for r in pool if len(r["answer"] or "") >= 80 and len(r["question"] or "") >= 80]
    gid = {g["representative_case_key"]: g["group_id"] for g in groups}
    rng = random.Random(SEED)
    rng.shuffle(pool)
    cat_n, items, taken = Counter(), [], set()
    for tag in TYPES:
        for i in range(round(N_PER * EXTRA)):
            x = min((r for r in pool if r["case_key"] not in taken), key=lambda r: cat_n[r["category_top"]])
            taken.add(x["case_key"])
            cat_n[x["category_top"]] += 1
            items.append({"tag": tag, "rank": i, "gold_group_id": gid[x["case_key"]], "source_case_key": x["case_key"], "gubun": "현장건의", "category": x["category_top"],
                          "answer_type": x["answer_type"], "reply_date": x["reply_date"], "_title": x["title"], "_question": x["question"], "_answer": f"검토결과: {x['answer_type']}\n{x['answer']}"})
    return items


def write_q(item):
    lo, hi = LEN[item["tag"]]
    src = "\n".join(f"{n}: {t}" for n, t in (("제목", item["_title"]), ("건의내용", item["_question"][:1500]), ("당국 검토", item["_answer"][:900])) if t)
    full = "\n".join((item["_title"], item["_question"], item["_answer"]))
    system = gen.BASE.format(lo=lo, hi=hi, guide=TYPES[item["tag"]]) + NOTE
    extra, q, copied = "", None, None
    for _ in range(3):
        q = (gen.call_json(system + extra, src).get("question") or "").strip()
        copied = gen.longest_copy(q, full)
        if q and lo <= len(q) <= hi and copied <= gen.MAX_COPY:
            break
        extra = f"\n- 앞선 답은 {len(q)}자, 원문과 같은 구간 {copied}자였습니다. {lo}~{hi}자, 같은 구간 {gen.MAX_COPY}자 이하로 다시 쓰세요."
        q = None
    out = {k: v for k, v in item.items() if not k.startswith("_")}
    if not q:
        return {**out, "question": None, "check": None}
    chk = gen.call_json(gen.CHECK, f"[질문]\n{q}\n\n[회신]\n제목: {item['_title']}\n질의요지: {item['_question'][:1500]}\n회답: {item['_answer'][:1200]}", "medium")
    return {**out, "question": q, "longest_copy": copied, "check": {"answerable": chk.get("answerable"), "leak": chk.get("leak"), "reason": chk.get("reason")}}


def make_devF():
    items = build_items()
    done = {x["source_case_key"]: x for x in map(json.loads, RAW.open(encoding="utf-8"))} if RAW.exists() else {}
    todo = [dict(x) for x in items if x["source_case_key"] not in done]
    with ThreadPoolExecutor(5) as ex, RAW.open("a", encoding="utf-8") as f:
        for r in ex.map(write_q, todo):
            done[r["source_case_key"]] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows, drop = [], Counter()
    for tag in TYPES:
        ok = []
        for x in sorted((i for i in items if i["tag"] == tag), key=lambda i: i["rank"]):
            r = done[x["source_case_key"]]
            if not r["question"]:
                drop["생성 실패"] += 1
            elif r["check"]["answerable"] is not True:
                drop["검수: 답이 안 됨"] += 1
            elif r["check"]["leak"] is not False:
                drop["검수: 결론 노출"] += 1
            else:
                ok.append(r)
        rows += ok[:N_PER]
    random.Random(SEED).shuffle(rows)
    with (EVAL / "devF.jsonl").open("w", encoding="utf-8") as f:
        for no, x in enumerate(rows, 1):
            f.write(json.dumps({"qid": f"DEVF-{no:03d}", "split": "dev", **{k: v for k, v in x.items() if k != "rank"}}, ensure_ascii=False) + "\n")
    return rows, dict(drop), len(todo)


def main():
    rows, drop, called = make_devF()
    print(f"devF {len(rows)}문항 (생성 호출 {called}, 탈락 {drop}) 유형 {dict(Counter(r['tag'] for r in rows))}", flush=True)
    chunks = read_jsonl("exmnt_chunks.jsonl")
    V = np.load(OUT / "embeddings_exmnt_kure-v1.npy")
    cgid = np.array([c["group_id"] for c in chunks])
    uniq, inv = np.unique(cgid, return_inverse=True)

    def ex_rank(qv):
        s = V @ qv
        best = np.full(len(uniq), -1.0)
        np.maximum.at(best, inv, s)
        order = np.argsort(-best)
        return [(int(uniq[i]), float(best[i])) for i in order[:10]]

    sets = {"devF": [json.loads(l) for l in open(EVAL / "devF.jsonl", encoding="utf-8")]}
    for n in DEV + ("unanswerable_20",):
        sets[n] = [json.loads(l) for l in open(EVAL / f"{n}.jsonl", encoding="utf-8")]
    rag.embed("준비")
    recs = []
    with rag.connect() as conn:
        for name, qs in sets.items():
            for q in qs:
                text = rag.mask_pii(q["question"])
                qv = np.array(rag.embed(text), dtype=np.float32)
                ex = ex_rank(qv)
                old = rag.search(text, 5, conn)
                gold = q.get("gold_group_id") if name == "devF" else None
                rank = next((i + 1 for i, (g, _) in enumerate(ex) if g == gold), None) if gold else None
                recs.append({"set": name, "qid": q["qid"], "tag": q.get("tag"), "ex": ex[:5], "old": [round(h["score"], 4) for h in old], "gold_rank": rank,
                             "gold_score": next((s for g, s in ex if g == gold), None) if gold else None})
    out = {"devF": {"n": len(rows), "dropped": drop, "tags": dict(Counter(r["tag"] for r in rows)), "answer_type": dict(Counter(r["answer_type"] for r in rows))}}
    F = [r for r in recs if r["set"] == "devF"]
    hit = lambda rs, k: round(sum(1 for r in rs if r["gold_rank"] and r["gold_rank"] <= k) / len(rs), 3)
    out["devF"].update({f"hit{k}": hit(F, k) for k in (1, 2, 3, 5, 10)})
    out["devF"]["by_tag"] = {t: {f"hit{k}": hit([r for r in F if r["tag"] == t], k) for k in (1, 2, 5)} for t in TYPES}
    out["devF"]["gold_beats_old_top1"] = sum(1 for r in F if r["gold_score"] is not None and r["gold_score"] > r["old"][0])
    grp = {"devF": F, "기존 개발셋(460)": [r for r in recs if r["set"] in DEV], "답변불가 20": [r for r in recs if r["set"] == "unanswerable_20"]}
    q = lambda v: [round(float(x), 3) for x in np.quantile(v, [0.1, 0.5, 0.9])]
    out["score_dist"] = {g: {"ex_top1_p10_p50_p90": q([r["ex"][0][1] for r in rs]), "old_top1_p10_p50_p90": q([r["old"][0] for r in rs]),
                             "ex_top1_gt_old_top1": sum(1 for r in rs if r["ex"][0][1] > r["old"][0]), "ex_top1_gt_old_5th": sum(1 for r in rs if r["ex"][0][1] > r["old"][-1]), "n": len(rs)}
                         for g, rs in grp.items()}
    rules = {"항상 1건": lambda r: 1, "항상 2건": lambda r: 2}
    for t in (0.55, 0.60, 0.65, 0.70):
        rules[f"1위 점수 ≥ {t:.2f}면 최대 2건"] = (lambda t: lambda r: sum(1 for _, s in r["ex"][:2] if s >= t))(t)
    rules["기존 5위 점수 이상인 것만 최대 2건"] = lambda r: sum(1 for _, s in r["ex"][:2] if s >= r["old"][-1])
    rules["기존 1위 점수 - 0.05 이상인 것만 최대 2건"] = lambda r: sum(1 for _, s in r["ex"][:2] if s >= r["old"][0] - 0.05)
    out["attach_rules"] = {}
    for name, f in rules.items():
        row = {}
        for g, rs in grp.items():
            k = [f(r) for r in rs]
            row[g] = {"붙는 비율": round(sum(1 for x in k if x > 0) / len(rs), 3), "평균 건수": round(sum(k) / len(rs), 2)}
            if g == "devF":
                row[g]["정답을 붙임"] = round(sum(1 for r, x in zip(rs, k) if r["gold_rank"] and r["gold_rank"] <= x) / len(rs), 3)
        out["attach_rules"][name] = row
    out["devF_items"] = [{"qid": r["qid"], "tag": r["tag"], "gold_rank": r["gold_rank"], "gold_score": r["gold_score"] and round(r["gold_score"], 3), "ex_top1": round(r["ex"][0][1], 3), "old_top1": r["old"][0]} for r in F]
    json.dump(out, open(EVAL / "x5_exmnt_search_test.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    manifest("x5_exmnt_search_test", devF=len(rows), questions=len(recs))
    print(json.dumps({k: v for k, v in out.items() if k != "devF_items"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
