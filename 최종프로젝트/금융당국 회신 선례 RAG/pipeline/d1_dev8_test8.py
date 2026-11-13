"""D1 dev8 50문항 · test8 50문항: 기존 회신(법령해석·비조치) 35 + 현장건의 15씩 (사용자 요청 2026-11-12 "open ai로 새로운 골드셋을 만들어봐. dev8, test8 각 50문항씩").
설계 (검색 점수를 보기 전에 정함, 검색 결과로 문항을 고르지 않는다)
- dev7·test7(`d1_dev7_test7.py`, 86번)과 같은 방식·같은 비율(기존 회신 70% : 현장건의 30%)이다. 생성·검수 함수와 프롬프트를 그대로 쓴다
- 기존 회신 35: 9가지 유형, 수는 dev7의 절반(실무 기획 5, 경계 조건 5, 비조치 사전검토 4, 우회 서술 4, 구어체 4, 조문 지정 4, 검색어 3, 복합 쟁점 3, 규제 효력 3)
- 현장건의 15: 5가지 유형 × 3 (실무 상황, 우회 서술, 구어체, 검색어, 경계 조건)
- 정답 묶음 제외 대상: 지금까지 만든 모든 세트(합성, dev1~7, test1~7, final_test, devF)의 정답 묶음. 회신·현장건의 모두 제외한다
  (dev7에서는 현장건의 쪽에서 devF만 뺐다. 이번에는 dev7·test7의 현장건의 정답도 뺀다)
- dev8와 test8는 정답 묶음이 겹치지 않는다. test8는 한 번만 평가하는 세트이고, 이 스크립트는 test8 질문을 출력하지 않는다
용도: 96번 답변 검증 단계처럼 개발셋을 본 뒤 정한 변경을 새 문항으로 확인하는 데 쓴다(dev8 개발용, test8 시험용)
출력: data/eval/dev8.jsonl, test8.jsonl    캐시: data/eval/_dev8_test8_raw.jsonl(기존), _dev8_test8_exmnt_raw.jsonl(현장건의)
"""
import json
import random
import sys
import time
from collections import Counter

import d1_dev7_test7 as d7          # 들여오면서 gen·xf의 유형·시드를 dev7용으로 바꿔 놓는다. 아래에서 dev8용으로 다시 정한다
from common import manifest, read_jsonl

gen, xf, EVAL = d7.gen, d7.xf, d7.EVAL
SEED = 20261111 + 8
SKIP = ("dev8", "test8", "unanswerable_20")
N_OLD = {"실무 기획형": 5, "비조치 사전검토형": 4, "경계 조건형": 5, "우회 서술형": 4, "구어체형": 4, "검색어형": 3, "조문 지정형": 4, "복합 쟁점형": 3, "규제 효력형": 3}
N_EX = 3


def used_groups():
    used = set()
    for p in EVAL.glob("*.jsonl"):
        if p.name.startswith("_") or p.stem in SKIP:
            continue
        for line in open(p, encoding="utf-8"):
            g = json.loads(line).get("gold_group_id")
            if g:
                used.add(g)
    return used


def build_exmnt():
    """d7.build_exmnt와 같되, 이미 쓴 현장건의 정답 묶음(devF, dev7, test7)을 모두 뺀다."""
    cases = {r["case_key"]: r for r in read_jsonl("exmnt_cases.jsonl")}
    groups = read_jsonl("exmnt_groups.jsonl")
    used = used_groups()
    gid = {g["representative_case_key"]: g["group_id"] for g in groups}
    pool = [cases[g["representative_case_key"]] for g in groups if g["group_id"] not in used]
    pool = [r for r in pool if len(r["answer"] or "") >= 80 and len(r["question"] or "") >= 80]
    rng = random.Random(SEED)
    rng.shuffle(pool)
    cat_n, ans_n, items, taken = Counter(), Counter(), [], set()
    for tag in xf.TYPES:
        for i in range(round(2 * N_EX * d7.EXTRA_EX)):
            x = min((r for r in pool if r["case_key"] not in taken), key=lambda r: (cat_n[r["category_top"]], ans_n[r["answer_type"]]))
            taken.add(x["case_key"])
            cat_n[x["category_top"]] += 1
            ans_n[x["answer_type"]] += 1
            items.append({"tag": tag, "rank": i, "gold_group_id": gid[x["case_key"]], "source_case_key": x["case_key"], "gubun": "현장건의", "category": x["category_top"],
                          "answer_type": x["answer_type"], "reply_date": x["reply_date"], "_title": x["title"], "_question": x["question"], "_answer": f"검토결과: {x['answer_type']}\n{x['answer']}"})
    return items, len(pool)


# dev8용으로 다시 정한다 (d7의 함수들은 d7 모듈의 전역을 읽는다)
gen.TYPES = {t: (N_OLD[t], N_OLD[t], *v[2:]) for t, v in gen.TYPES.items()}
gen.SEED = d7.SEED = SEED
gen.CACHE = EVAL / "_dev8_test8_raw.jsonl"
gen.used_groups = used_groups
d7.N_OLD, d7.N_EX = N_OLD, N_EX
d7.EX_RAW = EVAL / "_dev8_test8_exmnt_raw.jsonl"
d7.build_exmnt = build_exmnt


def main():
    t0 = time.time()
    rng = random.Random(SEED)
    od, ot, s_old = d7.make_old(rng)
    print("기존 회신 완료", json.dumps(s_old, ensure_ascii=False), flush=True)
    ed, et, s_ex = d7.make_exmnt(rng)
    print("현장건의 완료", json.dumps(s_ex, ensure_ascii=False), flush=True)
    for r in od + ot:
        r["corpus"] = "interp"
    for r in ed + et:
        r["corpus"] = "exmnt"
    dev, test = od + ed, ot + et
    rng.shuffle(dev)
    rng.shuffle(test)
    for name, rows, split, prefix in (("dev8", dev, "dev", "DEV8"), ("test8", test, "test", "TEST8")):
        with (EVAL / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for no, x in enumerate(rows, 1):
                f.write(json.dumps({"qid": f"{prefix}-{no:03d}", "split": split, **{k: v for k, v in x.items() if k != "rank"}}, ensure_ascii=False) + "\n")
    cnt = lambda rows, key: dict(Counter(f"{x['corpus']}:{x[key]}" for x in rows))
    stats = {"dev8": len(dev), "test8": len(test), "old": s_old, "exmnt": s_ex,
             "gubun_dev": dict(Counter(x["gubun"] for x in dev)), "gubun_test": dict(Counter(x["gubun"] for x in test)),
             "tags_dev": cnt(dev, "tag"), "tags_test": cnt(test, "tag"),
             "category_all": cnt(dev + test, "category"), "exmnt_answer_type": dict(Counter(x["answer_type"] for x in ed + et)),
             "seconds": round(time.time() - t0)}
    manifest("d1_dev8_test8", **stats)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
