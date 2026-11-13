"""D1 dev7 100문항 · test7 100문항: 기존 회신(법령해석·비조치) 70 + 현장건의 30씩 (사용자 요청 2026-11-11, 비율은 사용자 선택).
설계 (검색 점수를 보기 전에 정함, 검색 결과로 문항을 고르지 않는다)
- 기존 회신 부분: d1_dev5_test5의 후보 선정·생성·검수를 그대로 쓴다(9가지 유형, 비조치 약 45%, 덜 뽑힌 분야 우선).
  유형별 수는 dev5 dev 수의 1.4배로 dev·test 같게(합 70). 정답 묶음 제외 대상: 기존 모든 세트(dev1~6, test1~6, final_test, 합성 등)
- 현장건의 부분: x5_exmnt_search_test의 생성·검수(devF와 같은 방식)를 쓰되 유형을 5가지로 늘린다(실무 상황, 우회 서술, 구어체, 검색어, 경계 조건) × dev 6 · test 6.
  정답 후보: 현장건의 묶음 대표 중 건의·사유 80자 이상, devF 정답 제외. 과제분류(category_top)가 고르게, 같으면 검토결과(수용/불수용 등)도 고르게
- 현장건의는 지금 별도 후보군이다(84번). 정답 group_id가 1000000 이상이고 corpus 칸으로 구분한다(interp / exmnt)
- dev7와 test7는 정답 묶음이 겹치지 않는다. test7는 한 번만 평가하는 세트이고, 이 스크립트는 test7 질문을 출력하지 않는다
출력: data/eval/dev7.jsonl, test7.jsonl    캐시: data/eval/_dev7_test7_raw.jsonl(기존), _dev7_test7_exmnt_raw.jsonl(현장건의)
"""
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import d1_dev5_test5 as gen
import x5_exmnt_search_test as xf
from common import DATA, manifest, read_jsonl

EVAL = DATA / "eval"
SEED = 20261111 + 7
SKIP = ("dev7", "test7", "unanswerable_20", "devF")

# 기존 회신: dev5 유형 정의는 그대로, 수만 바꾼다
N_OLD = {"실무 기획형": 10, "비조치 사전검토형": 8, "경계 조건형": 10, "우회 서술형": 8, "구어체형": 8, "검색어형": 7, "조문 지정형": 7, "복합 쟁점형": 7, "규제 효력형": 5}
gen.TYPES = {t: (N_OLD[t], N_OLD[t], *v[2:]) for t, v in gen.TYPES.items()}
gen.SEED = SEED
gen.CACHE = EVAL / "_dev7_test7_raw.jsonl"


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


gen.used_groups = used_groups

# 현장건의: devF 유형 4가지 + 경계 조건형
xf.TYPES = {**xf.TYPES, "경계 조건형": "결론을 가르는 조건(금액·기간·대상·방식·횟수 등) 하나를 분명히 넣고 '~인 경우에도 ~해도 되는지', '~는 지금도 제한되는지' 형태로 쓰세요."}
xf.LEN = {**xf.LEN, "경계 조건형": (60, 220)}
N_EX, EXTRA_EX = 6, 1.6                     # 유형별 dev 6 · test 6
EX_RAW = EVAL / "_dev7_test7_exmnt_raw.jsonl"


def build_exmnt():
    cases = {r["case_key"]: r for r in read_jsonl("exmnt_cases.jsonl")}
    groups = read_jsonl("exmnt_groups.jsonl")
    devf = {json.loads(l)["gold_group_id"] for l in open(EVAL / "devF.jsonl", encoding="utf-8")}
    gid = {g["representative_case_key"]: g["group_id"] for g in groups}
    pool = [cases[g["representative_case_key"]] for g in groups if g["group_id"] not in devf]
    pool = [r for r in pool if len(r["answer"] or "") >= 80 and len(r["question"] or "") >= 80]
    rng = random.Random(SEED)
    rng.shuffle(pool)
    cat_n, ans_n, items, taken = Counter(), Counter(), [], set()
    for tag in xf.TYPES:
        for i in range(round(2 * N_EX * EXTRA_EX)):
            x = min((r for r in pool if r["case_key"] not in taken), key=lambda r: (cat_n[r["category_top"]], ans_n[r["answer_type"]]))
            taken.add(x["case_key"])
            cat_n[x["category_top"]] += 1
            ans_n[x["answer_type"]] += 1
            items.append({"tag": tag, "rank": i, "gold_group_id": gid[x["case_key"]], "source_case_key": x["case_key"], "gubun": "현장건의", "category": x["category_top"],
                          "answer_type": x["answer_type"], "reply_date": x["reply_date"], "_title": x["title"], "_question": x["question"], "_answer": f"검토결과: {x['answer_type']}\n{x['answer']}"})
    return items, len(pool)


def ok_rows(items, done, tag):
    ok, drop = [], Counter()
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
    return ok, drop


def make_exmnt(rng):
    items, n_pool = build_exmnt()
    done = {x["source_case_key"]: x for x in map(json.loads, EX_RAW.open(encoding="utf-8"))} if EX_RAW.exists() else {}
    todo = [dict(x) for x in items if x["source_case_key"] not in done]
    with ThreadPoolExecutor(5) as ex, EX_RAW.open("a", encoding="utf-8") as f:
        for r in ex.map(xf.write_q, todo):
            done[r["source_case_key"]] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dev, test, drop, short = [], [], Counter(), {}
    for tag in xf.TYPES:
        ok, d = ok_rows(items, done, tag)
        drop += d
        got = ok[:2 * N_EX]
        if len(got) < 2 * N_EX:
            short[tag] = 2 * N_EX - len(got)
        rng.shuffle(got)
        k = len(got) // 2
        dev += got[:k]
        test += got[k:]
    return dev, test, {"pool": n_pool, "called": len(todo), "dropped": dict(drop), "short": short}


def make_old(rng):
    done = {x["source_case_key"]: x for x in map(json.loads, gen.CACHE.open(encoding="utf-8"))} if gen.CACHE.exists() else {}
    items, n_pool = gen.build()
    called = gen.generate(items, done)
    reserve = {}
    for _ in range(3):                                   # 모자란 유형만 예비 후보 (d1_dev5_test5와 같은 규칙)
        need = {t: 2 * N_OLD[t] - gen.passed(items, done, t) for t in gen.TYPES}
        need = {t: n for t, n in need.items() if n > 0}
        if not need:
            break
        for t, n in need.items():
            reserve[t] = reserve.get(t, 0) + 2 * n
        items, n_pool = gen.build(reserve)
        called += gen.generate(items, done)
    dev, test, drop, short = [], [], Counter(), {}
    for tag in gen.TYPES:
        ok, d = ok_rows(items, done, tag)
        drop += d
        got = ok[:2 * N_OLD[tag]]
        if len(got) < 2 * N_OLD[tag]:
            short[tag] = 2 * N_OLD[tag] - len(got)
        rng.shuffle(got)
        k = len(got) // 2
        dev += got[:k]
        test += got[k:]
    return dev, test, {"pool": n_pool, "called": called, "reserve": reserve, "dropped": dict(drop), "short": short}


def main():
    t0 = time.time()
    rng = random.Random(SEED)
    od, ot, s_old = make_old(rng)
    print("기존 회신 완료", json.dumps(s_old, ensure_ascii=False), flush=True)
    ed, et, s_ex = make_exmnt(rng)
    print("현장건의 완료", json.dumps(s_ex, ensure_ascii=False), flush=True)
    for r in od + ot:
        r["corpus"] = "interp"
    for r in ed + et:
        r["corpus"] = "exmnt"
    dev, test = od + ed, ot + et
    rng.shuffle(dev)
    rng.shuffle(test)
    for name, rows, split, prefix in (("dev7", dev, "dev", "DEV7"), ("test7", test, "test", "TEST7")):
        with (EVAL / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for no, x in enumerate(rows, 1):
                f.write(json.dumps({"qid": f"{prefix}-{no:03d}", "split": split, **{k: v for k, v in x.items() if k != "rank"}}, ensure_ascii=False) + "\n")
    cnt = lambda rows, key: dict(Counter(f"{x['corpus']}:{x[key]}" for x in rows))
    stats = {"dev7": len(dev), "test7": len(test), "old": s_old, "exmnt": s_ex,
             "gubun_dev": dict(Counter(x["gubun"] for x in dev)), "gubun_test": dict(Counter(x["gubun"] for x in test)),
             "tags_dev": cnt(dev, "tag"), "tags_test": cnt(test, "tag"),
             "category_all": cnt(dev + test, "category"), "exmnt_answer_type": dict(Counter(x["answer_type"] for x in ed + et)),
             "seconds": round(time.time() - t0)}
    manifest("d1_dev7_test7", **stats)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
