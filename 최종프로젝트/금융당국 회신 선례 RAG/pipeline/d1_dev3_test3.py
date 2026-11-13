"""D1 dev3 70문항 · test3 40문항: final_test와 같은 방식(d1_final_test.py의 후보 뽑기·질문 생성 규칙)으로 만든 어려운 실무형 질문.
- 정답 후보: 혼동되기 쉬운 사례(질의요지 벡터가 다른 묶음과 CONFUSABLE 이상 비슷함) 우선, 거의 같은 질의(NEAR_DUP 이상)는 제외. 검색 점수를 보고 고르지 않는다
- LLM에 제목·질의요지·회답 일부를 주고 결론은 숨기며, 원문과 같은 글자 구간이 10자 이하가 되게 쓴다. 질문 유형 7가지를 final_test와 같은 비율로 나눈다
- 합성 질의셋, dev1·test1·dev2·test2, final_test와 정답 묶음이 겹치지 않는다. dev3와 test3도 서로 겹치지 않는다 (110개를 한 번에 뽑아 유형별로 무작위 분할)
- test3는 마지막에 한 번만 평가하는 세트다. 이 스크립트는 test3의 개별 질문을 출력하지 않는다
출력: data/eval/dev3.jsonl, test3.jsonl    이미 만든 항목은 다시 호출하지 않는다 (data/eval/_dev3_test3_raw.jsonl 캐시)
"""
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import d1_final_test as ft
from common import DATA, manifest

EVAL = DATA / "eval"
SEED = 20261111
DEV_QUOTA = {"복합 쟁점형": 10, "비조치 신청형": 7, "조문+상황형": 7, "조건 포함형": 7, "경계 조건형": 10, "구어체형": 7, "우회 서술형": 22}     # 70
TEST_QUOTA = {"복합 쟁점형": 6, "비조치 신청형": 4, "조문+상황형": 4, "조건 포함형": 4, "경계 조건형": 6, "구어체형": 4, "우회 서술형": 12}    # 40
ALL_QUOTA = {t: DEV_QUOTA[t] + TEST_QUOTA[t] for t in DEV_QUOTA}                                                                              # 110

# final_test.py의 규칙을 그대로 쓰되 시드·제외 세트·유형별 수만 바꾼다
ft.SEED = SEED
ft.USED = ft.USED + ("final_test.jsonl",)
ft.QUOTA = ALL_QUOTA


def main():
    items, dropped, n_pool = ft.build()
    cache = EVAL / "_dev3_test3_raw.jsonl"
    done = {x["source_case_key"]: x for x in map(json.loads, cache.open(encoding="utf-8"))} if cache.exists() else {}
    todo = [dict(x) for x in items if x["source_case_key"] not in done]
    t0 = time.time()
    with ThreadPoolExecutor(5) as ex, cache.open("a", encoding="utf-8") as f:
        for r in ex.map(ft.write_q, todo):
            done[r["source_case_key"]] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    chosen, short = [], {}                      # 유형별로 혼동 사례 우선(CONF_SHARE)으로 필요한 수만큼 고른다 (final_test와 같은 규칙)
    for tag, n in ALL_QUOTA.items():
        ok = [done[x["source_case_key"]] for x in sorted((i for i in items if i["tag"] == tag), key=lambda i: i["rank"]) if done[x["source_case_key"]]["question"]]
        conf = [x for x in ok if x["confusable_sim"] >= ft.CONFUSABLE]
        rest = [x for x in ok if x["confusable_sim"] < ft.CONFUSABLE]
        got = conf[:round(n * ft.CONF_SHARE)]
        got += rest[:n - len(got)]
        got += conf[len(conf[:round(n * ft.CONF_SHARE)]):][:n - len(got)]
        if len(got) < n:
            short[tag] = n - len(got)
        chosen.append((tag, got))

    rng = random.Random(SEED)
    dev, test = [], []
    for tag, got in chosen:                     # 유형별로 섞어서 dev3 / test3로 나눈다 (혼동 정도가 한쪽에 몰리지 않게)
        got = got[:]
        rng.shuffle(got)
        n_dev = round(len(got) * DEV_QUOTA[tag] / ALL_QUOTA[tag])
        dev += got[:n_dev]
        test += got[n_dev:]
    rng.shuffle(dev)
    rng.shuffle(test)
    for name, rows, split, prefix in (("dev3", dev, "dev", "DEV3"), ("test3", test, "test", "TEST3")):
        with (EVAL / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for no, x in enumerate(rows, 1):
                x = {k: v for k, v in x.items() if k != "rank"}
                f.write(json.dumps({"qid": f"{prefix}-{no:03d}", "split": split, **x}, ensure_ascii=False) + "\n")
    manifest("d1_dev3_test3", dev3=len(dev), test3=len(test), pool=n_pool, dropped=dict(dropped), short=short, called=len(todo),
             tags_dev=dict(Counter(x["tag"] for x in dev)), tags_test=dict(Counter(x["tag"] for x in test)), seconds=round(time.time() - t0))


if __name__ == "__main__":
    main()
