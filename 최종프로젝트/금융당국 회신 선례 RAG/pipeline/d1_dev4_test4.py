"""D1 dev4 50문항 · test4 20문항: dev3·test3와 같은 방식(d1_final_test.py의 후보 뽑기·질문 생성 규칙)에서 난이도를 조금 더 올린 어려운 실무형 질문.
dev3·test3에서 달라진 점 (모두 검색 점수를 보기 전에 정한 것)
- 혼동 사례(질의요지 벡터가 다른 묶음과 0.80 이상 비슷함) 비율 70% → 85%
- 원문과 같은 글자 구간 상한 10자 → 8자
- 질문마다 '비슷한 다른 회신과 구분되는 이 사례만의 사실관계·조건'을 반드시 넣게 하고, 핵심 용어를 그대로 쓰지 못하게 한다
- 각 문항에 KURE-v1 기준 가장 비슷한 다른 묶음과의 유사도(`confusable_sim_kure`)를 기록한다. 선택에는 쓰지 않는다 (혼동 사례를 OpenAI 벡터로만 고르는 편향을 나중에 분석하기 위해)
그대로인 것: 질문 유형 7가지와 비율, 정답 묶음이 합성 질의셋·dev1~3·test1~3·final_test와 겹치지 않음, dev4와 test4도 서로 겹치지 않음 (70개를 한 번에 뽑아 유형별 무작위 분할)
test4는 마지막에 한 번만 평가하는 세트다. 이 스크립트는 test4의 개별 질문을 출력하지 않는다
출력: data/eval/dev4.jsonl, test4.jsonl    이미 만든 항목은 다시 호출하지 않는다 (data/eval/_dev4_test4_raw.jsonl 캐시)
"""
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import d1_final_test as ft
from common import DATA, OUT, manifest, read_jsonl

EVAL = DATA / "eval"
SEED = 20261112
DEV_QUOTA = {"복합 쟁점형": 8, "비조치 신청형": 5, "조문+상황형": 5, "조건 포함형": 5, "경계 조건형": 8, "구어체형": 5, "우회 서술형": 14}     # 50
TEST_QUOTA = {"복합 쟁점형": 3, "비조치 신청형": 2, "조문+상황형": 2, "조건 포함형": 2, "경계 조건형": 4, "구어체형": 2, "우회 서술형": 5}      # 20
ALL_QUOTA = {t: DEV_QUOTA[t] + TEST_QUOTA[t] for t in DEV_QUOTA}                                                                              # 70

ft.SEED = SEED
ft.USED = ft.USED + ("final_test.jsonl", "dev3.jsonl", "test3.jsonl")
ft.QUOTA = ALL_QUOTA
ft.CONF_SHARE = 0.85
ft.MAX_COPY = 8
ft.BASE = ft.BASE.replace("10자를 넘으면", "8자를 넘으면") + "\n- 이 회신의 핵심 용어를 그대로 쓰지 말고, 비슷한 다른 회신과 구분되는 이 사례만의 사실관계나 조건 한 가지 이상을 반드시 질문에 넣으세요."


def kure_neighbors():
    """묶음별로 KURE-v1 기준 다른 묶음의 질의 카드와 가장 비슷한 정도. 선택에 쓰지 않고 기록만 한다."""
    chunks = read_jsonl("chunks_d3.jsonl")
    vec = np.load(OUT / "embeddings_d3_kure-v1.npy")
    idx = [i for i, c in enumerate(chunks) if c["chunk_type"] == "question_card"]
    q = vec[idx]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    gid = np.array([chunks[i]["group_id"] for i in idx])
    best = {}
    for g in np.unique(gid):
        s = q[gid == g] @ q.T
        s[:, gid == g] = -1
        best[int(g)] = round(float(s.max()), 4)
    return best


def main():
    items, dropped, n_pool = ft.build()
    cache = EVAL / "_dev4_test4_raw.jsonl"
    done = {x["source_case_key"]: x for x in map(json.loads, cache.open(encoding="utf-8"))} if cache.exists() else {}
    todo = [dict(x) for x in items if x["source_case_key"] not in done]
    t0 = time.time()
    with ThreadPoolExecutor(5) as ex, cache.open("a", encoding="utf-8") as f:
        for r in ex.map(ft.write_q, todo):
            done[r["source_case_key"]] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    chosen, short = [], {}
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

    kn = kure_neighbors()
    rng = random.Random(SEED)
    dev, test = [], []
    for tag, got in chosen:
        got = [{**x, "confusable_sim_kure": kn.get(x["gold_group_id"])} for x in got]
        rng.shuffle(got)
        n_dev = round(len(got) * DEV_QUOTA[tag] / ALL_QUOTA[tag])
        dev += got[:n_dev]
        test += got[n_dev:]
    rng.shuffle(dev)
    rng.shuffle(test)
    for name, rows, split, prefix in (("dev4", dev, "dev", "DEV4"), ("test4", test, "test", "TEST4")):
        with (EVAL / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for no, x in enumerate(rows, 1):
                x = {k: v for k, v in x.items() if k != "rank"}
                f.write(json.dumps({"qid": f"{prefix}-{no:03d}", "split": split, **x}, ensure_ascii=False) + "\n")
    manifest("d1_dev4_test4", dev4=len(dev), test4=len(test), pool=n_pool, dropped=dict(dropped), short=short, called=len(todo),
             tags_dev=dict(Counter(x["tag"] for x in dev)), tags_test=dict(Counter(x["tag"] for x in test)), seconds=round(time.time() - t0))


if __name__ == "__main__":
    main()
