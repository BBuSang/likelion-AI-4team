"""D1 dev9 100문항 · test9 100문항: 기존 회신(법령해석·비조치) 70 + 현장건의 30씩 (사용자 요청 2026-11-12 "새 시험 세트를 open ai로 dev9, test9를 만들고 진행해봐").
설계 (검색 점수를 보기 전에 정함, 검색 결과로 문항을 고르지 않는다)
- dev7·test7(86번), dev8·test8(97번)과 같은 방식·같은 비율(기존 회신 70% : 현장건의 30%). 생성·검수 함수와 프롬프트를 그대로 쓴다
- 문항 수는 사용자가 정하지 않았다. 세트당 100으로 했다: 99번에서 확인하려는 동작(질문과 맞지 않는 사례로 답하려 할 때 거절)이 약 1%의 문항에서만 일어나,
  50문항으로는 한 번도 일어나지 않을 수 있다(dev8·test8에서 각 1번)
- 유형별 수는 dev7과 같다: 기존 회신 9유형(실무 기획 10, 경계 조건 10, 비조치 사전검토 8, 우회 서술 8, 구어체 8, 검색어 7, 조문 지정 7, 복합 쟁점 7, 규제 효력 5), 현장건의 5유형 × 6
- 정답 묶음 제외 대상: 지금까지 만든 모든 세트(합성, dev1~8, test1~8, final_test, devF)의 정답 묶음. 회신·현장건의 모두
- dev9와 test9는 정답 묶음이 겹치지 않는다. test9는 한 번만 평가하는 세트이고, 이 스크립트는 test9 질문을 출력하지 않는다
출력: data/eval/dev9.jsonl, test9.jsonl    캐시: data/eval/_dev9_test9_raw.jsonl(기존), _dev9_test9_exmnt_raw.jsonl(현장건의)
"""
import json
import random
import sys
import time
from collections import Counter

import d1_dev8_test8 as d8          # 들여오면서 d7·gen·xf를 dev8용으로 바꿔 놓는다. 아래에서 dev9용으로 다시 정한다
from common import manifest

d7, gen, EVAL = d8.d7, d8.gen, d8.EVAL
SEED = 20261111 + 9
N_OLD = {"실무 기획형": 10, "비조치 사전검토형": 8, "경계 조건형": 10, "우회 서술형": 8, "구어체형": 8, "검색어형": 7, "조문 지정형": 7, "복합 쟁점형": 7, "규제 효력형": 5}
N_EX = 6

d8.SKIP = ("dev9", "test9", "unanswerable_20")        # d8.used_groups·build_exmnt가 읽는 전역
d8.SEED, d8.N_EX = SEED, N_EX
gen.TYPES = {t: (N_OLD[t], N_OLD[t], *v[2:]) for t, v in gen.TYPES.items()}
gen.SEED = d7.SEED = SEED
gen.CACHE = EVAL / "_dev9_test9_raw.jsonl"
d7.N_OLD, d7.N_EX = N_OLD, N_EX
d7.EX_RAW = EVAL / "_dev9_test9_exmnt_raw.jsonl"


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
    for name, rows, split, prefix in (("dev9", dev, "dev", "DEV9"), ("test9", test, "test", "TEST9")):
        with (EVAL / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for no, x in enumerate(rows, 1):
                f.write(json.dumps({"qid": f"{prefix}-{no:03d}", "split": split, **{k: v for k, v in x.items() if k != "rank"}}, ensure_ascii=False) + "\n")
    cnt = lambda rows, key: dict(Counter(f"{x['corpus']}:{x[key]}" for x in rows))
    stats = {"dev9": len(dev), "test9": len(test), "old": s_old, "exmnt": s_ex,
             "gubun_dev": dict(Counter(x["gubun"] for x in dev)), "gubun_test": dict(Counter(x["gubun"] for x in test)),
             "tags_dev": cnt(dev, "tag"), "tags_test": cnt(test, "tag"),
             "category_all": cnt(dev + test, "category"), "exmnt_answer_type": dict(Counter(x["answer_type"] for x in ed + et)),
             "seconds": round(time.time() - t0)}
    manifest("d1_dev9_test9", **stats)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
