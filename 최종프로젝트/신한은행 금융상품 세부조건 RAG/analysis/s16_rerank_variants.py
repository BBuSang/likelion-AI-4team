"""6-3 재순위 방식 비교 — 흔들림(후보 순서를 섞었을 때 1위가 바뀌는 정도)을 줄이는 방법 찾기

사용: python analysis/s16_rerank_variants.py [반복 수=3]
대상: .env 임베딩(KURE-v1)의 routed 상위 10개 후보, dev 105개. 후보 순서를 (질문·반복 번호 seed로) 섞어 반복
방식
  list   지금 방식 — 답이 있는 후보 번호를 순서대로 (s12 rerank·s13과 같은 지시문). s13 결과(rerank_stability.jsonl)를 그대로 씀
  score  후보마다 0~3점 (3: 질문에 직접 답함, 2: 답의 일부, 1: 관련 있지만 답 없음, 0: 무관) → 점수 높은 순, 같으면 원래 검색 순위
         점수는 각 후보를 따로 판단하게 하므로 목록 순서에 덜 흔들릴 것으로 기대
  vote   list를 순서를 섞어 3번 한 결과를 평균 순위로 합침 (호출 3배, 1번의 흔들림을 평균으로 줄임) — 반복 1번 = 투표 1번
  score_vote  score 3번의 점수를 더해 합침
지표: 반복별 H@1·H@5 평균과 표준편차, 반복 사이 1위 일치율
결과: data/eval/rerank_variants.jsonl (이미 한 호출은 다시 하지 않음)
"""
import hashlib
import json
import random
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import s12_retrieval as R  # noqa: E402
from s8_extract import load_env  # noqa: E402

EVAL = ROOT / "data" / "eval"
OUT = EVAL / "rerank_variants.jsonl"
SCORE_INSTR = ("은행 상품 문서 검색 결과를 평가한다. 후보마다 질문에 대한 쓸모를 0~3점으로 매긴다. "
               "3: 질문에 직접 답하는 근거가 있음, 2: 답의 일부가 있음, 1: 관련은 있지만 답이 없음, 0: 무관하거나 다른 상품. "
               "후보끼리 비교하지 말고 각 후보를 따로 판단한다. 모든 후보 번호에 점수를 준다.")


class Scored(BaseModel):
    class Item(BaseModel):
        id: int
        score: int
    items: list[Item]


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
    idx = R.Index()
    qs = [json.loads(l) for l in open(EVAL / "retrieval_dev.jsonl")]
    qv = R.embed_questions(qs, idx)
    cands = [[int(i) for i in idx.rank("routed", q["question"], v)[:10]] for q, v in zip(qs, qv)]

    # list: s13 결과 재사용 (같은 모델·같은 후보·같은 섞기 seed)
    lst = {}
    for l in open(EVAL / "rerank_stability.jsonl"):
        r = json.loads(l)
        if r["model"] == env["EMBED_MODEL"]:
            lst[(r["qid"], r["rep"])] = r["order"]

    done = {}
    if OUT.exists():
        for l in open(OUT):
            r = json.loads(l)
            done[r["key"]] = r["scores"]

    def call(job):
        qi, rep, perm, key = job
        if key in done:
            return job, done[key]
        body = "\n\n".join(f"<{n}>\n{idx.items[i]['text'][:1500]}" for n, i in enumerate(perm))
        r = client.responses.parse(model=env["CHAT_MODEL"], instructions=SCORE_INSTR, text_format=Scored,
                                   input=f"질문: {qs[qi]['question']}\n\n후보:\n{body}")
        sc = {perm[it.id]: it.score for it in r.output_parsed.items if 0 <= it.id < len(perm)}
        return job, {str(k): v for k, v in sc.items()}

    jobs = []
    for qi, (q, cand) in enumerate(zip(qs, cands)):
        for rep in range(reps):
            perm = cand[:]
            random.Random(f"{q['qid']}:{rep}").shuffle(perm)
            jobs.append((qi, rep, perm, hashlib.sha256(f"score|{env['EMBED_MODEL']}|{q['qid']}|{rep}|{perm}".encode()).hexdigest()))
    print(f"score 호출 {len(jobs)}번 중 새로 {sum(1 for j in jobs if j[3] not in done)}", flush=True)
    scores = {}
    with ThreadPoolExecutor(6) as ex, open(OUT, "a") as f:
        for job, sc in ex.map(call, jobs):
            if job[3] not in done:
                done[job[3]] = sc
                f.write(json.dumps({"key": job[3], "qid": qs[job[0]]["qid"], "rep": job[1], "scores": sc}) + "\n")
            scores[(job[0], job[1])] = {int(k): v for k, v in sc.items()}

    def order_list(qi, rep):
        o = lst.get((qs[qi]["qid"], rep))
        if o is None:
            return None
        return list(dict.fromkeys(o)) + [i for i in cands[qi] if i not in o]

    def order_score(qi, reps_used):
        tot = defaultdict(int)
        for rep in reps_used:
            for i, s in scores[(qi, rep)].items():
                tot[i] += s
        base = {i: r for r, i in enumerate(cands[qi])}
        return sorted(cands[qi], key=lambda i: (-tot.get(i, 0), base[i]))

    def order_vote(qi):
        pos = defaultdict(float)
        for rep in range(reps):
            o = order_list(qi, rep)
            for r, i in enumerate(o):
                pos[i] += r
        return sorted(cands[qi], key=lambda i: pos[i])

    def metrics(orders):
        h1 = sum(idx.is_gold(o[0], q["gold"]) for o, q in zip(orders, qs)) / len(qs)
        h5 = sum(any(idx.is_gold(i, q["gold"]) for i in o[:5]) for o, q in zip(orders, qs)) / len(qs)
        return h1, h5

    print("\n방식 | H@1 (반복별 → 평균±표준편차) | H@5 평균 | 반복 사이 1위 일치")
    for name, fn in (("list", lambda qi, rep: order_list(qi, rep)), ("score", lambda qi, rep: order_score(qi, [rep]))):
        per = [[fn(qi, rep) for qi in range(len(qs))] for rep in range(reps)]
        if any(o is None for p in per for o in p):
            print(f"{name}: list 결과 부족 (s13을 같은 반복 수로 먼저)")
            continue
        m = [metrics(p) for p in per]
        h1s = [x[0] for x in m]
        agree = sum(1 for qi in range(len(qs)) if len({per[r][qi][0] for r in range(reps)}) == 1) / len(qs)
        print(f"{name:11}| {[round(x, 3) for x in h1s]} → {statistics.mean(h1s):.3f}±{statistics.pstdev(h1s):.3f} | "
              f"{statistics.mean(x[1] for x in m):.3f} | {agree:.0%}")
    for name, orders in (("vote", [order_vote(qi) for qi in range(len(qs))]),
                         ("score_vote", [order_score(qi, range(reps)) for qi in range(len(qs))])):
        h1, h5 = metrics(orders)
        print(f"{name:11}| {h1:.3f} (호출 {reps}배) | {h5:.3f} | —")
    base = metrics([cands[qi] for qi in range(len(qs))])
    print(f"재순위 없음  | {base[0]:.3f} | {base[1]:.3f} | 100% (결정적)")


if __name__ == "__main__":
    main()
