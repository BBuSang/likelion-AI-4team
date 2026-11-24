"""5-3 재순위 흔들림 재기 — 같은 질문·같은 후보 10개를 순서만 섞어 CHAT_MODEL 재순위를 여러 번

사용: python analysis/s13_rerank_stability.py [반복 수=3]
- 임베딩 모델 두 개(OpenAI text-embedding-3-small, KURE-v1) 각각: dev 질문마다 routed 상위 10개 후보를 고정하고,
  후보 순서를 (질문·반복 번호로 정한 seed로) 섞어 재순위를 반복 → 반복마다 H@1·H@5, 반복 사이의 1위 일치율
- 결과는 data/eval/rerank_stability.jsonl에 저장(이미 한 호출은 다시 하지 않음)
- 후보 순서에 따라 1위가 바뀌는 정도 = 재순위가 흔들리는 정도. 두 모델 차이가 이 흔들림보다 작으면 '차이 없음'으로 봄
"""
import hashlib
import json
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import s8_extract  # noqa: E402

EVAL = ROOT / "data" / "eval"
OUT = EVAL / "rerank_stability.jsonl"
MODELS = ["text-embedding-3-small", "nlpai-lab/KURE-v1"]
INSTR = ("은행 상품 문서 검색 결과를 다시 줄 세운다. 질문에 직접 답하는 근거가 있는 후보 번호부터 순서대로 ids에 넣는다. "
         "질문의 상품과 다른 상품의 문서는 뒤로. 답이 없는 후보는 빼도 된다.")   # s12 재순위와 같은 지시문


class Order(BaseModel):
    ids: list[int]


def candidates(model):
    """그 임베딩 모델로 routed 상위 10개 (항목 번호) + 정답 판정 함수"""
    orig = s8_extract.load_env
    s8_extract.load_env = lambda: {**orig(), "EMBED_MODEL": model}
    import importlib
    import s12_retrieval as R
    importlib.reload(R)
    idx = R.Index()
    qs = [json.loads(l) for l in open(EVAL / "retrieval_dev.jsonl")]
    qv = R.embed_questions(qs, idx)
    s8_extract.load_env = orig
    cands = [[int(i) for i in idx.rank("routed", q["question"], v)[:10]] for q, v in zip(qs, qv)]
    return qs, cands, idx


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    env = s8_extract.load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
    done = {}
    if OUT.exists():
        for l in open(OUT):
            r = json.loads(l)
            done[r["key"]] = r["order"]

    summary = {}
    for model in MODELS:
        qs, cands, idx = candidates(model)
        jobs = []
        for qi, (q, cand) in enumerate(zip(qs, cands)):
            for rep in range(reps):
                perm = cand[:]
                random.Random(f"{q['qid']}:{rep}").shuffle(perm)
                key = hashlib.sha256(f"{model}|{q['qid']}|{rep}|{','.join(map(str, perm))}".encode()).hexdigest()
                jobs.append((qi, rep, perm, key))

        def call(job):
            qi, rep, perm, key = job
            if key in done:
                return job, done[key]
            body = "\n\n".join(f"<{n}>\n{idx.items[i]['text'][:1500]}" for n, i in enumerate(perm))
            r = client.responses.parse(model=env["CHAT_MODEL"], instructions=INSTR, text_format=Order,
                                       input=f"질문: {qs[qi]['question']}\n\n후보:\n{body}")
            order = [perm[n] for n in r.output_parsed.ids if 0 <= n < len(perm)]
            return job, order

        todo = sum(1 for j in jobs if j[3] not in done)
        print(f"[{model}] 재순위 {len(jobs)}번 중 새로 부를 것 {todo}", flush=True)
        with ThreadPoolExecutor(6) as ex, open(OUT, "a") as f:
            results = []
            for job, order in ex.map(call, jobs):
                if job[3] not in done:
                    done[job[3]] = order
                    f.write(json.dumps({"key": job[3], "model": model, "qid": qs[job[0]]["qid"], "rep": job[1], "order": order}) + "\n")
                results.append((job, order))

        tops = [[None] * reps for _ in qs]
        h1 = [[0] * len(qs) for _ in range(reps)]
        h5 = [[0] * len(qs) for _ in range(reps)]
        for (qi, rep, perm, key), order in results:
            full = list(dict.fromkeys(order)) + [i for i in perm if i not in order]
            tops[qi][rep] = full[0]
            h1[rep][qi] = int(idx.is_gold(full[0], qs[qi]["gold"]))
            h5[rep][qi] = int(any(idx.is_gold(i, qs[qi]["gold"]) for i in full[:5]))
        H1 = [sum(x) / len(qs) for x in h1]
        H5 = [sum(x) / len(qs) for x in h5]
        agree = sum(1 for t in tops if len(set(t)) == 1) / len(qs)
        any1 = sum(1 for qi in range(len(qs)) if any(h1[r][qi] for r in range(reps))) / len(qs)
        all1 = sum(1 for qi in range(len(qs)) if all(h1[r][qi] for r in range(reps))) / len(qs)
        in10 = sum(1 for qi, q in enumerate(qs) if any(idx.is_gold(i, q["gold"]) for i in cands[qi])) / len(qs)
        summary[model] = (H1, H5)
        print(f"  H@1 반복별 {[round(x, 3) for x in H1]} → 평균 {statistics.mean(H1):.3f} (표준편차 {statistics.pstdev(H1):.3f})")
        print(f"  H@5 반복별 {[round(x, 3) for x in H5]} → 평균 {statistics.mean(H5):.3f}")
        print(f"  {reps}번 모두 1위가 같은 질문 {agree:.3f} | 한 번이라도 1위 정답 {any1:.3f} | 매번 1위 정답 {all1:.3f} | 후보 10개 안 정답 {in10:.3f}")


if __name__ == "__main__":
    main()
