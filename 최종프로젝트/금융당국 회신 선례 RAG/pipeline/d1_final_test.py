"""D1 최종 평가용 최종 평가용 질문 셋(final_test) 100문항: 어렵게 만든 실무형 질문. 정답은 원본 사례의 묶음.
어렵게 만드는 방법 (검색 점수를 보고 고르지 않는다 — 점수를 보고 고르면 최종 평가가 오염된다)
- 정답 후보를 '혼동되기 쉬운 사례'(질의요지 벡터가 다른 묶음과 CONFUSABLE 이상 비슷함)에서 우선 뽑는다.
  단 정답이 둘로 갈리는 거의 같은 질의(NEAR_DUP 이상)는 뺀다.
- LLM에 제목·질의요지·회답 일부를 주고, 결론은 숨기되 이 사례만의 구체적 조건을 담게 한다.
- 원문의 전문 용어를 그대로 쓰지 못하게 한다 (원문과 같은 구간 MAX_COPY자 이하).
- dev1·test1·dev2·test2·합성 질의셋과 정답 묶음이 겹치지 않는다.
출력: data/eval/final_test.jsonl    이미 만든 항목은 다시 호출하지 않는다 (data/eval/_final_test_raw.jsonl 캐시).
"""
import json
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import rag
from common import DATA, OUT, manifest, read_jsonl
from d1_devtest import MULTI
from d1_rewrite import longest_copy

SEED, MAX_COPY, MAX_LEN, MIN_LEN = 20261110, 10, 220, 60
CONFUSABLE, NEAR_DUP, CONF_SHARE = 0.80, 0.93, 0.7
EVAL = DATA / "eval"
USED = ("rewrite_150.jsonl", "dev1.jsonl", "test1.jsonl", "dev2.jsonl", "test2.jsonl")
QUOTA = {"복합 쟁점형": 15, "비조치 신청형": 10, "조문+상황형": 10, "조건 포함형": 10, "경계 조건형": 15, "구어체형": 10, "우회 서술형": 30}
EXTRA = 1.5      # 생성에 실패하거나 검사에서 걸러질 몫을 대신할 예비 후보 배수
BASE = """은행 직원이 금융당국 회신 검색 도구에 입력하는 실제 업무 질문을 쓰세요. 검색 시스템의 최종 평가용이라 어렵게 만들어야 합니다.
- 아래 회신이 정답이 되는 질문이어야 합니다. 이 회신만 답할 수 있는 구체적인 쟁점(대상, 방식, 조건)을 질문에 담되, 회신의 결론이나 당국의 판단은 쓰지 마세요.
- 회신에 쓰인 전문 용어와 표현을 그대로 가져오지 말고, 직원이 업무에서 쓰는 다른 말로 풀어 쓰세요. 원문과 같은 글자 구간이 10자를 넘으면 안 됩니다.
- 비슷한 주제의 다른 회신과 혼동되기 쉽게 쓰되, 이 회신 하나로 답이 정해져야 합니다.
- 회사명, 문서 번호, 사람 이름은 쓰지 마세요. 원문에 없는 사실도 더하지 마세요.
- 60~200자로 쓰세요.
JSON으로만 답하세요: {"question": "..."}"""
GUIDE = {
    "복합 쟁점형": "\n- 이 회신이 다루는 쟁점 두 가지를 한 질문에 함께 묻는 형태로 쓰세요. 조문 번호는 쓰지 마세요.",
    "비조치 신청형": "\n- 직원이 새 상품·업무를 하려는데 법령 위반이나 제재 소지가 있는지, 비조치 의견을 구할 수 있는지 묻는 형태로 쓰세요. 조문 번호는 쓰지 마세요.",
    "조문+상황형": "\n- 반드시 다음 조문을 질문에 그대로 언급하세요: {cite}\n- 같은 조문을 다루는 회신이 많으므로, 조문만으로는 정답이 정해지지 않게 이 회신의 구체적인 상황을 함께 쓰세요.",
    "조건 포함형": "\n- 반드시 '{gubun}' 회신 중에서, {year}년 이후 회신으로 한정해 찾아 달라는 조건을 문장에 넣으세요. 조문 번호는 쓰지 마세요.",
    "경계 조건형": "\n- 결론이 갈리는 경계 조건(금액·기간·대상·방식 등)이 질문에 분명히 드러나게 쓰세요. '~인 경우에도 ~해도 되는지' 같은 단서 형태가 좋습니다. 조문 번호는 쓰지 마세요.",
    "구어체형": "\n- 영업점이나 현업 직원이 메신저에 급히 치듯이 짧은 구어체로 쓰세요. 약어나 줄임말, 불완전한 문장을 써도 됩니다. 조문 번호는 쓰지 마세요.",
    "우회 서술형": "\n- 법령·제도 이름이나 전문 용어를 거의 쓰지 말고, 일어난 업무 상황을 평범한 말로 길게 설명하는 형태로 쓰세요. 조문 번호는 쓰지 마세요.",
}


def neighbors():
    """묶음별로 다른 묶음의 질의요지 조각과 가장 비슷한 정도(코사인)와 그 묶음."""
    chunks = read_jsonl("chunks.jsonl")
    vec = np.load(OUT / "embeddings.npy")
    idx = [i for i, c in enumerate(chunks) if c["chunk_type"] == "question"]
    q = vec[idx]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    gid = np.array([chunks[i]["group_id"] for i in idx])
    best = {}
    for g in np.unique(gid):
        rows = np.where(gid == g)[0]
        s = q[rows] @ q.T
        s[:, gid == g] = -1
        top = s.max(axis=0)
        order = np.argsort(-top)[:3]
        best[int(g)] = (float(top[order[0]]), [(int(gid[o]), float(top[o])) for o in order if top[o] >= CONFUSABLE])
    return best


def build():
    used = {json.loads(l)["gold_group_id"] for name in USED for l in open(EVAL / name, encoding="utf-8")}
    rep = {g["group_id"]: g["representative_case_key"] for g in read_jsonl("case_groups.jsonl")}
    near = neighbors()
    with rag.connect() as c:
        rows = c.execute("""select case_key, group_id, gubun, category_final, reply_date::text reply_date, doc_class, title, question, answer, reason from interp.cases
                            where indexable and not body_suspect and doc_class = 'general' and reply_date is not null order by case_key""").fetchall()
        arts = c.execute("""select a.case_key, l.law_name, a.raw from interp.case_articles a join interp.laws l using (law_id)
                            where a.resolve_method = 'explicit' and a.raw like '제%%' and l.law_kind <> '행정규칙' order by a.id""").fetchall()
    cites = defaultdict(list)
    for a in arts:
        cites[a["case_key"]].append(f"{a['law_name']} {a['raw']}")
    rng = random.Random(SEED)
    rng.shuffle(rows)
    pool, dropped = [], Counter()
    for r in rows:
        g = r["group_id"]
        if g in used or rep.get(g) != r["case_key"]:
            continue
        if len(r["question"] or "") < 80 or len(r["answer"] or "") < 40 or g not in near:
            dropped["짧은 칸 또는 질의요지 조각 없음"] += 1
            continue
        sim, nb = near[g]
        if sim >= NEAR_DUP:
            dropped["거의 같은 질의가 다른 묶음에 있음"] += 1
            continue
        pool.append({**r, "confusable_sim": round(sim, 4), "similar_groups": [{"group_id": a, "sim": round(b, 4)} for a, b in nb], "confusable": sim >= CONFUSABLE})

    taken, items = set(), []

    def pick(tag, n, ok):
        cands = [x for x in pool if x["group_id"] not in taken and ok(x)]
        conf = [x for x in cands if x["confusable"]]
        rest = [x for x in cands if not x["confusable"]]
        want = round(n * EXTRA)
        k = round(want * CONF_SHARE)
        chosen = conf[:k] + rest[:want - min(k, len(conf))]
        chosen += conf[k:][:want - len(chosen)]
        for x in chosen:
            taken.add(x["group_id"])
            cite, flt = None, None
            if tag == "조문+상황형":
                cite = rng.choice(cites[x["case_key"]])
            if tag == "조건 포함형":
                y = max(2014, int(x["reply_date"][:4]) - rng.randint(0, 2))
                flt = {"gubun": x["gubun"], "date_from": f"{y}-01-01"}
            items.append({"tag": tag, "rank": len([i for i in items if i["tag"] == tag]), "gold_group_id": x["group_id"], "source_case_key": x["case_key"],
                          "gubun": x["gubun"], "category": x["category_final"], "reply_date": x["reply_date"], "doc_class": x["doc_class"],
                          "cited_article": cite, "filter": flt, "source_question": x["question"], "confusable_sim": x["confusable_sim"],
                          "similar_groups": x["similar_groups"], "_title": x["title"], "_answer": x["answer"], "_reason": x["reason"]})

    pick("복합 쟁점형", QUOTA["복합 쟁점형"], lambda x: MULTI.search(x["question"]))
    pick("비조치 신청형", QUOTA["비조치 신청형"], lambda x: x["gubun"] == "비조치의견서")
    pick("조문+상황형", QUOTA["조문+상황형"], lambda x: bool(cites.get(x["case_key"])))
    for tag in ("조건 포함형", "경계 조건형", "구어체형", "우회 서술형"):
        pick(tag, QUOTA[tag], lambda x: True)
    return items, dropped, len(pool)


def write_q(item):
    title, answer, reason = item.pop("_title") or "", item.pop("_answer") or "", item.pop("_reason") or ""
    cite = item.get("cited_article")
    src = "\n".join(f"{n}: {t}" for n, t in (("제목", title), ("질의요지", item["source_question"][:1500]), ("회답", answer[:600])) if t)
    full = "\n".join((title, item["source_question"], answer, reason[:1500]))
    guide = GUIDE[item["tag"]].format(cite=cite, gubun=item["gubun"], year=(item["filter"] or {}).get("date_from", "")[:4])
    extra = ""
    for _ in range(4):
        r = rag.client.chat.completions.create(model=rag.CHAT_MODEL, reasoning_effort="low", response_format={"type": "json_object"},
                                               messages=[{"role": "system", "content": BASE + guide + extra}, {"role": "user", "content": src}])
        try:
            q = json.loads(r.choices[0].message.content)["question"].strip()
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        copied = longest_copy(q.replace(cite, "") if cite else q, full)
        ok_cite = not cite or re.sub(r"\s", "", cite) in re.sub(r"\s", "", q)
        if q and MIN_LEN <= len(q) <= MAX_LEN and copied <= MAX_COPY and ok_cite:
            return {**item, "question": q, "longest_copy": copied, "model": rag.CHAT_MODEL}
        extra = f"\n- 앞선 답은 {len(q)}자이고 원문과 같은 구간이 {copied}자이거나 지정 조문이 빠졌습니다. {MIN_LEN}~{MAX_LEN}자로, 원문과 같은 구간이 {MAX_COPY}자 이하가 되게 다시 쓰세요."
    return {**item, "question": None, "longest_copy": None, "model": rag.CHAT_MODEL}


def main():
    items, dropped, n_pool = build()
    cache_path = EVAL / "_final_test_raw.jsonl"
    done = {x["source_case_key"]: x for x in map(json.loads, cache_path.open(encoding="utf-8"))} if cache_path.exists() else {}
    todo = [dict(x) for x in items if x["source_case_key"] not in done]
    t0 = time.time()
    with ThreadPoolExecutor(5) as ex, cache_path.open("a", encoding="utf-8") as f:
        for r in ex.map(write_q, todo):
            done[r["source_case_key"]] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    final, short = [], {}
    for tag, n in QUOTA.items():
        ok = [done[x["source_case_key"]] for x in sorted((i for i in items if i["tag"] == tag), key=lambda i: i["rank"]) if done[x["source_case_key"]]["question"]]
        conf = [x for x in ok if x["confusable_sim"] >= CONFUSABLE]
        rest = [x for x in ok if x["confusable_sim"] < CONFUSABLE]
        got = conf[:round(n * CONF_SHARE)]
        got += rest[:n - len(got)]
        got += conf[len(conf[:round(n * CONF_SHARE)]):][:n - len(got)]
        if len(got) < n:
            short[tag] = n - len(got)
        final += got
    random.Random(SEED).shuffle(final)
    with (EVAL / "final_test.jsonl").open("w", encoding="utf-8") as f:
        for no, x in enumerate(final, 1):
            x = {k: v for k, v in x.items() if k != "rank"}
            f.write(json.dumps({"qid": f"FINAL-{no:03d}", "split": "final_test", **x}, ensure_ascii=False) + "\n")
    manifest("d1_final_test", n=len(final), pool=n_pool, dropped=dict(dropped), short=short, called=len(todo),
             tags=dict(Counter(x["tag"] for x in final)), confusable=sum(x["confusable_sim"] >= CONFUSABLE for x in final), seconds=round(time.time() - t0))


if __name__ == "__main__":
    main()
