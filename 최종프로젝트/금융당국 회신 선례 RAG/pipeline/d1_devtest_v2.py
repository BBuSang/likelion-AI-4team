"""기획안 수량의 개발셋(Dev) 60, 테스트셋(Test) 50을 OpenAI로 새로 만든다. 정답은 원본 사례의 묶음.
- 예비 dev_80·test_60의 한계(회답을 보고 써서 질문이 구체적이고 쉬움)를 줄이려고 LLM에 제목과 질의요지만 준다
- 합성 질의셋, 예비 dev·test와 정답 묶음이 겹치지 않고, 새 dev와 test도 서로 겹치지 않는다
출력: data/eval/dev2.jsonl, test2.jsonl    이미 만든 항목은 다시 호출하지 않는다.
"""
import json
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import rag
from common import DATA, manifest
from d1_devtest import GUIDE, MULTI, load_done
from d1_rewrite import longest_copy

N_DEV, N_TEST, N_RULE, SEED, MAX_COPY, MAX_LEN = 60, 50, 11, 20261109, 15, 140
EVAL = DATA / "eval"
USED = ("rewrite_150.jsonl", "dev1.jsonl", "test1.jsonl")
BASE = """은행 직원이 사내 규제 검색 도구에 입력하는 실제 업무 질문을 쓰세요.
- 아래 질의에 대한 금융당국 회신을 찾으려는 질문이어야 합니다. 원문에 없는 사실은 더하지 마세요.
- 세부 수치와 부수적인 사정은 빼고 핵심 쟁점이 드러나게 쓰세요. 40~100자, 1~2문장의 자연스러운 말투로 쓰세요.
- 회사명, 문서 번호는 쓰지 마세요. 원문의 표현을 그대로 옮기지 말고 다른 말로 풀어 쓰세요.
JSON으로만 답하세요: {"question": "..."}"""


def write_q(item):
    src = item.pop("_source")
    extra, cite = "", item.get("cited_article")
    base = item["source_question"].replace(cite, "") if cite else item["source_question"]
    guide = GUIDE[item["tag"]].format(cite=cite, gubun=item["gubun"], year=(item["filter"] or {}).get("date_from", "")[:4])
    for _ in range(4):
        r = rag.client.chat.completions.create(model=rag.CHAT_MODEL, reasoning_effort="low", response_format={"type": "json_object"},
                                               messages=[{"role": "system", "content": BASE + guide + extra}, {"role": "user", "content": src}])
        try:
            q = json.loads(r.choices[0].message.content)["question"].strip()
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        copied = longest_copy(q.replace(cite, "") if cite else q, base)
        if q and copied <= MAX_COPY and len(q) <= MAX_LEN and (not cite or re.sub(r"\s", "", cite) in re.sub(r"\s", "", q)):
            return {**item, "question": q, "longest_copy": copied, "model": rag.CHAT_MODEL}
        extra = f"\n- 앞선 답은 {len(q)}자이고 원문과 {copied}자가 같거나 지정 조문이 빠졌습니다. {MAX_LEN}자 이내로 다시 쓰세요."
    return {**item, "question": None, "longest_copy": None, "model": rag.CHAT_MODEL}


def build():
    used = {json.loads(l)["gold_group_id"] for name in USED for l in open(EVAL / name, encoding="utf-8")}
    with rag.connect() as c:
        rows = c.execute("""select case_key, group_id, gubun, category_final, reply_date::text reply_date, doc_class, title, question from interp.cases
                            where indexable and not body_suspect and doc_class in ('general', 'rule_review_2016') and reply_date is not null order by case_key""").fetchall()
        arts = c.execute("""select a.case_key, l.law_name, a.raw from interp.case_articles a join interp.laws l using (law_id)
                            where a.resolve_method = 'explicit' and a.raw like '제%%' and l.law_kind <> '행정규칙' order by a.id""").fetchall()
    cites = defaultdict(list)
    for a in arts:
        cites[a["case_key"]].append(f"{a['law_name']} {a['raw']}")
    rng = random.Random(SEED)
    rng.shuffle(rows)
    seen, gen, rule = set(used), defaultdict(list), []
    for r in rows:
        if r["group_id"] in seen:
            continue
        if r["doc_class"] == "general" and len(r["question"] or "") >= 50:
            seen.add(r["group_id"])
            gen[r["category_final"]].append(r)
        elif r["doc_class"] == "rule_review_2016" and len(r["title"]) >= 10:
            seen.add(r["group_id"])
            rule.append(r)
    n_gen, size = N_DEV + N_TEST - N_RULE, sum(map(len, gen.values()))
    quota = {k: round(n_gen * len(v) / size) for k, v in gen.items()}
    for k in sorted(gen, key=lambda k: -len(gen[k])):
        if sum(quota.values()) == n_gen:
            break
        quota[k] += 1 if sum(quota.values()) < n_gen else -1
    picks = [x for k in sorted(gen, key=str) for x in gen[k][:quota[k]]]
    rng.shuffle(picks)
    multi = [x for x in picks if MULTI.search(x["question"])][:round(n_gen * 0.1)]   # 복합 쟁점형은 질의가 나뉜 사례에서 먼저 채운다
    multi_keys = {x["case_key"] for x in multi}
    rest = [x for x in picks if x["case_key"] not in multi_keys]
    weights = ["상황 설명형"] * 62 + ["조문 지정형"] * 22 + ["조건 포함형"] * 16
    items = [(x, "복합 쟁점형", None, None) for x in multi]
    for i, x in enumerate(rest):
        tag = weights[int(i * 100 / len(rest))]
        cite, flt = None, None
        if tag == "조문 지정형":
            if cites.get(x["case_key"]):
                cite = rng.choice(cites[x["case_key"]])
            else:
                tag = "상황 설명형"
        if tag == "조건 포함형":
            y = max(2014, int(x["reply_date"][:4]) - rng.randint(0, 2))
            flt = {"gubun": x["gubun"], "date_from": f"{y}-01-01"}
        items.append((x, tag, cite, flt))
    items += [(x, "규제 효력형", None, None) for x in rule[:N_RULE]]
    rng.shuffle(items)

    def make(x, tag, cite, flt, split):
        body = "\n".join(f"{n}: {x[s][:1500]}" for n, s in (("제목", "title"), ("질의요지", "question")) if x[s])
        return {"split": split, "gold_group_id": x["group_id"], "source_case_key": x["case_key"], "gubun": x["gubun"], "category": x["category_final"],
                "reply_date": x["reply_date"], "doc_class": x["doc_class"], "tag": tag, "cited_article": cite, "filter": flt, "source_question": x["question"] or x["title"], "_source": body}

    out = [make(x, tag, cite, flt, "dev" if no < N_DEV else "test") for no, (x, tag, cite, flt) in enumerate(items)]
    spare = [x for k in sorted(gen, key=str) for x in gen[k][quota[k]:]]   # 생성에 실패한 항목을 대신할 예비 사례
    rng.shuffle(spare)
    return out, [make(x, "상황 설명형", None, None, "spare") for x in spare[:20]]


def main():
    items, spare = build()
    paths = {"dev": EVAL / "dev2.jsonl", "test": EVAL / "test2.jsonl"}
    done = {**load_done(paths["dev"]), **load_done(paths["test"])}
    todo = [dict(x) for x in items if x["source_case_key"] not in done]
    t0 = time.time()
    with ThreadPoolExecutor(5) as ex:
        for r in ex.map(write_q, todo):
            done[r["source_case_key"]] = r
    replaced = 0
    for i, x in enumerate(items):
        while not done[x["source_case_key"]]["question"]:     # 실패 항목은 같은 분할의 예비 사례로 바꾼다
            alt = {**spare.pop(0), "split": x["split"]}
            done[alt["source_case_key"]] = write_q(dict(alt))
            items[i] = x = alt
            replaced += 1
    counters = {"dev": 0, "test": 0}
    with paths["dev"].open("w", encoding="utf-8") as d, paths["test"].open("w", encoding="utf-8") as t:
        for x in items:
            r = {k: v for k, v in done[x["source_case_key"]].items() if k != "_source"}
            s = x["split"]
            counters[s] += 1
            r["qid"] = f"{s.upper()}2-{counters[s]:03d}"
            (d if s == "dev" else t).write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest("d1_devtest_v2", dev=counters["dev"], test=counters["test"], called=len(todo), replaced=replaced,
             tags=dict(Counter(done[x["source_case_key"]]["tag"] for x in items)), seconds=round(time.time() - t0))


if __name__ == "__main__":
    main()
