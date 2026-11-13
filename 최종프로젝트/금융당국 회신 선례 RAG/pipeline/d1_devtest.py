"""D1 예비 평가용 dev 80, test 60: 회신을 바탕으로 LLM이 실무형 질문을 만든다. 정답은 원본 사례의 묶음.
성능 수준을 먼저 확인하는 용도이며, 최종 평가는 별도의 메인 질문 셋으로 한다.
- 합성 질의셋(rewrite_150)과 정답 묶음이 겹치지 않고, dev와 test도 서로 겹치지 않는다
- 분야 비율대로 뽑고 질문 유형을 섞는다 (상황 설명 / 조문 지정 / 조건 포함 / 복합 쟁점 / 규제 효력)
출력: data/eval/dev1.jsonl, test1.jsonl    이미 만든 항목은 다시 호출하지 않는다.
"""
import json
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import rag
from common import DATA, manifest
from d1_rewrite import longest_copy

N_DEV, N_TEST, SEED, MAX_COPY = 80, 60, 20261111, 15
EVAL = DATA / "eval"
BASE = """은행 직원이 사내 규제 검색 도구에 입력하는 실제 업무 질문을 쓰세요.
- 아래 회신이 답이 되는 질문이어야 하고, 다른 회신과 구분될 만큼 이 사안의 구체적인 사실을 담으세요.
- 회신의 결론이나 당국의 판단을 질문에 쓰지 마세요. 원문에 없는 사실도 더하지 마세요.
- 회사명, 문서 번호는 쓰지 마세요. 1~3문장의 자연스러운 말투로 쓰세요.
- 원문의 표현을 그대로 옮기지 말고 다른 말로 풀어 쓰세요.
JSON으로만 답하세요: {"question": "..."}"""
GUIDE = {
    "상황 설명형": "\n- 직원이 자기 업무 상황을 설명하며 묻는 형태로 쓰세요. 조문 번호는 쓰지 마세요.",
    "조문 지정형": "\n- 반드시 다음 조문을 질문에 그대로 언급하세요: {cite}",
    "조건 포함형": "\n- 반드시 '{gubun}' 회신 중에서, {year}년 이후 회신으로 한정해 찾아 달라는 조건을 문장에 넣으세요. 조문 번호는 쓰지 마세요.",
    "복합 쟁점형": "\n- 이 회신의 쟁점 두 가지를 함께 묻는 한 질문으로 쓰세요. 조문 번호는 쓰지 마세요.",
    "규제 효력형": "\n- 이 회신은 행정지도·모범규준의 효력을 다룹니다. 해당 규제가 지금도 효력이 있는지, 계속 지켜야 하는지를 묻는 질문으로 쓰세요. 규제 이름은 써도 됩니다.",
}
MULTI = re.compile(r"①|\(1\)|질의\s*1|\(가\)|1\)")


def write_q(item):
    src = item.pop("_source")
    extra, cite = "", item.get("cited_article")
    base = re.sub(re.escape(cite) if cite else r"$^", "", item["source_question"]) if cite else item["source_question"]
    guide = GUIDE[item["tag"]].format(cite=cite, gubun=item["gubun"], year=(item["filter"] or {}).get("date_from", "")[:4])
    for _ in range(4):
        r = rag.client.chat.completions.create(model=rag.CHAT_MODEL, reasoning_effort="low", response_format={"type": "json_object"},
                                               messages=[{"role": "system", "content": BASE + guide + extra}, {"role": "user", "content": src}])
        try:
            q = json.loads(r.choices[0].message.content)["question"].strip()
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        copied = longest_copy(q.replace(cite, "") if cite else q, base)
        if q and copied <= MAX_COPY and (not cite or re.sub(r"\s", "", cite) in re.sub(r"\s", "", q)):
            return {**item, "question": q, "longest_copy": copied}
        extra = f"\n- 앞선 답은 원문과 {copied}자가 같거나 지정 조문이 빠졌습니다. 다시 쓰세요."
    return {**item, "question": None, "longest_copy": None}


def build():
    used = {json.loads(l)["gold_group_id"] for l in open(EVAL / "rewrite_150.jsonl", encoding="utf-8")}
    with rag.connect() as c:
        rows = c.execute("""select case_key, group_id, gubun, category_final, reply_date::text reply_date, doc_class, title, question, answer, reason from interp.cases
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
        if r["doc_class"] == "general" and len(r["question"]) >= 50:
            seen.add(r["group_id"])
            gen[r["category_final"]].append(r)
        elif r["doc_class"] == "rule_review_2016" and len(r["title"]) >= 10:
            seen.add(r["group_id"])
            rule.append(r)
    total, n_rule = N_DEV + N_TEST, 14
    n_gen, size = total - n_rule, sum(map(len, gen.values()))
    quota = {k: round(n_gen * len(v) / size) for k, v in gen.items()}
    for k in sorted(gen, key=lambda k: -len(gen[k])):
        if sum(quota.values()) == n_gen:
            break
        quota[k] += 1 if sum(quota.values()) < n_gen else -1
    picks = [x for k in sorted(gen, key=str) for x in gen[k][:quota[k]]]
    rng.shuffle(picks)
    weights = ["상황 설명형"] * 55 + ["조문 지정형"] * 20 + ["조건 포함형"] * 15 + ["복합 쟁점형"] * 10
    items = []
    for i, x in enumerate(picks):
        tag = weights[int(i * 100 / len(picks))]
        cite, flt = None, None
        if tag == "조문 지정형":
            if cites.get(x["case_key"]):
                cite = rng.choice(cites[x["case_key"]])
            else:
                tag = "상황 설명형"
        if tag == "복합 쟁점형" and not MULTI.search(x["question"]):
            tag = "상황 설명형"
        if tag == "조건 포함형":
            y = max(2014, int(x["reply_date"][:4]) - rng.randint(0, 2))
            flt = {"gubun": x["gubun"], "date_from": f"{y}-01-01"}
        items.append((x, tag, cite, flt))
    items += [(x, "규제 효력형", None, None) for x in rule[:n_rule]]
    rng.shuffle(items)

    def make(x, tag, cite, flt, split):
        body = "\n".join(f"{n}: {x[s][:1500]}" for n, s in (("제목", "title"), ("질의요지", "question"), ("회답", "answer"), ("이유", "reason")) if x[s])
        return {"split": split, "gold_group_id": x["group_id"], "source_case_key": x["case_key"], "gubun": x["gubun"], "category": x["category_final"],
                "reply_date": x["reply_date"], "doc_class": x["doc_class"], "tag": tag, "cited_article": cite, "filter": flt, "source_question": x["question"] or x["title"], "_source": body}

    out = [make(x, tag, cite, flt, "dev" if no < N_DEV else "test") for no, (x, tag, cite, flt) in enumerate(items)]
    spare = [x for k in sorted(gen, key=str) for x in gen[k][quota[k]:]]   # 생성에 실패한 항목을 대신할 예비 사례
    rng.shuffle(spare)
    return out, [make(x, "상황 설명형", None, None, "spare") for x in spare[:20]]


def load_done(path):
    return {x["source_case_key"]: x for x in map(json.loads, path.open(encoding="utf-8")) if x.get("question")} if path.exists() else {}


def main():
    items, spare = build()
    paths = {"dev": EVAL / "dev1.jsonl", "test": EVAL / "test1.jsonl"}
    done = {**load_done(paths["dev"]), **load_done(paths["test"])}
    todo = [x for x in items if x["source_case_key"] not in done]
    t0 = time.time()
    with ThreadPoolExecutor(5) as ex:
        for r in ex.map(write_q, todo):
            done[r["source_case_key"]] = r
    replaced = 0
    for i, x in enumerate(items):
        while not done[x["source_case_key"]]["question"]:     # 실패 항목은 같은 분할의 예비 사례로 바꾼다
            alt = {**spare.pop(0), "split": x["split"]}
            if alt["source_case_key"] not in done or not done[alt["source_case_key"]]["question"]:
                done[alt["source_case_key"]] = write_q(dict(alt))
            done[alt["source_case_key"]]["split"] = x["split"]
            items[i] = x = alt
            replaced += 1
    counters = {"dev": 0, "test": 0}
    with paths["dev"].open("w", encoding="utf-8") as d, paths["test"].open("w", encoding="utf-8") as t:
        for x in items:
            r = {k: v for k, v in done[x["source_case_key"]].items() if k != "_source"}
            s = x["split"]
            counters[s] += 1
            r["qid"] = f"{s.upper()}{counters[s]:03d}"
            (d if s == "dev" else t).write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest("d1_devtest", dev=counters["dev"], test=counters["test"], failed=sum(not done[x["source_case_key"]]["question"] for x in items),
             called=len(todo), replaced=replaced, tags=dict(Counter(done[x["source_case_key"]]["tag"] for x in items)), seconds=round(time.time() - t0))


if __name__ == "__main__":
    main()
