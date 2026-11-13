"""D1 재작성 세트: 질의요지를 LLM으로 직원 말투로 바꿔 쓴다. 개발 중 빠른 비교용이며 원문과 표현이 비슷해 점수가 높게 나오므로 참고용이다.
표본은 분야 비율대로 뽑고 묶음당 한 건만 쓴다. 정답은 묶음(group_id) 단위. 질의요지만 API로 보낸다.
출력: data/eval/rewrite_150.jsonl   이미 만든 항목은 다시 호출하지 않는다.
"""
import json
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

from psycopg.rows import dict_row

import rag
from common import DATA, manifest

N, SEED, MIN_Q, MAX_COPY = 150, 20261110, 50, 15
OUT_FILE = DATA / "eval" / "rewrite_150.jsonl"
SYSTEM = """은행 직원이 사내 규제 검색 도구에 묻는 질문으로 바꿔 쓰세요.
- 아래 원문의 사안과 쟁점을 바꾸지 말고, 원문에 없는 사실을 더하지 마세요.
- 직원이 자기 업무 상황을 설명하며 묻는 자연스러운 말투로 1~2문장으로 쓰세요.
- 회사명, 날짜, 문서 번호, 법령 조문 번호는 쓰지 마세요. 법령 이름은 꼭 필요할 때만 쓰세요.
- 원문의 표현을 그대로 옮기지 말고 다른 말로 풀어 쓰세요.
JSON으로만 답하세요: {"question": "..."}"""


def longest_copy(a, b):
    """두 글에서 공백을 뺀 가장 긴 공통 구간의 길이."""
    x, y = re.sub(r"\s+", "", a), re.sub(r"\s+", "", b)
    return SequenceMatcher(None, x, y, autojunk=False).find_longest_match(0, len(x), 0, len(y)).size


def rewrite(item):
    extra = ""
    for _ in range(4):
        r = rag.client.chat.completions.create(model=rag.CHAT_MODEL, reasoning_effort="low", response_format={"type": "json_object"},
                                               messages=[{"role": "system", "content": SYSTEM + extra}, {"role": "user", "content": item["source_question"]}])
        try:
            q = json.loads(r.choices[0].message.content)["question"].strip()
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        copied = longest_copy(q, item["source_question"])
        if q and copied <= MAX_COPY:
            return {**item, "question": q, "longest_copy": copied}
        extra = f"\n- 앞선 답은 원문과 {copied}자가 똑같았습니다. {MAX_COPY}자 넘게 같은 표현을 쓰지 마세요."
    return {**item, "question": None, "longest_copy": None}


def sample():
    with rag.connect() as c:
        rows = c.execute("""select c.case_key, c.group_id, c.gubun, c.category_final, c.reply_date::text reply_date, c.question from interp.cases c
                            where c.indexable and not c.body_suspect and c.doc_class = 'general' and length(c.question) >= %s order by c.case_key""", (MIN_Q,)).fetchall()
    rng = random.Random(SEED)
    rng.shuffle(rows)
    used, by = set(), defaultdict(list)
    for r in rows:
        if r["group_id"] not in used:
            used.add(r["group_id"])
            by[r["category_final"]].append(r)
    total = sum(map(len, by.values()))
    quota = {k: round(N * len(v) / total) for k, v in by.items()}
    for k in sorted(by, key=lambda k: -len(by[k])):      # 반올림 차이는 큰 분야에서 맞춘다
        if sum(quota.values()) == N:
            break
        quota[k] += 1 if sum(quota.values()) < N else -1
    return [{"qid": f"RW{i:03d}", "gold_group_id": x["group_id"], "source_case_key": x["case_key"], "gubun": x["gubun"], "category": x["category_final"], "reply_date": x["reply_date"], "tag": "상황 설명형", "source_question": x["question"]}
            for i, x in enumerate((x for k in sorted(by, key=str) for x in by[k][:quota[k]]), 1)]


def main():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    items = sample()
    done = {}
    if OUT_FILE.exists():
        done = {x["qid"]: x for x in map(json.loads, OUT_FILE.open(encoding="utf-8")) if x.get("question")}
    todo = [x for x in items if x["qid"] not in done or done[x["qid"]]["source_case_key"] != x["source_case_key"]]
    started = time.time()
    with ThreadPoolExecutor(5) as ex:
        for r in ex.map(rewrite, todo):
            done[r["qid"]] = r
    rows = [done[x["qid"]] for x in items]
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest("d1_rewrite", n=len(rows), failed=sum(not r["question"] for r in rows), called=len(todo), category=dict(Counter(r["category"] for r in rows)), gubun=dict(Counter(r["gubun"] for r in rows)), seconds=round(time.time() - started))


if __name__ == "__main__":
    main()
