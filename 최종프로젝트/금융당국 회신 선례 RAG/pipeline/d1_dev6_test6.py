"""D1 dev6 50문항 · test6 50문항: d1_dev5_test5.py와 같은 설계로 새 정답 묶음에서 만든 세트 (사용자 요청 2026-11-11).
- dev5 설계와 다른 점: 유형별 문항 수를 dev·test 모두 dev5의 dev 수와 같게 해서 각 50문항, 시드·캐시 분리, 제외 대상에 dev5·test5 포함
설계 (검색 점수를 보기 전에 정함, 검색 결과로 문항을 고르지 않는다)
- 정답 후보: 검색 대상 대표 사례 중 질의요지 80자·회답 40자 이상, 기존 모든 평가 세트(합성·dev1~4·test1~4·final_test)의 정답 묶음과 겹치지 않음(dev5·test5 포함). 기존 파일은 이 제외에만 쓴다
- 구분 비율: 비조치의견서를 약 45%로 늘린다(기존 세트는 법령해석 약 73%). 유형별 지정 구분이 없으면 법령해석·비조치를 번갈아 배정
- 분야(category_final)가 고르게 섞이도록, 같은 유형 안에서 지금까지 덜 뽑힌 분야부터 고른다
- 질문 유형 9가지(아래 TYPES): 실무 기획, 비조치 사전검토, 경계 조건, 우회 서술, 구어체, 검색어 나열, 조문 지정, 복합 쟁점, 규제 효력(2016 규제 정비 건)
- 생성: OpenAI(.env의 CHAT_MODEL)로 회신 제목·질의요지·회답 일부를 보고 질문을 쓴다. 결론은 쓰지 않고, 원문과 같은 글자 구간 10자 이하
- 검수: 별도 LLM 호출로 (1) 이 회신이 질문의 답이 되는가 (2) 질문에 회신의 결론이 드러났는가를 판정해, 통과한 것만 쓴다. 예비 후보(1.5배)로 채운다
- dev6와 test6는 정답 묶음이 겹치지 않는다(유형별로 한꺼번에 뽑아 무작위 분할). test6는 한 번만 평가하는 세트이고, 이 스크립트는 test6 질문을 출력하지 않는다
출력: data/eval/dev6.jsonl, test6.jsonl    캐시: data/eval/_dev6_test6_raw.jsonl (이미 만든 항목은 다시 호출하지 않는다)
"""
import json
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import rag
from common import DATA, manifest

EVAL = DATA / "eval"
SEED = 20261111 + 6
MAX_COPY, EXTRA = 10, 1.5
CACHE = EVAL / "_dev6_test6_raw.jsonl"
# 유형: (dev 수, test 수, 구분 지정, doc_class, 길이 범위, 안내)
TYPES = {
    "실무 기획형": (7, 7, None, "general", (60, 220), "새 상품·서비스·업무 방식을 기획 중인 담당자가, 그 방식이 규제에 걸리는지 선례를 찾는 질문으로 쓰세요. 기획 내용(대상, 방식, 구조)을 구체적으로 쓰세요."),
    "비조치 사전검토형": (6, 6, "비조치의견서", "general", (60, 220), "시행 전에 제재 가능성을 확인하려는 질문으로 쓰세요. '이렇게 하면 나중에 조치받을 수 있는지', '비조치 의견을 받은 사례가 있는지' 같은 형태가 좋습니다."),
    "경계 조건형": (7, 7, None, "general", (60, 220), "결론을 가르는 조건(금액·기간·대상·방식·횟수 등) 하나를 분명히 넣고 '~인 경우에도 ~해도 되는지' 형태로 쓰세요."),
    "우회 서술형": (6, 6, None, "general", (80, 220), "법령·제도 이름이나 전문 용어를 거의 쓰지 말고, 실제 업무에서 벌어진 상황을 평범한 말로 설명하는 형태로 쓰세요."),
    "구어체형": (6, 6, None, "general", (30, 120), "영업점 직원이 메신저에 급히 치듯 짧은 구어체로 쓰세요. 줄임말, 불완전한 문장을 써도 됩니다."),
    "검색어형": (5, 5, None, "general", (10, 60), "문장이 아니라 검색창에 넣는 핵심어 3~6개를 띄어쓰기로 나열하세요(예: '적금 제휴카드 캐시백 특별이익 회신'). 조사·서술어는 쓰지 마세요."),
    "조문 지정형": (5, 5, None, "general", (60, 220), "반드시 다음 조문을 질문에 그대로 쓰세요: {cite}\n- 같은 조문을 다룬 회신이 많으므로 이 회신의 구체적인 상황을 함께 쓰세요."),
    "복합 쟁점형": (5, 5, None, "general", (80, 220), "이 회신이 다루는 쟁점 두 가지를 한 질문에 함께 묻는 형태로 쓰세요."),
    "규제 효력형": (3, 3, "비조치의견서", "rule_review_2016", (40, 200), "예전에 당국이 내린 지도·지침·모범규준 같은 규율이 지금도 지켜야 하는 것인지 묻는 형태로 쓰세요. 규율의 내용을 평범한 말로 설명하세요."),
}
BASE = """은행·금융회사 직원이 금융당국 회신(법령해석·비조치의견서) 검색 도구에 입력할 질문 하나를 쓰세요.
- 아래 회신이 정답이 되는 질문이어야 합니다. 회신이 다루는 대상·행위·조건을 담되, 회신의 결론(가능/불가, 해당/비해당, 당국의 판단)은 쓰지 마세요.
- 회신에 쓰인 문장을 그대로 옮기지 말고 직원의 말로 바꾸세요. 원문과 같은 글자 구간이 10자를 넘으면 안 됩니다.
- 회사명, 문서 번호, 사람 이름, 회신 날짜는 쓰지 마세요. 원문에 없는 사실을 더하지 마세요.
- {lo}~{hi}자로 쓰세요.
- 유형: {guide}
JSON으로만 답하세요: {{"question": "..."}}"""
CHECK = """금융당국 회신 검색 평가용 질문을 검수합니다. [질문]과 [회신]을 보고 JSON으로만 답하세요.
1) answerable: 이 회신이 질문이 묻는 쟁점에 직접 답하는가 (true/false). 회신 내용과 질문의 사실관계가 어긋나거나, 회신이 다루지 않는 것을 물으면 false
2) leak: 질문에 회신의 결론(가능/불가, 해당/비해당, 당국 판단)이 이미 드러나 있는가 (true/false)
출력: {"answerable": true, "leak": false, "reason": "한 문장"}"""


def longest_copy(q, src):
    q, src = re.sub(r"\s+", "", q), re.sub(r"\s+", "", src)
    if not q or not src:
        return 0
    return SequenceMatcher(None, q, src, autojunk=False).find_longest_match(0, len(q), 0, len(src)).size


def used_groups():
    used = set()
    for p in EVAL.glob("*.jsonl"):
        if p.name.startswith("_") or p.stem in ("dev6", "test6", "unanswerable_20"):
            continue
        for line in open(p, encoding="utf-8"):
            g = json.loads(line).get("gold_group_id")
            if g:
                used.add(g)
    return used


def build(reserve=None):
    """reserve: {유형: 추가 후보 수}. 검수로 모자란 유형만 맨 뒤에 예비 후보를 더 뽑는다(앞서 뽑은 후보는 그대로)."""
    used = used_groups()
    with rag.connect() as c:
        rows = c.execute("""select c.case_key, c.group_id, c.gubun, c.category_final, c.reply_date::text reply_date, c.doc_class, c.title, c.question, c.answer, c.reason
                            from interp.cases c join interp.case_groups g on g.representative_case_key = c.case_key
                            where c.indexable and not c.body_suspect and c.reply_date is not null order by c.case_key""").fetchall()
        arts = c.execute("""select a.case_key, l.law_name, a.raw from interp.case_articles a join interp.laws l using (law_id)
                            where a.resolve_method = 'explicit' and a.raw like '제%%' and l.law_kind <> '행정규칙' order by a.id""").fetchall()
    cites = defaultdict(list)
    for a in arts:
        cites[a["case_key"]].append(f"{a['law_name']} {a['raw']}")
    pool = [r for r in rows if r["group_id"] not in used and len(r["question"] or "") >= 80 and len(r["answer"] or "") >= 40]
    rng = random.Random(SEED)
    rng.shuffle(pool)
    taken, cat_n, items = set(), Counter(), []
    alt = 0
    plan = [(tag, round((nd + nt) * EXTRA), 0) for tag, (nd, nt, *_r) in TYPES.items()]
    plan += [(tag, n, round((TYPES[tag][0] + TYPES[tag][1]) * EXTRA)) for tag, n in (reserve or {}).items()]
    for tag, want, start in plan:
        _, _, gubun, dclass, _, _ = TYPES[tag]
        for i in range(start, start + want):
            g = gubun
            if g is None:                               # 지정이 없으면 번갈아 배정 (비조치 약 45%)
                g = "비조치의견서" if alt % 9 in (1, 3, 5, 7) else "법령해석"
                alt += 1
            cands = [x for x in pool if x["group_id"] not in taken and x["gubun"] == g and x["doc_class"] == dclass and (tag != "조문 지정형" or cites.get(x["case_key"]))]
            if not cands:
                cands = [x for x in pool if x["group_id"] not in taken and x["doc_class"] == dclass and (tag != "조문 지정형" or cites.get(x["case_key"]))]
            x = min(cands, key=lambda r: cat_n[r["category_final"]])          # 덜 뽑힌 분야부터 (같으면 섞인 순서)
            taken.add(x["group_id"])
            cat_n[x["category_final"]] += 1
            items.append({"tag": tag, "rank": i, "gold_group_id": x["group_id"], "source_case_key": x["case_key"], "gubun": x["gubun"], "category": x["category_final"],
                          "reply_date": x["reply_date"], "doc_class": x["doc_class"], "cited_article": rng.choice(cites[x["case_key"]]) if tag == "조문 지정형" else None,
                          "_title": x["title"] or "", "_question": x["question"] or "", "_answer": x["answer"] or "", "_reason": x["reason"] or ""})
    return items, len(pool)


def call_json(system, user, effort="low"):
    r = rag.client.chat.completions.create(model=rag.CHAT_MODEL, reasoning_effort=effort, response_format={"type": "json_object"},
                                           messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    try:
        return json.loads(r.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return {}


def write_q(item):
    tag, cite = item["tag"], item["cited_article"]
    lo, hi = TYPES[tag][4]
    src = "\n".join(f"{n}: {t}" for n, t in (("구분", item["gubun"]), ("제목", item["_title"]), ("질의요지", item["_question"][:1500]), ("회답", item["_answer"][:700])) if t)
    full = "\n".join((item["_title"], item["_question"], item["_answer"], item["_reason"]))
    system = BASE.format(lo=lo, hi=hi, guide=TYPES[tag][5].format(cite=cite))
    extra, q, copied = "", None, None
    for _ in range(3):
        q = (call_json(system + extra, src).get("question") or "").strip()
        copied = longest_copy(q.replace(cite, "") if cite else q, full)
        ok_cite = not cite or re.sub(r"\s", "", cite) in re.sub(r"\s", "", q)
        bad = re.search(r"[LO]\d{3,}|\d{4}[.-]\d{1,2}[.-]\d{1,2}", q)
        if q and lo <= len(q) <= hi and copied <= MAX_COPY and ok_cite and not bad:
            break
        extra = f"\n- 앞선 답은 {len(q)}자, 원문과 같은 구간 {copied}자였거나 지정 조문이 빠졌습니다. {lo}~{hi}자, 같은 구간 {MAX_COPY}자 이하로 다시 쓰세요."
        q = None
    out = {k: v for k, v in item.items() if not k.startswith("_")}
    if not q:
        return {**out, "question": None, "check": None, "model": rag.CHAT_MODEL}
    chk = call_json(CHECK, f"[질문]\n{q}\n\n[회신]\n제목: {item['_title']}\n질의요지: {item['_question'][:1500]}\n회답: {item['_answer'][:1000]}\n이유: {item['_reason'][:800]}", "medium")
    return {**out, "question": q, "longest_copy": copied, "check": {"answerable": chk.get("answerable"), "leak": chk.get("leak"), "reason": chk.get("reason")}, "model": rag.CHAT_MODEL}


def generate(items, done):
    todo = [dict(x) for x in items if x["source_case_key"] not in done]
    with ThreadPoolExecutor(5) as ex, CACHE.open("a", encoding="utf-8") as f:
        for r in ex.map(write_q, todo):
            done[r["source_case_key"]] = r
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(todo)


def passed(items, done, tag):
    return sum(1 for x in items if x["tag"] == tag and done[x["source_case_key"]]["question"] and done[x["source_case_key"]]["check"]["answerable"] is True and done[x["source_case_key"]]["check"]["leak"] is False)


def main():
    t0 = time.time()
    done = {x["source_case_key"]: x for x in map(json.loads, CACHE.open(encoding="utf-8"))} if CACHE.exists() else {}
    items, n_pool = build()
    called = generate(items, done)
    reserve = {}
    for _ in range(3):                                   # 모자란 유형만 예비 후보를 더한다 (모자란 수의 2배씩, 최대 3번)
        need = {tag: TYPES[tag][0] + TYPES[tag][1] - passed(items, done, tag) for tag in TYPES}
        need = {t: n for t, n in need.items() if n > 0}
        if not need:
            break
        for t, n in need.items():
            reserve[t] = reserve.get(t, 0) + 2 * n
        items, n_pool = build(reserve)
        called += generate(items, done)
    rng = random.Random(SEED)
    dev, test, short, drop = [], [], {}, Counter()
    for tag, (nd, nt, *_rest) in TYPES.items():
        ok = []
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
        got = ok[:nd + nt]
        if len(got) < nd + nt:
            short[tag] = nd + nt - len(got)
        rng.shuffle(got)
        k = round(len(got) * nd / (nd + nt))
        dev += got[:k]
        test += got[k:]
    rng.shuffle(dev)
    rng.shuffle(test)
    for name, rows, split, prefix in (("dev6", dev, "dev", "DEV6"), ("test6", test, "test", "TEST6")):
        with (EVAL / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for no, x in enumerate(rows, 1):
                f.write(json.dumps({"qid": f"{prefix}-{no:03d}", "split": split, **{k: v for k, v in x.items() if k != "rank"}}, ensure_ascii=False) + "\n")
    stats = {"dev6": len(dev), "test6": len(test), "pool": n_pool, "called": called, "reserve": reserve, "dropped": dict(drop), "short": short,
             "gubun_dev": dict(Counter(x["gubun"] for x in dev)), "gubun_test": dict(Counter(x["gubun"] for x in test)),
             "category_all": dict(Counter(x["category"] for x in dev + test)), "tags_dev": dict(Counter(x["tag"] for x in dev)), "tags_test": dict(Counter(x["tag"] for x in test)),
             "seconds": round(time.time() - t0)}
    manifest("d1_dev6_test6", **stats)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
