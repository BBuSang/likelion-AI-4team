"""5-6-1 평가 질문-정답 세트(골드셋) 초안 만들기: dev1~3, test1~3 각 50문항 (OpenAI)

1) 조문을 법령·유형별로 뽑아 '칸(slot)'을 만든다 (조문은 300칸 모두 서로 다름)
2) 칸마다 OpenAI가 질문을 쓴다 (LLM 호출은 5칸 묶음 1번. 다시 부르는 건 '건너뛰기·조문 베끼기'로 걸린 칸만)
3) eval/queries.csv, eval/gold.csv, eval/gold_review.csv 를 쓴다

정답 조문은 '질문을 쓴 원본 조문'이라 AI가 정답을 고르지 않는다 (복수법령형의 두 번째 정답만 후보 중 AI 선택).
정답 확정은 사용자 확인 뒤 (기획안 5-6-1) -> status = 'AI초안'.
"""
import argparse
import csv
import json
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

import pandas as pd
import requests

from common import ENV, ROOT, WORK

EVAL = ROOT / "eval"
SEED = 20261113
SETS = ["dev1", "dev2", "dev3", "test1", "test2", "test3"]
PER_SET = 50
DIFF_MIX = {}  # 난이도 섞기 (비어 있으면 난이도 미지정)
TYPE_MIX = {"상황형": 20, "용어형": 8, "조문지정형": 6, "하위법령형": 8, "복수법령형": 8}  # 세트당 50
SCOPE_W = {"A": 0.50, "B": 0.15, "C": 0.15, "D": 0.20}  # 핵심 44개 법률을 절반으로
LAW_CAP_SET, LAW_CAP_ALL = 3, 7  # 한 세트에 같은 법령 최대 3, 전체 최대 7
GENERIC = {  # 어느 법에나 있는 조문 제목: 질문만으로 어느 법인지 못 가림
    "목적", "벌칙", "준용규정", "과태료", "양벌규정", "다른 법률과의 관계", "규제의 재검토", "적용범위",
    "청문", "과태료의 부과기준", "업무의 위탁", "권한의 위탁", "업무", "재검토기한", "수수료", "정관",
    "임원", "이사회", "미수범", "권한의 위임", "권한의 위임·위탁", "시행일", "경과조치", "벌칙 적용 시 공무원 의제",
    "벌칙 적용에서 공무원 의제", "위임규정", "시행세칙", "기타", "보칙", "고유식별정보의 처리",
    "민감정보 및 고유식별정보의 처리", "고유식별정보의 처리", "과징금", "감독", "정의", "용어의 정의",
}
DEF_TITLES = {"정의", "용어의 정의"}
NUM = re.compile(r"\d+\s*(일|개월|월|년|원|만원|억원|%|퍼센트|배|명|세|이내|이상|이하|미만|초과)|\d+분의|100분의|이내")


# ---------------------------------------------------------------- 데이터
def load():
    a = pd.read_csv(WORK / "02_clean" / "articles.csv", dtype=str, keep_default_na=False)
    laws = pd.read_csv(WORK / "01_fetch" / "laws.csv", dtype=str, keep_default_na=False)
    for c in ("searchable", "has_image", "has_table", "has_delegation", "is_reference_list"):
        a[c] = a[c].str.lower().eq("true")
    a["n_chars"] = a["n_chars"].astype(int)
    pool = a[a.searchable & ~a.has_image & ~a.has_table & ~a.is_reference_list & a.n_chars.between(90, 1200)].copy()
    # 같은 글이 여러 법에 반복되는 조문은 뺀다 (정답이 하나로 안 정해짐)
    pool = pool[~pool.duplicated("text", keep=False)]
    pool = pool[~pool.text.str.contains(r"^\s*(?:삭제|종전 제)", regex=True)]
    return a, laws, pool


def short_name(n):  # 법령명 그대로 (질문에 쓰기 위함)
    return n


# ---------------------------------------------------------------- 칸 뽑기
def build_slots(pool, laws, allart):
    rnd = random.Random(SEED)
    lawinfo = laws.set_index("uid")
    children = defaultdict(list)
    for _, r in laws.iterrows():
        if r.parent_uid:
            children[r.parent_uid].append(r.uid)
    used_art, law_all = set(), Counter()
    if (EVAL / "gold.csv").exists():  # 앞서 만든 세트와 조문이 겹치지 않게
        used_art |= set(pd.read_csv(EVAL / "gold.csv", dtype=str).article_uid)
    by_law = {u: g for u, g in pool.groupby("law_uid")}
    slots = []

    def eligible(kind, g):
        if kind == "용어형":
            return g[g.title.isin(DEF_TITLES)]
        g = g[~g.title.isin(GENERIC)]
        if kind == "하위법령형":
            return g[(g.law_level != "법률") & g.text.str.contains(NUM)]
        if kind == "복수법령형":
            return g[(g.law_level == "법률") & g.has_delegation & g.law_uid.map(lambda u: len(children.get(u, [])) > 0)]
        return g  # 상황형·조문지정형

    order = []
    for si, s in enumerate(SETS):  # 세트마다 유형 칸 만들고 섞음
        for k, n in TYPE_MIX.items():
            order += [(s, k)] * n
    # 까다로운 유형부터 뽑아야 조문이 안 모자람
    prio = {"복수법령형": 0, "하위법령형": 1, "용어형": 2, "조문지정형": 3, "상황형": 4}
    order.sort(key=lambda x: (prio[x[1]], x[0]))
    set_law = defaultdict(Counter)
    for s, kind in order:
        for _try in range(400):
            sc = rnd.choices(list(SCOPE_W), weights=list(SCOPE_W.values()))[0]
            cands = [u for u in by_law if lawinfo.loc[u, "scope"] == sc and law_all[u] < LAW_CAP_ALL and set_law[s][u] < LAW_CAP_SET]
            if not cands:
                continue
            u = rnd.choice(cands)
            g = eligible(kind, by_law[u])
            g = g[~g.article_uid.isin(used_art)]
            if len(g):
                r = g.iloc[rnd.randrange(len(g))]
                used_art.add(r.article_uid)
                law_all[u] += 1
                set_law[s][u] += 1
                slots.append({"set": s, "q_type": kind, "article_uid": r.article_uid})
                break
        else:
            raise SystemExit(f"{s} {kind}: 뽑을 조문이 모자람")
    for s in SETS:  # 세트마다 난이도 섞기 (유형과 무관하게 무작위)
        ns = [i for i, x in enumerate(slots) if x["set"] == s]
        ds = [d for d, n in DIFF_MIX.items() for _ in range(n)]
        rnd.shuffle(ds)
        for i, d in zip(ns, ds):
            slots[i]["difficulty"] = d
    return slots, children, by_law, used_art, law_all, set_law, lawinfo


def replacement(slot, pool, laws, children, by_law, used_art, law_all, set_law, lawinfo, rnd):
    """건너뛴 칸의 새 조문 (같은 세트·같은 유형)"""
    s, kind = slot["set"], slot["q_type"]
    for _try in range(800):
        sc = rnd.choices(list(SCOPE_W), weights=list(SCOPE_W.values()))[0]
        cands = [u for u in by_law if lawinfo.loc[u, "scope"] == sc and law_all[u] < LAW_CAP_ALL + 2 and set_law[s][u] < LAW_CAP_SET + 1]
        if not cands:
            continue
        u = rnd.choice(cands)
        g = by_law[u]
        if kind == "용어형":
            g = g[g.title.isin(DEF_TITLES)]
        else:
            g = g[~g.title.isin(GENERIC)]
            if kind == "하위법령형":
                g = g[(g.law_level != "법률") & g.text.str.contains(NUM)]
            elif kind == "복수법령형":
                g = g[(g.law_level == "법률") & g.has_delegation & g.law_uid.map(lambda x: len(children.get(x, [])) > 0)]
        g = g[~g.article_uid.isin(used_art)]
        if len(g):
            r = g.iloc[rnd.randrange(len(g))]
            used_art.add(r.article_uid)
            law_all[u] += 1
            set_law[s][u] += 1
            return {"set": s, "q_type": kind, "article_uid": r.article_uid, "difficulty": slot.get("difficulty", "")}
    raise SystemExit("대체 조문 없음")


# ---------------------------------------------------------------- 복수법령형 후보
def bigrams(t):
    t = re.sub(r"\s+", "", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def decree_candidates(row, pool, children, k=5):
    base = bigrams(row.title * 2 + row.text[:400])
    cands = pool[pool.law_uid.isin(children.get(row.law_uid, [])) & ~pool.title.isin(GENERIC)]
    sc = []
    for _, c in cands.iterrows():
        b = bigrams(c.title * 2 + c.text[:400])
        sc.append((len(base & b) / (len(base | b) or 1), c.article_uid))
    sc.sort(reverse=True)
    return [u for _, u in sc[:k]]


# ---------------------------------------------------------------- OpenAI
SYSTEM = """너는 한국 법령 검색(RAG) 평가용 질문을 만드는 출제자다. 조문을 보고, 그 조문이 정답이 되는 질문을 쓴다.

규칙:
- 질문은 실제 사람이 챗봇에 치는 말투(구어체, 한 문장 또는 두 문장). 조문 문장을 베끼지 말고 쉬운 말로 바꿔 쓴다.
- 질문만 읽고 이 조문이 정답이라는 것이 분명해야 한다. 조문의 핵심 주제어(업무·대상·상황)는 질문에 넣되, 조문의 문장·구절을 그대로 옮기지 않는다 (연속 15글자 이상 같으면 안 됨).
- 질문에 조문 번호(제N조)는 '조문지정형'일 때만 쓴다. 그 외 유형은 번호를 쓰지 않는다.
- 조문 안에 답이 있어야 한다. 조문에 없는 내용을 묻지 않는다.
- 너무 일반적이어서(목적·정의·벌칙·위임·시행일 등) 다른 법과 구분되지 않거나, 질문을 만들 수 없는 조문이면 skip=true.
유형:
- 상황형: 실제 업무·생활 상황을 일상어로 (예: "~하려면 어떻게 해야 해?", "~하면 어떻게 돼?")
- 용어형: 조문에 정의된 법률 용어 하나의 뜻을 묻는다. 어느 법의 용어인지 알 수 있게 분야를 질문에 넣는다
- 조문지정형: 법령명(또는 흔히 쓰는 줄임말)과 조문 번호를 넣어 그 조문의 내용을 묻는다 (예: "자본시장법 제9조에서 정한 …은 뭐야?")
- 하위법령형: 금액·기간·비율·기준 같은 구체적 수치·요건을 묻는다
- 복수법령형: 법률 조문과 후보 하위법령 조문이 함께 있어야 완전히 답할 수 있는 질문. 후보 중 실제로 그 질문의 답을 담은 조문 id 만 골라 gold_ids 에 넣는다. 맞는 후보가 없으면 skip=true.
난이도(입력에 difficulty 가 있을 때 반드시 따른다. 조문 자체는 바뀌지 않고 질문 표현만 달라진다):
- 쉬움: 조문의 핵심 주제어를 질문에 그대로 쓰는 직설적인 질문. 한 가지만 묻는다.
- 보통: 핵심 법률 용어를 일상어로 풀어 쓰거나 우회해서 묻는다. 주제어를 직접 쓰지 않아도 된다.
- 어려움: 구체적인 사례·상황(인물, 금액, 날짜, 사정 등)을 서술한 2~3문장의 질문. 법률 용어를 거의 쓰지 않고, 질문 속 여러 정보 중 조문이 답하는 부분을 찾아야 한다. 비슷한 다른 조문과 헷갈릴 만한 세부 조건을 하나 넣는다. 그래도 답은 이 조문 안에 있어야 한다. (조문지정형이면 법령명·조문번호는 넣되 상황은 길게, 용어형이면 용어를 직접 쓰지 말고 그 뜻을 설명하는 상황으로 묻는다)
출력은 JSON 하나: {"items":[{"id":"입력의 id","skip":false,"question":"...","gold_ids":[복수법령형만: 후보 id들],"note":"이 조문이 답하는 핵심 한 줄(20자 이상 60자 이하)"}]}"""


def call(items):
    body = {"model": ENV["LLM_MODEL"], "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": json.dumps({"items": items}, ensure_ascii=False)}]}
    last = None
    for a in range(4):
        try:
            r = requests.post(ENV["LLM_API_BASE"] + "/chat/completions", headers={"Authorization": "Bearer " + ENV["LLM_API_KEY"]}, json=body, timeout=180)
            r.raise_for_status()
            return json.loads(r.json()["choices"][0]["message"]["content"])["items"]
        except Exception as e:
            last = str(e)[:200].replace(ENV["LLM_API_KEY"], "***")
            time.sleep(2 ** a)
    raise RuntimeError(last)


def copied(question, art_text, law_name):
    q = question.replace(law_name, "")
    m = SequenceMatcher(None, q, art_text, autojunk=False).find_longest_match(0, len(q), 0, len(art_text))
    return m.size


# ---------------------------------------------------------------- 실행
def main():
    global SETS, SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_set", type=int, default=PER_SET)
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--diff", default="", help="예: 쉬움=15,보통=20,어려움=15")
    ap.add_argument("--types", default="", help="예: 상황형=24,용어형=4,조문지정형=6,하위법령형=8,복수법령형=8")
    ap.add_argument("--dry", action="store_true", help="LLM 호출 없이 칸만 확인")
    args = ap.parse_args()
    SETS = args.sets.split(",")
    if args.types:
        TYPE_MIX.update({k: int(v) for k, v in (x.split("=") for x in args.types.split(","))})
    if args.diff:
        DIFF_MIX.update({k: int(v) for k, v in (x.split("=") for x in args.diff.split(","))})
    if SETS[0] != "dev1":
        SEED += 100 * int(re.sub(r"[^0-9]", "", SETS[0]) or 1)  # 앞선 세트와 다른 조문 순서
    allart, laws, pool = load()
    art = pool.set_index("article_uid")
    slots, children, by_law, used_art, law_all, set_law, lawinfo = build_slots(pool, laws, allart)
    rnd = random.Random(SEED + 1)
    print("칸", len(slots), Counter(s["set"] for s in slots))
    if args.dry:
        return

    results = {}  # 칸 번호 -> dict
    pending = list(range(len(slots)))
    rounds = 0
    while pending and rounds < 6:
        rounds += 1
        batches = []
        for i in range(0, len(pending), 5):
            idx = pending[i:i + 5]
            items = []
            for n in idx:
                s = slots[n]
                r = art.loc[s["article_uid"]]
                it = {"id": str(n), "q_type": s["q_type"], "법령명": r.law_name, "구분": r.law_level, "조문": f"{r.article_label}({r.title})", "본문": r.text}
                if s.get("difficulty"):
                    it["difficulty"] = s["difficulty"]
                if s["q_type"] == "복수법령형":
                    cu = decree_candidates(r, pool, children)
                    s["cands"] = cu
                    it["후보_하위법령"] = [{"id": u, "법령명": art.loc[u].law_name, "조문": f"{art.loc[u].article_label}({art.loc[u].title})", "본문": art.loc[u].text[:700]} for u in cu]
                    if not cu:
                        it["후보_하위법령"] = []
                items.append(it)
            batches.append((idx, items))
        with ThreadPoolExecutor(4) as ex:
            outs = list(ex.map(lambda b: call(b[1]), batches))
        redo = []
        for (idx, items), out in zip(batches, outs):
            got = {str(o.get("id")): o for o in out}
            for n in idx:
                o, s = got.get(str(n)), slots[n]
                r = art.loc[s["article_uid"]]
                bad = None
                if not o or o.get("skip") or not str(o.get("question", "")).strip():
                    bad = "skip"
                else:
                    q = o["question"].strip()
                    if copied(q, r.text, r.law_name) >= 15:
                        bad = "조문 베낌"
                    elif s["q_type"] != "조문지정형" and re.search(r"제\s*\d+\s*조", q):
                        bad = "조문번호"
                    elif s["q_type"] == "조문지정형" and not re.search(r"제\s*\d+\s*조", q):
                        bad = "번호없음"
                    elif s["q_type"] == "복수법령형":
                        g = [x for x in o.get("gold_ids", []) if x in s.get("cands", [])]
                        if not g:
                            bad = "후보 못 고름"
                        o["gold_ids"] = g
                if bad:
                    print(f"  재뽑기 {slots[n]['set']} {s['q_type']} {r.law_name} {r.article_label} ({bad})")
                    redo.append(n)
                else:
                    results[n] = {"question": o["question"].strip(), "note": str(o.get("note", "")).strip(), "gold_ids": o.get("gold_ids", [])}
        for n in redo:
            slots[n] = replacement(slots[n], pool, laws, children, by_law, used_art, law_all, set_law, lawinfo, rnd)
        pending = redo
    if pending:
        raise SystemExit(f"끝내 못 만든 칸 {len(pending)}")

    # ---- 저장
    EVAL.mkdir(exist_ok=True)
    byset = defaultdict(list)
    for n, s in enumerate(slots):
        byset[s["set"]].append(n)
    qrows, grows, rrows = [], [], []
    for s in SETS:
        ns = byset[s]
        rnd2 = random.Random(SEED + SETS.index(s))
        rnd2.shuffle(ns)
        for j, n in enumerate(ns, 1):
            sl, res = slots[n], results[n]
            qid = f"{s}-{j:03d}"
            r = art.loc[sl["article_uid"]]
            qrows.append({"qid": qid, "split": "dev" if s.startswith("dev") else "test", "set": s, "question": res["question"],
                          "q_type": sl["q_type"], "difficulty": sl.get("difficulty", ""), "note": res["note"], "status": "AI초안"})
            gold = [(sl["article_uid"], True)] + [(u, False) for u in res["gold_ids"]]
            for u, pr in gold:
                g = art.loc[u]
                grows.append({"qid": qid, "article_uid": u, "primary_gold": pr, "law_name": g.law_name, "law_level": g.law_level, "article_label": g.article_label, "title": g.title})
            rrows.append({"qid": qid, "set": s, "q_type": sl["q_type"], "난이도": sl.get("difficulty", ""), "question": res["question"], "정답": " / ".join(f"{art.loc[u].law_name} {art.loc[u].article_label}({art.loc[u].title})" for u, _ in gold),
                          "정답 본문(앞 300자)": r.text[:300].replace("\n", " "), "AI 메모": res["note"], "확인(O/X/수정)": ""})
    def w(path, rows):
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    for name, rows in (("queries", qrows), ("gold", grows), ("gold_review", rrows)):
        f = EVAL / f"{name}.csv"
        if f.exists():  # 앞서 만든 세트 뒤에 이어 붙임 (새 칸은 앞 행엔 빈 값)
            old = pd.read_csv(f, dtype=str, keep_default_na=False)
            if SETS[0] in set(old.get("set", old["qid"].str.split("-").str[0])):
                raise SystemExit(f"{SETS[0]} 이미 있음")
            rows = old.to_dict("records") + rows
            cols = list(dict.fromkeys(c for r in rows for c in r))
            rows = [{c: r.get(c, "") for c in cols} for r in rows]
        w(f, rows)
    print("저장:", len(qrows), "문항, 정답", len(grows), "행")


if __name__ == "__main__":
    main()
