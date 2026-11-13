"""09c 법률류 인용 항·호 비교: 조문이 개정된 인용에 대해 회신 당시 버전과 현행 버전의 해당 항(또는 호) 본문을 비교한다.
항까지 인용했으면 항 전체(딸린 호·목 포함)가 같아야 '동일'이다. 찾지 못하면 판단하지 않는다.
입력: processed/case_articles.jsonl, cases_clean.jsonl, law_ids.jsonl, law_article_history.jsonl
출력: processed/law_para_compare.jsonl
"""
import re
from collections import Counter, defaultdict
from datetime import date

import lawapi
from common import manifest, read_jsonl, write_jsonl
from lawapi import call
from s09_laws import search_all
from s09b_admrul_text import norm, ymd

USED = ("explicit", "context")
TEXT = ("조문내용", "항내용", "호내용", "목내용")
L = lambda x: x if isinstance(x, list) else ([x] if x else [])


def text_of(u):
    out = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in TEXT and isinstance(v, str):
                    out.append(v)
                elif isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(x, list):
            for i in x:
                if isinstance(i, str):
                    out.append(i)
                else:
                    walk(i)

    walk(u)
    return norm("".join(out))


def find_para(u, n):
    hit = [p for p in L(u.get("항")) if isinstance(p, dict) and (p.get("항번호") or "").strip()[:1]
           and ord(p["항번호"].strip()[0]) - 0x245F == n]
    return hit[0] if len(hit) == 1 else None


def find_item(u, n, sub):
    """항이 없는 조문의 호. 같은 번호가 여러 곳이면 판단하지 않는다."""
    hit = []
    for p in L(u.get("항")):
        for h in L(p.get("호") if isinstance(p, dict) else None):
            m = re.match(r"\s*(\d+)(?:\s*의\s*(\d+))?", str(h.get("호번호", ""))) if isinstance(h, dict) else None
            if m and int(m.group(1)) == n and (int(m.group(2)) if m.group(2) else None) == sub:
                hit.append(h)
    return hit[0] if len(hit) == 1 else None


def main():
    today = date.today().strftime("%Y%m%d")
    idx = {r["case_key"]: ymd(r["reply_date"]) for r in read_jsonl("cases_clean.jsonl") if r["indexable"]}
    ids = {x["law_name"]: x for x in read_jsonl("law_ids.jsonl")}
    hist = defaultdict(list)
    for h in read_jsonl("law_article_history.jsonl"):
        if h.get("change_date"):
            hist[(h["law_name"], h["article"], h["article_sub"])].append(h)

    def changed_after(a, rd):
        return any(rd < ymd(h["change_date"]) <= today and ymd(h.get("effective_date")) <= today
                   for h in hist[(a["law_name"], a["article"], a["article_sub"])])

    memo = {}

    def at(name, day):
        if name not in memo:
            ok = ids[name]["law_ids_all"]
            memo[name] = [h for h in search_all({"target": "eflaw", "query": name, "nw": "1,3"}, "LawSearch", "law") if h["법령ID"] in ok]
        past = [v for v in memo[name] if ymd(v["시행일자"]) <= day]
        return max(past, key=lambda v: (ymd(v["시행일자"]), int(v["법령일련번호"]))) if past else None

    # (버전) → 이 버전에서 꺼낼 조문들
    wanted, todo = defaultdict(set), []
    seen = set()
    for a in read_jsonl("case_articles.jsonl"):
        rd = idx.get(a["case_key"])
        if not rd or a["resolve_method"] not in USED or a["supplementary"] or not a["article"].isdigit() or not (a["paragraph"] or a["item"]):
            continue
        if not ids.get(a["law_name"], {}).get("law_ids_all") or ids[a["law_name"]].get("repealed_on") or not changed_after(a, rd):
            continue
        k = (a["law_name"], a["article"], a["article_sub"], a["paragraph"], a["item"], a["item_sub"], rd)
        if k in seen:
            continue
        seen.add(k)
        v1, v2 = at(a["law_name"], rd), at(a["law_name"], today)
        if not v1 or not v2 or v1["법령일련번호"] == v2["법령일련번호"]:
            continue
        for v in (v1, v2):
            wanted[(v["법령일련번호"], v["시행일자"])].add((str(a["article"]), str(a["article_sub"] or "")))
        todo.append((k, (v1["법령일련번호"], v1["시행일자"]), (v2["법령일련번호"], v2["시행일자"])))

    print(f"[항 비교] 인용 {len(todo)}건, 법령 버전 {len(wanted)}개", flush=True)
    units = {}
    for i, (ver, arts) in enumerate(sorted(wanted.items()), 1):
        d = call("lawService", {"target": "eflaw", "MST": ver[0], "efYd": ver[1]})["response"].get("법령", {})
        for u in L((d.get("조문") or {}).get("조문단위")):
            k = (str(u.get("조문번호")), str(u.get("조문가지번호") or ""))
            if k in arts and u.get("조문여부") != "전문":
                units[(ver, k)] = u
        if i % 50 == 0:
            print(f"[항 비교] {i}/{len(wanted)}", flush=True)

    rows = []
    for (name, art, sub, para, item, isub, rd), v1, v2 in todo:
        k = (str(art), str(sub or ""))
        u1, u2 = units.get((v1, k)), units.get((v2, k))
        level, same = ("para" if para else "item"), None
        if u1 and u2:
            x1, x2 = (find_para(u1, para), find_para(u2, para)) if para else (find_item(u1, item, isub), find_item(u2, item, isub))
            if x1 and x2:
                same = text_of(x1) == text_of(x2)
        rows.append({"law_name": name, "article": art, "article_sub": sub, "paragraph": para, "item": item, "item_sub": isub,
                     "reply_ymd": rd, "level": level, "cited_part_same": same})
    write_jsonl("law_para_compare.jsonl", rows)
    manifest("09c_law_text", citations=len(rows), versions=len(wanted), result=dict(Counter((r["level"], r["cited_part_same"]) .__str__() for r in rows)),
             requests=lawapi.n_requests)


if __name__ == "__main__":
    main()
