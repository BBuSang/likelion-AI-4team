"""09 법령: 인용 법령의 ID, 조문별 변경 이력, 행정규칙 버전을 국가법령정보 API에서 받는다.
- 법률·시행령·시행규칙: 현행법령(시행일) 목록(eflaw)에서 ID를 찾고 조문별 변경 이력(lsJoHstInf)을 받는다
- 행정규칙: admrul에서 ID를 찾고 같은 행정규칙ID의 현행·연혁 버전을 모은다
- 조문 이력은 explicit·context로 해석된 조문만 받는다
입력: processed/laws.jsonl, case_articles.jsonl, cases_clean.jsonl
출력: processed/law_ids.jsonl, law_source_ids.jsonl, law_full_revisions.jsonl, law_article_history.jsonl, admrul_versions.jsonl (+csv), b4_progress.log
응답 원본은 raw/law/ 캐시 → 다시 돌리면 이어서 받는다.
"""
import re

import lawapi
from common import OUT, manifest, read_jsonl, write_csv, write_jsonl
from lawapi import call

key = lambda s: re.sub(r"[\s·ㆍ・․‧]", "", s or "")     # 띄어쓰기와 가운뎃점 종류는 무시
fkey = lambda s: key(s).replace("의", "")
USED = ("explicit", "context")
REPEAL = ("폐지", "타법폐지")                          # '폐지제정'은 다시 제정된 것이라 폐지가 아니다
log = open(OUT / "b4_progress.log", "a", encoding="utf-8")


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


def items(resp, root, name):
    d = resp.get(root, {})
    x = d.get(name, [])
    return (x if isinstance(x, list) else [x]), int(d.get("totalCnt", 0) or 0)


def search_all(params, root, name, max_pages=20):
    out, page = [], 1
    while True:
        r = call("lawSearch", {**params, "display": "100", "page": str(page)})["response"]
        got, total = items(r, root, name)
        out += got
        if len(out) >= total or not got or page >= max_pages:
            return out
        page += 1


def pick(cands, name, field):
    """이름 일치: 띄어쓰기 무시(exact) → 조사 '의'까지 무시(loose)"""
    for label, f in (("exact", key), ("loose", fkey)):
        hit = [c for c in cands if f(c.get(field)) == f(name)]
        if hit:
            return hit, label
    return [], None


def find_adm(name):
    for nw in ("1", "2"):     # 현행에 없으면(폐지 등) 연혁에서
        hit, how = pick(search_all({"target": "admrul", "query": name, "nw": nw}, "AdmRulSearch", "admrul"), name, "행정규칙명")
        if hit:
            return {"match": how, "admrul_id": hit[0]["행정규칙ID"], "current": any(h.get("현행연혁구분") == "현행" for h in hit)}
    return None


def find_law(name):
    cands = search_all({"target": "eflaw", "query": name, "nw": "1,3"}, "LawSearch", "law")
    hit, how = pick(cands, name, "법령명한글")
    if not hit:
        return None
    law_ids = sorted({h["법령ID"] for h in hit})
    # 이름이 바뀐 뒤의 버전까지 포함해 마지막 버전을 본다. 개명 법령은 폐지가 아니다
    last = max((c for c in cands if c["법령ID"] in law_ids), key=lambda c: (c["시행일자"], int(c["법령일련번호"])))
    repealed = last.get("제개정구분명") in REPEAL
    return {"match": how, "law_id": law_ids[0], "law_ids_all": law_ids, "current": not repealed,
            "repealed_on": last["시행일자"] if repealed else None,
            "abbrev": next((h.get("법령약칭명") for h in hit if h.get("법령약칭명")), None)}


def article_history(name, art, sub, law_ids):
    jo = f"{int(art):04d}{int(sub or 0):02d}"
    out = []
    for lid in law_ids:           # 한시법 재제정 등으로 법령 ID가 여러 개면 전부 모은다
        page, rows = 1, []
        while True:
            d = call("lawService", {"target": "lsJoHstInf", "ID": lid, "JO": jo, "display": "100", "page": str(page)})["response"].get("LawService", {})
            got = d.get("law", [])
            got = got if isinstance(got, list) else [got]
            rows += got
            total = int(d.get("totalCnt", len(rows)) or len(rows))
            if len(rows) >= total or len(got) < 100:
                break
            page += 1
        for g in rows:
            ji, li = g.get("조문정보", {}), g.get("법령정보", {})
            out.append({"law_name": name, "source_id": lid, "article": art, "article_sub": sub, "jo": jo,
                        "change_date": ji.get("조문변경일"), "change_reason": ji.get("변경사유"),
                        "effective_date": li.get("시행일자"), "promulgation_date": li.get("공포일자"),
                        "revision_type": li.get("제개정구분명"), "law_serial": li.get("법령일련번호")})
    return out or [{"law_name": name, "source_id": law_ids[0], "article": art, "article_sub": sub, "jo": jo, "change_date": None}]


def main():
    laws = read_jsonl("laws.jsonl")
    idx = {r["case_key"] for r in read_jsonl("cases_clean.jsonl") if r["indexable"]}
    arts = [a for a in read_jsonl("case_articles.jsonl") if a["case_key"] in idx and a["resolve_method"] in USED]

    ids = []
    for i, L in enumerate(laws, 1):
        name = L["law_name"]
        # 이름에 '규정'이 있어도 대통령령인 경우, '규칙'이 행정규칙인 경우가 있어 양쪽을 모두 찾아본다
        found = (find_adm(name) or find_law(name)) if L["law_kind"] == "행정규칙" else (find_law(name) or find_adm(name))
        ids.append({"law_name": name, "law_kind": L["law_kind"], "match": None, **(found or {})})
        if i % 25 == 0:
            say(f"[ID] {i}/{len(laws)}")
    source =[{"law_name": x["law_name"], "source_kind": "lawgo_law", "source_id": s} for x in ids for s in x.get("law_ids_all", [])]
    source += [{"law_name": x["law_name"], "source_kind": "lawgo_admrul", "source_id": x["admrul_id"]} for x in ids if x.get("admrul_id")]
    write_jsonl("law_source_ids.jsonl", source)
    write_csv("law_source_ids.csv", source, ["law_name", "source_kind", "source_id"])
    idmap = {x["law_name"]: x for x in ids}
    say("[ID] 완료", len({s["law_name"] for s in source}), "/", len(ids), "매칭")

    # 전부개정은 조문 번호가 통째로 바뀔 수 있어 시점 판정에서 따로 본다
    full = []
    for x in ids:
        if x.get("law_ids_all"):
            hits = search_all({"target": "eflaw", "query": x["law_name"], "nw": "1,3"}, "LawSearch", "law")
            full += [{"law_name": x["law_name"], "effective_date": d}
                     for d in sorted({h["시행일자"] for h in hits if h["법령ID"] in x["law_ids_all"] and (h.get("제개정구분명") or "").startswith("전부개정")})]
    write_jsonl("law_full_revisions.jsonl", full)

    pairs = sorted({(a["law_name"], a["article"], a["article_sub"]) for a in arts
                    if not a["supplementary"] and "-" not in a["article"] and idmap.get(a["law_name"], {}).get("law_id")},
                   key=lambda p: (p[0], int(p[1]), p[2] or 0))
    hist = []
    for i, (name, art, sub) in enumerate(pairs, 1):
        hist += article_history(name, art, sub, idmap[name]["law_ids_all"])
        if i % 100 == 0:
            say(f"[조문이력] {i}/{len(pairs)}")
    write_jsonl("law_article_history.jsonl", hist)
    write_csv("law_article_history.csv", hist, ["law_name", "source_id", "article", "article_sub", "change_date", "change_reason",
                                                "effective_date", "promulgation_date", "revision_type"])
    say("[조문이력] 완료", len(pairs), "쌍", len(hist), "행")

    vers, seen = [], set()
    for x in ids:
        if not x.get("admrul_id"):
            continue
        for nw in ("1", "2"):
            for c in search_all({"target": "admrul", "query": x["law_name"], "nw": nw}, "AdmRulSearch", "admrul"):
                if c.get("행정규칙ID") == x["admrul_id"] and c.get("행정규칙일련번호") not in seen:
                    seen.add(c.get("행정규칙일련번호"))
                    vers.append({"law_name": x["law_name"], "admrul_id": x["admrul_id"], "name_at_time": c.get("행정규칙명"),
                                 "status": c.get("현행연혁구분"), "issue_date": c.get("발령일자"),
                                 "effective_date": c.get("시행일자"), "revision_type": c.get("제개정구분명"),
                                 "serial": c.get("행정규칙일련번호")})
    vers.sort(key=lambda v: (v["law_name"], v["issue_date"] or ""))
    for x in ids:                 # 행정규칙의 현행 여부는 마지막 버전으로 정한다
        vs = [v for v in vers if v["law_name"] == x["law_name"]]
        if x.get("admrul_id") and vs:
            last = max(vs, key=lambda v: (v["effective_date"] or "", v["issue_date"] or ""))
            repealed = last["revision_type"] in REPEAL
            x.update(current=not repealed, repealed_on=last["effective_date"] if repealed else None)
    write_jsonl("law_ids.jsonl", ids)
    write_jsonl("admrul_versions.jsonl", vers)
    write_csv("admrul_versions.csv", vers, ["law_name", "admrul_id", "serial", "name_at_time", "status", "issue_date", "effective_date", "revision_type"])
    say("[행정규칙] 완료", len({v["law_name"] for v in vers}), "개 규정", len(vers), "버전")
    manifest("09_laws", laws=len(ids), matched=len({s["law_name"] for s in source}), article_pairs=len(pairs), history=len(hist),
             no_history=sum(h["change_date"] is None for h in hist), admrul_versions=len(vers), requests=lawapi.n_requests)


if __name__ == "__main__":
    main()
