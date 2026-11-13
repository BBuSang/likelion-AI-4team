"""10 시점: 회신일 뒤 인용 조문(행정규칙은 규정)이 바뀌었는지 판정한다. 판정은 시간이 지나면 달라지므로 기준일을 남긴다.
changed/unchanged: 법률류 조문 이력, 본문 분리 검증을 통과한 행정규칙 조문 비교, 회신일 뒤 폐지   reg_changed/reg_unchanged: 그 밖의 행정규칙(규정 단위)
scheduled: 공포됐지만 기준일까지 시행되지 않은 개정만 있음   unknown: 확인 불가
cited_part_same: 조문이 개정된 법률류 인용에서 인용한 항·호 본문이 그대로면 True, 바뀌었으면 False, 비교 못 하면 None
입력: processed/cases_clean.jsonl, case_articles.jsonl, law_ids.jsonl, law_full_revisions.jsonl, law_article_history.jsonl, admrul_versions.jsonl, admrul_article_hash.jsonl, admrul_text_gate.jsonl, law_para_compare.jsonl
출력: processed/case_article_status.jsonl (+csv), case_time_check.jsonl
"""
from collections import Counter, defaultdict
from datetime import date

from common import manifest, read_jsonl, write_csv, write_jsonl
from s09b_admrul_text import version_at, ymd

USED = ("explicit", "context")
SCHEDULED_REG = "시행 전인 규정 개정 있음 (조문 변경 미확인)"
NEW_ARTICLE = ("제정", "본조신설")


def judge(a, rd, today, info, hist, vers, hashes, full, compared):
    """(status, why, later_changes, n_later)"""
    if a["resolve_method"] not in USED:
        return "unknown", "법령 연결이 불확실한 조문", [], 0
    if not rd:
        return "unknown", "회신일 없음", [], 0
    if not (info.get("law_id") or info.get("admrul_id")):
        return "unknown", "법령을 찾지 못함 (협회·거래소 규정, 표기 오류 등)", [], 0
    if a["supplementary"]:
        return "unknown", "부칙", [], 0
    repealed = ymd(info.get("repealed_on"))
    if repealed and repealed <= today:
        if rd < repealed:
            return "changed", "회신일 뒤 법령 폐지", [{"date": repealed, "reason": "폐지"}], 1
        return "unknown", "회신 당시 이미 폐지된 법령", [], 0
    if info.get("admrul_id"):
        vs = vers.get(a["law_name"], [])
        if not vs:
            return "unknown", "행정규칙 버전 없음", [], 0
        later = sorted({v["effective_date"] for v in vs if rd < ymd(v["effective_date"]) <= today})
        planned = sorted({v["effective_date"] for v in vs if ymd(v["effective_date"]) > today})
        then, now = version_at(vs, rd), version_at(vs, today)
        if a["law_name"] not in compared or then is None:   # 본문 비교를 믿을 수 없으면 규정 단위로만 표시
            if not later and planned:
                return "scheduled", SCHEDULED_REG, [{"date": d} for d in planned[:10]], 0
            return ("reg_changed" if later else "reg_unchanged"), None, [{"date": d} for d in later[:10]], len(later)
        k = (a["article"], a["article_sub"])
        h1, h2 = hashes.get((a["law_name"], then["serial"]), {}), hashes.get((a["law_name"], now["serial"]), {})
        if not h1 or not h2:
            return "unknown", "규정 본문의 조문을 읽지 못함", [], 0
        if k not in h1:
            return "unknown", "회신 당시 규정에 없는 조문 번호", [], 0
        if k not in h2:
            return "changed", "현행 규정에 없는 조문 (삭제 또는 이동)", [{"date": d} for d in later[:10]], len(later)
        if h1[k] != h2[k]:
            return "changed", None, [{"date": d} for d in later[:10]], len(later)
        if planned:
            return "scheduled", SCHEDULED_REG, [{"date": d} for d in planned[:10]], 0
        return "unchanged", None, [], 0
    if "-" in a["article"]:
        return "unknown", "법률에 없는 형식의 조문 번호", [], 0
    hs = [h for h in hist.get((a["law_name"], a["article"], a["article_sub"]), []) if h.get("change_date")]
    if not hs:
        return "unknown", "조문 이력 없음", [], 0
    first = min(hs, key=lambda h: ymd(h["change_date"]))
    if rd < ymd(first["change_date"]) and first.get("change_reason") in NEW_ARTICLE:
        return "unknown", "회신 당시 없던 조문 번호", [], 0
    def planned(h):              # 공포됐지만 기준일까지 시행되지 않은 개정
        return ymd(h["change_date"]) > today or ymd(h.get("effective_date")) > today
    later = [h for h in hs if rd < ymd(h["change_date"]) or planned(h)]
    past = {(h["change_date"], h.get("change_reason")) for h in later if not planned(h)}
    items = [{"date": d, "reason": r} for d, r in sorted({(h["change_date"], h.get("change_reason")) for h in later})[:10]]
    if past:
        return "changed", None, items, len(past)
    if later:
        return "scheduled", "공포됐으나 아직 시행 전인 개정", items, 0
    if any(rd < d <= today for d in full.get(a["law_name"], ())):
        return "unknown", "회신일 뒤 법령 전부개정 (조문 번호가 달라졌을 수 있음)", [], 0
    return "unchanged", None, [], 0


def main():
    today = date.today().strftime("%Y%m%d")
    checked_on = date.today().isoformat()
    idx = {r["case_key"]: r for r in read_jsonl("cases_clean.jsonl") if r["indexable"]}
    ids = {x["law_name"]: x for x in read_jsonl("law_ids.jsonl")}
    hist = defaultdict(list)
    for h in read_jsonl("law_article_history.jsonl"):
        hist[(h["law_name"], h["article"], h["article_sub"])].append(h)
    vers = defaultdict(list)
    for v in read_jsonl("admrul_versions.jsonl"):
        vers[v["law_name"]].append(v)
    hashes = defaultdict(dict)
    for x in read_jsonl("admrul_article_hash.jsonl"):
        hashes[(x["law_name"], x["serial"])][(x["article"], x["article_sub"])] = x["hash"]

    full = defaultdict(list)
    for x in read_jsonl("law_full_revisions.jsonl"):
        full[x["law_name"]].append(ymd(x["effective_date"]))
    compared = {g["law_name"] for g in read_jsonl("admrul_text_gate.jsonl") if g["pass"]}
    parts = {(x["law_name"], x["article"], x["article_sub"], x["paragraph"], x["item"], x["item_sub"], x["reply_ymd"]): x["cited_part_same"]
             for x in read_jsonl("law_para_compare.jsonl")}

    rows = []
    for a in read_jsonl("case_articles.jsonl"):
        if a["case_key"] not in idx:
            continue
        r = idx[a["case_key"]]
        status, why, later, n = judge(a, ymd(r["reply_date"]), today, ids.get(a["law_name"], {}), hist, vers, hashes, full, compared)
        part = parts.get((a["law_name"], a["article"], a["article_sub"], a["paragraph"], a["item"], a["item_sub"], ymd(r["reply_date"]))) if status == "changed" and not why else None
        rows.append({"case_key": a["case_key"], "law_name": a["law_name"], "law_kind": a["law_kind"], "article": a["article"],
                     "article_sub": a["article_sub"], "paragraph": a["paragraph"], "item": a["item"], "item_sub": a["item_sub"],
                     "supplementary": a["supplementary"], "section": a["section"], "pos": a["pos"], "raw": a["raw"],
                     "resolve_method": a["resolve_method"], "reply_date": r["reply_date"], "checked_on": checked_on,
                     "status": status, "why": why, "n_later": n, "later_changes": later, "cited_part_same": part})
    write_jsonl("case_article_status.jsonl", rows)
    write_csv("case_article_status.csv", rows, ["case_key", "law_name", "law_kind", "article", "article_sub", "paragraph", "item", "item_sub", "raw",
                                                 "resolve_method", "reply_date", "checked_on", "status", "why", "n_later", "cited_part_same"])

    by_case = defaultdict(Counter)
    for s in rows:
        by_case[s["case_key"]][s["status"]] += 1
    cases = []
    for k in idx:
        c = by_case.get(k)
        if not c:
            v = None
        elif c["changed"]:
            v = "article_changed"
        elif c["reg_changed"]:
            v = "reg_changed_only"
        elif c["scheduled"]:
            v = "scheduled"
        elif c["unknown"]:
            v = "unknown"
        else:
            v = "unchanged"
        cases.append({"case_key": k, "time_check": v, "counts": dict(c or {}), "checked_on": checked_on})
    write_jsonl("case_time_check.jsonl", cases)
    manifest("10_status", checked_on=checked_on, articles=len(rows), status=dict(Counter(s["status"] for s in rows)),
             why=dict(Counter(s["why"] for s in rows if s["status"] == "unknown")),
             compared_regs=len(compared), cited_part_same=dict(Counter(str(s["cited_part_same"]) for s in rows if s["status"] == "changed" and not s["why"])),
             cases=dict(Counter(c["time_check"] for c in cases)))


if __name__ == "__main__":
    main()
