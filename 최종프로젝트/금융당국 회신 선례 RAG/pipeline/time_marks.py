"""E2 ③ 시점 표시: 인용 조문의 시점 판정(`case_article_status`)을 사례 카드·답변에 붙일 표시로 바꾼다 (기획안 5-5, 7-5).
- 조문별 표시: ⚠️ 회신일 이후 조문 개정·폐지 / ✅ 이후 개정 없음 / 🔶 규정 개정됨(조문 변경 미확인) / 🕒 시행 예정 개정 / ❔ 확인 불가
- 사례 대표 표시: 사례가 인용한 **모든** 조문 중 가장 주의가 필요한 것 (⚠️ > 🔶 > 🕒 > ❔ > ✅). 법령 연결이 불확실한(default·unresolved) 조문과 부칙도
  확인하지 못한 인용으로 ❔에 넣는다. ✅를 틀리게 붙이는 것이 ❔보다 나쁘기 때문이다. `cases.time_check`(s10)와 같은 기준이다
- 조문이 개정됐다고 회신이 무효가 되는 것은 아니므로 문구는 "현재 적용 여부 확인 필요"로 한다
이 모듈은 판정을 새로 하지 않고 DB의 판정을 읽어 표시로 바꾸기만 한다.
"""
from collections import Counter, defaultdict

ORDER = {"⚠️": 0, "🔶": 1, "🕒": 2, "❔": 3, "✅": 4}
ICON = {"changed": "⚠️", "unchanged": "✅", "reg_unchanged": "✅", "reg_changed": "🔶", "scheduled": "🕒"}
NEED_CHECK = "현재 적용 여부 확인 필요"
HEADLINE = {"⚠️": f"회신일 이후 개정·폐지된 인용 조문이 있습니다 — {NEED_CHECK}",
            "🔶": "인용 규정이 회신일 이후 개정됐습니다 (조문 변경 여부 미확인)",
            "🕒": "시행 예정인 개정이 있습니다",
            "❔": "시점을 확인하지 못한 인용이 있습니다",
            "✅": "확인한 인용 조문 모두 회신일 이후 개정이 없습니다"}

CITATION_SQL = """
    select a.case_key, a.id, a.resolve_method, a.supplementary, a.raw, a.article, a.article_sub, a.law_id, l.law_name,
           s.status, s.why, s.cited_part_same, s.later_changes, s.checked_on
    from interp.case_articles a left join interp.laws l on l.law_id = a.law_id left join interp.case_article_status s on s.case_article_id = a.id
    where a.case_key = any(%s) order by a.id"""


def fmt_date(d):
    d = str(d or "")
    return f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 and d.isdigit() else d


def citation_mark(r):
    """인용 조문 한 줄(case_articles + case_article_status)의 표시. 반환: (아이콘, 문구, 개정일 목록)"""
    st, why = r["status"], r["why"] or ""
    if r["supplementary"]:
        return "❔", "부칙 인용 (시점 확인 대상 아님)", []
    if r["resolve_method"] not in ("explicit", "context"):
        return "❔", "법령 연결이 불확실한 조문 (확인 불가)", []
    dates = sorted(fmt_date(x.get("date")) for x in (r["later_changes"] or []) if x.get("date"))
    icon = ICON.get(st, "❔")
    if st == "changed":
        if "폐지" in why:
            return icon, f"회신일 이후 폐지 — {NEED_CHECK}", dates[:3]
        same = {True: " (인용 부분은 동일)", False: " (인용 부분도 변경)"}.get(r["cited_part_same"], "")
        return icon, f"회신일 이후 조문 개정{same} — {NEED_CHECK}", dates[:3]
    if st == "unchanged":
        return icon, "이후 개정 없음", []
    if st == "reg_unchanged":
        return icon, "이후 규정 개정 없음", []
    if st == "reg_changed":
        return icon, "규정 개정됨 (조문 변경 여부 미확인)", dates[:3]
    if st == "scheduled":
        return icon, "시행 예정 개정 있음", dates[:3]
    return "❔", why or "확인 불가 (시점 판정 없음)", []


def summarize(citations):
    """인용 조문 표시 목록 → 사례 대표 표시. 인용이 없으면 표시도 없다."""
    if not citations:
        return {"mark": None, "label": "조문 인용 없음", "counts": {}}
    counts = Counter(c["icon"] for c in citations)
    mark = min(counts, key=ORDER.get)
    detail = " · ".join(f"{i} {counts[i]}" for i in sorted(counts, key=ORDER.get))
    return {"mark": mark, "label": f"{HEADLINE[mark]} (인용 {len(citations)}건: {detail})", "counts": dict(counts)}


def load_marks(conn, case_keys, article=None):
    """사례별 시점 표시. article=(law_id, article, article_sub)를 주면 그 조문의 인용만 본다 (② 조문 조회 결과용).
    반환: {case_key: {mark, label, counts, citations: [{law, raw, icon, label, dates, resolve}]}} — 인용이 없는 사례도 키가 있다."""
    keys = list(case_keys)
    by = defaultdict(dict)
    for r in conn.execute(CITATION_SQL, (keys,)).fetchall():
        if article and (r["law_id"], r["article"], r["article_sub"]) != tuple(article):
            continue
        icon, label, dates = citation_mark(r)
        k = (r["law_name"], r["raw"])
        old = by[r["case_key"]].get(k)
        if old is None or ORDER[icon] < ORDER[old["icon"]]:        # 같은 조문을 여러 번 인용하면 가장 주의가 필요한 표시 하나
            by[r["case_key"]][k] = {"law": r["law_name"], "raw": r["raw"], "icon": icon, "label": label, "dates": dates, "resolve": r["resolve_method"]}
    out = {}
    for ck in keys:
        cites = sorted(by[ck].values(), key=lambda c: (ORDER[c["icon"]], c["law"] or "", c["raw"]))
        out[ck] = {**summarize(cites), "citations": cites}
    return out


def format_card(m, limit=4):
    """사례 시점 표시를 글로 (콘솔·데모용)."""
    if not m["mark"]:
        return "조문 인용 없음"
    lines = [f"{m['mark']} {m['label']}"]
    for c in m["citations"][:limit]:
        d = f" [{', '.join(c['dates'])}]" if c["dates"] else ""
        lines.append(f"  {c['icon']} {c['law'] or '법령 미확인'} {c['raw']}: {c['label']}{d}")
    if len(m["citations"]) > limit:
        lines.append(f"  … 외 {len(m['citations']) - limit}건")
    return "\n".join(lines)
