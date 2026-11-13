"""E2 ③ 시점 표시 점검 (읽기 전용). 표시 단계가 DB의 시점 판정을 그대로, 보수적으로 옮기는지 전수 확인한다.
1. 사례 대표 표시 = 저장된 사례 판정(cases.time_check)과 일치하는가 (조문 인용이 있는 사례 전부)
2. ✅ 표시는 조문 이력(law_article_history)에서 독립적으로 다시 확인: 회신일 뒤 변경 이력이 없고, 이력이 존재해야 한다
3. ⚠️·🕒의 날짜 규칙: 시행 예정은 전부 기준일 뒤, 개정은 기준일 이내 날짜가 있어야 한다
4. 문구: 모든 표시에 정해진 문구가 붙는지, ✅가 확인 불가 인용과 섞여 나가지 않는지
이 점검은 DB 판정 → 표시 단계의 일관성이다. 판정이 법령 원문과 맞는지는 B7·잠금 평가(본문 대조 80/80, 30/30)에서 쟀다.
사용: python e2_check.py     출력: data/eval/e2_check.json
"""
import json
import sys
from collections import Counter, defaultdict

import rag
import time_marks as tm
from common import DATA, manifest

sys.stdout.reconfigure(encoding="utf-8")
EXPECT = {"article_changed": "⚠️", "reg_changed_only": "🔶", "scheduled": "🕒", "unknown": "❔", "unchanged": "✅"}


def main():
    out = {}
    with rag.connect() as conn:
        cases = conn.execute("select case_key, reply_date, time_check, time_checked_on from interp.cases where indexable").fetchall()
        keys = [c["case_key"] for c in cases]
        marks = {}
        for i in range(0, len(keys), 500):
            marks.update(tm.load_marks(conn, keys[i:i + 500]))

        # 1. 대표 표시 = 저장된 사례 판정
        bad, dist = [], Counter()
        for c in cases:
            m = marks[c["case_key"]]
            want = EXPECT.get(c["time_check"])
            dist[m["mark"]] += 1
            if m["mark"] != want:
                bad.append((c["case_key"], c["time_check"], m["mark"]))
        out["1_representative_vs_stored"] = {"cases": len(cases), "mismatch": len(bad), "examples": bad[:5], "distribution": {str(k): v for k, v in dist.items()}}

        # 2·3. 판정을 원자료에서 독립적으로 다시 확인 (법률류: 조문 이력 표, 행정규칙: 현행 본문의 개정 표시)
        hist = defaultdict(list)
        for h in conn.execute("select law_id, article, article_sub, change_date, effective_date from interp.law_article_history").fetchall():
            hist[(h["law_id"], h["article"], h["article_sub"])].append(h)
        rows = conn.execute("""select a.case_key, a.law_id, l.law_name, l.law_kind, a.article, a.article_sub, a.resolve_method, a.supplementary, s.status, s.why, s.checked_on, x.reply_date
                               from interp.case_articles a join interp.case_article_status s on s.case_article_id = a.id join interp.cases x using (case_key)
                               join interp.laws l on l.law_id = a.law_id where x.indexable""").fetchall()
        vers = defaultdict(list)
        for v in conn.execute("select serial, law_id, effective_date, issue_date from interp.admrul_versions").fetchall():
            vers[v["law_id"]].append(v)
        law_rows = [r for r in rows if r["law_kind"] != "행정규칙"]
        # 2a 법률류 ✅: 회신일 뒤 변경 이력이 없고 이력이 있어야 한다
        ok_n = viol_later = no_hist = 0
        ex2 = []
        for r in law_rows:
            if r["status"] != "unchanged":
                continue
            ok_n += 1
            hs = hist.get((r["law_id"], r["article"], r["article_sub"]), [])
            if not hs:
                no_hist += 1
            if any(h["change_date"] and h["change_date"] > r["reply_date"] for h in hs):
                viol_later += 1
                ex2.append((r["case_key"], r["law_name"], r["article"], r["article_sub"]))
        # 3 법률류 ⚠️·🕒: 이력으로 날짜 규칙 확인. 개정(⚠️)은 기준일까지 시행된 변경이 있어야 하고, 시행 예정(🕒)은 시행된 변경이 없어야 한다
        sched_n = sched_bad = chg_n = chg_bad = 0
        for r in law_rows:
            if r["status"] not in ("changed", "scheduled") or (r["why"] or "").find("폐지") >= 0:
                continue
            ck = r["checked_on"]
            hs = [h for h in hist.get((r["law_id"], r["article"], r["article_sub"]), []) if h["change_date"] and h["change_date"] > r["reply_date"]]
            past = [h for h in hs if h["change_date"] <= ck and (h["effective_date"] is None or h["effective_date"] <= ck)]
            if r["status"] == "scheduled":
                sched_n += 1
                sched_bad += bool(past) or not hs
            else:
                chg_n += 1
                chg_bad += not past
        out["2a_law_unchanged_recheck"] = {"checked": ok_n, "later_change_in_history": viol_later, "no_history": no_hist, "examples": ex2[:5]}
        out["3_law_changed_scheduled_recheck"] = {"scheduled": sched_n, "scheduled_violations": sched_bad, "changed": chg_n, "changed_without_past_change": chg_bad}

        # 2b 행정규칙 ✅: 현행 버전 조문 본문에 회신일 뒤 날짜의 개정 표시가 있으면 모순 (판정은 조문 본문 지문 비교였고 개정 표시는 지문에서 뺀다)
        import re
        import s09b_admrul_text as s9
        from lawapi import call
        DATE = re.compile(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.")
        cache, n_chk, n_marker_later, n_nocache, n_nokey, ex2b = {}, 0, 0, 0, 0, []
        for r in rows:
            if r["law_kind"] != "행정규칙" or r["status"] not in ("unchanged", "reg_unchanged") or r["resolve_method"] not in ("explicit", "context"):
                continue
            n_chk += 1
            ck = r["checked_on"].strftime("%Y%m%d")
            cur = max((v for v in vers[r["law_id"]] if v["effective_date"] and v["effective_date"].strftime("%Y%m%d") <= ck),
                      key=lambda v: (v["effective_date"], v["issue_date"] or v["effective_date"], v["serial"]), default=None)
            if cur is None:
                n_nocache += 1
                continue
            if cur["serial"] not in cache:
                try:
                    jo = call("lawService", {"target": "admrul", "ID": cur["serial"]})["response"].get("AdmRulService", {}).get("조문내용")
                except RuntimeError:
                    jo = None
                parts = (s9.split_run_on(jo) if isinstance(jo, str) and "\n" not in jo.strip() and s9.HYPHEN.search(jo) else (jo if isinstance(jo, list) else s9.SPLIT.split(jo or ""))) if jo else []
                arts = {}
                for t in parts:
                    t = t if isinstance(t, str) else str(t)
                    m = s9.JO.match(t)
                    if m:
                        arts.setdefault((m.group(1), int(m.group(2)) if m.group(2) else None), t)
                cache[cur["serial"]] = arts
            t = cache[cur["serial"]].get((r["article"], r["article_sub"]))
            if t is None:
                n_nokey += 1
                continue
            later = [f"{y}{int(m_):02d}{int(d):02d}" for mk in s9.MARK.findall(t) for y, m_, d in DATE.findall(mk)
                     if f"{y}{int(m_):02d}{int(d):02d}" > r["reply_date"].strftime("%Y%m%d")]
            if later:
                n_marker_later += 1
                ex2b.append((r["case_key"], r["law_name"], r["article"], r["article_sub"], sorted(set(later))[:2]))
        out["2b_admrul_unchanged_recheck"] = {"checked": n_chk, "no_current_version": n_nocache, "article_not_found_in_text": n_nokey,
                                              "marker_date_after_reply": n_marker_later, "examples": ex2b[:8]}

        # 4. 문구와 ✅ 혼입
        cite_marks = Counter()
        icon_label_missing = ok_mixed = unlabeled = 0
        for m in marks.values():
            if m["mark"] and not m["label"]:
                unlabeled += 1
            if m["mark"] == "✅" and any(c["icon"] != "✅" for c in m["citations"]):
                ok_mixed += 1
            for c in m["citations"]:
                cite_marks[c["icon"]] += 1
                if not c["label"]:
                    icon_label_missing += 1
                if c["icon"] == "⚠️" and tm.NEED_CHECK not in c["label"]:
                    icon_label_missing += 1
        out["4_labels"] = {"citation_marks": dict(cite_marks), "label_missing": icon_label_missing, "case_without_label": unlabeled, "check_ok_mixed_with_unchecked": ok_mixed}
        out["examples"] = {k: tm.format_card(marks[k]) for k in ("L3760", "O1437", "L5146") if k in marks}
    with open(DATA / "eval" / "e2_check.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    manifest("e2_check", mismatch=out["1_representative_vs_stored"]["mismatch"], law_unchanged_violations=out["2a_law_unchanged_recheck"]["later_change_in_history"],
             admrul_marker_after_reply=out["2b_admrul_unchanged_recheck"]["marker_date_after_reply"])
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
