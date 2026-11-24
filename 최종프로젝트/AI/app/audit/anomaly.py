"""이상 접근 탐지 (3주차 선택 과제 '복원 기록 기반 이상 접근 탐지').

vault.restore_log(실명 보기 기록)만 보고, 이유를 설명할 수 있는 규칙으로 경고를 만든다. 실명·원문은 쓰지 않음
(토큰·문서 ID·개수만). 시간은 한국 시간(KST) 기준.

규칙 (기준값은 RULES 에서 바꿈)
  burst       대량 복원         한 사람이 짧은 시간(10분)에 많이(10번) 실명 보기
  spike       평소보다 급증     하루 건수가 본인 평소(이전 14일 중 활동한 날의 중앙값)의 3배 이상이면서 20건 이상 (감사 60건)
                                (평소 기록이 3일 미만이면 하루 30건 이상)
  off_hours   업무 외 시간      22시~7시 또는 주말 실명 보기 (1~2번 '주의', 3번 이상 '경고')
  breadth     넓은 범위         하루에 서로 다른 문서를 많이 복원 (사원·팀장 30건, 감사 60건)
  purpose     형식적인 사유     같은 사유로 하루 10번 이상, 또는 4자 미만 사유 3번 이상 (사유가 필요한 팀장·감사)
  probing     권한 밖 시도 반복 1시간 안에 거절(사유 없음)·권한 밖 토큰이 남은 요청 5번 이상
  same_person 특정인 반복 조회  같은 사람(토큰)을 하루에 5번 이상, 서로 다른 문서 3건 이상에서 복원
"""
import collections
import datetime as dt
import hashlib
import statistics

from app.core import db

KST = dt.timezone(dt.timedelta(hours=9))

RULES = {
    "burst": {"window_min": 10, "count": 10},
    # 감사는 정기 점검으로 하루 수십 건이 정상 업무 → 넓은 범위(breadth)처럼 감사 기준을 따로 (모의 기록에서 잘못된 경고 전부가 이 경우, 38장)
    "spike": {"history_days": 14, "ratio": 3.0, "min_count": 20, "min_count_auditor": 60, "min_history_days": 3, "abs_count": 30},
    "off_hours": {"start_hour": 22, "end_hour": 7, "warn_count": 3},
    "breadth": {"docs": 30, "docs_auditor": 60},
    "purpose": {"same_count": 10, "short_len": 4, "short_count": 3},
    "probing": {"window_min": 60, "count": 5},
    "same_person": {"restores": 5, "docs": 3},
}
RULE_KO = {"burst": "대량 복원", "spike": "평소보다 급증", "off_hours": "업무 외 시간", "breadth": "넓은 범위",
           "purpose": "형식적인 사유", "probing": "권한 밖 시도 반복", "same_person": "특정인 반복 조회"}
SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def load(days=14, until=None, conn=None, actor=None):
    """기간 [until - days - 기준선 14일, until] 의 기록 (actor 를 주면 그 사람 것만). 반환: [dict] (시간은 KST)"""
    until = until or dt.datetime.now(dt.timezone.utc)
    since = until - dt.timedelta(days=days + RULES["spike"]["history_days"])
    sql = """SELECT id, actor, role, emp_id, team, purpose, tokens, doc_ids, denied, refused, created_at
             FROM vault.restore_log WHERE created_at >= %s AND created_at <= %s"""
    params = [since, until]
    if actor:
        sql += " AND actor = %s"
        params.append(actor)
    sql += " ORDER BY created_at"
    own = conn is None
    conn = conn or db.vault_conn()
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        if own:
            conn.close()
    keys = ("id", "actor", "role", "emp_id", "team", "purpose", "tokens", "doc_ids", "denied", "refused", "at")
    out = []
    for r in rows:
        d = dict(zip(keys, r))
        d["at"] = d["at"].astimezone(KST)
        out.append(d)
    return out


def _alert(rule, severity, actor, rows, count, detail):
    rows = sorted(rows, key=lambda r: r["at"])
    first, last = rows[0], rows[-1]
    # id 는 내용까지 넣어 겹치지 않게 — 같은 사람·같은 시각에 토큰만 다른 '특정인 반복 조회'가 여럿 나옴 (39장, 화면 정렬이 섞이던 원인)
    uid = hashlib.sha1(f"{rule}|{actor}|{first['id']}|{last['id']}|{detail}".encode()).hexdigest()[:12]
    return {"id": f"{rule}:{uid}", "rule": rule, "rule_ko": RULE_KO[rule],
            "severity": severity, "actor": actor, "role": last.get("role"), "team": last.get("team"),
            "start": first["at"].isoformat(), "end": last["at"].isoformat(), "count": count, "detail": detail,
            "log_ids": [r["id"] for r in rows][:100]}


def _windows(rows, minutes, need):
    """시간순 rows 에서 'minutes 분 안에 need 개 이상'인 구간들 (겹치면 하나로 합침)"""
    span = dt.timedelta(minutes=minutes)
    hit, j = set(), 0
    for i in range(len(rows)):  # 조건을 채운 구간에 든 기록에 표시
        while rows[i]["at"] - rows[j]["at"] > span:
            j += 1
        if i - j + 1 >= need:
            hit.update(range(j, i + 1))
    out, cur = [], []
    for k in sorted(hit):  # 표시된 기록을 '앞 기록과 span 이내'면 같은 묶음으로
        if cur and rows[k]["at"] - cur[-1]["at"] > span:
            out.append(cur)
            cur = []
        cur.append(rows[k])
    if cur:
        out.append(cur)
    return out


def detect(rows, days=14, until=None):
    """rows: load() 결과. 최근 days 일 안에 시작하는 경고만 반환 (그 전 기록은 기준선으로만 씀)"""
    until = (until or dt.datetime.now(dt.timezone.utc)).astimezone(KST)
    start = until - dt.timedelta(days=days)
    by_actor = collections.defaultdict(list)
    for r in rows:
        by_actor[r["actor"]].append(r)
    alerts = []
    for actor, rs in by_actor.items():
        ok = [r for r in rs if not r["refused"]]                 # 실제로 복원한 요청
        recent = [r for r in ok if r["at"] >= start]
        day = lambda r: r["at"].date()  # noqa: E731
        by_day = collections.defaultdict(list)
        for r in recent:
            by_day[day(r)].append(r)

        # burst
        c = RULES["burst"]
        for w in _windows(recent, c["window_min"], c["count"]):
            alerts.append(_alert("burst", "high", actor, w, len(w),
                                 f"{c['window_min']}분 안에 {len(w)}번 실명 보기"))

        # spike (기준선 = 이전 history_days 일 중 활동한 날의 하루 건수 중앙값)
        c = RULES["spike"]
        all_days = collections.Counter(day(r) for r in ok)
        for d, drs in by_day.items():
            hist = [all_days[d - dt.timedelta(days=k)] for k in range(1, c["history_days"] + 1)]
            hist = [h for h in hist if h]
            n = len(drs)
            floor = c["min_count_auditor"] if drs[-1].get("role") == "AUDITOR" else c["min_count"]
            if len(hist) >= c["min_history_days"]:
                base = statistics.median(hist)
                if n >= floor and n >= c["ratio"] * base:
                    alerts.append(_alert("spike", "medium", actor, drs, n,
                                         f"하루 {n}건 (평소 {base:g}건의 {n / max(base, 1):.1f}배)"))
            elif n >= max(c["abs_count"], floor):
                alerts.append(_alert("spike", "medium", actor, drs, n, f"하루 {n}건 (평소 기록 부족, 기준 {c['abs_count']}건)"))

        # off_hours
        c = RULES["off_hours"]
        for d, drs in by_day.items():
            off = [r for r in drs if r["at"].hour >= c["start_hour"] or r["at"].hour < c["end_hour"]
                   or r["at"].weekday() >= 5]
            if off:
                sev = "medium" if len(off) >= c["warn_count"] else "low"
                when = "주말" if off[0]["at"].weekday() >= 5 else "야간"
                alerts.append(_alert("off_hours", sev, actor, off, len(off), f"{when} {len(off)}번 실명 보기"))

        # breadth
        c = RULES["breadth"]
        for d, drs in by_day.items():
            docs = {x for r in drs for x in (r["doc_ids"] or [])}
            limit = c["docs_auditor"] if drs[-1].get("role") == "AUDITOR" else c["docs"]
            if len(docs) >= limit:
                alerts.append(_alert("breadth", "medium", actor, drs, len(docs), f"하루에 서로 다른 문서 {len(docs)}건"))

        # purpose (사유가 필요한 역할만)
        c = RULES["purpose"]
        for d, drs in by_day.items():
            need = [r for r in drs if r.get("role") in ("LEAD", "AUDITOR")]
            same = collections.Counter((r["purpose"] or "").strip() for r in need)
            for p, n in same.items():
                if p and n >= c["same_count"]:
                    alerts.append(_alert("purpose", "medium", actor, [r for r in need if (r["purpose"] or "").strip() == p],
                                         n, f"같은 사유로 {n}번 (사유 글자 수 {len(p)})"))
            short = [r for r in need if len((r["purpose"] or "").strip()) < c["short_len"]]
            if len(short) >= c["short_count"]:
                alerts.append(_alert("purpose", "low", actor, short, len(short), f"{c['short_len']}자 미만 사유 {len(short)}번"))

        # probing (거절 + 권한 밖 토큰이 남은 요청)
        c = RULES["probing"]
        tries = [r for r in rs if r["at"] >= start and (r["refused"] or r["denied"])]
        for w in _windows(tries, c["window_min"], c["count"]):
            n_ref = sum(1 for r in w if r["refused"])
            alerts.append(_alert("probing", "high", actor, w, len(w),
                                 f"{c['window_min']}분 안에 거절 {n_ref}번 · 권한 밖 토큰이 남은 요청 {len(w) - n_ref}번"))

        # same_person
        c = RULES["same_person"]
        for d, drs in by_day.items():
            hits = collections.defaultdict(list)
            for r in drs:
                for t in set(r["tokens"] or []):
                    if t.startswith("<PERSON_"):
                        hits[t].append(r)
            for t, trs in hits.items():
                docs = {x for r in trs for x in (r["doc_ids"] or [])}
                if len(trs) >= c["restores"] and len(docs) >= c["docs"]:
                    alerts.append(_alert("same_person", "high", actor, trs, len(trs),
                                         f"같은 사람 {t} 를 {len(trs)}번 · 문서 {len(docs)}건에서 복원"))
    alerts.sort(key=lambda a: (SEVERITY_ORDER[a["severity"]], a["start"]), reverse=False)
    return alerts


def run(days=14):
    """API 용: 최근 days 일 경고 + 규칙별 개수"""
    days = max(1, min(int(days), 90))
    alerts = detect(load(days), days)
    by_rule = collections.Counter(a["rule"] for a in alerts)
    return {"days": days, "rules": {k: {"name": RULE_KO[k], **v} for k, v in RULES.items()},
            "summary": {"total": len(alerts), "by_rule": dict(by_rule),
                        "by_severity": dict(collections.Counter(a["severity"] for a in alerts))},
            "alerts": alerts}


def check_latest(actor):
    """실명 보기 직후 본인에게 보여 줄 경고: 방금 요청까지 넣어 이 사람 기록에 규칙을 적용하고,
    '방금 요청이 들어간' 경고만 돌려줌. 막지는 않음. 반환: [{rule, rule_ko, severity, detail}]"""
    rows = load(days=1, actor=actor)
    if not rows:
        return []
    latest = max(r["id"] for r in rows)
    out = []
    for a in detect(rows, days=1):
        if latest in a["log_ids"] and not any(o["rule"] == a["rule"] for o in out):
            out.append({k: a[k] for k in ("rule", "rule_ko", "severity", "detail")})
    return out
