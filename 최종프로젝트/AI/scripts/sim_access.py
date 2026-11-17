"""이상 접근 탐지 평가용 모의 복원 기록 (worklog 38장).

    DB_PORT=5435 python scripts/sim_access.py --eval                 # 메모리에서 만들고 탐지·채점만 (DB 에 안 씀)
    python scripts/sim_access.py --insert --clean                    # 화면 시연용: 지금 DB 의 restore_log 에 넣음 (actor 가 sim-…@saebom.test)
    python scripts/sim_access.py --clean                             # 넣었던 모의 기록만 지움

정상 사용: 직원 30명(사원 20 · 팀장 7 · 감사 3), 4주(앞 2주 = 기준선, 뒤 2주 = 평가), 평일 근무 시간에 하루 0~8건.
  헷갈리기 쉬운 정상도 넣음: 바쁜 날(평소 2배), 21시대 야근, 감사의 넓은 점검(문서 40건), 같은 고객 2~3번 다시 보기
이상 시나리오 7종 × 강도 3단계(약·중·강): 약한 것은 일부러 기준 아래 → 놓치는 게 맞는 경우도 있음 (기준값은 미리 정한 그대로)
문서 ID·사람 토큰은 DB 에 실제 있는 것을 씀 (없으면 가짜 ID)
"""
import argparse
import collections
import datetime as dt
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.audit.anomaly import KST, detect  # noqa: E402
from app.core import config, db  # noqa: E402

SIM_DOMAIN = "@saebom.test"
SIM_PREFIX = "sim-"
TEAMS = ["고객센터", "여신심사팀", "리스크관리팀", "카드운영팀", "인사팀"]
GOOD_PURPOSES = ["고객 민원 회신 전 본인 확인", "한도 심사 결과 안내 대상 확인", "분실 신고 처리 이력 확인", "연체 안내 통화 전 확인",
                 "보이스피싱 피해 접수 경과 확인", "감사 지적 사항 조치 확인", "우편 반송 재발송 주소 확인", "대리인 위임 서류 확인"]


def corpus(rnd):
    """(문서 ID 목록, 문서별 사람 토큰). DB 에 없으면 가짜"""
    try:
        with db.vault_conn() as c:
            rows = c.execute("SELECT doc_id, array_agg(DISTINCT token) FROM vault.person_mentions GROUP BY doc_id").fetchall()
        if rows:
            return [r[0] for r in rows], {r[0]: r[1] for r in rows}
    except Exception:
        pass
    docs = [f"H{i:04d}" for i in range(1, 188)]
    return docs, {d: [f"<PERSON_{rnd.getrandbits(32):08x}>" for _ in range(2)] for d in docs}


def make(seed=7, start=None):
    rnd = random.Random(seed)
    docs, toks = corpus(rnd)
    start = start or dt.datetime(2026, 10, 19, tzinfo=KST)  # 월요일, 4주
    users = ([("STAFF", i) for i in range(20)] + [("LEAD", i) for i in range(7)] + [("AUDITOR", i) for i in range(3)])
    people = []
    for role, i in users:
        team = rnd.choice(TEAMS)
        people.append({"actor": f"{SIM_PREFIX}{role.lower()}{i:02d}{SIM_DOMAIN}", "role": role, "team": team,
                       "emp_id": f"S{len(people):03d}", "rate": rnd.uniform(1, 5) if role != "AUDITOR" else rnd.uniform(3, 8),
                       "docs": rnd.sample(docs, min(len(docs), 25 if role == "STAFF" else 60))})
    rows, labels = [], []
    nid = iter(range(1, 10 ** 7))

    def add(p, when, purpose=None, doc=None, token=None, denied=(), refused=None):
        doc = doc or rnd.choice(p["docs"])
        tk = token or (rnd.choice(toks.get(doc) or ["<PERSON_00000000>"]))
        if p["role"] in ("LEAD", "AUDITOR") and purpose is None and not refused:
            purpose = rnd.choice(GOOD_PURPOSES) + f" ({rnd.randint(100, 999)})"
        rows.append({"id": next(nid), "actor": p["actor"], "role": p["role"], "emp_id": p["emp_id"], "team": p["team"],
                     "purpose": purpose, "tokens": [] if refused else [tk], "doc_ids": [] if refused else [doc],
                     "denied": list(denied), "refused": refused, "at": when})

    def work_time(day):
        return day.replace(hour=rnd.randint(9, 17), minute=rnd.randint(0, 59), second=rnd.randint(0, 59))

    # 정상 사용 (+ 헷갈리기 쉬운 정상)
    for d in range(28):
        day = start + dt.timedelta(days=d)
        if day.weekday() >= 5:
            continue
        for p in people:
            n = max(0, int(rnd.gauss(p["rate"], 1.2)))
            if rnd.random() < 0.05:
                n *= 2                                            # 바쁜 날
            for _ in range(n):
                add(p, work_time(day))
            if rnd.random() < 0.03:
                add(p, day.replace(hour=21, minute=rnd.randint(0, 50)))  # 21시대 야근 (기준 22시 전)
            if p["role"] == "AUDITOR" and rnd.random() < 0.08:
                for doc in rnd.sample(docs, 40):                   # 감사의 넓은 점검 (감사 기준 60건 아래)
                    add(p, work_time(day), purpose="분기 정기 감사 표본 점검 " + doc, doc=doc)
            if rnd.random() < 0.05:
                doc = rnd.choice(p["docs"])
                for _ in range(rnd.randint(2, 3)):                 # 같은 고객 다시 보기
                    add(p, work_time(day), doc=doc)

    # 이상 시나리오 (평가 기간 = 뒤 2주)
    def scenario(kind, p, day, level):
        t0 = work_time(day)
        before = len(rows)
        if kind == "burst":        # 10분 안에 n번
            n = {"약": 7, "중": 12, "강": 30}[level]
            for k in range(n):
                add(p, t0 + dt.timedelta(seconds=int(k * 540 / n)))
        elif kind == "spike":      # 하루 종일 평소의 여러 배 (몰아서가 아니라 퍼뜨려서)
            n = {"약": int(p["rate"] * 2) + 2, "중": max(22, int(p["rate"] * 4)), "강": 45}[level]
            for _ in range(n):
                add(p, work_time(day))
        elif kind == "off_hours":  # 야간·주말
            n = {"약": 1, "중": 3, "강": 8}[level]
            night = day.replace(hour=23, minute=5)
            for k in range(n):
                add(p, night + dt.timedelta(minutes=k * 4))
        elif kind == "breadth":    # 서로 다른 문서 여러 건
            n = {"약": 20, "중": 35, "강": 70}[level]
            for doc in rnd.sample(docs, min(n, len(docs))):
                add(p, work_time(day), doc=doc)
        elif kind == "purpose":    # 같은 짧은 사유 반복
            n = {"약": 6, "중": 12, "강": 25}[level]
            for _ in range(n):
                add(p, work_time(day), purpose="확인")
        elif kind == "probing":    # 사유 없이 거절 + 권한 밖 토큰
            n = {"약": 3, "중": 6, "강": 15}[level]
            for k in range(n):
                when = t0 + dt.timedelta(minutes=k * 50 // n)
                if k % 2:
                    add(p, when, denied=("<PERSON_ffffffff>",))
                else:
                    add(p, when, refused="no_purpose")
        elif kind == "same_person":  # 한 사람을 여러 문서에서 반복
            tk = rnd.choice(toks.get(rnd.choice(docs)) or ["<PERSON_abcdef01>"])
            n = {"약": 3, "중": 6, "강": 12}[level]
            for _ in range(n):
                add(p, work_time(day), token=tk, doc=rnd.choice(docs))
        labels.append({"kind": kind, "level": level, "actor": p["actor"], "day": day.date().isoformat(),
                       "ids": [r["id"] for r in rows[before:]]})

    kinds = ["burst", "spike", "off_hours", "breadth", "purpose", "probing", "same_person"]
    leads = [p for p in people if p["role"] == "LEAD"]
    pool = rnd.sample(people, len(people))
    k = 0
    for kind in kinds:
        for level in ("약", "중", "강"):
            p = leads[k % len(leads)] if kind in ("purpose", "probing") else pool[k % len(pool)]
            day = start + dt.timedelta(days=14 + (k % 10) + (2 if (start + dt.timedelta(days=14 + k % 10)).weekday() >= 5 else 0))
            if kind == "off_hours" and level == "강":
                day = start + dt.timedelta(days=19)  # 토요일
            scenario(kind, p, day, level)
            k += 1
    rows.sort(key=lambda r: r["at"])
    return rows, labels, people


def evaluate(rows, labels, until, days=14):
    alerts = detect(rows, days=days, until=until)
    by_id = {}
    for a in alerts:
        for i in a["log_ids"]:
            by_id.setdefault(i, []).append(a)
    res = collections.defaultdict(dict)
    hit_alerts = set()
    for lb in labels:
        found = {a["rule"] for i in lb["ids"] for a in by_id.get(i, [])}
        for i in lb["ids"]:
            hit_alerts.update(a["id"] for a in by_id.get(i, []))
        res[lb["kind"]][lb["level"]] = sorted(found)
    false_alerts = [a for a in alerts if a["id"] not in hit_alerts]
    return alerts, res, false_alerts


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--insert", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    if a.clean:
        with db.vault_conn() as c:
            n = c.execute("DELETE FROM vault.restore_log WHERE actor LIKE %s", (SIM_PREFIX + "%" + SIM_DOMAIN,)).rowcount
        print(f"모의 기록 {n}건 지움 (DB 포트 {config.DB_PORT})")
    if a.eval:
        rows, labels, people = make(a.seed)
        until = dt.datetime(2026, 11, 15, 23, 59, tzinfo=KST)
        alerts, res, false_alerts = evaluate(rows, labels, until)
        n_norm_days = len({(r["actor"], r["at"].date()) for r in rows if r["at"] >= until - dt.timedelta(days=14)})
        print(f"기록 {len(rows)}건 · 직원 {len(people)}명 · 시나리오 {len(labels)}개 · 경고 {len(alerts)}건")
        print(f"{'시나리오':<14}{'약':<28}{'중':<28}{'강':<28}")
        for kind, lv in res.items():
            print(f"{kind:<14}" + "".join(f"{(', '.join(lv.get(x, [])) or '—'):<28}" for x in ("약", "중", "강")))
        print(f"잘못된 경고(시나리오와 무관): {len(false_alerts)}건 / 활동한 직원-날 {n_norm_days}개")
        for f in false_alerts[:10]:
            print("   ", f["rule"], f["severity"], f["actor"], f["start"][:16], f["detail"])
        out = config.OUT_DIR / "anomaly_sim"
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(json.dumps({"scenarios": res, "alerts": len(alerts),
                                                      "false_alerts": [{k: f[k] for k in ("rule", "severity", "actor", "start", "detail")}
                                                                       for f in false_alerts]},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    if a.insert:
        # 시연용: 지금 시각 기준 최근 4주로 옮겨서 넣음
        now = dt.datetime.now(KST)
        start = (now - dt.timedelta(days=27)).replace(hour=0, minute=0, second=0, microsecond=0)
        start -= dt.timedelta(days=start.weekday())
        rows, labels, _ = make(a.seed, start=start)
        rows = [r for r in rows if r["at"] <= now]
        with db.vault_conn() as c:
            c.cursor().executemany(
                """INSERT INTO vault.restore_log (actor, purpose, tokens, role, doc_ids, denied, emp_id, team, refused, created_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                [(r["actor"], r["purpose"], r["tokens"], r["role"], r["doc_ids"], r["denied"], r["emp_id"], r["team"],
                  r["refused"], r["at"]) for r in rows])
        print(f"모의 기록 {len(rows)}건 넣음 (DB 포트 {config.DB_PORT}, actor {SIM_PREFIX}…{SIM_DOMAIN})")


if __name__ == "__main__":
    main()
