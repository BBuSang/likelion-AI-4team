"""이상 접근 탐지 규칙. DB 없이 기록을 직접 만들어 detect() 에 넣음."""
import datetime as dt

from app.audit.anomaly import KST, detect

NOW = dt.datetime(2026, 11, 13, 18, 0, tzinfo=KST)  # 금요일 18시
_id = iter(range(1, 10 ** 6))


def row(actor="a@x", at=None, role="LEAD", purpose="민원 처리 확인", tokens=("<PERSON_00000001>",), docs=("H0001",),
        denied=(), refused=None):
    return {"id": next(_id), "actor": actor, "role": role, "emp_id": "E1", "team": "고객센터", "purpose": purpose,
            "tokens": list(tokens), "doc_ids": list(docs), "denied": list(denied), "refused": refused, "at": at}


def at(day_offset=0, h=10, m=0):
    return (NOW - dt.timedelta(days=day_offset)).replace(hour=h, minute=m)


def rules(alerts):
    return {a["rule"] for a in alerts}


def normal_history(actor="a@x", days=14, per_day=4):
    """평일 근무 시간에 하루 몇 건, 사유·문서·사람이 다양"""
    out = []
    for d in range(1, days + 1):
        if (NOW - dt.timedelta(days=d)).weekday() >= 5:
            continue
        for k in range(per_day):
            out.append(row(actor, at(d, 9 + k * 2, 5), purpose=f"민원 {d}-{k} 처리 확인",
                           tokens=(f"<PERSON_{d:04d}{k:04d}>",), docs=(f"H{d:02d}{k:02d}",)))
    return out


def test_normal_use_no_alert():
    rows = normal_history() + [row("a@x", at(0, 11), purpose="오늘 민원 확인", tokens=("<PERSON_99990000>",), docs=("H9999",))]
    assert detect(rows, days=1, until=NOW) == []


def test_burst():
    rows = normal_history() + [row("a@x", at(0, 14, m), purpose=f"확인 건 {m}", tokens=(f"<PERSON_7777{m:04d}>",),
                                   docs=(f"H7{m:03d}",)) for m in range(10)]
    assert "burst" in rules(detect(rows, days=1, until=NOW))


def test_spike_and_breadth():
    rows = normal_history() + [row("a@x", at(0, 9 + (i % 8), i % 60), purpose=f"점검 {i}", tokens=(f"<PERSON_5555{i:04d}>",),
                                   docs=(f"H5{i:03d}",)) for i in range(35)]
    r = rules(detect(rows, days=1, until=NOW))
    assert {"spike", "breadth"} <= r


def test_off_hours():
    rows = normal_history() + [row("a@x", at(0, 23, 10 + i), purpose=f"야간 확인 {i}", tokens=(f"<PERSON_4444{i:04d}>",),
                                   docs=(f"H4{i:03d}",)) for i in range(3)]
    a = [x for x in detect(rows, days=1, until=NOW) if x["rule"] == "off_hours"]
    assert a and a[0]["severity"] == "medium"


def test_purpose_repeated():
    rows = normal_history() + [row("a@x", at(0, 9 + i % 8, (i * 7) % 60), purpose="확인", tokens=(f"<PERSON_3333{i:04d}>",),
                                   docs=(f"H3{i:03d}",)) for i in range(10)]
    assert "purpose" in rules(detect(rows, days=1, until=NOW))


def test_probing():
    rows = normal_history() + [row("a@x", at(0, 15, i * 5), purpose=None, tokens=(), refused="no_purpose") for i in range(3)] \
        + [row("a@x", at(0, 15, 20 + i * 5), denied=("<PERSON_00000009>",)) for i in range(2)]
    assert "probing" in rules(detect(rows, days=1, until=NOW))


def test_same_person():
    rows = normal_history() + [row("a@x", at(0, 10 + i, 0), purpose=f"상담 이력 {i}", tokens=("<PERSON_deadbeef>",),
                                   docs=(f"H2{i:03d}",)) for i in range(5)]
    assert "same_person" in rules(detect(rows, days=1, until=NOW))


def test_old_history_only_baseline():
    """기준선 기간의 기록은 경고를 만들지 않음 (최근 days 일 안에 시작한 것만)"""
    rows = [row("a@x", at(5, 14, m), purpose=f"확인 {m}", tokens=(f"<PERSON_1111{m:04d}>",), docs=(f"H1{m:03d}",))
            for m in range(10)]
    assert detect(rows, days=1, until=NOW) == []


def test_alert_ids_unique():
    """같은 사람·같은 시각에 토큰만 다른 경고들도 id 가 겹치지 않음 (화면 정렬이 섞이던 원인, 39장)"""
    rows = normal_history()
    for i in range(5):
        rows.append(row("a@x", at(0, 10 + i, 0), purpose=f"상담 이력 {i}", tokens=("<PERSON_aaaa0001>", "<PERSON_aaaa0002>"),
                        docs=(f"H8{i:03d}",)))
    alerts = detect(rows, days=1, until=NOW)
    assert sum(a["rule"] == "same_person" for a in alerts) == 2
    assert len({a["id"] for a in alerts}) == len(alerts)
