"""문서에서 직원 목록과 문서 메타데이터(담당자·팀·공개 범위)를 추론한다 (worklog 1-5, 3-4-1, 10장).

안내문: 담당자·소속은 주지 않으니 문서에서 추론해 직원 목록 {emp_id, name, team, position} 과
문서 메타데이터 {id, owner_emp_id, team} 을 만들어 권한에 쓴다. 끝까지 모르면 더 엄격하게.

입력: 문서 [{id, text, doc_type}] + 탐지 구간 {id: [{label,start,end}]} (이름 구간은 탐지기 결과. 정답 라벨은 안 씀)
순서
  1. 직원 언급 찾기 — 이름 구간마다 2주차 mention_features(역할·소속) + 이름 뒤 직급 + 앞의 항목 이름(담당자: 등)
     고객·외부(수탁사 등)는 제외. 팀은 '(고객센터)' '채권관리팀 이상훈' '이상훈 사원, 채권관리팀' 처럼 같은 줄 가까이
  2. 직원 목록 — 이름별로 모으고, 한 이름이 여러 팀에서 나오면 팀마다 다른 직원(동명이인). 팀 없는 언급은
     그 이름의 팀이 하나뿐일 때만 그 직원에 붙임. 직급 = 가장 많이 나온 직급, 역할 = 직급·팀으로 (3-4-1)
  3. 문서 메타데이터 — 담당자: 작성자·서명 > 담당자 항목 > 처리자·상담원 > 신청자 > 직급 붙은 첫 직원.
     팀: 담당자의 팀 > 문서의 소속 항목 > 문서 속 직원들의 팀이 하나 > 문서에 나온 팀 이름이 하나 > 모름
     공개 범위: public(공지·부고·전 임직원) / team(팀을 앎) / restricted(팀 모름 → 감사만)
"""
import collections
import re

from app.pii.person_resolve import TITLES, mention_features
from app.pii.pii_pipeline import normalize, split_name

_TITLE_RE = "|".join(sorted(TITLES, key=len, reverse=True))
# 우리 회사 조직 이름: …팀 / …센터 / …본부 ('면접실' '회의실' 같은 장소는 빼려고 '실' 은 안 씀)
_TEAM = r"[가-힣A-Za-z]{1,10}?(?:팀|센터|본부)"  # 짧게: '인사팀팀장' → 인사팀
# 뒤에 조사·'입니다'·'장'(인사팀장)이 붙어도 팀 이름으로 ('새봄카드 고객센터입니다')
TEAM_RE = re.compile(rf"(?<![가-힣A-Za-z])({_TEAM})(?=$|[^가-힣]|입니다|에서|에게|의|에|로|은|는|이|가|과|와|를|을|장)")
_NOT_TEAM = {"콜센터", "고객님센터", "본부", "센터", "팀"}
# 줄 맨 앞 '항목:' — 'n. ', '- ', '| ' 같은 글머리 뒤
_KEY = re.compile(r"[\s\-*•·\d.)\]|]*([^:：|\n]{1,15}?)\s*[:：]")
# 문서의 소속 항목: '소속팀: IT운영팀', '요청 부서: 인사팀', '부서/직급: 인사팀/과장', '작성: 고객보호팀'
_TEAM_KEY = re.compile(r"소속|부서|작성$|발신$|팀$")
# 담당자를 가리키는 항목 이름 → 종류 (위에서부터 먼저 맞는 것)
KEY_KINDS = [
    ("approver", r"승인|확인자|결재|참조|수신|참석"),
    ("author", r"작성자|^작성$|보고자|발신|보낸\s*사람|소명자"),
    ("applicant", r"신청자|요청자|이수자|대상\s*직원|보고\s*대상"),
    ("handler", r"처리자|상담원|상담사|조치자|접수자|심사역|점검자|확인\s*담당"),
    ("owner", r"담당"),
]
# 콜론 없이 이름 바로 앞: '담당 심사역 양서현', '확인 담당 오예준 대리', '담당 정예린이'
_INLINE = re.compile(r"(?:^|[\s(])(담당\s*직원|처리\s*담당|확인\s*담당|담당|심사역|상담원|상담사|처리자)\s*$")
PRIORITY = {"signature": 0, "author": 0, "owner": 1, "handler": 2, "applicant": 3, "titled": 4, "approver": 5}
_SIGN_OFF = re.compile(r"(?:regards|sincerely|감사합니다|올림|드림)[\s,.!]*$", re.I)

# 흔한 한 글자 성 (외자 이름 '안윤' 을 성+이름으로 볼 때만 씀)
_SURNAME1 = set("김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진나지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용예봉경")
LEAD_TITLES = {"팀장", "차장", "부장", "센터장", "실장"}
AUDIT_TITLES = {"이사", "본부장", "대표"}
AUDIT_TEAMS = {"정보보호팀", "감사팀", "준법감시팀", "내부통제팀"}
PUBLIC_TITLE = re.compile(r"공지|부고|경조사|전\s*임직원")
# 감사 보고서: 종류 또는 첫 줄에 '감사 지적'·'내부 감사' (감사팀 내부 문서 → 감사 역할만)
AUDIT_DOC = re.compile(r"감사\s*지적|내부\s*감사|감사\s*결과\s*보고")


def _is_team(s):
    return bool(s) and bool(re.fullmatch(_TEAM, s)) and s not in _NOT_TEAM


def clean_team(t):
    """표기 정리: '인사팀팀' → 인사팀, '고객센터팀' → 고객센터 (synth3 원문에 있는 오타)"""
    if not t:
        return t
    t = re.sub(r"(팀|센터|본부)팀$", lambda m: m.group(1), t)
    return t


def _line(text, s, e):
    ls = text.rfind("\n", 0, s) + 1
    le = text.find("\n", e)
    return ls, (len(text) if le < 0 else le)


def _near_team(text, s, e, title_end):
    """같은 줄에서 이름 앞 15자 안 / 직급 뒤 15자 안의 팀 이름 (가장 가까운 것)."""
    ls, le = _line(text, s, e)
    best = None
    for m in TEAM_RE.finditer(text, ls, le):
        if not _is_team(m.group(1)):
            continue
        if m.end() <= s:
            d = s - m.end()
            between = text[m.end():s]
        elif m.start() >= title_end:
            d = m.start() - title_end
            between = text[title_end:m.start()]
        else:
            continue
        # 사이에 다른 이름·문장이 끼면 그 사람 팀이 아님 (쉼표·괄호·표 칸·직급·'소속'만 허용)
        if d <= 25 and _GAP.fullmatch(between) and (best is None or d < best[0]):
            best = (d, m.group(1))
    return best[1] if best else None


_GAP = re.compile(rf"[\s,(（)）/:：·|\-]*(?:(?:소속|{_TITLE_RE})[\s,(（)）/:：·|\-]*)*")


def _key_kind(key):
    for k, p in KEY_KINDS:
        if re.search(p, key):
            return k
    return None


def doc_team_fields(text):
    """문서의 소속 항목 값: '부서/직급: 인사팀/과장' → 인사팀"""
    out = []
    for line in text.split("\n"):
        m = _KEY.match(line)
        if m and _TEAM_KEY.search(m.group(1).strip()):
            t = TEAM_RE.match(line, m.end() + len(line[m.end():]) - len(line[m.end():].lstrip()))
            if t and _is_team(t.group(1)):
                out.append(clean_team(t.group(1)))
    return out


def staff_mentions(doc, spans):
    """문서 1건의 직원 언급: [{name, team, title, kind, start, end}] (kind = 담당자 판단용 항목 종류)."""
    text = doc["text"]
    spans = sorted(spans, key=lambda s: (s["start"], s["end"]))
    out = []
    for i, sp in enumerate(spans):
        if sp["label"] != "person":
            continue
        s, e = sp["start"], sp["end"]
        name = normalize("person", text[s:e])
        two = len(name) == 2 and name[0] in _SURNAME1  # 외자 이름 후보 ('안윤', '임린') — 아래에서 단서가 강할 때만
        if not (split_name(name)[0] or two):
            continue  # 성 없는 이름('지영 씨', '동현 님')은 누구인지 몰라 직원 목록에 안 넣음
        f = mention_features(text, spans, i)
        tm = re.match(rf"\s*({_TITLE_RE})(?![가-힣])", text[e:e + 8])
        title = tm.group(1) if tm else None
        title_end = e + (tm.end() if tm else 0)
        aff = f["aff"]
        if f["role"] in ("customer", "external") or (aff and not _is_team(aff)):
            continue  # 고객, 협력사·수탁사 사람
        ls, le = _line(text, s, e)
        before = text[ls:s]
        tb = re.search(rf"({_TITLE_RE})[\s/|,·]*$", before)  # '팀장 이지영', '리스크관리팀 / 부장 / 김성호'
        if not title and tb:
            title = tb.group(1)
        kind = None
        km = _KEY.match(text, ls)
        if km and km.end() <= s:  # 이 이름이 '항목: 값' 의 값 쪽
            kind = _key_kind(km.group(1).strip())
        im = _INLINE.search(before)
        if im and kind in (None, "titled"):
            kind = _key_kind(im.group(1)) or "owner"
        after_line = text[title_end:le]
        prev = text[:ls].rstrip().rsplit("\n", 1)[-1]
        if re.match(r"\s*(?:드림|올림)\s*$", after_line) or re.match(r"\s*입니다", after_line) or (
                not before.strip() and not after_line.strip() and _SIGN_OFF.search(prev)):
            kind = "signature"  # 'OOO 드림', '서은서 차장입니다', 'Best regards,\n김서현'
        team = clean_team(aff if _is_team(aff) else _near_team(text, s, e, title_end))
        if f["role"] != "staff" and not title and kind is None and team is None:
            continue  # 직원이라는 단서 없음
        if two and not (title or team or kind in ("author", "owner", "handler", "applicant", "signature")):
            continue  # 두 글자는 이름만 쓴 표기일 수 있어 강한 단서가 있을 때만
        out.append({"name": name, "team": team, "title": title, "kind": kind or ("titled" if title else None),
                    "start": s, "end": e})
    # 신청서·소명서처럼 문서에 소속 항목이 하나뿐이면 팀 없는 작성자·신청자에게 그 팀
    # (처리자·담당자에게는 안 줌: '요청 부서' 는 요청한 쪽 팀이라 처리한 사람 팀과 다름 — synth2 에서 확인)
    fields = set(doc_team_fields(text))
    if len(fields) == 1:
        for m in out:
            if m["team"] is None and m["kind"] in ("author", "applicant", "signature"):
                m["team"], m["team_from"] = next(iter(fields)), "field"
    return out


def _role_of(team, position):
    if team in AUDIT_TEAMS or position in AUDIT_TITLES:
        return "AUDITOR"
    if position in LEAD_TITLES:
        return "LEAD"
    return "STAFF"


def build(docs, spans_by_doc):
    """반환: (employees, doc_meta, mentions_by_doc, emp_key, emp_id)
    employees: [{emp_id, name, team, position, role, n_mentions, n_docs}]
    doc_meta : [{id, owner_emp_id, team, visibility, basis}]  basis = 담당자·팀을 어떻게 정했나"""
    mentions = {d["id"]: staff_mentions(d, spans_by_doc.get(d["id"], [])) for d in docs}

    # 두 글자 이름: 다른 문서의 성 포함 이름의 '이름' 부분이면 그 사람 ('지우 대리' → 엄지우), 여럿이면 버림
    full_by_given = collections.defaultdict(set)
    for ms in mentions.values():
        for m in ms:
            if len(m["name"]) >= 3 and split_name(m["name"])[0]:
                full_by_given[split_name(m["name"])[1]].add(m["name"])
    for doc_id, ms in mentions.items():
        kept = []
        for m in ms:
            if len(m["name"]) == 2 and m["name"] in full_by_given:
                fulls = full_by_given[m["name"]]
                if len(fulls) != 1:
                    continue
                m["name"] = next(iter(fulls))
            kept.append(m)
        mentions[doc_id] = kept

    # 2. 직원 목록: 이름 → 팀별
    teams_of = collections.defaultdict(collections.Counter)
    for ms in mentions.values():
        for m in ms:
            if m["team"]:
                teams_of[m["name"]][m["team"]] += 1
    emp_key = {}  # (doc_id, start) -> (name, team)
    agg = collections.defaultdict(lambda: {"titles": collections.Counter(), "n": 0, "docs": set()})
    for doc_id, ms in mentions.items():
        for m in ms:
            team = m["team"]
            if team is None:
                known = teams_of.get(m["name"])
                team = next(iter(known)) if known and len(known) == 1 else None
            k = (m["name"], team)
            emp_key[(doc_id, m["start"])] = k
            a = agg[k]
            a["n"] += 1
            a["docs"].add(doc_id)
            if m["title"] and m["title"] != "상담원":
                a["titles"][m["title"]] += 1
            elif m["title"]:
                a["titles"].setdefault("상담원", 0)
    keys = sorted(agg, key=lambda k: (k[1] is None, k[1] or "", k[0]))
    emp_id = {k: f"E{i:03d}" for i, k in enumerate(keys, 1)}
    employees = []
    for k in keys:
        a = agg[k]
        position = a["titles"].most_common(1)[0][0] if a["titles"] else None
        employees.append({"emp_id": emp_id[k], "name": k[0], "team": k[1], "position": position,
                          "role": _role_of(k[1], position), "n_mentions": a["n"], "n_docs": len(a["docs"])})
    team_of_emp = {emp_id[k]: k[1] for k in keys}

    # 3. 문서 메타데이터
    meta = []
    for d in docs:
        text, ms = d["text"], mentions[d["id"]]
        cands = sorted((m for m in ms if m["kind"] in PRIORITY), key=lambda m: (PRIORITY[m["kind"]], m["start"]))
        owner = emp_id[emp_key[(d["id"], cands[0]["start"])]] if cands else None
        owner_basis = cands[0]["kind"] if cands else None
        fields = doc_team_fields(text)
        staff_teams = {team_of_emp[emp_id[emp_key[(d["id"], m["start"])]]] for m in ms} - {None}
        named_teams = {clean_team(t) for t in TEAM_RE.findall(text) if _is_team(t)}
        if owner and team_of_emp[owner]:
            team, team_basis = team_of_emp[owner], "owner"
        elif len(set(fields)) == 1:
            team, team_basis = fields[0], "field"
        elif len(staff_teams) == 1:
            team, team_basis = next(iter(staff_teams)), "staff"
        elif len(named_teams) == 1:
            team, team_basis = next(iter(named_teams)), "named"
        else:
            team, team_basis = None, None
        first_line = text.strip().split("\n", 1)[0]
        audit = bool(AUDIT_DOC.search(d.get("doc_type") or "") or AUDIT_DOC.search(first_line))
        if audit:
            vis = "restricted"  # 감사 보고서는 지적받은 팀이 아니라 감사만 (사용자 결정 10-4). 팀은 기록용으로 남김
        elif "공지" in (d.get("doc_type") or "") or PUBLIC_TITLE.search(first_line) or re.search(r"대상\s*[:：]\s*전\s*임직원", text):
            vis = "public"
        elif team:
            vis = "team"
        else:
            vis = "restricted"
        meta.append({"id": d["id"], "owner_emp_id": owner, "team": team, "visibility": vis,
                     "basis": {"owner": owner_basis, "team": team_basis}})
    return employees, meta, {k: v for k, v in mentions.items()}, emp_key, emp_id
