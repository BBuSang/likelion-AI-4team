"""문서 1건을 저장할 때 이름 언급을 **이미 DB 에 있는 사람**과 잇기 (worklog 3-7-1).

2주차 ingest 는 업로드 전체를 한꺼번에 보고 묶었지만(person_resolve.resolve), 그 묶음 key 는 업로드마다 달라질 수 있어
다음 파일의 같은 사람이 다른 토큰이 됐다. 여기서는
  1. 문서 안에서는 2주차 resolve() 를 그대로 써서 언급을 묶고 단서(닻·소속·역할)를 뽑고
  2. 그 묶음을 rag.persons / rag.person_attrs 의 기존 인물과 맞춰 본다.
     닻(전화·이메일·계좌 등) 토큰은 사람과 상관없이 항상 같으므로 원문 없이 토큰만으로 비교할 수 있다.
  3. 이은 사람은 vault.person_keys 의 key 를 그대로 써서 같은 사람 토큰을 받는다.

판단 (잘못 합치는 것이 쪼개지는 것보다 위험하므로 애매하면 잇지 않는다)
  닻이 맞는 기존 인물 1명                    → anchor_link
  닻이 맞는 기존 인물 2명 이상                → ambiguous: 검수자가 고르기 전까지 이름 기준 미확정 토큰
  닻 있음, 맞는 사람 없음                     → new (새 key)
  닻 없음, 같은 이름·같은 소속 1명            → aff_link
  닻 없음, 소속 있음, 맞는 사람 없음           → new
  닻·소속 없음, 같은 이름의 확정 인물이 1명뿐 → unique_link (약함, 검수 화면에 표시)
  그 밖                                       → unresolved (이름 기준 토큰, 2주차 name_only 와 같음)
검수자가 choices 로 고르면 그 결정을 따른다: {그룹 id: 기존 pid | "new" | "unresolved"}
"""
import collections
import secrets

from app.pii import person_resolve, pii_pipeline

WEAK = {"unique_link", "ambiguous"}  # 검수 화면에서 확인을 권하는 판단


def pid_of(name, key):
    return "P_" + pii_pipeline._hmac(f"pid|{name}|{key or ''}")


def _candidates(rag_cur, name):
    """같은 이름의 기존 인물과 그 닻 토큰."""
    rag_cur.execute(
        """SELECT p.pid, p.role, p.affiliations, p.status,
                  COALESCE(array_agg(a.label || ' ' || a.token) FILTER (WHERE a.token IS NOT NULL), '{}')
           FROM rag.persons p LEFT JOIN rag.person_attrs a USING (pid)
           WHERE p.name_token = %s GROUP BY p.pid""",
        (pii_pipeline.name_token(name),))
    out = []
    for pid, role, affs, status, attrs in rag_cur.fetchall():
        by_label = collections.defaultdict(set)
        for x in attrs:
            lab, tok = x.split(" ", 1)
            by_label[lab].add(tok)
        out.append({"pid": pid, "role": role, "affs": set(affs or []), "status": status, "anchors": by_label})
    return out


def _role_ok(a, b):
    return not a or not b or a == b


def _conflicts(group_anchors, cand):
    """같은 종류 식별자가 서로 다르면 다른 사람 (2주차 resolve.compatible 와 같은 규칙)."""
    return any(group_anchors[lab] and cand["anchors"].get(lab) and not group_anchors[lab] & cand["anchors"][lab]
               for lab in group_anchors)


def _compatible(anchors, affs, role, cand):
    """서로 어긋나지 않으면 같은 사람일 수 있음 (2주차 resolve.compatible 를 기존 인물 기준으로)."""
    if not _role_ok(role, cand["role"]) or _conflicts(anchors, cand):
        return False
    if affs and cand["affs"] and not affs & cand["affs"]:
        return False  # 소속이 다름
    if bool(affs) != bool(cand["affs"]) and not (role and role == cand["role"]):
        return False  # 소속 있는 쪽과 없는 쪽은 역할이 같다고 확인될 때만
    return True


def link(doc_id, text, spans, rag_cur, vault_cur, choices=None):
    """spans: 확정된 구간 [{label,start,end}].
    반환: (spans — person 구간에 normalized·person_key·basis·role·aff 를 붙인 것, groups — 사람 묶음별 판단)
    groups: [{"id", "mentions": [[start,end]], "basis", "pid", "candidates": [pid], "evidence": [라벨], "anchors": {(라벨, 토큰)}}]"""
    choices = choices or {}
    resolved, _ = person_resolve.resolve([{"id": doc_id, "text": text}], {doc_id: spans})
    out = resolved[doc_id]

    grouped = collections.defaultdict(list)
    for s in out:
        if s["label"] == "person":
            grouped[(s["normalized"], s.get("person_key"))].append(s)

    groups = []
    for gi, ((name, local_key), ss) in enumerate(sorted(grouped.items(), key=lambda kv: kv[1][0]["start"])):
        gid = f"g{gi}"
        anchors = collections.defaultdict(set)
        anchor_pairs = set()
        for s in ss:
            for lab, v in s.get("anchors", []):
                tok = pii_pipeline.make_token(lab, v)
                anchors[lab].add(tok)
                anchor_pairs.add((lab, tok))
        affs = {s["aff"] for s in ss if s.get("aff")}
        roles = collections.Counter(s["role"] for s in ss if s.get("role"))
        role = roles.most_common(1)[0][0] if roles else None
        cands = [c for c in _candidates(rag_cur, name) if _role_ok(role, c["role"])]

        evidence, matched = [], []
        by_anchor = [c for c in cands if any(anchors[lab] & c["anchors"].get(lab, set()) for lab in anchors)
                     and not _conflicts(anchors, c)] if anchors else []
        by_aff = [c for c in cands if c["affs"] & affs] if affs else []
        # 이름 기준 미확정 인물(unresolved)은 '다른 사람'으로 세지 않음
        compat = [c for c in cands if c["status"] == "resolved" and _compatible(anchors, affs, role, c)]
        if by_anchor:
            matched, evidence = by_anchor, sorted({lab for c in by_anchor for lab in anchors
                                                   if anchors[lab] & c["anchors"].get(lab, set())})
            basis = "anchor_link" if len(matched) == 1 else "ambiguous"
        elif by_aff:
            matched, evidence = by_aff, ["affiliation"]
            basis = "aff_link" if len(matched) == 1 else "ambiguous"
        elif len(compat) == 1:
            # 2주차 resolve 3단계(서로 어긋나지 않는 묶음 합치기)와 같은 판단을 기존 인물과 함
            matched = compat
            basis, evidence = ("compat_link", ["compatible"]) if anchors or affs else ("unique_link", ["name"])
        elif not compat:
            basis = "new"  # 처음 나온 사람. 단서가 없어도 새 사람으로 두어야 뒤 문서가 이 사람과 이어질 수 있음
        else:
            matched = compat
            basis = "new" if anchors or affs else "unresolved"  # 단서 있는 사람은 따로, 없으면 미확정

        choice = choices.get(gid)
        if choice == "new":
            basis = "new"
        elif choice == "unresolved":
            basis = "unresolved"
        elif choice:
            if choice not in {c["pid"] for c in cands}:
                raise ValueError(f"{gid}: 고른 인물 {choice} 는 같은 이름의 후보가 아님")
            matched, basis = [c for c in cands if c["pid"] == choice], "chosen_link"

        if basis in ("anchor_link", "aff_link", "compat_link", "unique_link", "chosen_link"):
            pid = matched[0]["pid"]
            vault_cur.execute("SELECT person_key FROM vault.person_keys WHERE pid = %s", (pid,))
            row = vault_cur.fetchone()
            if row is None:  # rag 에만 있고 key 가 없는 인물 (생기면 안 됨) → 새 사람으로
                key, basis = f"id:{secrets.token_hex(6)}", "new"
            else:
                key = row[0]
        elif basis == "new":
            key = f"id:{secrets.token_hex(6)}"
        else:  # unresolved / ambiguous → 이름 기준 토큰
            key = None
        pid = pid_of(name, key)

        for s in ss:
            s["person_key"], s["basis"] = key, basis
        groups.append({"id": gid, "name": name, "key": key, "pid": pid, "basis": basis, "role": role,
                       "affs": affs, "mentions": [[s["start"], s["end"]] for s in ss],
                       "candidates": [c["pid"] for c in matched], "evidence": evidence,
                       "anchors": anchor_pairs if key else set()})
    return out, groups


def public_groups(groups):
    """API 로 내보내는 판단 요약 (이름 원문·key 는 빼고 위치·근거만. 이름은 화면이 원문 위치로 보여 줌)."""
    return [{"id": g["id"], "mentions": g["mentions"], "basis": g["basis"], "pid": g["pid"],
             "candidates": g["candidates"], "evidence": g["evidence"], "needs_check": g["basis"] in WEAK}
            for g in groups]
