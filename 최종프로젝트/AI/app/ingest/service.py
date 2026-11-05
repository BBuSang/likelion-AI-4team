"""적재 흐름 (worklog 3-6, 3-7): 업로드 → 탐지 → 검수 대기 → 승인/반려 → 사람 잇기 → 가명처리 → 청크·임베딩 → 저장.

- 원문은 vault.staging_doc 에만 둔다 (vault_user). 외부 API(임베딩)로는 가명처리 + guard 를 거친 청크만 나간다
- 정답 라벨(gold)이 있는 문서: 탐지 결과 = 정답이면 자동 승인(approved_by='gold'), 다르면 검수 대기
- 저장은 한 번에 한 건 (advisory lock). 같은 사람이 동시에 두 번 새로 생기지 않게
- 검수 기록(review_log)에는 값 없이 라벨·위치·개수만
"""
import collections
import json
import logging
import threading

from psycopg.types.json import Jsonb

from app.core import config, db
from app.ingest import linking
from app.ingest.ingest import _display, chunk
from app.pii import pii_pipeline
from app.rag.embed import embed

logger = logging.getLogger("ingest")

LABELS = ("person", "phone_number", "email", "address", "resident_registration_number", "account_number", "card_number")
MAX_CHARS = 600            # 청크 크기 (2주차와 같음)
MAX_DOC_CHARS = 20000      # 문서 1건 최대 길이
MAX_DOCS = 2000            # 업로드 1번 최대 건수
_STORE_LOCK_KEY = 74_210_501  # pg_advisory_xact_lock 번호 (저장 단계 한 건씩)
_detect_lock = threading.Lock()  # 탐지 모델을 여러 업로드가 동시에 쓰지 않게


class IngestError(ValueError):
    """요청이 잘못됨 (400)"""


class Conflict(Exception):
    """상태가 맞지 않음 (409)"""


# ---------------------------------------------------------------- 입력 읽기
def parse_upload(source, content, doc_type=None):
    """source: jsonl (1·2주차 형식: 한 줄에 {id, text, doc_type, entities, ...}) / text (줄글 1건).
    반환: [{ext_id, text, doc_type, meta, gold}]"""
    if source == "text":
        if not content.strip():
            raise IngestError("빈 글")
        docs = [{"ext_id": None, "text": content, "doc_type": doc_type, "meta": {}, "gold": None}]
    elif source == "jsonl":
        docs = []
        for n, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                raise IngestError(f"{n}번째 줄: JSON 이 아님")
            if not isinstance(r, dict) or not isinstance(r.get("text"), str) or not r["text"].strip():
                raise IngestError(f"{n}번째 줄: text 가 없음")
            gold = None
            if isinstance(r.get("entities"), list):
                gold = sorted({(e["label"], int(e["start"]), int(e["end"])) for e in r["entities"]
                               if e.get("label") in LABELS}, key=lambda x: (x[1], x[2]))
                gold = [{"label": l, "start": s, "end": e} for l, s, e in gold]
            meta = {k: v for k, v in r.items() if k not in ("id", "text", "doc_type", "entities", "split")}
            docs.append({"ext_id": str(r["id"]) if r.get("id") is not None else None, "text": r["text"],
                         "doc_type": r.get("doc_type") or doc_type, "meta": meta, "gold": gold})
    else:
        raise IngestError(f"모르는 입력 형식: {source}")
    if not docs:
        raise IngestError("문서가 없음")
    if len(docs) > MAX_DOCS:
        raise IngestError(f"한 번에 {MAX_DOCS}건까지")
    for i, d in enumerate(docs, 1):
        if len(d["text"]) > MAX_DOC_CHARS:
            raise IngestError(f"{i}번째 문서가 {MAX_DOC_CHARS}자를 넘음")
    return docs


# ---------------------------------------------------------------- 구간 확인
def validate_spans(text, spans):
    """검수 화면이 보낸 구간 확인. 위치는 파이썬 글자 단위(코드포인트).
    span 에 text 를 같이 보내면 원문과 같은지도 본다 (화면 UTF-16 위치가 어긋났는지 잡기)."""
    out = []
    for i, s in enumerate(spans):
        try:
            label, start, end = s["label"], int(s["start"]), int(s["end"])
        except (KeyError, TypeError, ValueError):
            raise IngestError(f"{i}번째 구간: label·start·end 필요")
        if label not in LABELS:
            raise IngestError(f"{i}번째 구간: 모르는 라벨 {label}")
        if not 0 <= start < end <= len(text):
            raise IngestError(f"{i}번째 구간: 위치가 본문 밖")
        if "text" in s and s["text"] != text[start:end]:
            raise IngestError(f"{i}번째 구간: 위치와 글자가 맞지 않음 (화면 위치 계산 확인)")
        if not text[start:end].strip():
            raise IngestError(f"{i}번째 구간: 공백만 있음")
        out.append({"label": label, "start": start, "end": end})
    out.sort(key=lambda s: (s["start"], s["end"]))
    for a, b in zip(out, out[1:]):
        if b["start"] < a["end"]:
            raise IngestError(f"구간이 겹침: {a['start']}-{a['end']} / {b['start']}-{b['end']}")
    return out


def _key(s):
    return (s["label"], s["start"], s["end"])


def diff(detected, final):
    """탐지기 결과 대비 검수자가 바꾼 개수. 같은 위치 다른 라벨 = relabeled, 위치가 바뀌면 지움+추가."""
    d, f = {_key(s) for s in detected}, {_key(s) for s in final}
    d_pos = {(s[1], s[2]): s[0] for s in d}
    f_pos = {(s[1], s[2]): s[0] for s in f}
    relabeled = sum(1 for p, lab in f_pos.items() if p in d_pos and d_pos[p] != lab)
    added = sum(1 for p in f_pos if p not in d_pos)
    removed = sum(1 for p in d_pos if p not in f_pos)
    return {"added": added, "removed": removed, "relabeled": relabeled,
            "detail": {"added": sorted([lab, *p] for p, lab in f_pos.items() if p not in d_pos),
                       "removed": sorted([lab, *p] for p, lab in d_pos.items() if p not in f_pos),
                       "relabeled": sorted([d_pos[p], lab, *p] for p, lab in f_pos.items()
                                           if p in d_pos and d_pos[p] != lab)}}


# ---------------------------------------------------------------- 업로드
def create_job(uploader, source, filename, content, doc_type=None):
    """job 만 만들고 id 를 돌려줌. 탐지는 process_job() 이 (서버에서는 백그라운드로)."""
    docs = parse_upload(source, content, doc_type)
    with db.vault_conn() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO vault.ingest_job (uploader, source, filename, total) VALUES (%s,%s,%s,%s) RETURNING id",
                    (uploader, source, filename, len(docs)))
        job_id = cur.fetchone()[0]
    return job_id, docs


def process_job(job_id, docs):
    """문서마다 탐지 → staging 저장. 정답이 있고 탐지 = 정답이면 바로 저장(자동 승인)."""
    try:
        auto = []
        with db.vault_conn() as conn, conn.cursor() as cur:
            for seq, d in enumerate(docs, 1):
                with _detect_lock:
                    det = sorted(pii_pipeline.detect(d["text"]), key=lambda s: (s["start"], s["end"]))
                det = [{"label": s["label"], "start": s["start"], "end": s["end"]} for s in det]
                doc_id = f"J{job_id}-{d['ext_id'] or seq}"
                cur.execute(
                    """INSERT INTO vault.staging_doc (job_id, seq, ext_id, doc_id, doc_type, text, meta, detected, gold)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                    (job_id, seq, d["ext_id"], doc_id, d["doc_type"], d["text"], Jsonb(d["meta"]), Jsonb(det),
                     Jsonb(d["gold"]) if d["gold"] is not None else None))
                sid = cur.fetchone()[0]
                if d["gold"] is not None and {_key(s) for s in det} == {_key(s) for s in d["gold"]}:
                    auto.append(sid)
            cur.execute("UPDATE vault.ingest_job SET status = 'ready' WHERE id = %s", (job_id,))
        for sid in auto:  # 업로드 순서대로 한 건씩 저장
            try:
                store(sid, actor="gold", action="auto_approve")
            except Exception:
                logger.exception("자동 승인 저장 실패: staging %s", sid)
    except Exception as e:
        logger.exception("job %s 처리 실패", job_id)
        with db.vault_conn() as conn, conn.cursor() as cur:
            cur.execute("UPDATE vault.ingest_job SET status = 'failed', error = %s WHERE id = %s",
                        (type(e).__name__, job_id))


# ---------------------------------------------------------------- 조회
def job_summary(job_id):
    with db.vault_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, uploader, source, filename, total, status, created_at FROM vault.ingest_job WHERE id = %s",
                    (job_id,))
        row = cur.fetchone()
        if row is None:
            return None
        cur.execute("SELECT status, count(*) FROM vault.staging_doc WHERE job_id = %s GROUP BY status", (job_id,))
        counts = dict(cur.fetchall())
        cur.execute("SELECT count(*) FROM vault.staging_doc WHERE job_id = %s AND approved_by = 'gold'", (job_id,))
        auto = cur.fetchone()[0]
    keys = ("id", "uploader", "source", "filename", "total", "status", "created_at")
    return dict(zip(keys, row)) | {"counts": counts, "auto_approved": auto}


def list_docs(job_id):
    with db.vault_conn() as conn, conn.cursor() as cur:
        cur.execute("""SELECT id, seq, ext_id, doc_id, doc_type, status, approved_by, jsonb_array_length(detected),
                              gold IS NOT NULL
                       FROM vault.staging_doc WHERE job_id = %s ORDER BY seq""", (job_id,))
        keys = ("id", "seq", "ext_id", "doc_id", "doc_type", "status", "approved_by", "n_detected", "has_gold")
        return [dict(zip(keys, r)) for r in cur.fetchall()]


def _load(cur, staging_id, for_update=False):
    cur.execute("SELECT id, job_id, doc_id, doc_type, text, meta, detected, gold, final_spans, status, approved_by "
                "FROM vault.staging_doc WHERE id = %s" + (" FOR UPDATE" if for_update else ""), (staging_id,))
    row = cur.fetchone()
    if row is None:
        return None
    keys = ("id", "job_id", "doc_id", "doc_type", "text", "meta", "detected", "gold", "final_spans", "status",
            "approved_by")
    return dict(zip(keys, row))


def get_doc(staging_id, actor):
    """검수 화면용: 원문 + 탐지 구간 + 사람 잇기 미리보기. 원문이 화면으로 가는 유일한 길이라 조회 기록을 남김."""
    with db.vault_conn() as conn, conn.cursor() as cur:
        d = _load(cur, staging_id)
        if d is None:
            return None
        cur.execute("INSERT INTO vault.review_log (staging_doc_id, actor, action) VALUES (%s, %s, 'view')",
                    (staging_id, actor))
    spans = d["final_spans"] or d["detected"]
    d["links"] = preview_links(staging_id, spans, _doc=d) if d["status"] in ("pending", "failed") else []
    d["spans"] = spans
    return d


def preview_links(staging_id, spans, choices=None, _doc=None):
    """지금 구간으로 저장하면 사람이 어떻게 이어지는지 (저장 안 함)."""
    if _doc is None:
        with db.vault_conn() as conn, conn.cursor() as cur:
            _doc = _load(cur, staging_id)
        if _doc is None:
            return None
    spans = validate_spans(_doc["text"], spans)
    with db.rag_conn() as rc, rc.cursor() as rcur, db.vault_conn() as vc, vc.cursor() as vcur:
        _, groups = linking.link(_doc["doc_id"], _doc["text"], spans, rcur, vcur, choices)
    return linking.public_groups(groups)


# ---------------------------------------------------------------- 검수
def review(staging_id, actor, action, spans=None, reason=None, choices=None):
    if action == "reject":
        with db.vault_conn() as conn, conn.cursor() as cur:
            d = _load(cur, staging_id, for_update=True)
            if d is None:
                return None
            if d["status"] not in ("pending", "failed"):
                raise Conflict(f"이미 처리됨: {d['status']}")
            cur.execute("""UPDATE vault.staging_doc SET status='rejected', approved_by=%s, reviewed_at=now(),
                           reject_reason=%s WHERE id=%s""", (actor, (reason or "")[:500], staging_id))
            cur.execute("INSERT INTO vault.review_log (staging_doc_id, actor, action, detail) VALUES (%s,%s,'reject',%s)",
                        (staging_id, actor, Jsonb({"reason_given": bool(reason)})))
        return {"status": "rejected"}
    if action != "approve":
        raise IngestError("action 은 approve / reject")
    if spans is None:
        raise IngestError("승인하려면 spans 필요")
    return store(staging_id, actor=actor, action="approve", spans=spans, choices=choices)


def store(staging_id, actor, action, spans=None, choices=None):
    """승인된 1건 저장. spans 가 없으면 탐지 결과(자동 승인)."""
    vconn = db.vault_conn()
    try:
        with vconn.cursor() as vcur:
            vcur.execute("SELECT pg_advisory_xact_lock(%s)", (_STORE_LOCK_KEY,))
            d = _load(vcur, staging_id, for_update=True)
            if d is None:
                return None
            if d["status"] not in ("pending", "failed"):
                raise Conflict(f"이미 처리됨: {d['status']}")
            final = validate_spans(d["text"], spans if spans is not None else d["detected"])
            changes = diff(d["detected"], final)
            try:
                result = _store_doc(d, final, vcur, choices)
            except (IngestError, Conflict):
                raise
            except Exception as e:
                vconn.rollback()
                with vconn.cursor() as c2:
                    c2.execute("UPDATE vault.staging_doc SET status='failed', error=%s WHERE id=%s",
                               (type(e).__name__, staging_id))
                    c2.execute("INSERT INTO vault.review_log (staging_doc_id, actor, action) VALUES (%s,%s,'store_failed')",
                               (staging_id, actor))
                vconn.commit()
                raise
            vcur.execute("""UPDATE vault.staging_doc SET status='stored', final_spans=%s, approved_by=%s,
                            reviewed_at=now(), error=NULL WHERE id=%s""", (Jsonb(final), actor, staging_id))
            vcur.execute("""INSERT INTO vault.review_log (staging_doc_id, actor, action, added, removed, relabeled, detail)
                            VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                         (staging_id, actor, action, changes["added"], changes["removed"], changes["relabeled"],
                          Jsonb(changes["detail"] | {"links": [{"basis": g["basis"], "n": len(g["mentions"])}
                                                               for g in result["links"]]})))
        vconn.commit()
        return {"status": "stored", "doc_id": d["doc_id"], **{k: changes[k] for k in ("added", "removed", "relabeled")},
                "chunks": result["chunks"], "links": result["links"]}
    finally:
        vconn.close()


def _store_doc(d, final, vcur, choices):
    """사람 잇기 → 가명처리 → guard → 청크 → 임베딩 → rag 저장(커밋) → vault 쓰기(호출한 쪽이 커밋).
    rag 를 먼저 커밋하고 vault 가 실패하면 토큰은 있는데 복원 매핑이 없는 상태가 됨 (유출은 아님). 드묾."""
    doc_id, text = d["doc_id"], d["text"]
    with db.rag_conn() as rconn, rconn.cursor() as rcur:
        linked, groups = linking.link(doc_id, text, final, rcur, vcur, choices)
        masked, records = pii_pipeline.pseudonymize(text, linked)
        masked, _ = pii_pipeline.guard(masked)  # 탐지·검수 모두 놓친 번호·이메일 모양 (API·DB 에 원문이 남지 않게)

        mapping, persons, mentions = {}, {}, []
        extra = collections.defaultdict(set)
        group_of = {(m[0], m[1]): g for g in groups for m in g["mentions"]}
        for r in records:
            extra[r["token"]].update(r["search_tokens"])
            m = mapping.setdefault(r["token"], {"label": r["label"], "normalized": r["normalized"],
                                                "originals": collections.Counter(), "pid": None})
            m["originals"][r["original"]] += 1
            if r["label"] == "person":
                g = group_of[(r["start"], r["end"])]
                m["pid"] = g["pid"]
                p = persons.setdefault(r["token"], {"g": g, "mentions": 0})
                p["mentions"] += 1
                mentions.append((doc_id, r["start"], r["end"], r["token"], g["pid"], g["basis"], g["role"]))

        rows = []
        for i, c in enumerate(chunk(masked, MAX_CHARS)):
            body = set(pii_pipeline.TOKEN_PATTERN.findall(c))
            rows.append({"chunk_id": f"{doc_id}#{i}", "doc_id": doc_id, "chunk_idx": i, "doc_type": d["doc_type"],
                         "masked_text": c, "tokens": sorted(body | {x for t in body for x in extra[t]})})
        vecs = embed([r["masked_text"] for r in rows])  # 가명처리 + guard 된 글만 나감

        rcur.executemany(
            """INSERT INTO rag.chunks (chunk_id, doc_id, chunk_idx, doc_type, masked_text, tokens, embedding)
               VALUES (%(chunk_id)s, %(doc_id)s, %(chunk_idx)s, %(doc_type)s, %(masked_text)s, %(tokens)s, %(embedding)s)""",
            [r | {"embedding": v} for r, v in zip(rows, vecs)])
        for tok, p in persons.items():
            g = p["g"]
            rcur.execute(
                """INSERT INTO rag.persons (pid, person_token, name_token, given_token, role, affiliations, status, mentions)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (pid) DO UPDATE SET mentions = rag.persons.mentions + EXCLUDED.mentions,
                       role = COALESCE(rag.persons.role, EXCLUDED.role),
                       affiliations = ARRAY(SELECT DISTINCT unnest(rag.persons.affiliations || EXCLUDED.affiliations))""",
                (g["pid"], tok, pii_pipeline.name_token(g["name"]),
                 pii_pipeline.given_token(pii_pipeline.split_name(g["name"])[1]), g["role"], sorted(g["affs"]),
                 "resolved" if g["key"] else "unresolved", p["mentions"]))
            rcur.executemany("INSERT INTO rag.person_attrs (pid, label, token) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                             [(g["pid"], lab, t) for lab, t in sorted(g["anchors"])])
        rconn.commit()

    vcur.executemany(
        """INSERT INTO vault.token_map (token, label, normalized, originals, pid, display)
           VALUES (%s,%s,%s,%s,%s,%s)
           ON CONFLICT (token) DO UPDATE SET originals =
               ARRAY(SELECT DISTINCT unnest(vault.token_map.originals || EXCLUDED.originals)),
               pid = COALESCE(EXCLUDED.pid, vault.token_map.pid)""",
        [(t, m["label"], m["normalized"], sorted(m["originals"]), m["pid"], _display(m)) for t, m in mapping.items()])
    vcur.executemany(
        """INSERT INTO vault.person_mentions (doc_id, start_pos, end_pos, token, pid, basis, role)
           VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""", mentions)
    vcur.executemany("INSERT INTO vault.person_keys (pid, name, person_key) VALUES (%s,%s,%s) ON CONFLICT DO NOTHING",
                     [(g["pid"], g["name"], g["key"]) for g in groups])
    return {"chunks": len(rows), "links": linking.public_groups(groups)}
