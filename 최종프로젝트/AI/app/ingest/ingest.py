"""적재 파이프라인: 문서 → 탐지 → (동명이인 구분) → 가명처리 → 청크 → 임베딩 → DB

사용법
    python -m app.ingest.ingest --reset                              # 탐지부터 전부
    python -m app.ingest.ingest --reset --detect-cache               # tools/detect_cache.py 결과 재사용 (모델 로딩 없음)
    python -m app.ingest.ingest --reset --detect-cache --no-resolve  # 동명이인 구분 끔 (이름 문자열 기준 토큰, 비교용)

- 입력 파일의 entities(정답 라벨)는 쓰지 않습니다. 탐지는 pii_pipeline.detect() 가 합니다.
- 전체 문서를 먼저 탐지한 뒤 person_resolve.resolve() 로 이름 언급마다 추정 인물을 정하고 나서 토큰을 만듭니다.
- 저장
    rag.chunks         마스킹된 청크, tokens(본문 토큰 + 검색 보조 토큰 NAME/GIVEN/ADDRP), 임베딩
    rag.persons        추정 인물 (사람 토큰, 이름 토큰, 역할, 소속)            ← 실명 없음
    rag.person_attrs   추정 인물별 전화·이메일 등의 토큰                       ← 실명 없음
    vault.token_map    토큰 ↔ 정규화 값·원문 표기, pid, 복원용 대표 표기
    vault.person_mentions  이름 언급별 위치·판단 근거
- outputs/pseudo_<파일명>.jsonl 에 문서별 가명처리 결과를 남깁니다 (check_token_consistency.py 입력).
"""
import argparse
import collections
import json
import time
from pathlib import Path

from app.core import config, db
from app.pii import pii_pipeline, person_resolve
from app.rag.embed import embed


def load_docs(path, limit=None):
    with open(path, encoding="utf-8") as f:
        docs = [json.loads(l) for l in f if l.strip()]
    return docs[:limit] if limit else docs


def chunk(text, max_chars=600):
    """빈 줄·줄바꿈 단위로 나누고 max_chars 안에서 이어 붙입니다. 토큰이 잘리지 않도록 줄 단위로만 자릅니다."""
    parts, cur = [], ""
    for line in text.split("\n"):
        if cur and len(cur) + len(line) + 1 > max_chars:
            parts.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur.strip():
        parts.append(cur)
    return parts or [text]


def detect_all(docs, data_path, use_cache):
    cache_path = config.OUT_DIR / f"detect_{Path(data_path).stem}.jsonl"
    if use_cache and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cache = {r["id"]: r["spans"] for r in (json.loads(l) for l in f if l.strip())}
        if all(d["id"] in cache for d in docs):
            print(f"탐지 결과 재사용: {cache_path}")
            return {d["id"]: cache[d["id"]] for d in docs}
    return {d["id"]: pii_pipeline.detect(d["text"]) for d in docs}


def pid_of(name, key):
    return "P_" + pii_pipeline._hmac(f"pid|{name}|{key or ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(config.DATA_DIR / "holdout_labeled.jsonl"))
    ap.add_argument("--reset", action="store_true", help="기존 데이터 비우고 시작")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-chars", type=int, default=600)
    ap.add_argument("--detect-cache", action="store_true", help="outputs/detect_<파일명>.jsonl 이 있으면 탐지 대신 사용")
    ap.add_argument("--no-resolve", action="store_true", help="동명이인 구분 끄기 (이름 문자열 기준 토큰)")
    a = ap.parse_args()

    if pii_pipeline.IS_DEMO and config.EMBED_PROVIDER == "api":
        print("⚠ 데모용 마스킹은 이름 등을 탐지하지 못해 원문 일부가 임베딩 API로 전송됩니다.")

    docs = load_docs(a.data, a.limit)
    config.OUT_DIR.mkdir(exist_ok=True)
    pseudo_path = config.OUT_DIR / f"pseudo_{Path(a.data).stem}.jsonl"

    # 1. 전체 탐지 → 2. 동명이인 구분 (문서를 한 번에 봐야 하므로 가명처리 전에)
    t0 = time.perf_counter()
    spans = detect_all(docs, a.data, a.detect_cache)
    t_detect = time.perf_counter() - t0
    t1 = time.perf_counter()
    if not a.no_resolve:
        spans, _ = person_resolve.resolve(docs, spans)

    # 3. 가명처리 → 청크
    rows, mapping, pseudo, collisions, mentions = [], {}, [], set(), []
    persons = {}  # person_token -> 추정 인물
    n_guard = 0
    for d in docs:
        masked, records = pii_pipeline.pseudonymize(d["text"], spans[d["id"]])
        masked, g = pii_pipeline.guard(masked)  # 탐지기가 놓친 번호·이메일 모양 (임베딩 API·DB 에 원문이 남지 않게)
        n_guard += g
        pseudo.append({"id": d["id"], "entities": [
            {k: r[k] for k in ("label", "start", "end", "token")} for r in records]})
        extra = collections.defaultdict(set)  # 본문 토큰 → 같이 넣을 검색 보조 토큰
        for r in records:
            extra[r["token"]].update(r["search_tokens"])
            m = mapping.setdefault(r["token"], {"label": r["label"], "normalized": r["normalized"],
                                                "originals": collections.Counter(), "pid": None})
            m["originals"][r["original"]] += 1
            if m["normalized"] != r["normalized"]:
                collisions.add(r["token"])
            if r["label"] == "person":
                pid = pid_of(r["normalized"], r.get("person_key"))
                m["pid"] = pid
                p = persons.setdefault(r["token"], {
                    "pid": pid, "name": r["normalized"], "roles": collections.Counter(), "affs": set(),
                    "attrs": set(), "mentions": 0,
                    "status": "resolved" if r.get("person_key") else "unresolved"})
                p["mentions"] += 1
                if r.get("role"):
                    p["roles"][r["role"]] += 1
                if r.get("aff"):
                    p["affs"].add(r["aff"])
                mentions.append((d["id"], r["start"], r["end"], r["token"], pid, r.get("basis", "off"), r.get("role")))
        for r in records:  # 닻으로 붙은 식별자 → 그 사람의 가명처리된 개인정보
            if r["label"] == "person" and r.get("person_key"):
                for lab, v in (s for s in _anchors_of(spans[d["id"]], r)):
                    persons[r["token"]]["attrs"].add((lab, pii_pipeline.make_token(lab, v)))
        for i, c in enumerate(chunk(masked, a.max_chars)):
            body = set(pii_pipeline.TOKEN_PATTERN.findall(c))
            rows.append({"chunk_id": f"{d['id']}#{i}", "doc_id": d["id"], "chunk_idx": i,
                         "doc_type": d.get("doc_type"), "masked_text": c,
                         "tokens": sorted(body | {x for t in body for x in extra[t]})})
    t_mask = time.perf_counter() - t1

    # 4. 임베딩 (마스킹된 청크만 전송)
    t2 = time.perf_counter()
    vecs = embed([r["masked_text"] for r in rows])
    t_embed = time.perf_counter() - t2

    # 5. 저장
    with db.rag_conn() as conn, conn.cursor() as cur:
        if a.reset:
            cur.execute("TRUNCATE rag.chunks, rag.person_attrs, rag.persons")
        cur.executemany(
            """INSERT INTO rag.chunks (chunk_id, doc_id, chunk_idx, doc_type, masked_text, tokens, embedding)
               VALUES (%(chunk_id)s, %(doc_id)s, %(chunk_idx)s, %(doc_type)s, %(masked_text)s, %(tokens)s, %(embedding)s)
               ON CONFLICT (chunk_id) DO UPDATE SET masked_text = EXCLUDED.masked_text,
                   tokens = EXCLUDED.tokens, embedding = EXCLUDED.embedding, doc_type = EXCLUDED.doc_type""",
            [r | {"embedding": v} for r, v in zip(rows, vecs)])
        cur.executemany(
            """INSERT INTO rag.persons (pid, person_token, name_token, given_token, role, affiliations, status, mentions)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (pid) DO UPDATE SET mentions = EXCLUDED.mentions, role = EXCLUDED.role,
                   affiliations = EXCLUDED.affiliations""",
            [(p["pid"], tok, pii_pipeline.name_token(p["name"]),
              pii_pipeline.given_token(pii_pipeline.split_name(p["name"])[1]),
              p["roles"].most_common(1)[0][0] if p["roles"] else None, sorted(p["affs"]), p["status"], p["mentions"])
             for tok, p in persons.items()])
        cur.executemany(
            "INSERT INTO rag.person_attrs (pid, label, token) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
            [(p["pid"], lab, t) for p in persons.values() for lab, t in sorted(p["attrs"])])

    with db.vault_conn() as conn, conn.cursor() as cur:
        if a.reset:
            cur.execute("TRUNCATE vault.token_map, vault.person_mentions")
        cur.executemany(
            """INSERT INTO vault.token_map (token, label, normalized, originals, pid, display)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (token) DO UPDATE SET originals =
                   ARRAY(SELECT DISTINCT unnest(vault.token_map.originals || EXCLUDED.originals)),
                   pid = EXCLUDED.pid, display = EXCLUDED.display""",
            [(t, m["label"], m["normalized"], sorted(m["originals"]), m["pid"], _display(m))
             for t, m in mapping.items()])
        cur.executemany(
            """INSERT INTO vault.person_mentions (doc_id, start_pos, end_pos, token, pid, basis, role)
               VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING""", mentions)

    with open(pseudo_path, "w", encoding="utf-8") as f:
        for p in pseudo:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    n = len(docs)
    if collisions:
        print(f"⚠ 토큰 충돌 {len(collisions)}종: 서로 다른 값이 같은 토큰을 씀 (예: {sorted(collisions)[:3]})")
    n_res = sum(p["status"] == "resolved" for p in persons.values())
    print(f"문서 {n}건 → 청크 {len(rows)}개, 토큰 {len(mapping)}종, 추정 인물 {len(persons)}명 "
          f"(확정 {n_res}, 미확정 {len(persons) - n_res}) 적재  [동명이인 구분 {'끔' if a.no_resolve else '켬'}]")
    print(f"마지막 차단(guard)으로 가린 번호·이메일 모양: {n_guard}개")
    print(f"탐지 {t_detect * 1000 / n:.1f} ms/문서{' (캐시)' if a.detect_cache else ''}, "
          f"구분+가명처리 {t_mask * 1000 / n:.1f} ms/문서, 임베딩 {t_embed * 1000 / n:.1f} ms/문서 ({config.EMBED_PROVIDER})")
    print(f"가명처리 결과: {pseudo_path}")


def _anchors_of(doc_spans, record):
    """resolve() 가 이름 구간에 붙여 둔 닻 (label, normalized)."""
    for s in doc_spans:
        if s["label"] == "person" and s["start"] == record["start"]:
            return s.get("anchors", [])
    return []


def _display(m):
    """복원할 때 보여줄 대표 표기: 사람이면 가장 많이 나온 성 포함 표기, 그 외엔 가장 많이 나온 원래 서식."""
    if m["label"] == "person":
        full = [o for o, _ in m["originals"].most_common() if pii_pipeline.split_name(o.replace(" ", ""))[0]]
        return full[0].replace(" ", "") if full else m["normalized"]
    return m["originals"].most_common(1)[0][0]


if __name__ == "__main__":
    main()
