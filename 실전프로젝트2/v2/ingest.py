"""v2 적재 (plan_v2 P3, 청크 없이 문서 = 청크 1개). run_pseudo.py 결과를 rag·vault 테이블에 넣는다.

    python -X utf8 v2/ingest.py          (실전프로젝트2 폴더, v2 DB 실행 상태)

rag  (ingest_user): documents · chunks(kind=doc) · tokens · mentions · address_regions · doc_fields
vault(vault_user) : token_map · token_surface · mention_raw
entities 는 entity/resolve.py 가 채움. 임베딩에는 가명 본문만 보냄 (캐시: outputs/emb_cache.json)
매번 rag·vault 를 비우고 다시 넣음 (TRUNCATE, 테이블·권한은 그대로)
"""
import collections
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

V2 = Path(__file__).resolve().parent
sys.path.insert(0, str(V2))
import config  # noqa: E402
import db  # noqa: E402
from clients import embed  # noqa: E402
from pseudo.pipeline import KEY_VER  # noqa: E402

OUT = V2 / "outputs"
CACHE = OUT / "emb_cache.json"
PIPELINE_VER = "v2.0"
STATUS_RE = re.compile(r"(처리 완료|처리 중|접수 완료|진행 중|승인|반려|보류)")
CASE_RE = re.compile(r"\b([A-Z]{2,5}-\d{3,})\b")
DATE_RE = re.compile(r"(20\d\d)\s*[-.년]\s*(\d{1,2})\s*[-.월]\s*(\d{1,2})")


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def hk(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def embed_cached(texts):
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [t for t in dict.fromkeys(texts) if hk(t) not in cache]
    for i, t in enumerate(todo, 1):
        cache[hk(t)] = embed(t).tolist()
        if i % 20 == 0 or i == len(todo):
            CACHE.write_text(json.dumps(cache), encoding="utf-8")
            print(f"  임베딩 {i}/{len(todo)}", flush=True)
    return {t: np.asarray(cache[hk(t)], dtype=np.float32) for t in texts}


def doc_date(masked):
    m = DATE_RE.search(masked)
    return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}" if m else None


def main():
    docs = jl(OUT / "mentions_holdout.jsonl")
    raw_len = {d["id"]: len(d["text"]) for d in jl(config.DATA_DIR / "holdout_labeled.jsonl")}
    t0 = time.perf_counter()
    emb = embed_cached([d["masked"] for d in docs])
    print(f"임베딩 {len(docs)}건 ({time.perf_counter() - t0:.0f}초, 캐시 포함)")

    doc_freq = collections.Counter(t for d in docs for t in {m["token"] for m in d["mentions"]})
    token_info, surfaces, regions = {}, collections.defaultdict(set), {}
    for d in docs:
        for m in d["mentions"]:
            token_info[m["token"]] = (m["label"], m["normalized"])
            surfaces[(m["token"], m["raw"])].add(d["id"])
            if m["sido"]:
                regions[m["token"]] = (m["sido"], m["sigungu"])

    with db.ingest_conn() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE rag.doc_fields, rag.address_regions, rag.entity_links, rag.mentions, "
                    "rag.entities, rag.tokens, rag.chunks, rag.documents")
        cur.executemany("INSERT INTO rag.tokens (token, label, doc_freq, key_ver) VALUES (%s, %s, %s, %s)",
                        [(t, lab, doc_freq[t], KEY_VER) for t, (lab, _) in token_info.items()])
        cur.executemany("INSERT INTO rag.address_regions (token, sido, sigungu) VALUES (%s, %s, %s)",
                        [(t, s, g) for t, (s, g) in regions.items()])
        mid = 0
        for d in docs:
            masked = d["masked"]
            title = next((l for l in masked.split("\n") if l.strip()), "")
            chunk_id = f"{d['id']}#d"
            cur.execute("INSERT INTO rag.documents (doc_id, split, doc_type, title_masked, doc_date, masked_text, "
                        "char_len, pipeline_ver) VALUES (%s, 'holdout', %s, %s, %s, %s, %s, %s)",
                        (d["id"], d["doc_type"], title, doc_date(masked), masked, raw_len[d["id"]], PIPELINE_VER))
            cur.execute("INSERT INTO rag.chunks (chunk_id, doc_id, kind, header, masked_text, tokens, embedding) "
                        "VALUES (%s, %s, 'doc', %s, %s, %s, %s)",
                        (chunk_id, d["id"], f"[{d['doc_type']}] {title}", masked,
                         sorted({m["token"] for m in d["mentions"]}), emb[masked]))
            for m in d["mentions"]:
                mid += 1
                m["mention_id"] = mid
                cur.execute("INSERT INTO rag.mentions (mention_id, doc_id, chunk_id, token, label, m_start, m_end, "
                            "line_no, form, role, own_role, title, org_unit) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                            (mid, d["id"], chunk_id, m["token"], m["label"], m["m_start"], m["m_end"],
                             m["line_no"], m["form"], m["role"], m["own_role"], m["title"], m["org"]))
            for i, line in enumerate(masked.split("\n")):
                for field, rx in (("status", STATUS_RE), ("case_no", CASE_RE)):
                    for v in rx.findall(line):
                        cur.execute("INSERT INTO rag.doc_fields (doc_id, chunk_id, line_no, field, value) "
                                    "VALUES (%s, %s, %s, %s, %s)", (d["id"], chunk_id, i, field, v))
        conn.commit()

    with db.vault_conn() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE vault.token_surface, vault.mention_raw, vault.token_map")
        cur.executemany("INSERT INTO vault.token_map (token, label, normalized, key_ver) VALUES (%s, %s, %s, %s)",
                        [(t, lab, norm, KEY_VER) for t, (lab, norm) in token_info.items()])
        cur.executemany("INSERT INTO vault.token_surface (token, surface, doc_count) VALUES (%s, %s, %s)",
                        [(t, s, len(ds)) for (t, s), ds in surfaces.items()])
        cur.executemany("INSERT INTO vault.mention_raw (mention_id, doc_id, o_start, o_end, raw_text) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        [(m["mention_id"], d["id"], m["o_start"], m["o_end"], m["raw"])
                         for d in docs for m in d["mentions"]])
        conn.commit()

    with db.rag_conn() as conn:                       # 검색 계정으로 읽어 확인
        counts = {t: conn.execute(f"SELECT count(*) FROM rag.{t}").fetchone()[0]
                  for t in ("documents", "chunks", "tokens", "mentions", "address_regions", "doc_fields")}
    print("rag:", counts)
    print(f"vault: token_map {len(token_info)} · token_surface {len(surfaces)} · mention_raw {mid}")


if __name__ == "__main__":
    main()
