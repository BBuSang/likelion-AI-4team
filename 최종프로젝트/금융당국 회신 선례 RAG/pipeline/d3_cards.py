"""D3 검색 단위 후보: 질의 카드(제목 + 질의요지) + 판단 본문(회답 + 이유).
- 질의 카드는 베이스라인의 question 청크와 글이 같다 (머리말에 제목이 이미 들어 있다). 임베딩을 그대로 쓴다
- 판단 본문은 회답과 이유를 이어 붙인 뒤 s12와 같은 규칙(900자 초과 시 항목 경계에서 600자 근처까지 묶기)으로 자른다
- 2016 규제 정비 건의 rule_card는 그대로 둔다
- 개인정보 형식이 발견되면 저장하지 않고 멈춘다
입력: processed/cases_enriched.jsonl, case_group_map.jsonl, chunks.jsonl, embedding_cache.sqlite
출력: processed/chunks_d3.jsonl, embeddings_d3.npy (같은 순서). 베이스라인 파일과 DB는 건드리지 않는다
"""
import sqlite3
import time
from collections import Counter

import numpy as np
from kiwipiepy import Kiwi

from common import OUT, manifest, read_jsonl, write_jsonl
from s12_chunks import KEEP, PII, SECTIONS, chunk_text, header, no_info
from s13_embeddings import BATCH, DIM, MODEL, embed, text_hash


def build():
    base = read_jsonl("chunks.jsonl")
    rows = [r for r in read_jsonl("cases_enriched.jsonl") if r["indexable"]]
    maps = {m["case_key"]: m for m in read_jsonl("case_group_map.jsonl")}
    cases = {r["case_key"]: r for r in rows}
    out = []
    for c in base:                                     # 질의 카드·rule_card는 베이스라인에서 가져온다 (글 동일)
        if c["chunk_type"] == "question":
            out.append({**c, "chunk_type": "question_card"})
        elif c["chunk_type"] == "rule_card":
            out.append(dict(c))
    in_base = {c["case_key"] for c in base}
    n_judg_cases, dropped = 0, Counter()
    for r in rows:
        m = maps[r["case_key"]]
        if m["body_suspect"] or r["doc_class"] == "rule_review_2016" or r["case_key"] not in in_base:
            continue
        parts = []
        for s in ("answer", "reason"):
            t = (r[s] or "").strip()
            if not t:
                continue
            if no_info(t):
                dropped[s] += 1
                continue
            parts.append(t)
        if not parts:
            continue
        n_judg_cases += 1
        for no, t in enumerate(chunk_text("\n".join(parts)), 1):
            out.append({"case_key": r["case_key"], "group_id": m["group_id"], "chunk_type": "judgment", "chunk_no": no, "text": t,
                        "embed_text": header(r) + "\n" + t, "doc_class": r["doc_class"], "gubun": r["gubun"],
                        "category_final": r["category_final"], "reply_date": r["reply_date"]})
    found = Counter(n for c in out for n, rx in PII.items() if rx.search(c["text"]))
    if found:
        raise SystemExit(f"개인정보 형식 발견, 저장 중단: {dict(found)}")
    todo = [c for c in out if "bm25_tokens" not in c]      # 판단 본문의 형태소 색인 (s12와 같은 방식: 제목 + 본문의 명사·용언·숫자·영문)
    kiwi = Kiwi()
    for c, toks in zip(todo, kiwi.tokenize((cases[c["case_key"]]["title"] or "") + "\n" + c["text"] for c in todo)):
        c["bm25_tokens"] = " ".join(t.form.lower() for t in toks if t.tag.split("-")[0] in KEEP)
    return out, n_judg_cases, dropped


def main():
    chunks, n_judg_cases, dropped = build()
    hashes = [text_hash(c["embed_text"]) for c in chunks]
    db = sqlite3.connect(OUT / "embedding_cache.sqlite")
    have = {h for (h,) in db.execute("select hash from cache")}
    todo, seen = [], set()
    for h, c in zip(hashes, chunks):
        if h not in have and h not in seen:
            seen.add(h)
            todo.append((h, c["embed_text"]))
    print(f"청크 {len(chunks)}, 고유 글 {len(set(hashes))}, 캐시 있음 {len(set(hashes)) - len(todo)}, 호출 대상 {len(todo)}", flush=True)
    tokens, started = 0, time.time()
    for i in range(0, len(todo), BATCH):
        part = todo[i:i + BATCH]
        vecs, used = embed([t for _, t in part])
        tokens += used
        db.executemany("insert or replace into cache values (?,?,?,?)", [(h, MODEL, DIM, np.asarray(v, dtype=np.float32).tobytes()) for (h, _), v in zip(part, vecs)])
        db.commit()
    got = {h: b for h, b in db.execute("select hash, vec from cache where model = ?", (MODEL,))}
    missing = [h for h in hashes if h not in got]
    if missing:
        raise SystemExit(f"캐시에 없는 글 {len(missing)}개")
    arr = np.vstack([np.frombuffer(got[h], dtype=np.float32) for h in hashes])
    write_jsonl("chunks_d3.jsonl", chunks)
    np.save(OUT / "embeddings_d3.npy", arr)
    ls = sorted(len(c["text"]) for c in chunks if c["chunk_type"] == "judgment")
    manifest("d3_cards", chunks=len(chunks), types=dict(Counter(c["chunk_type"] for c in chunks)), judgment_cases=n_judg_cases, dropped_no_info=dict(dropped),
             called=len(todo), tokens=tokens, shape=list(arr.shape), judgment_len={"median": ls[len(ls) // 2], "p90": ls[int(len(ls) * .9)], "max": ls[-1]},
             over_max=sum(x > 900 for x in ls), seconds=round(time.time() - started))


if __name__ == "__main__":
    main()
