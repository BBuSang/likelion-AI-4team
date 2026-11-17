"""4-2 청크·임베딩 (2-5~2-7 섹션 → 검색 단위)

사용:
  python analysis/s10_chunks.py build     # data/chunks/chunks.jsonl, clauses.jsonl, glossary.jsonl
  python analysis/s10_chunks.py embed     # OpenAI 임베딩 (data/chunks/embed_cache.jsonl 에 내용 해시로 저장 → 다시 돌려도 새 글만 호출)
  python analysis/s10_chunks.py load      # DB chunk 적재 + common_clause·glossary_term 임베딩 채움 (s9_load_db.py 다음)
  python analysis/s10_chunks.py search "질문"   # 벡터 검색 상위 5개 (동작 확인용, OpenAI 모델일 때)
  python analysis/s10_chunks.py export-colab  # KURE 등 GPU 모델용: data/colab/embed_input.jsonl (청크·공통 조항·용어·평가 질문)
  python analysis/s10_chunks.py import-colab data/colab/embed_output.jsonl   # 코랩 결과를 캐시에 합침
  python analysis/s10_chunks.py embed-questions-local data/eval/retrieval_test2.jsonl   # 맥에서 KURE로 질문만 임베딩 (7-3)
  python analysis/s10_chunks.py embed-local      # 맥에서 KURE로 청크·공통 조항·용어 전체 (코랩 대신, 캐시에 없는 것만)

임베딩 모델은 .env의 EMBED_MODEL·EMBED_DIM을 따른다. 캐시(embed_cache.jsonl)는 (모델, 내용 해시)로 구분해서
OpenAI·KURE 결과가 한 파일에 같이 있어도 섞이지 않는다.

청크 규칙
  - 문서별 청크: 정형 문구가 아닌 섹션(2-7 is_boilerplate=false)의 2-6 정리본(text)
    · 짧은 섹션 합치기: 같은 문서·같은 큰 제목 안에서 이어지는 섹션을 MIN_CHARS 이상이 될 때까지 묶음 (MAX_CHARS를 넘지 않게)
      조문형처럼 큰 제목만 있는 섹션(제1조·제2조, 머리말)은 이웃끼리도 묶음
      — 섹션 하나는 출처 단위라 청크마다 섹션 번호 목록(section_seqs)을 남김
    · 긴 섹션 나누기: SPLIT_CHARS를 넘으면 줄 경계에서 TARGET_CHARS 안팎으로 나누고 앞 조각 마지막 줄을 겹침
  - 정형 문구·공통 조항: 문서별 청크에 넣지 않고 common_clause 한 줄 = 임베딩 하나 (문서 연결은 document_clause)
  - 용어집: glossary_term 한 줄 = 임베딩 하나
  - embed_text 머리말: "[문서 제목] 큰 제목 > 섹션 이름 (p.쪽)" + 본문. 공유 문서는 상품명이 아니라 문서 제목
  - tsv: Kiwi 형태소(명사·동사·형용사 어간·외국어·숫자)를 공백으로 이어 'simple' 설정으로 → 키워드(BM25류) 검색용
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECS, OUT = ROOT / "data" / "sections", ROOT / "data" / "chunks"
DSN = os.environ.get("DATABASE_URL", "postgresql://postgres:fss_local@127.0.0.1:5433/fss_rag")
MIN_CHARS, MAX_CHARS, SPLIT_CHARS, TARGET_CHARS = 300, 900, 1500, 900
EMBED_BATCH = 256


def is_openai(model: str) -> bool:
    return model.startswith("text-embedding")


def read_cache(model: str) -> dict:
    """embed_cache.jsonl에서 그 모델의 벡터만 {해시: 벡터}"""
    cache, path = {}, OUT / "embed_cache.jsonl"
    if path.exists():
        for l in open(path):
            r = json.loads(l)
            if r["model"] == model:
                cache[r["hash"]] = r["vec"]
    return cache


def h(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def l1(path: str) -> str:
    return path.split(" > ")[0]


def split_long(text: str):
    """줄 경계에서 TARGET_CHARS 안팎으로 나누고, 다음 조각 앞에 앞 조각 마지막 줄을 겹친다"""
    lines, pieces, cur = text.split("\n"), [], []
    for ln in lines:
        if cur and sum(len(x) + 1 for x in cur) + len(ln) > TARGET_CHARS:
            pieces.append(cur)
            cur = [cur[-1]] if len(cur[-1]) < TARGET_CHARS // 3 else []
        cur.append(ln)
    if cur:
        pieces.append(cur)
    return ["\n".join(p) for p in pieces]


def build():
    import csv
    docs = {r["doc_id"]: r for r in csv.DictReader(open(ROOT / "data" / "meta" / "documents.csv"))}
    OUT.mkdir(parents=True, exist_ok=True)
    chunks = []
    for p in sorted(SECS.glob("*.jsonl")):
        d = p.stem
        secs = [json.loads(l) for l in open(p)]
        groups, cur = [], []
        for s in secs:
            if s["is_boilerplate"]:
                continue   # 정형 문구는 건너뛰기만 함 (앞뒤 상품 섹션은 같은 큰 제목이면 계속 묶음)
            n = len(s["text"])
            # 같은 큰 제목이면 묶음. 조문형처럼 큰 제목만 있는 섹션(제1조, 제2조 …)이나 머리말은 짧으면 이웃 조문과도 묶음
            flat = lambda x: " > " not in x["section_path"]   # noqa: E731
            same = cur and (l1(cur[0]["section_path"]) == l1(s["section_path"]) or (flat(s) and all(flat(x) for x in cur)))
            size = sum(len(x["text"]) for x in cur)
            if cur and (not same or size >= MIN_CHARS or size + n > MAX_CHARS):
                groups.append(cur)
                cur = []
            cur.append(s)
        if cur:
            groups.append(cur)
        for g in groups:
            text = "\n".join(s["text"] for s in g)
            names = [s["section"] for s in g if s["section_path"] != "(머리말)"]
            path = l1(g[0]["section_path"])
            if len(g) == 1 and " > " in g[0]["section_path"]:
                path = g[0]["section_path"]
            elif all(" > " not in s["section_path"] for s in g) and len(g) > 1:
                path = " · ".join(dict.fromkeys(s["section"] for s in g))
            elif names and any(" > " in s["section_path"] for s in g):
                path = f"{path} > " + " · ".join(dict.fromkeys(n for n in names if n != path))
            pieces = split_long(text) if len(text) > SPLIT_CHARS else [text]
            for i, piece in enumerate(pieces):
                sec = path if len(pieces) == 1 else f"{path} ({i + 1}/{len(pieces)})"
                head = f"[{docs[d]['title']}] {sec} (p.{g[0]['page_start']})"
                chunks.append({"chunk_key": f"{d}-{g[0]['seq']}-{i}", "doc_id": d, "section": sec,
                               "section_seqs": [s["seq"] for s in g], "page": g[0]["page_start"], "page_end": g[-1]["page_end"],
                               "content": piece, "embed_text": f"{head}\n{piece}"})
    with open(OUT / "chunks.jsonl", "w") as f:
        for c in chunks:
            c["content_hash"] = h(c["embed_text"])
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    clauses = list(csv.DictReader(open(ROOT / "data" / "meta" / "common_clause.csv")))
    with open(OUT / "clauses.jsonl", "w") as f:
        for c in clauses:
            et = f"[공통 조항 · {c['n_docs']}개 문서] {c['title']}\n{c['content']}"
            f.write(json.dumps({"clause_id": int(c["clause_id"]), "embed_text": et, "content_hash": h(et)}, ensure_ascii=False) + "\n")
    gloss = list(csv.DictReader(open(ROOT / "data" / "meta" / "glossary.csv")))
    with open(OUT / "glossary.jsonl", "w") as f:
        for g in gloss:
            et = f"[용어] {g['term']}\n{g['definition']}"
            f.write(json.dumps({"term": g["term"], "embed_text": et, "content_hash": h(et)}, ensure_ascii=False) + "\n")
    L = sorted(len(c["content"]) for c in chunks)
    print(f"청크 {len(chunks):,}개 (문서 {len({c['doc_id'] for c in chunks})}개) · 길이 p10 {L[len(L)//10]} / 중앙 {L[len(L)//2]} / "
          f"p90 {L[len(L)*9//10]} / 최대 {L[-1]} · 100자 미만 {sum(1 for x in L if x < 100)}")
    print(f"공통 조항 {len(clauses)}개 · 용어 {len(gloss)}개")


def all_items():
    items = []
    for name in ("chunks", "clauses", "glossary"):
        items += [json.loads(l) for l in open(OUT / f"{name}.jsonl")]
    return items


def embed():
    sys.path.insert(0, str(ROOT / "analysis"))
    from s8_extract import load_env
    from openai import OpenAI
    env = load_env()
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
    model, dim = env["EMBED_MODEL"], int(env["EMBED_DIM"])
    if not is_openai(model):
        sys.exit(f"{model}은 OpenAI 모델이 아님 → export-colab으로 코랩에서 임베딩한 뒤 import-colab으로 가져오기")
    cache_path = OUT / "embed_cache.jsonl"
    cache = read_cache(model)
    todo = list({it["content_hash"]: it["embed_text"] for it in all_items() if it["content_hash"] not in cache}.items())
    print(f"임베딩할 글 {len(todo):,}개 (캐시 {len(cache):,}개)", flush=True)
    tokens = 0
    with open(cache_path, "a") as f:
        for i in range(0, len(todo), EMBED_BATCH):
            batch = todo[i:i + EMBED_BATCH]
            r = client.embeddings.create(model=model, input=[t for _, t in batch])
            tokens += r.usage.total_tokens
            for (hh, _), e in zip(batch, r.data):
                assert len(e.embedding) == dim
                f.write(json.dumps({"hash": hh, "model": model, "vec": e.embedding}) + "\n")
            print(f"  {min(i + EMBED_BATCH, len(todo)):,}/{len(todo):,}", flush=True)
    print(f"토큰 {tokens:,}")


_KIWI = None
_KIWI_LOCK = __import__("threading").Lock()


def kiwi_tokens(texts):
    """14-1: Kiwi는 한 번만 만들어 다시 씀 (호출마다 새로 만들면 0.3초씩 — 응답 시간의 코드 부분 대부분). 결과는 같음"""
    global _KIWI
    keep = ("NN", "VV", "VA", "SL", "SN", "XR")
    out = []
    with _KIWI_LOCK:   # 여러 상품 재순위를 동시에 돌릴 때 한 Kiwi를 같이 쓰지 않게
        if _KIWI is None:
            from kiwipiepy import Kiwi
            _KIWI = Kiwi()
        for toks in _KIWI.tokenize(texts):
            out.append(" ".join(t.form for t in toks if t.tag.startswith(keep)))
    return out


def load():
    import psycopg
    sys.path.insert(0, str(ROOT / "analysis"))
    from s8_extract import load_env
    env = load_env()
    cache = read_cache(env["EMBED_MODEL"])
    missing = [x for x in all_items() if x["content_hash"] not in cache]
    if missing:
        sys.exit(f"{env['EMBED_MODEL']} 임베딩이 없는 글 {len(missing)}개 — embed(OpenAI) 또는 export-colab/import-colab 먼저")
    vec = lambda x: "[" + ",".join(f"{v:.7f}" for v in cache[x["content_hash"]]) + "]"   # noqa: E731
    chunks = [json.loads(l) for l in open(OUT / "chunks.jsonl")]
    for c in chunks:   # PDF에서 딸려 온 NUL 문자는 PostgreSQL 글자 칸에 못 넣음 → DB에 넣을 때만 뺌 (해시·임베딩은 원래 글 기준)
        c["content"], c["embed_text"] = c["content"].replace("\x00", ""), c["embed_text"].replace("\x00", "")
    toks = kiwi_tokens([c["embed_text"] for c in chunks])
    with psycopg.connect(DSN) as con, con.cursor() as cur:
        cur.execute("TRUNCATE chunk RESTART IDENTITY")
        cur.executemany(
            "INSERT INTO chunk (doc_id, section, section_seqs, page, page_end, content, embed_text, embedding, tsv, content_hash)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector, to_tsvector('simple', %s), %s)",
            [(c["doc_id"], c["section"], c["section_seqs"], c["page"], c["page_end"], c["content"], c["embed_text"], vec(c), t,
              c["content_hash"]) for c, t in zip(chunks, toks)])
        for c in (json.loads(l) for l in open(OUT / "clauses.jsonl")):
            cur.execute("UPDATE common_clause SET embedding = %s::vector WHERE clause_id = %s", (vec(c), c["clause_id"]))
        for g in (json.loads(l) for l in open(OUT / "glossary.jsonl")):
            cur.execute("UPDATE glossary_term SET embedding = %s::vector WHERE term = %s", (vec(g), g["term"]))
        con.commit()
        for t, w in (("chunk", "embedding IS NOT NULL"), ("common_clause", "embedding IS NOT NULL"),
                     ("glossary_term", "embedding IS NOT NULL")):
            cur.execute(f"SELECT count(*), count(*) FILTER (WHERE {w}) FROM {t}")
            print(f"{t}: {cur.fetchone()}")


def search(q: str, k=5):
    sys.path.insert(0, str(ROOT / "analysis"))
    from s8_extract import load_env
    from openai import OpenAI
    import psycopg
    env = load_env()
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
    v = client.embeddings.create(model=env["EMBED_MODEL"], input=[q]).data[0].embedding
    vs = "[" + ",".join(map(str, v)) + "]"
    with psycopg.connect(DSN) as con, con.cursor() as cur:
        cur.execute("""
          (SELECT 'chunk' AS src, c.doc_id || ' ' || d.title AS who, c.section, 1 - (c.embedding <=> %s::vector) AS sim
             FROM chunk c JOIN document d USING (doc_id) ORDER BY c.embedding <=> %s::vector LIMIT %s)
          UNION ALL
          (SELECT 'clause', '공통 조항', title, 1 - (embedding <=> %s::vector) FROM common_clause ORDER BY embedding <=> %s::vector LIMIT 2)
          UNION ALL
          (SELECT 'term', '용어집', term, 1 - (embedding <=> %s::vector) FROM glossary_term ORDER BY embedding <=> %s::vector LIMIT 1)
          ORDER BY sim DESC""", (vs, vs, k, vs, vs, vs, vs))
        print(f"Q: {q}")
        for src, who, sec, sim in cur.fetchall():
            print(f"  {sim:.3f} [{src}] {who[:30]} | {sec[:50]}")


def export_colab():
    """코랩 입력: 청크·공통 조항·용어(embed_text) + 검색 평가 질문. 같은 글은 한 번만"""
    rows = {it["content_hash"]: {"hash": it["content_hash"], "kind": "passage", "text": it["embed_text"]} for it in all_items()}
    for sp in ("dev", "test"):
        p = ROOT / "data" / "eval" / f"retrieval_{sp}.jsonl"
        if p.exists():
            for q in (json.loads(l) for l in open(p)):
                rows[h(q["question"])] = {"hash": h(q["question"]), "kind": "query", "text": q["question"]}
    out = ROOT / "data" / "colab"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "embed_input.jsonl", "w") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_q = sum(r["kind"] == "query" for r in rows.values())
    print(f"data/colab/embed_input.jsonl — 글 {len(rows) - n_q:,}개 + 질문 {n_q}개 "
          f"({(out / 'embed_input.jsonl').stat().st_size / 1e6:.1f}MB) → 코랩 노트북에 올리기")


def local_model(model: str):
    """KURE 등 sentence-transformers 모델을 맥에서 직접 (MPS 있으면 MPS) — 7-3"""
    import torch
    from sentence_transformers import SentenceTransformer
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = SentenceTransformer(model, device=dev)
    m.max_seq_length = 2048
    return m


def embed_local(texts_by_hash: dict, model_name: str):
    """{해시: 글} 중 캐시에 없는 것만 로컬 모델로 임베딩해 embed_cache.jsonl에 추가 (정규화)"""
    cache = read_cache(model_name)
    todo = [(h_, t) for h_, t in texts_by_hash.items() if h_ not in cache]
    if not todo:
        return 0
    m = local_model(model_name)
    vecs = m.encode([t for _, t in todo], batch_size=16, normalize_embeddings=True, convert_to_numpy=True)
    with open(OUT / "embed_cache.jsonl", "a") as f:
        for (h_, _), v in zip(todo, vecs):
            f.write(json.dumps({"hash": h_, "model": model_name, "vec": [round(float(x), 6) for x in v]}) + "\n")
    return len(todo)


def import_colab(path: str):
    """코랩 결과({hash, model, vec})를 embed_cache.jsonl에 합침 — 입력과 해시·개수·차원이 맞는지 확인"""
    sys.path.insert(0, str(ROOT / "analysis"))
    from s8_extract import load_env
    env = load_env()
    want = {json.loads(l)["hash"] for l in open(ROOT / "data" / "colab" / "embed_input.jsonl")}
    rows = [json.loads(l) for l in open(path)]
    models = {r["model"] for r in rows}
    dims = {len(r["vec"]) for r in rows}
    got = {r["hash"] for r in rows}
    print(f"가져온 벡터 {len(rows):,}개, 모델 {models}, 차원 {dims}, 입력 중 빠진 것 {len(want - got)}")
    if len(models) != 1 or len(dims) != 1 or want - got:
        sys.exit("모델·차원이 하나가 아니거나 빠진 글이 있음 — 노트북을 다시 확인")
    if models != {env["EMBED_MODEL"]} or dims != {int(env["EMBED_DIM"])}:
        print(f"주의: .env는 EMBED_MODEL={env['EMBED_MODEL']}, EMBED_DIM={env['EMBED_DIM']} — 이 결과를 쓰려면 .env를 맞추기")
    old = read_cache(next(iter(models)))
    with open(OUT / "embed_cache.jsonl", "a") as f:
        for r in rows:
            if r["hash"] not in old:
                f.write(json.dumps({"hash": r["hash"], "model": r["model"], "vec": r["vec"]}) + "\n")
    print("embed_cache.jsonl에 합침")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("export-colab", "import-colab"):
        export_colab() if sys.argv[1] == "export-colab" else import_colab(sys.argv[2])
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == "embed-local":   # 청크·공통 조항·용어 전체를 로컬 모델로 (코랩 대신)
        sys.path.insert(0, str(ROOT / "analysis"))
        from s8_extract import load_env
        print("새로 임베딩한 글", embed_local({it["content_hash"]: it["embed_text"] for it in all_items()}, load_env()["EMBED_MODEL"]))
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == "embed-questions-local":   # 평가 질문을 로컬 모델로 (예: test2)
        sys.path.insert(0, str(ROOT / "analysis"))
        from s8_extract import load_env
        files = sys.argv[2:]
        qs = {h(q["question"]): q["question"] for f in files for q in map(json.loads, open(f))}
        print("새로 임베딩한 질문", embed_local(qs, load_env()["EMBED_MODEL"]))
        sys.exit()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    {"build": build, "embed": embed, "load": load}.get(cmd, lambda: search(" ".join(sys.argv[2:])))()
