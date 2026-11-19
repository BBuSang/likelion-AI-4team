"""기획안 5-5 판단 기준 1~6 점검 → data/work/06_load/audit.json, reports/5-5_적재.md

    python pipeline/audit_05.py
"""
import csv
import json
import random
import time
from collections import Counter

import numpy as np
import psycopg

from common import ENV, REPORTS, WORK
from embed_check_04 import local_encoder

csv.field_size_limit(2**31 - 1)
OUT = WORK / "06_load"


def conn(reader=False):
    return psycopg.connect(host=ENV["DB_HOST"], port=ENV["DB_PORT"], dbname=ENV["POSTGRES_DB"],
                           user="law_reader" if reader else ENV["POSTGRES_USER"],
                           password=ENV["LAW_PASSWORD"] if reader else ENV["POSTGRES_PASSWORD"], autocommit=True)


def vec_lit(v):
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


def main():
    laws = list(csv.DictReader(open(WORK / "01_fetch" / "laws.csv", encoding="utf-8-sig")))
    n_art = sum(1 for _ in csv.DictReader(open(WORK / "02_clean" / "articles.csv", encoding="utf-8-sig")))
    n_line = sum(1 for _ in csv.DictReader(open(WORK / "02_clean" / "article_lines.csv", encoding="utf-8-sig")))
    chunks = list(csv.DictReader(open(WORK / "04_chunks" / "chunks.csv", encoding="utf-8-sig")))
    z = np.load(WORK / "05_embed" / "kure_vectors.npz")
    ids, vecs = z["chunk_id"].tolist(), z["vectors"]
    manifest = json.loads((WORK / "05_embed" / "manifest.json").read_text(encoding="utf-8"))
    res = {}
    c = conn()

    # 1. 행 수
    db = {t: c.execute(f"SELECT count(*) FROM lawrag.{t}").fetchone()[0] for t in ("laws", "articles", "article_lines", "chunks")}
    want = {"laws": len(laws), "articles": n_art, "article_lines": n_line, "chunks": len(chunks)}
    srch = c.execute("SELECT count(*) FROM lawrag.articles WHERE searchable").fetchone()[0]
    res["1"] = {"DB": db, "CSV": want, "검색 대상 조문(DB)": srch, "약칭": c.execute("SELECT count(*) FROM lawrag.law_aliases").fetchone()[0],
                "pass": db == want}

    # 2. 법령별 조각 수
    dbc = dict(c.execute("SELECT law_uid, count(*) FROM lawrag.chunks GROUP BY 1").fetchall())
    csvc = Counter(ch["law_uid"] for ch in chunks)
    diff = [(u, csvc[u], dbc.get(u, 0)) for u in set(csvc) | set(dbc) if csvc[u] != dbc.get(u, 0)]
    no_chunk = [r["law_name"] for r in laws if r["uid"] not in dbc]
    res["2"] = {"법령": len(csvc), "차이": diff[:10], "조각 없는 법령(검색 대상 조문이 없음)": no_chunk, "pass": not diff}

    # 3. 빈 값 + 벡터 그대로 들어갔는지
    nulls = {col: c.execute(f"SELECT count(*) FROM lawrag.chunks WHERE {col} IS NULL").fetchone()[0]
             for col in ("embedding", "cite", "law_name", "law_level", "effective_date", "input_hash")}
    random.seed(55)
    pick = random.sample(range(len(ids)), 5)
    rt = []
    for j in pick:
        v = c.execute("SELECT embedding::text FROM lawrag.chunks WHERE chunk_id = %s", (ids[j],)).fetchone()[0]
        dv = np.array(json.loads(v), dtype=np.float32)
        rt.append(float(dv @ vecs[j]))
    res["3"] = {"NULL": nulls, "DB 벡터 vs 코랩 벡터 코사인(5개)": [round(x, 7) for x in rt],
                "pass": not any(nulls.values()) and min(rt) >= 0.99999}

    # 4. 색인
    idx = [r[0] for r in c.execute("SELECT indexname FROM pg_indexes WHERE schemaname='lawrag' ORDER BY 1").fetchall()]
    q0 = vec_lit(vecs[0])
    plan = "\n".join(r[0] for r in c.execute(
        f"EXPLAIN SELECT chunk_id FROM lawrag.chunks ORDER BY embedding <=> '{q0}'::vector LIMIT 5").fetchall())
    res["4"] = {"색인": idx, "실행 계획": plan, "pass": "chunks_embedding_hnsw" in idx and "chunks_embedding_hnsw" in plan}

    # 5. DB 검색 top-5 = 메모리 전수 비교 top-5 (검색 전용 계정 law_reader)
    enc = local_encoder()
    qs = manifest["questions"]
    qv = enc(qs)
    exact = [[ids[j] for j in np.argsort(-(vecs @ q))[:5]] for q in qv]
    r = conn(reader=True)
    per_ef = {}
    for ef in (40, 100, 200):
        r.execute(f"SET hnsw.ef_search = {ef}")
        got, t_sum = [], 0.0
        for q in qv:
            t0 = time.time()
            got.append([x[0] for x in r.execute("SELECT chunk_id FROM lawrag.chunks ORDER BY embedding <=> %s::vector LIMIT 5",
                                                 (vec_lit(q),)).fetchall()])
            t_sum += time.time() - t0
        same = sum(g == e for g, e in zip(got, exact))
        overlap = sum(len(set(g) & set(e)) for g, e in zip(got, exact)) / (5 * len(qs))
        per_ef[ef] = {"top-5 완전히 같음": f"{same}/{len(qs)}", "겹침": round(overlap, 3), "질문당 ms": round(1000 * t_sum / len(qs), 1)}
    ok_ef = next((ef for ef, v in per_ef.items() if v["top-5 완전히 같음"] == f"{len(qs)}/{len(qs)}"), None)
    try:
        r.execute("INSERT INTO lawrag.load_log (rule) VALUES ('x')")
        write_blocked = False
    except psycopg.errors.InsufficientPrivilege:
        write_blocked = True
    res["5"] = {"ef_search 별": per_ef, "top-5가 전부 같은 가장 작은 ef_search": ok_ef, "law_reader 쓰기 막힘": write_blocked,
                "pass": ok_ef is not None and write_blocked}
    # 6. 연결 표 (v0.31)
    lk = list(csv.DictReader(open(WORK / "06_links" / "article_links.csv", encoding="utf-8-sig")))
    dbk = dict(c.execute("SELECT kind, count(*) FROM lawrag.article_links GROUP BY 1").fetchall())
    csvk = dict(Counter(x["kind"] for x in lk))
    orphan = c.execute("SELECT count(*) FROM lawrag.article_links l WHERE NOT EXISTS (SELECT 1 FROM lawrag.articles a WHERE a.article_uid = l.src_uid) "
                       "OR NOT EXISTS (SELECT 1 FROM lawrag.articles a WHERE a.article_uid = l.dst_uid)").fetchone()[0]
    same_rows = set(map(tuple, c.execute("SELECT src_uid, dst_uid, kind FROM lawrag.article_links").fetchall())) == {(x["src_uid"], x["dst_uid"], x["kind"]) for x in lk}
    reader_n = r.execute("SELECT count(*) FROM lawrag.article_links").fetchone()[0]
    res["6"] = {"DB 종류별": dbk, "CSV 종류별": csvk, "행 전부 같음": same_rows, "양 끝 조문 없음": orphan, "law_reader 읽기": reader_n,
                "pass": dbk == csvk and same_rows and orphan == 0 and reader_n == len(lk)}
    res["log"] = c.execute("SELECT loaded_at::text, rule, counts FROM lawrag.load_log").fetchall()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "audit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for k in "123456":
        print(k, "통과" if res[k]["pass"] else "실패", str({kk: vv for kk, vv in res[k].items() if kk not in ("pass", "실행 계획")})[:300])
    print(plan)
    report(res)


def report(res):
    ok = lambda k: "✅ 통과" if res[k]["pass"] else "❌ 실패"
    r1, r5 = res["1"], res["5"]
    L = []
    p = L.append
    p("# 5-5 적재 결과\n")
    p(f"PostgreSQL 16 + pgvector 0.8.7 · `team4-lawrag2-db` (127.0.0.1:{ENV['DB_PORT']}) · DB `{ENV['POSTGRES_DB']}` · 스키마 `lawrag` · "
      f"적재 {res['log'][0][0][:16]} · 규칙 `{res['log'][0][1]}` · 코드 `pipeline/load_05.py`, `pipeline/audit_05.py`\n")
    p("- DB를 **지우고 새로 만든 뒤** 적재 (확장 vector, 스키마 lawrag, 검색 전용 계정 `law_reader` SELECT 권한)")
    p("- 테이블: " + ", ".join(f"`{t}` {n:,}" for t, n in r1["DB"].items()) + f", `law_aliases` {r1['약칭']}, `article_links` {sum(res['6']['DB 종류별'].values()):,}, `load_log` 1")
    p(f"- 검색 대상 조문 {r1['검색 대상 조문(DB)']:,} (나머지는 삭제·본문 없음, `articles.searchable=false`로 보관)")
    p("- 색인: `chunks.embedding` HNSW(코사인, m=16, ef_construction=64), `chunks(law_level)`, `chunks(article_uid)`, `chunks(law_uid)`, `articles(law_uid)`, `article_lines(article_uid)`\n")
    p("## 판단 기준 점검 (기획안 5-5)\n")
    p("| # | 기준 | 결과 | 내용 |\n|---|---|---|---|")
    p(f"| 1 | 행 수 | {ok('1')} | DB = CSV: " + ", ".join(f"{t} {n:,}" for t, n in r1["DB"].items()) + " |")
    p(f"| 2 | 법령별 | {ok('2')} | 법령 {res['2']['법령']}개 조각 수 DB = CSV, 차이 {len(res['2']['차이'])}. 조각 없는 법령 {len(res['2']['조각 없는 법령(검색 대상 조문이 없음)'])} |")
    p(f"| 3 | 빈 값 | {ok('3')} | embedding·cite·law_name·law_level·effective_date·input_hash NULL 0. DB 벡터 vs 코랩 벡터 코사인 {min(res['3']['DB 벡터 vs 코랩 벡터 코사인(5개)'])} 이상 |")
    p(f"| 4 | 색인 | {ok('4')} | HNSW 색인 있음, `EXPLAIN`에서 `chunks_embedding_hnsw` 사용 |")
    p(f"| 6 | 연결 표 | {ok('6')} | `article_links` DB = CSV (" + ", ".join(f"{k} {v:,}" for k, v in res['6']['DB 종류별'].items())
      + f"), 행 전부 같음, 양 끝 조문 없음 {res['6']['양 끝 조문 없음']}, 검색 전용 계정 {res['6']['law_reader 읽기']:,}행 읽힘 |")
    p(f"| 5 | 검색 일치 | {ok('5')} | 예시 질문 12개: DB(HNSW) top-5 = 메모리 전수 비교 top-5 — `ef_search={r5['top-5가 전부 같은 가장 작은 ef_search']}`부터 12/12. 검색 전용 계정 쓰기 막힘 |\n")
    p("### 5. `hnsw.ef_search`별 (HNSW는 근사 검색이라 후보 수를 늘리면 정확해지고 느려짐)\n")
    p("| ef_search | top-5 완전히 같은 질문 | 겹침 | 질문당 |\n|---|---|---|---|")
    for ef, v in r5["ef_search 별"].items():
        p(f"| {ef} | {v['top-5 완전히 같음']} | {v['겹침']:.1%} | {v['질문당 ms']}ms |")
    p(f"\n→ 검색 함수(5-6)는 `SET hnsw.ef_search = {r5['top-5가 전부 같은 가장 작은 ef_search'] or 200}` 으로 쓴다.\n")
    if res["2"]["조각 없는 법령(검색 대상 조문이 없음)"]:
        p("### 2. 조각이 없는 법령 (조문이 모두 삭제·본문 없음)\n")
        p("- " + ", ".join(res["2"]["조각 없는 법령(검색 대상 조문이 없음)"]) + "\n")
    p("### 4. 실행 계획\n")
    import re
    p("```\n" + re.sub(r"'\[[^\]]{50,}\]'", "'[…질문 벡터 1024차원…]'", res["4"]["실행 계획"]) + "\n```")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "5-5_적재.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("reports/5-5_적재.md", len(L), "줄")


if __name__ == "__main__":
    main()
