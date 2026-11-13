"""D1 조문 조회셋 후보: 조문 50개를 뽑고, 추출 결과(DB)와 본문 문자열 검색을 대조해 정답 후보를 만든다.
두 방법이 모두 찾은 사례는 자동 인정하고, 한쪽만 찾은 사례는 수기 확인 대상이다. 추출 결과만 정답으로 쓰면 자기 채점이 되기 때문이다.
출력: data/eval/article_candidates.jsonl
"""
import json
import random
import re
import sys
from collections import Counter, defaultdict

import rag
from common import DATA, manifest

N, SEED = {"법률": 25, "시행령": 10, "행정규칙": 15}, 20261112
OUT_FILE = DATA / "eval" / "article_candidates.jsonl"
SECS = ("title", "question", "answer", "reason")


def flex(name):
    return r"\s*".join(map(re.escape, re.sub(r"\s+", "", name)))


def art_rx(article, sub):
    num = r"\s*".join(map(re.escape, article))
    tail = rf"\s*의\s*{sub}(?!\d)" if sub else r"(?!\s*의\s*\d)"
    return rf"(?:제\s*{num}\s*조{tail}|§\s*{num}(?:\s*조)?{tail})"


def names_of(law, kind, aliases, base_aliases):
    out = {law} | set(aliases)
    if kind in ("시행령", "시행규칙"):
        out |= {f"{a}{kind}" for a in base_aliases}
    return sorted(out, key=len, reverse=True)


def main():
    with rag.connect() as c:
        rows = c.execute("""select l.law_id, l.law_name, l.law_kind, a.article, a.article_sub, a.case_key, a.resolve_method, x.group_id from interp.case_articles a
                            join interp.laws l using (law_id) join interp.cases x using (case_key)
                            where x.indexable and not a.supplementary""").fetchall()
        aliases = defaultdict(list)
        for r in c.execute("select l.law_name, a.alias from interp.law_aliases a join interp.laws l using (law_id)").fetchall():
            aliases[r["law_name"]].append(r["alias"])
        cases = {r["case_key"]: r for r in c.execute("select case_key, group_id, title, question, answer, reason, indexable, body_suspect from interp.cases where indexable").fetchall()}
    linked = defaultdict(lambda: defaultdict(set))
    for r in rows:
        linked[(r["law_name"], r["law_kind"], r["article"], r["article_sub"])][r["resolve_method"]].add(r["case_key"])
    rng = random.Random(SEED)
    pool = defaultdict(list)
    for k, v in sorted(linked.items(), key=lambda kv: str(kv[0])):
        n = len(v["explicit"] | v["context"])
        if 2 <= n <= 40:
            pool[k[1]].append((k, n))
    picks = []
    for kind, m in N.items():
        hi = [k for k, n in pool[kind] if n >= 8]
        lo = [k for k, n in pool[kind] if n < 8]
        rng.shuffle(hi)
        rng.shuffle(lo)
        a = min(len(hi), m // 2)
        picks += hi[:a] + lo[:m - a]

    out, stats = [], Counter()
    for no, (law, kind, art, sub) in enumerate(picks, 1):
        base = re.sub(r"\s*(시행령|시행규칙)$", "", law)
        vs = names_of(law, kind, aliases[law], aliases.get(base, []) + [base])
        name = "|".join(flex(v) for v in vs)
        ar = art_rx(art, sub)
        strict = re.compile(rf"(?:{name})[\s」｣”\"'’)]*(?:\(\s*이하[^)\n]*\))?[\s」｣”\"'’)]*(?:상|의)?\s*{ar}")
        whole = re.compile(ar)
        anyname = re.compile(name)
        db = linked[(law, kind, art, sub)]
        items = {}
        for k, c in cases.items():
            text = "\n".join(c[s] or "" for s in SECS)
            m = strict.search(text)
            m2 = whole.search(text) if not m else None
            in_db = k in db["explicit"] | db["context"] | db["default"] | db["unresolved"]
            if not (m or in_db or (m2 and anyname.search(text))):
                continue
            method = next((x for x in ("explicit", "context", "default", "unresolved") if k in db[x]), None)
            hit = m or m2
            snip = text[max(0, hit.start() - 70):hit.end() + 30].replace("\n", "/") if hit else (text[:120].replace("\n", "/"))
            kind_src = "both" if m and method in ("explicit", "context") else ("db_only" if method in ("explicit", "context") else "text_only" if m else "weak")
            items[k] = {"case_key": k, "group_id": c["group_id"], "source": kind_src, "method": method, "snippet": snip}
            stats[kind_src] += 1
        out.append({"aid": f"ART{no:02d}", "law_name": law, "law_kind": kind, "article": art, "article_sub": sub, "query": f"{law} 제{art}조" + (f"의{sub}" if sub else ""), "candidates": list(items.values())})
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    manifest("d1_articles_candidates", articles=len(out), kinds=dict(Counter(x["law_kind"] for x in out)), candidate_sources=dict(stats))


def finalize():
    """수기 확인 결과를 반영해 조문 조회셋을 확정한다. 정답 = 양쪽 일치 + 추출만(수기 오답 제외) + 수기로 올린 후보."""
    review = json.loads((DATA / "eval" / "article_review.json").read_text(encoding="utf-8"))
    rows, stats = [], Counter()
    for line in OUT_FILE.open(encoding="utf-8"):
        x = json.loads(line)
        accept, reject = set(review["accept"].get(x["aid"], [])), set(review["reject"].get(x["aid"], []))
        gold = [c for c in x["candidates"] if (c["source"] in ("both", "db_only") and c["case_key"] not in reject) or c["case_key"] in accept]
        pred = {c["case_key"] for c in x["candidates"] if c["method"] in ("explicit", "context")}
        g = {c["case_key"] for c in gold}
        stats["tp"] += len(g & pred)
        stats["fp"] += len(pred - g)
        stats["fn"] += len(g - pred)
        rows.append({"aid": x["aid"], "query": x["query"], "law_name": x["law_name"], "law_kind": x["law_kind"], "article": x["article"], "article_sub": x["article_sub"],
                     "gold_case_keys": sorted(g), "gold_group_ids": sorted({c["group_id"] for c in gold}), "n_gold": len(g)})
    with (DATA / "eval" / "article_lookup_50.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    p, r = stats["tp"] / (stats["tp"] + stats["fp"]), stats["tp"] / (stats["tp"] + stats["fn"])
    manifest("d1_articles_final", articles=len(rows), gold_total=sum(x["n_gold"] for x in rows), kinds=dict(Counter(x["law_kind"] for x in rows)),
             extraction_vs_gold={**stats, "precision": round(p, 4), "recall": round(r, 4)})


if __name__ == "__main__":
    finalize() if "--finalize" in sys.argv else main()
