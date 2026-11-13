"""06 조문: 사례별 언급 법령과 인용 조문을 뽑는다. 불확실한 조문은 법령을 비워 둔다.
입력: processed/cases_clean.jsonl   출력: processed/case_laws.jsonl, case_articles.jsonl, laws.jsonl (+csv)
"""
import json
import re
from collections import Counter, defaultdict

from common import OUT, manifest, read_jsonl, write_csv, write_jsonl
from law_rules import (ALIASES, METHOD_RANK, QUOTED, SUFFIX, Canon, build_bare_pattern, category_of, extract, is_generic,
                       kind_of, local_aliases, local_generic, split_tail, key)

SECS = ("title", "question", "answer", "reason")


def official_aliases():
    """이전 법령 수집 결과의 공식 약칭(법령약칭명)을 사전에 보탠다."""
    p = OUT / "law_ids.jsonl"
    if not p.exists():
        return 0
    n = 0
    for line in open(p, encoding="utf-8"):
        x = json.loads(line)
        ab, name = x.get("abbrev"), x["law_name"]
        if ab and len(key(ab)) >= 3 and not split_tail(ab)[1] and not is_generic(ab) and key(ab) != key(name):
            if ALIASES.setdefault(ab, name) == name:
                n += 1
    return n


def main():
    n_official = official_aliases()
    rows = read_jsonl("cases_clean.jsonl")
    spell = Counter()
    for r in rows:
        for s in SECS:
            for m in QUOTED.finditer(r[s] or ""):
                name = re.sub(r"\s+", " ", m.group(1)).strip()
                if re.search(SUFFIX + r"$", name):
                    spell[name] += 1
    canon = Canon(spell)
    names = {split_tail(s)[0] for s in spell} | set(ALIASES) | set(ALIASES.values())
    bare = build_bare_pattern(names)

    arts_out, laws_out = [], []
    for r in rows:
        text_all = "\n".join(r[s] or "" for s in SECS)
        local = local_aliases(text_all, canon.known)
        gen = local_generic(text_all, canon.known)
        pre = Counter()
        for s in SECS:
            pre.update(extract(r[s] or "", s, canon, bare, local, gen)[0])
        main_law = next((n for n, _ in pre.most_common() if kind_of(n) == "법률"), None)
        main_reg = next((n for n, _ in pre.most_common() if kind_of(n) == "행정규칙" and not n.endswith("시행세칙")), None)
        single = len(pre) == 1
        found, seen = [], Counter()
        for s in SECS:
            mentions, arts = extract(r[s] or "", s, canon, bare, local, gen, main_law, main_reg, single)
            seen.update(mentions)
            found += arts
        # 같은 조문이 여러 번 나오면 가장 확실한 해석 하나만 남긴다
        best = {}
        for a in found:
            k = (a["law_name"], a["article"], a["article_sub"], a["paragraph"], a["item"], a["item_sub"])
            if k not in best or METHOD_RANK[a["resolve_method"]] < METHOD_RANK[best[k]["resolve_method"]]:
                best[k] = a
        resolved = {(a["article"], a["article_sub"]) for a in best.values() if a["law_name"]}
        for a in best.values():
            if a["law_name"] or (a["article"], a["article_sub"]) not in resolved:
                arts_out.append({"case_key": r["case_key"], **a})
        for name, c in seen.items():
            laws_out.append({"case_key": r["case_key"], "law_name": name, "law_kind": kind_of(name), "mentions": c})

    law_cases, art_cases = defaultdict(set), defaultdict(set)
    for x in laws_out:
        law_cases[x["law_name"]].add(x["case_key"])
    for a in arts_out:
        if a["law_name"]:
            art_cases[a["law_name"]].add(a["case_key"])
            law_cases[a["law_name"]].add(a["case_key"])
    laws = [{"law_name": n, "law_kind": kind_of(n), "category": category_of(n), "n_cases": len(c),
             "n_cases_with_article": len(art_cases.get(n, ()))}
            for n, c in sorted(law_cases.items(), key=lambda kv: -len(kv[1]))]
    write_jsonl("case_laws.jsonl", laws_out)
    write_jsonl("case_articles.jsonl", arts_out)
    write_jsonl("laws.jsonl", laws)
    write_csv("case_laws.csv", laws_out, ["case_key", "law_name", "law_kind", "mentions"])
    write_csv("case_articles.csv", arts_out, ["case_key", "law_name", "law_kind", "article", "article_sub", "paragraph", "item", "item_sub",
                                               "supplementary", "section", "raw", "resolve_method", "law_guess"])
    write_csv("laws.csv", laws, ["law_name", "law_kind", "category", "n_cases", "n_cases_with_article"])
    manifest("06_articles", official_aliases=n_official, laws=len(laws), articles=len(arts_out),
             resolve_method=dict(Counter(a["resolve_method"] for a in arts_out)))


if __name__ == "__main__":
    main()
