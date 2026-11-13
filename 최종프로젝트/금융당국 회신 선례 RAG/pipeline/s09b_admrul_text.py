"""09b 행정규칙 본문: 인용 사례의 회신 당시 버전과 현행 버전의 조문을 받아 조문별 지문(hash)을 만든다.
시점 판정(s10)이 두 버전의 지문을 비교한다. 본문은 raw/law/ 캐시에만 있고 processed에는 지문만 남긴다.
조문 분리를 믿을 수 있는 규정만 조문 단위로 비교한다. 기준(GATE)은 결과를 보기 전에 정했다.
입력: processed/admrul_versions.jsonl, case_articles.jsonl, cases_clean.jsonl   출력: processed/admrul_article_hash.jsonl, admrul_text_gate.jsonl
"""
import hashlib
import re
from collections import Counter, defaultdict
from datetime import date
from statistics import median

import lawapi
from common import manifest, read_jsonl, write_jsonl
from lawapi import call

GATE = {"cited_found_min": 0.90, "adjacent_match_median_min": 0.80}
USED = ("explicit", "context")
JO = re.compile(r"\s*제\s*(\d+(?:-\d+)*)\s*조(?:\s*의\s*(\d+))?(?=\s*[\(\[<]|\s|$)")
SPLIT = re.compile(r"(?m)(?=^\s*제\s*\d+(?:-\d+)*\s*조(?:\s*의\s*\d+)?\s*\()")
# 개정 표시는 본문이 바뀐 것이 아니다
MARK = re.compile(r"<[^<>]{0,80}(?:개정|신설|이동|삭제)[^<>]*>|〈[^〉]{0,80}〉|\[[^\]]{0,80}(?:신설|개정|이동|종전|삭제)[^\]]*\]")
DOTS = re.compile(r"[\s·ㆍ・․‧]")
ymd = lambda d: (d or "").replace("-", "")


def norm(t):
    t = DOTS.sub("", MARK.sub("", t))
    return t.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("｢", "「").replace("｣", "」")


HYPHEN = re.compile(r"제\s*(\d+)-(\d+)\s*조(?:\s*의\s*(\d+))?(?=\(|\s+(?!제\s*\d|및|또는|내지|에서|부터))")
HEADING_TAIL = re.compile(r"(?:제\s*\d+\s*(?:편|장|절|관)[^.<\[]{0,40})+$")


def split_run_on(text):
    """조문 구분 없는 긴 문자열 ('…한다.<개정 …>제1-2조(정의) …')을 나눈다.
    본문의 다른 조문 인용과 섞이지 않도록 조번호가 차례로 늘어나는 것만 제목으로 인정한다."""
    cuts, last = [], (0, 0, -1)
    for m in HYPHEN.finditer(text):
        c = (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0))
        if c <= last or c[0] not in (last[0], last[0] + 1):
            continue
        if (c[0] == last[0] and c[1] > last[1] + 30) or (c[0] == last[0] + 1 and c[1] > 5):
            continue
        close = text.find(")", m.end()) if text[m.end():m.end() + 1] == "(" else -1
        if close != -1 and text[close + 1:close + 2] in tuple("에의를을은는이가과와로도만"):
            continue                    # '제4-20조(불건전 영업행위)에 따라' 같은 본문 인용
        cuts.append(m.start())
        last = c
    parts = [text[a:b] for a, b in zip(cuts, cuts[1:] + [len(text)])]
    return [HEADING_TAIL.sub("", p.rstrip()) for p in parts]


def split_articles(jo):
    if isinstance(jo, str) and "\n" not in jo.strip() and HYPHEN.search(jo):
        items = split_run_on(jo)
    else:
        items = jo if isinstance(jo, list) else SPLIT.split(jo or "")
    out = {}
    for t in items:
        t = t if isinstance(t, str) else str(t)
        m = JO.match(t)
        if m:
            out.setdefault((m.group(1), int(m.group(2)) if m.group(2) else None), norm(t))
    return out


def version_at(vers, day):
    """그 날짜에 시행 중이던 버전. 날짜 이전 버전이 없으면 None."""
    past = [v for v in vers if ymd(v["effective_date"]) and ymd(v["effective_date"]) <= day]
    return max(past, key=lambda v: (ymd(v["effective_date"]), ymd(v["issue_date"]), v["serial"])) if past else None


def main():
    today = date.today().strftime("%Y%m%d")
    vers = defaultdict(list)
    for v in read_jsonl("admrul_versions.jsonl"):
        vers[v["law_name"]].append(v)
    idx = {r["case_key"]: r for r in read_jsonl("cases_clean.jsonl") if r["indexable"]}
    need, cited = set(), defaultdict(list)
    for a in read_jsonl("case_articles.jsonl"):
        if a["case_key"] in idx and a["resolve_method"] in USED and a["law_name"] in vers and not a["supplementary"]:
            v = version_at(vers[a["law_name"]], ymd(idx[a["case_key"]]["reply_date"]))
            if v:
                need.add((a["law_name"], v["serial"], v["effective_date"]))
                cited[a["law_name"]].append((v["serial"], (a["article"], a["article_sub"])))
    for name in cited:
        cur = version_at(vers[name], today)
        if cur:
            need.add((name, cur["serial"], cur["effective_date"]))

    rows, parsed = [], defaultdict(dict)
    for i, (name, serial, eff) in enumerate(sorted(need), 1):
        body = call("lawService", {"target": "admrul", "ID": serial})["response"].get("AdmRulService", {})
        arts = split_articles(body.get("조문내용"))
        parsed[name][(eff, serial)] = {k: hashlib.sha1(t.encode()).hexdigest()[:16] for k, t in arts.items()}
        for (art, sub), t in arts.items():
            rows.append({"law_name": name, "serial": serial, "effective_date": eff, "article": art, "article_sub": sub,
                         "hash": hashlib.sha1(t.encode()).hexdigest()[:16], "n_chars": len(t)})
        if i % 25 == 0:
            print(f"[본문] {i}/{len(need)}", flush=True)
    write_jsonl("admrul_article_hash.jsonl", rows)

    gate = []
    for name, by_ver in parsed.items():
        seq = [by_ver[k] for k in sorted(by_ver)]
        by_serial = {s: h for (_, s), h in by_ver.items()}
        found = sum(k in by_serial[s] for s, k in cited[name]) / len(cited[name])
        sims = [sum(a[k] == b.get(k) for k in a) / len(a) for a, b in zip(seq, seq[1:]) if a]
        match = median(sims) if sims else 1.0
        ok = all(seq) and found >= GATE["cited_found_min"] and match >= GATE["adjacent_match_median_min"]
        gate.append({"law_name": name, "pass": ok, "versions": len(seq), "cited": len(cited[name]), "cited_found": round(found, 3),
                     "adjacent_match_median": round(match, 3), "empty_versions": sum(not h for h in seq)})
    write_jsonl("admrul_text_gate.jsonl", gate)
    manifest("09b_admrul_text", versions=len(need), regs=len(gate), passed=sum(g["pass"] for g in gate),
             failed=[g["law_name"] for g in gate if not g["pass"]], articles=len(rows), requests=lawapi.n_requests)


if __name__ == "__main__":
    main()
