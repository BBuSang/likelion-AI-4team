"""기획안 5-6-6 개선 ③: 위임·준용 연결 만들기 → data/work/06_links/article_links.csv

규칙 (일반 구조만, 법령·조문 이름을 코드에 쓰지 않음)
  1. 약칭 풀기: 본문의 「법명」(이하 "X"이라 한다, X가 법·영·령·규칙·규정으로 끝날 때만) / 같은 법 시행령(이하 "영"이라 한다) 정의.
     시행령·시행규칙은 정의가 없으면 "법" = 상위 법률, "영" = 같은 법률의 시행령, "규칙" = 같은 법률의 시행규칙
  2. 참조 찾기: 법 제N조 · 영 제N조 · 규칙 제N조 · 「법명」 제N조 · 같은 법 제N조 (+ 이어지는 ", 제M조")
  3. 연결 종류: 위임 = 아래 위계 조문 → 자기 법령 가족('법'·'영'·'규칙'이 가리키는 법령과 그 시행령·시행규칙) 중 위 위계 조문 / 준용 = '준용' 문장 안의 다른 법령 조문

    python pipeline/links_06.py
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "work" / "06_links"
csv.field_size_limit(10 ** 9)

RANK = {"법률": 0, "시행령": 1, "시행규칙": 2, "행정규칙": 3}
Q = "\"“”'‘’"
DEF = re.compile(r"「([^」]+)」[^「」\n]{0,40}?\(이하\s*[" + Q + r"]([^" + Q + r"]{1,15})[" + Q + r"]\s*(?:이?라|라)\s*한다\)")
SAME_DEF = re.compile(r"같은\s*법\s*(시행령|시행규칙)\s*\(이하\s*[" + Q + r"]([^" + Q + r"]{1,15})[" + Q + r"]\s*(?:이?라|라)\s*한다\)")
ART = r"제(\d+)조(?:의(\d+))?"
ART_DASH = r"제(\d+)-(\d+)조(?:의(\d+))?"
TAIL = r"(?:\s*제\d+항)?(?:\s*제\d+호)?(?:\s*[가-하]목)?(?:\s*(?:각\s*호|단서|본문|전단|후단)(?:\s*외의\s*부분)?)?"
FAMILY = {"법", "영", "규칙", "시행령", "시행규칙"}
LAWWORD = re.compile(r"(?:법|영|령|규칙|규정)$")
SENT = re.compile(r"[^.。\n]*(?:[.。]|\n|$)")


def label(m_no, m_sub=None, dash=None):
    if dash is not None:
        return f"제{m_no}-{dash}조" + (f"의{m_sub}" if m_sub else "")
    return f"제{m_no}조" + (f"의{m_sub}" if m_sub else "")


def main():
    laws = {r["uid"]: r for r in csv.DictReader(open(ROOT / "data/work/01_fetch/laws.csv", encoding="utf-8-sig"))}
    by_name = {}
    for u, r in laws.items():
        by_name[re.sub(r"\s+", "", r["law_name"])] = u
        if r.get("abbrev"):
            by_name.setdefault(re.sub(r"\s+", "", r["abbrev"]), u)
    kids = defaultdict(dict)  # 법률 uid → {시행령: uid, 시행규칙: uid}
    for u, r in laws.items():
        if r["parent_uid"] and r["level"] in ("시행령", "시행규칙"):
            kids[r["parent_uid"]].setdefault(r["level"], u)

    arts = [a for a in csv.DictReader(open(ROOT / "data/work/02_clean/articles.csv", encoding="utf-8-sig"))
            if a["searchable"] == "True"]
    key = {(a["law_uid"], a["article_label"]): a["article_uid"] for a in arts}
    by_law = defaultdict(list)
    for a in arts:
        by_law[a["law_uid"]].append(a)

    def find_law(name):
        return by_name.get(re.sub(r"\s+", "", name))

    # 1. 약칭 풀기 (법령마다)
    alias, family = {}, {}
    for lu, items in by_law.items():
        lv = laws[lu]["level"]
        al = {}
        if lv in ("시행령", "시행규칙") and laws[lu]["parent_uid"]:
            p = laws[lu]["parent_uid"]
            al["법"] = p
            if "시행령" in kids[p]:
                al["영"] = kids[p]["시행령"]
            if "시행규칙" in kids[p]:
                al["규칙"] = kids[p]["시행규칙"]
        last = None
        for a in items:
            for m in re.finditer(DEF.pattern + "|" + SAME_DEF.pattern, a["text"]):
                if m.group(1):
                    u = find_law(m.group(1))
                    # 약칭이 법령을 가리키는 말일 때만 (예: "법", "영", "○○법") — 용어 정의("금융지주회사")는 제외
                    if u and u != lu and LAWWORD.search(m.group(2).strip()):
                        al[m.group(2).strip()] = u
                        last = u
                elif m.group(3) and last:
                    base = laws[last]["parent_uid"] or last
                    u = kids[base].get(m.group(3))
                    if u:
                        al[m.group(4).strip()] = u
        alias[lu] = al
        # 위임 상대 = 자기 법령 가족: "법"·"영"·"규칙"(·"시행령"·"시행규칙")으로 가리킨 법령과 그 법률의 시행령·시행규칙
        fam = {u for w, u in al.items() if w in FAMILY}
        for u in list(fam):
            base = laws[u]["parent_uid"] or u
            fam.add(base)
            fam.update(kids[base].values())
        family[lu] = fam - {lu}

    # 2·3. 참조 찾기 → 연결
    edges, why = {}, Counter()
    for lu, items in by_law.items():
        lv = RANK[laws[lu]["level"]]
        al = alias[lu]
        names = sorted(al, key=len, reverse=True)
        ali = "|".join(re.escape(n) for n in names) if names else "(?!)"
        ref = re.compile(r"(?:「([^」]+)」|같은\s*법|(?<![가-힣」])(?<!이 )(" + ali + r"))\s*(?:" + ART_DASH + "|" + ART + ")" + TAIL)
        cont = re.compile(r"\s*(?:,|ㆍ|·|및|또는|와|과|부터)\s*(?:" + ART_DASH + "|" + ART + ")" + TAIL)
        for a in items:
            for s in SENT.findall(a["text"]):
                junyong = "준용" in s
                prev = None
                for m in ref.finditer(s):
                    if m.group(1) is not None:
                        tl = find_law(m.group(1))
                    elif m.group(2) is not None:
                        tl = al.get(m.group(2))
                    else:
                        tl = prev
                    prev = tl
                    if not tl or tl == lu:
                        continue
                    labs = [label(m.group(3), m.group(5), m.group(4)) if m.group(3) else label(m.group(6), m.group(7))]
                    pos = m.end()
                    while True:
                        c = cont.match(s, pos)
                        if not c:
                            break
                        labs.append(label(c.group(1), c.group(3), c.group(2)) if c.group(1) else label(c.group(4), c.group(5)))
                        pos = c.end()
                    related = tl in family[lu]
                    kind = "위임" if related and RANK[laws[tl]["level"]] < lv else ("준용" if junyong else None)
                    if not kind:
                        why["일반 인용(안 씀)"] += 1
                        continue
                    for lb in labs:
                        dst = key.get((tl, lb))
                        if not dst:
                            why["대상 조문 없음"] += 1
                            continue
                        e = (a["article_uid"], dst)
                        if e not in edges or edges[e] == "준용":
                            edges[e] = kind
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "article_links.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_uid", "dst_uid", "kind"])
        for (s, d), k in sorted(edges.items()):
            w.writerow([s, d, k])
    kinds = Counter(edges.values())
    deg = Counter()
    for s, d in edges:
        deg[s] += 1
        deg[d] += 1
    stats = {"연결 수": len(edges), "종류": dict(kinds), "연결된 조문 수": len(deg),
             "조문당 연결 최대": max(deg.values()) if deg else 0, "버린 참조": dict(why),
             "약칭 있는 법령": sum(1 for v in alias.values() if v), "법령 수": len(by_law)}
    json.dump({"stats": stats, "alias": {laws[k]["law_name"]: {a: laws[u]["law_name"] for a, u in v.items()} for k, v in alias.items() if v}},
              open(OUT / "links_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
