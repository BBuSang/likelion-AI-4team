"""E1 ② 조문 → 사례 조회: 질문 글에서 법령·조문을 읽어 `case_articles`를 직접 조회한다 (검색·임베딩·LLM을 쓰지 않는다).
- 법령 이름은 정식 명칭, 약칭(`law_aliases`), 시행령·시행규칙 약칭(`금소법 시행령`)을 모두 인식한다. 띄어쓰기·「」는 무시한다
- 조문 표기: `제27조의2`, `27조의2`, `제 27 조의 2`, `§27`, `제4-20조`, 뒤에 `제1항`·`제4호`
- 법령 이름이 없으면 조회하지 않고 후보 법령을 돌려준다 (틀리게 붙이느니 비워 둔다)
- `explicit`·`context`로 연결된 사례가 본 목록, `default`만 있는 사례는 '참고'. `unresolved`·부칙·검색 제외 사례는 쓰지 않는다
- 같은 묶음(group_id)의 사례는 한 줄로 합치고 `n_same`으로 알린다
- ③ 시점 표시: 각 사례가 **그 조문**을 인용한 부분의 시점 판정(⚠️ ✅ 🔶 🕒 ❔)을 `time_mark`로 붙인다. 묶음의 대표 사례 기준이고 `group_marks`에 묶음 안 표시 수를 둔다
사용: python articles_lookup.py "은행법 제27조의2 해석 사례"     (DB가 떠 있어야 함. 키·접속 값은 출력하지 않는다)
"""
import re
import sys
from collections import defaultdict

from psycopg.rows import dict_row

from common import ROOT, pg_connect
from time_marks import ORDER, load_marks

ART = re.compile(r"(?:제\s*(\d+(?:\s*-\s*\d+)?)\s*조|(?<![\d-])(\d+(?:-\d+)?)\s*조(?!\s*원)|§\s*(\d+(?:-\d+)?))"
                 r"(?:\s*의\s*(\d+))?(?:\s*제\s*(\d+)\s*항)?(?:\s*제\s*(\d+)\s*호)?")
STRIP = re.compile(r"[\s「」『』\"'“”‘’()]")
KIND_SUFFIX = re.compile(r"\s*(시행령|시행규칙)$")
RANK = {"explicit": 0, "context": 1, "default": 2}
SHOW_MAIN = ("explicit", "context")


def norm(s):
    return STRIP.sub("", s)


class Lookup:
    def __init__(self, conn):
        self.conn = conn
        self.laws = {r["law_id"]: r for r in conn.execute("select law_id, law_name, law_kind from interp.laws").fetchall()}
        aliases = defaultdict(list)
        for r in conn.execute("select law_id, alias from interp.law_aliases").fetchall():
            aliases[r["law_id"]].append(r["alias"])
        by_name = {r["law_name"]: r["law_id"] for r in self.laws.values()}
        names = defaultdict(set)            # 정규화한 이름 → 법령 ID들
        for lid, law in self.laws.items():
            names[norm(law["law_name"])].add(lid)
            for a in aliases[lid]:
                names[norm(a)].add(lid)
            m = KIND_SUFFIX.search(law["law_name"])
            if m:                           # 시행령·시행규칙: 모법의 약칭 + 종류
                base = by_name.get(KIND_SUFFIX.sub("", law["law_name"]))
                for a in aliases.get(base, []):
                    names[norm(a + m.group(1))].add(lid)
        self.names = {k: v for k, v in names.items() if len(k) >= 2}
        self.ambiguous = {k: v for k, v in self.names.items() if len(v) > 1}

    def find_law(self, segment):
        """글 조각에서 마지막에 나온(같으면 가장 긴) 법령 이름을 찾는다. 이름이 여러 법령이면 모두 돌려준다."""
        seg = norm(segment)
        best = None
        for name, ids in self.names.items():
            p = seg.rfind(name)
            if p >= 0 and (best is None or (p + len(name), len(name)) > (best[0], best[1])):
                best = (p + len(name), len(name), name, ids)
        return (best[2], sorted(best[3])) if best else (None, [])

    def parse(self, text):
        """질문에서 (법령, 조문) 언급을 뽑는다. 법령 이름이 없으면 앞 언급의 법령을 잇고, 그래도 없으면 비워 둔다."""
        out, prev_end, prev_law = [], 0, (None, [])
        for m in ART.finditer(text):
            law = self.find_law(text[prev_end:m.start()])
            if not law[0]:
                law = prev_law
            prev_end, prev_law = m.end(), law
            art = re.sub(r"\s+", "", m.group(1) or m.group(2) or m.group(3))
            out.append({"law_name_text": law[0], "law_ids": law[1], "article": art, "sub": int(m.group(4)) if m.group(4) else None,
                        "paragraph": int(m.group(5)) if m.group(5) else None, "item": int(m.group(6)) if m.group(6) else None, "span": m.group(0)})
        return out

    def candidate_laws(self, article, sub, limit=5):
        """법령 이름이 없을 때 그 조문이 인용된 법령을 사례 수 순으로 알려 준다."""
        rows = self.conn.execute("""select a.law_id, count(distinct a.case_key) n from interp.case_articles a join interp.cases c using (case_key)
            where a.article = %s and a.article_sub is not distinct from %s and a.law_id is not null and not a.supplementary and c.indexable
              and a.resolve_method in ('explicit', 'context') group by 1 order by 2 desc limit %s""", (article, sub, limit)).fetchall()
        return [{"law_id": r["law_id"], "law_name": self.laws[r["law_id"]]["law_name"], "n_cases": r["n"]} for r in rows]

    def cases_of(self, law_id, article, sub, paragraph=None, order="desc"):
        rows = self.conn.execute("""select a.case_key, a.paragraph, a.item, a.resolve_method, a.raw, c.group_id, c.title, c.gubun, c.reply_date, c.answer_type, c.doc_class, c.url
            from interp.case_articles a join interp.cases c using (case_key)
            where a.law_id = %s and a.article = %s and a.article_sub is not distinct from %s and not a.supplementary and c.indexable
              and a.resolve_method in ('explicit', 'context', 'default')""", (law_id, article, sub)).fetchall()
        by_case = {}
        for r in rows:
            c = by_case.setdefault(r["case_key"], {**{k: r[k] for k in ("case_key", "group_id", "title", "gubun", "reply_date", "answer_type", "doc_class", "url")},
                                                  "methods": set(), "raws": [], "paragraphs": set()})
            c["methods"].add(r["resolve_method"])
            if r["raw"] not in c["raws"]:
                c["raws"].append(r["raw"])
            if r["paragraph"] is not None:
                c["paragraphs"].add(r["paragraph"])
        cases = list(by_case.values())
        for c in cases:
            c["method"] = min(c["methods"], key=RANK.get)
            c["reference"] = c["method"] not in SHOW_MAIN                     # default로만 연결된 사례는 참고
            c["paragraph_match"] = bool(paragraph and paragraph in c["paragraphs"])
        date_key = lambda c: c["reply_date"].toordinal() if c["reply_date"] else 0
        cases.sort(key=lambda c: (c["reference"], not c["paragraph_match"] if paragraph else False, -date_key(c) if order == "desc" else date_key(c)))
        return cases

    def merge_groups(self, cases):
        """같은 묶음의 사례는 한 줄로 합친다 (대표는 목록에서 먼저 나온 것 = 정렬 기준 첫 사례). case_keys는 본 목록, ref_keys는 참고 사례."""
        groups, out = {}, []
        for c in cases:
            g = groups.get(c["group_id"])
            if g is None:
                g = groups[c["group_id"]] = {**c, "case_keys": [], "ref_keys": [], "n_same": 0, "group_marks": {}}
                out.append(g)
            else:
                g["n_same"] += 1
            g["group_marks"][c.get("time_mark")] = g["group_marks"].get(c.get("time_mark"), 0) + 1
            g["ref_keys" if c["reference"] else "case_keys"].append(c["case_key"])
        return out

    def lookup(self, text, order=None):
        """질문 → {status, parsed, results}. status: ok / need_law(법령 이름이 없거나 못 찾음, 후보를 알려 줌) / no_article / no_cases."""
        order = order or ("asc" if re.search(r"오래된|옛날|과거부터|처음부터", text) else "desc")
        mentions = self.parse(text)
        if not mentions:
            return {"status": "no_article", "parsed": [], "results": [], "message": "조문 번호(예: 제27조의2)를 찾지 못했습니다."}
        results, parsed, statuses = [], [], []
        for m in mentions:
            info = {k: m[k] for k in ("law_name_text", "article", "sub", "paragraph", "item", "span")}
            if not m["law_ids"]:
                info.update(status="need_law", candidates=self.candidate_laws(m["article"], m["sub"]))
            elif len(m["law_ids"]) > 1:
                info.update(status="need_law", candidates=[{"law_id": i, "law_name": self.laws[i]["law_name"]} for i in m["law_ids"]])
            else:
                lid = m["law_ids"][0]
                cases = self.cases_of(lid, m["article"], m["sub"], m["paragraph"], order)
                marks = load_marks(self.conn, [c["case_key"] for c in cases], article=(lid, m["article"], m["sub"])) if cases else {}
                for c in cases:
                    mk = marks[c["case_key"]]
                    c["time_mark"], c["time_label"], c["time_citations"] = mk["mark"], mk["label"], mk["citations"]
                info.update(law_id=lid, law_name=self.laws[lid]["law_name"], n_cases=len(cases), status="ok" if cases else "no_cases")
                results += [{**g, "mention": m["span"], "law_name": self.laws[lid]["law_name"]} for g in self.merge_groups(cases)]
            parsed.append(info)
            statuses.append(info["status"])
        status = "ok" if "ok" in statuses else ("need_law" if "need_law" in statuses else "no_cases")
        return {"status": status, "parsed": parsed, "results": results}


def connect():
    from dotenv import load_dotenv
    import os
    load_dotenv(ROOT / ".env", override=True)
    return pg_connect(os.environ["RAG_DSN"], row_factory=dict_row)


def show(res):
    print("상태:", res["status"])
    for p in res["parsed"]:
        extra = f" 후보 {[c['law_name'] for c in p.get('candidates', [])]}" if p.get("candidates") else ""
        print(f"  언급 '{p['span']}' → {p.get('law_name') or p['law_name_text'] or '법령 없음'} {p['status']} {p.get('n_cases', '')}{extra}")
    main = [r for r in res["results"] if not r["reference"]]
    for r in main[:15]:
        print(f"  {r.get('time_mark') or '  '} [{r['case_key']}] {r['reply_date']} {r['gubun']} {r['answer_type'] or '-'} | {r['title'][:50]} ({r['method']}{', 같은 취지 ' + str(r['n_same']) + '건' if r['n_same'] else ''})")
    if len(main) > 15:
        print(f"  … 외 {len(main) - 15}건")
    if main:
        from collections import Counter
        c = Counter(r.get("time_mark") for r in main)
        print("  시점 표시 (묶음 대표 기준):", " · ".join(f"{k} {c[k]}" for k in sorted(c, key=lambda x: ORDER.get(x, 9))), "— ⚠️는 현재 적용 여부 확인 필요")
    ref = [r for r in res["results"] if r["reference"]]
    if ref:
        print(f"  참고(법령 추정 연결) {len(ref)}건")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    with connect() as conn:
        lk = Lookup(conn)
        for q in sys.argv[1:]:
            print(f"\n질문: {q}")
            show(lk.lookup(q))
