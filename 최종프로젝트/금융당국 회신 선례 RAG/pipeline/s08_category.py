"""08 분야: 사례에 언급된 법령으로 분야를 추정하고 검색·표시에 쓸 최종 분야를 정한다. 포털 값은 그대로 둔다.
추정 순서: ① 업권 법령 언급이 다른 법령 언급 이상이면 그 업권 → ② 제목·공문 부서코드·질의 키워드 → ③ 법령만 있으면 '공통' → ④ 포털 값
입력: processed/cases_clean.jsonl, case_laws.jsonl, case_articles.jsonl   출력: processed/cases_enriched.jsonl (+csv)
"""
import re
from collections import Counter, defaultdict

from common import manifest, read_jsonl, write_csv, write_jsonl
from law_rules import category_of

USED = ("explicit", "context")
# 어느 업권에서나 쓰는 법령: 업권 근거로 세지 않는다
CROSS = re.compile(r"전자문서|전자서명|금융지주회사")
KEYWORDS = [
    ("상호·저축은행업", r"저축은행|상호금융|신협|신용협동조합|농협|수협|새마을금고|회원조합|단위조합"),
    ("여신전문금융업", r"신용카드|체크카드|카드사|카드회원|카드발급|가맹점|캐피탈|할부금융|대부업|여신전문|여전|MS카드|IC카드|하이패스카드"),
    ("보험", r"(?<!예금)보험"),
    ("전자금융", r"망분리|전자금융|선불전자|가상자산|IT보안|클라우드|해킹|공개용 서버|침해사고"),
    ("자본시장", r"펀드|집합투자|투자일임|ISA|증권|금융투자|운용보수|자산운용|투자자문|투자회사"),
]
BANK = re.compile(r"은행")
# 금감원 공문 제목 끝의 부서코드: '(저축총괄-00667)', '(IT보안-00071)'
DEPT = re.compile(r"\(\s*([가-힣A-Za-z]{2,6})\s*-\s*\d+")
DEPT_MAP = [("저축", "상호·저축은행업"), ("여전", "여신전문금융업"), ("IT", "전자금융"), ("은행", "은행"), ("은총", "은행"),
            ("보생", "보험"), ("보손", "보험"), ("보험", "보험"), ("증권", "자본시장"), ("금투", "자본시장"), ("자산운용", "자본시장")]


def by_words(text):
    """키워드가 한 업권만 가리킬 때 그 업권. '은행'은 다른 업권 표시가 없을 때만 본다 (저축은행과 겹침)."""
    hit = {c for c, p in KEYWORDS if re.search(p, text)}
    if len(hit) == 1:
        return hit.pop()
    return "은행" if not hit and BANK.search(text) else None


def by_keyword(r):
    title = r["title"] or ""
    for text in (title, (r["question"] or "")[:300]):
        c = by_words(text)
        if c:
            return c
        if text is title:
            m = DEPT.search(title)
            if m:
                c = next((v for k, v in DEPT_MAP if m.group(1).startswith(k)), None)
                if c:
                    return c
    return None


def infer(r, cats, others, has_law):
    tie = False
    if cats and sum(cats.values()) >= others:
        top = cats.most_common()
        tied = sorted(c for c, n in top if n == top[0][1])
        if len(tied) == 1:
            return tied[0], False
        if r["category"] in tied:
            return r["category"], True
        tie = True
        first = tied[0]
    else:
        first = None
    kw = by_keyword(r)
    if kw:
        return kw, tie
    return first or ("공통" if has_law else None), tie


def main():
    rows = read_jsonl("cases_clean.jsonl")
    by_case, others, any_law = defaultdict(Counter), Counter(), Counter()
    for x in read_jsonl("case_laws.jsonl"):
        k = x["case_key"]
        any_law[k] += 1
        c = None if CROSS.search(x["law_name"].replace(" ", "")) else category_of(x["law_name"])
        if c:
            by_case[k][c] += x["mentions"]
        else:
            others[k] += x["mentions"]
    n_art = Counter(a["case_key"] for a in read_jsonl("case_articles.jsonl") if a["resolve_method"] in USED)

    for r in rows:
        k = r["case_key"]
        r["category_inferred"], r["category_inferred_tie"] = infer(r, by_case.get(k), others[k], any_law[k] > 0)
        if r["category_inferred"]:
            r["category_final"], r["category_final_source"] = r["category_inferred"], "inferred"
        elif r["category"]:
            r["category_final"], r["category_final_source"] = r["category"], "portal"
        else:
            r["category_final"], r["category_final_source"] = None, None
        r["n_laws"], r["n_articles"] = any_law[k], n_art[k]

    write_jsonl("cases_enriched.jsonl", rows)
    write_csv("cases_enriched.csv", rows, list(rows[0].keys()))
    idx = [r for r in rows if r["indexable"]]
    manifest("08_category", cases=len(rows), indexable=len(idx),
             source=dict(Counter(r["category_final_source"] for r in idx)),
             final=dict(Counter(r["category_final"] for r in idx)),
             ties=sum(r["category_inferred_tie"] for r in idx),
             no_law=sum(r["n_laws"] == 0 for r in idx), no_article=sum(r["n_articles"] == 0 for r in idx))


if __name__ == "__main__":
    main()
