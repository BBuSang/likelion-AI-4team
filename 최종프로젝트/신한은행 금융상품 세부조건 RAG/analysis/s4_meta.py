"""2-4 문서 메타 추출 (정규식, LLM 없음) → data/meta/doc_meta.csv

필드 (schema.sql document 테이블):
  review_no        준법감시인 심의필 번호            ← 2-2 로그 s2_removed.csv (rule=review)
  valid_from/to    심의필 유효기간                    ← 2-2 로그 (rule=valid)
  info_base_date   금리 등 기준일 (가장 많이 나온 값) ← 본문: 'YYYY.MM.DD 현재' / 이자율·금리 문맥의 'YYYY.MM.DD 기준'
                                                     / '이자율 (YYYY.MM.DD …' 괄호 안 날짜
                                                     / 문서 맨 앞 300자 안 '(YYYY.MM.DD 기준)'
  base_dates       본문에 나온 기준일 전부 (서로 다르면 여러 개, ';' 구분)
  effective_date   약관·특약 시행일 (가장 늦은 값)    ← '이 특약/약관은 YYYY.MM.DD부터 (개정) 시행'
  pages_total / pages_body / format / 예외 표시      ← 2-1 doc_survey.csv, 2-3 s3_pages.csv
"""
import collections
import csv
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S3, META = ROOT / "data" / "stage" / "s3", ROOT / "data" / "meta"

D_FULL = r"(\d{4})\s*[.년/-]\s*(\d{1,2})\s*[.월/-]\s*(\d{1,2})\s*일?"
D_SHORT = r"[’‘']\s*(\d{2})\s*[.]\s*(\d{1,2})\s*[.]\s*(\d{1,2})"
DATE = f"(?:{D_FULL}|{D_SHORT})"
CURRENT = re.compile(DATE + r"\.?\s*\)?\s*현재")
RATE_BASIS = re.compile(r"(?:이자율|금리|이율)[^.。]{0,30}?" + DATE + r"\.?\s*\)?\s*기준")
# '기본이자율 (2026.04.15 USD 기준' / '기본이자율 (2026.09.11 보통예금 : 연 0.1%' — 이자율 바로 뒤 괄호 안 날짜
RATE_PAREN = re.compile(r"(?:이자율|금리|이율)\*?\s*\(\s*" + DATE)
# 문서 제목 바로 뒤 '(2026. 07.23 기준)' — 문서 맨 앞 300자 안에서만
DOC_BASIS = re.compile(r"\(\s*" + DATE + r"\s*\.?\s*기준\s*\)")
EFFECTIVE = re.compile(r"이\s*(?:특약|약관|약정서?|규정)[은는이]?\s*" + DATE + r"\s*부터\s*(?:개정\s*|변경\s*)?시행")
VALID = re.compile(r"\(\s*" + D_FULL + r"\s*\.?\s*~\s*" + D_FULL + r"\s*\.?\s*\)")


def to_date(g: tuple) -> str | None:
    """정규식 그룹 (full y,m,d, short y,m,d) → 'YYYY-MM-DD'"""
    y, m, d = (g[0], g[1], g[2]) if g[0] else ("20" + g[3], g[4], g[5])
    try:
        return date(int(y), int(m), int(d)).isoformat()
    except (TypeError, ValueError):
        return None


def main() -> None:
    survey = {r["doc_id"]: r for r in csv.DictReader(open(META / "doc_survey.csv"))}
    pages = collections.defaultdict(collections.Counter)
    for r in csv.DictReader(open(META / "s3_pages.csv")):
        pages[r["doc_id"]][r["role"]] += 1
    review, valid = {}, {}
    for r in csv.DictReader(open(META / "s2_removed.csv")):
        if r["rule"] == "review" and r["doc_id"] not in review:
            review[r["doc_id"]] = re.sub(r"\s+", "", re.search(r"제?\s*([\d\s-]+)호", r["line"]).group(1))
        if r["rule"] == "valid" and r["doc_id"] not in valid:
            m = VALID.search(r["line"])
            valid[r["doc_id"]] = (to_date(m.groups()[0:3] + (None,) * 3),
                                  to_date(m.groups()[3:6] + (None,) * 3))

    rows = []
    for did, sv in survey.items():
        body = re.sub(r"\s+", " ", re.sub(r"=== p\.\d+ ===", " ", (S3 / f"{did}.txt").read_text()))
        bases = collections.Counter()
        for pat in (CURRENT, RATE_BASIS, RATE_PAREN):
            for m in pat.finditer(body):
                if "예시" in body[max(0, m.start() - 15):m.start()]:
                    continue
                dt = to_date(m.groups()[-6:])
                if dt:
                    bases[dt] += 1
        for m in DOC_BASIS.finditer(body[:300]):
            dt = to_date(m.groups()[-6:])
            if dt:
                bases[dt] += 1
        eff = [d for d in (to_date(m.groups()) for m in EFFECTIVE.finditer(body)) if d]
        vf, vt = valid.get(did, (None, None))
        rows.append({
            "doc_id": did, "title": sv["title"], "doc_type": sv["doc_type"], "format": sv["format"],
            "review_no": review.get(did, ""), "valid_from": vf or "", "valid_to": vt or "",
            "info_base_date": bases.most_common(1)[0][0] if bases else "",
            "base_dates": ";".join(sorted(bases)),
            "effective_date": max(eff) if eff else "",
            "pages_total": sv["pages"], "pages_body": pages[did]["body"],
            "dup_copy": sv["dup_copy"], "no_space": sv["no_space"], "ocr": sv["ocr"],
            "summary": sv["summary"],
        })
    with open(META / "doc_meta.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by_type = collections.defaultdict(list)
    for r in rows:
        by_type[r["doc_type"]].append(r)
    print(f"{'문서종류':8} {'수':>4} {'심의필':>5} {'유효기간':>6} {'기준일':>5} {'시행일':>5}")
    for t, rs in sorted(by_type.items(), key=lambda x: -len(x[1])):
        c = lambda k: sum(bool(r[k]) for r in rs)
        print(f"{t:8} {len(rs):4} {c('review_no'):5} {c('valid_from'):6} {c('info_base_date'):5} {c('effective_date'):5}")
    print("기준일이 2개 이상인 문서:", sum(";" in r["base_dates"] for r in rows))


if __name__ == "__main__":
    main()
