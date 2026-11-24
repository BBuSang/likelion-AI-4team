"""판매 중지 상품 목록(raw) → data/meta/discontinued.csv (LLM 없음)

입력: data/meta/raw/discontinued_YYYY-MM-DD.tsv  (대출·예금, 사이트 목록 그대로)
      data/meta/raw/eld_discontinued_YYYY-MM-DD.tsv (ELD, 회차 패턴으로 압축)
출력: data/meta/discontinued.csv
  category, subcategory, site_no, name, also_on_sale
  - name: 끝의 '(YYYY.MM.DD 판매중지)' / '(판매중지)' 표시를 뗀 이름
  - 판매중지 날짜는 쓰지 않는다 (2026-11-09 결정: 대출만 날짜가 있고 예금·ELD는 없어 일관되지 않음)
  - also_on_sale: 같은 이름이 판매 중 목록(products.csv)에도 있으면 1 (사이트 목록끼리 겹침)
검증: 탭별 행 수 = 사이트 총건수
"""
import csv
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "meta"
DATE = sys.argv[1] if len(sys.argv) > 1 else "2026-11-09"
EXPECTED = {"대출상품|기업대출": 74, "대출상품|주택담보대출": 14, "대출상품|기타담보대출": 17,
            "대출상품|신용대출": 15, "저축상품|적립식예금": 148, "저축상품|거치식예금": 35,
            "저축상품|입출금자유예금": 29, "복합금융상품|복합금융상품": 1006}
STOP = re.compile(r"\s*\(\s*(?:(\d{4})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*)?판매\s*중지\s*\)\s*$")


def norm(s: str) -> str:
    return re.sub(r"[\s()（）,，·ㆍ.\[\]]", "", unicodedata.normalize("NFC", s)).lower()


def rows_of(path: Path):
    lines = [l for l in path.read_text().splitlines() if l and not l.startswith("#")]
    return [l.split("\t") for l in lines[1:]]


def main() -> None:
    on_sale = {norm(p["name"]) for p in csv.DictReader(open(META / "products.csv"))}
    out = []
    for tab, no, raw in rows_of(META / "raw" / f"discontinued_{DATE}.tsv"):
        name = STOP.sub("", raw).strip()
        cat, sub = tab.split("|")
        out.append({"category": cat, "subcategory": sub, "site_no": int(no), "name": name})
    eld_no = 0
    for pattern, count, rounds in rows_of(META / "raw" / f"eld_discontinued_{DATE}.tsv"):
        rs = rounds.split(",")
        assert len(rs) == int(count), pattern
        for r in rs:
            eld_no += 1
            out.append({"category": "복합금융상품", "subcategory": "복합금융상품", "site_no": eld_no,
                        "name": pattern.replace("{회차}", r)})
    for r in out:
        r["also_on_sale"] = int(norm(r["name"]) in on_sale)

    got = Counter(f"{r['category']}|{r['subcategory']}" for r in out)
    bad = {k: (got[k], v) for k, v in EXPECTED.items() if got[k] != v}
    if bad:
        sys.exit(f"건수 불일치: {bad}")

    with open(META / "discontinued.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)

    uniq = {(r["subcategory"], r["name"]) for r in out}
    print(f"판매 중지 {len(out)}행 (고유 이름 {len(uniq)})")
    for k in EXPECTED:
        print(f"  {k:22} {got[k]:5}")
    print("판매 중 목록에도 같은 이름:", sorted({r["name"] for r in out if r["also_on_sale"]}))


if __name__ == "__main__":
    main()
