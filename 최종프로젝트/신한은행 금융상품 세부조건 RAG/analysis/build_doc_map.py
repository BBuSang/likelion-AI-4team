"""신한은행 상품공시 연결표(xlsx) → 상품·문서 연결 DB(CSV 3개).

입력: data/meta/신한은행_상품공시_설명서_연결표.xlsx (사용자가 상품공시실에서 정리, 2026-11-06 판매분)
출력 (data/meta/):
  products.csv      product_id, category(분류), subcategory(구분), site_no, name, dup_name
  documents.csv     doc_id, title, file, role(explainer/terms), doc_type, n_products
  product_docs.csv  product_id, doc_id, role
검증: 연결표의 모든 문서가 data/docs/*.pdf 에 있는지, 쓰이지 않는 PDF가 없는지 확인한다.
"""
import collections
import csv
import sys
import unicodedata
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
META, DOCS = ROOT / "data" / "meta", ROOT / "data" / "docs"
XLSX = META / "신한은행_상품공시_설명서_연결표.xlsx"
SHEETS = {"저축상품": "SAV", "기타상품": "ETC", "복합금융상품": "ELD", "대출상품": "LOAN"}
NONE = {"", "(없음)"}


def nfc(x) -> str:
    return unicodedata.normalize("NFC", str(x or "")).strip()


def doc_type(title: str, role: str) -> str:
    for key, t in [("간이투자설명서", "간이투자설명서"), ("투자설명서", "투자설명서"),
                   ("상품설명서", "상품설명서"), ("설명서", "상품설명서"), ("특약", "특약"),
                   ("약정서", "약정서"), ("계약서", "약정서"), ("약관", "약관")]:
        if key in title:
            return t
    # 이름에 종류가 없으면 사이트 폴더(열) 기준
    return "상품설명서" if role == "explainer" else "특약"


def main() -> None:
    files = {nfc(p.stem): p.name for p in DOCS.glob("*.pdf")}
    wb = openpyxl.load_workbook(XLSX, read_only=True, data_only=True)

    products, links = [], []
    for sheet, prefix in SHEETS.items():
        for r in list(wb[sheet].iter_rows(values_only=True))[1:]:
            if not r[2]:
                continue
            sub, no, name, expl, terms = (nfc(x) for x in r[:5])
            pid = f"{prefix}-{sub}-{no}"
            products.append({"product_id": pid, "category": sheet, "subcategory": sub,
                             "site_no": no, "name": name})
            for col, role in [(expl, "explainer"), (terms, "terms")]:
                for t in (s.strip() for s in col.split("\n")):
                    if t and t not in NONE:
                        links.append((pid, t, role))

    # 문서 표: 제목 하나 = PDF 하나
    titles = sorted({t for _, t, _ in links})
    missing = [t for t in titles if t not in files]
    unused = sorted(set(files) - set(titles))
    if missing or unused:
        sys.exit(f"불일치: 폴더에 없는 문서 {missing[:5]} / 연결표에 없는 PDF {unused[:5]}")

    role_of = {}
    for _, t, role in links:
        role_of.setdefault(t, role)
    n_prod = collections.Counter(t for _, t, _ in set(links))
    doc_id = {t: f"D{i:03d}" for i, t in enumerate(titles, 1)}
    documents = [{"doc_id": doc_id[t], "title": t, "file": files[t], "role": role_of[t],
                  "doc_type": doc_type(t, role_of[t]), "n_products": n_prod[t]} for t in titles]

    name_count = collections.Counter(p["name"] for p in products)
    for p in products:
        p["dup_name"] = int(name_count[p["name"]] > 1)

    def write(name, rows, fields):
        with open(META / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    write("products.csv", products,
          ["product_id", "category", "subcategory", "site_no", "name", "dup_name"])
    write("documents.csv", documents,
          ["doc_id", "title", "file", "role", "doc_type", "n_products"])
    write("product_docs.csv",
          [{"product_id": p, "doc_id": doc_id[t], "role": r} for p, t, r in sorted(set(links))],
          ["product_id", "doc_id", "role"])

    print(f"상품 {len(products)} (이름 중복 행 {sum(p['dup_name'] for p in products)}) / "
          f"문서 {len(documents)} / 연결 {len(set(links))}")
    print("문서 종류:", collections.Counter(d["doc_type"] for d in documents).most_common())
    print("공유 문서(상품 2개 이상):", sum(d["n_products"] > 1 for d in documents))
    print("설명서 없는 상품:", [p["name"] for p in products
                          if not any(l[0] == p["product_id"] and l[2] == "explainer" for l in links)])


if __name__ == "__main__":
    main()
