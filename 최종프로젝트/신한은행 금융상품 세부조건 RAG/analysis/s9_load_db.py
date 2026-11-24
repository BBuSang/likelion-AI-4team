"""4-1 DB 적재 — schema.sql을 새로 만들고 수집·전처리·추출 결과를 넣는다 (임베딩·청크는 다음 단계)

사용: python analysis/s9_load_db.py            # 스키마를 지우고 다시 만든 뒤 전부 적재 (다시 돌려도 같은 결과)
DB:  환경변수 DATABASE_URL, 없으면 로컬 개발용 postgresql://postgres:fss_local@127.0.0.1:5433/fss_rag
     (docker 컨테이너 fss_rag_db, pgvector/pgvector:pg16, 볼륨 fss_rag_pgdata — 실전 프로젝트의 pii_rag_db와 분리)

넣는 것
  product            ← data/meta/products.csv (상품공시 2026-11-06)
  document           ← documents.csv + doc_meta.csv(2-4 심의필·유효기간·기준일·쪽수) + PDF sha256
                       + data/extract/full(3-2 요약 칸·detail·extract_status·extract_notes)
  product_document   ← product_docs.csv
  doc_rate / doc_fee / doc_benefit ← data/extract/full (근거 섹션·원문 포함)
  common_clause / document_clause / glossary_term ← 2-7 결과 (임베딩은 비워 둠)
  rate_notice (+sale_status) / rate_notice_product ← data/rates (2026-11-09)
  discontinued_product ← data/meta/discontinued.csv (2026-11-09)
"""
import csv
import hashlib
import json
import os
import re
from pathlib import Path

import sys

import psycopg
from psycopg.types.json import Jsonb

ROOT = Path(__file__).resolve().parents[1]
META, RATES, EXT = ROOT / "data" / "meta", ROOT / "data" / "rates", ROOT / "data" / "extract" / "full"
DSN = os.environ.get("DATABASE_URL", "postgresql://postgres:fss_local@127.0.0.1:5433/fss_rag")
PRODUCT_SNAPSHOT, COLLECT_DATE = "2026-11-06", "2026-11-09"


def rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def num(x):
    x = (x or "").strip().replace(",", "")
    return float(x) if x not in ("", "-") else None


def date_or_none(x):
    return x if x and re.match(r"^\d{4}-\d{2}-\d{2}$", x) else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    products, documents = rows(META / "products.csv"), rows(META / "documents.csv")
    meta = {r["doc_id"]: r for r in rows(META / "doc_meta.csv")}
    ext = {p.stem: json.loads(p.read_text()) for p in sorted(EXT.glob("D*.json"))}
    with psycopg.connect(DSN, autocommit=False) as con, con.cursor() as cur:
        cur.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
        sys.path.insert(0, str(ROOT / "analysis"))
        from s8_extract import load_env
        dim = int(load_env().get("EMBED_DIM", 1536))
        cur.execute((ROOT / "models" / "schema.sql").read_text().replace("vector(1536)", f"vector({dim})"))   # 임베딩 모델 차원

        cur.executemany("INSERT INTO product (product_id, product_name, category, sub_category, site_no, dup_name, snapshot_date)"
                        " VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        [(r["product_id"], r["name"], r["category"], r["subcategory"], int(r["site_no"]) if r["site_no"] else None,
                          r["dup_name"] in ("1", "True", "true"), PRODUCT_SNAPSHOT) for r in products])

        doc_rows = []
        for r in documents:
            m, e = meta.get(r["doc_id"], {}), ext.get(r["doc_id"])
            x = e["extraction"] if e else {}
            doc_rows.append((
                r["doc_id"], r["title"], r["file"], sha256(ROOT / "data" / "docs" / r["file"]), r["doc_type"], int(r["n_products"]),
                x.get("summary"), x.get("target_customer"), x.get("join_channels") or None, x.get("deposit_protected"),
                x.get("risk_grade"), Jsonb(x.get("detail") or {}),
                m.get("review_no") or None, date_or_none(m.get("valid_from")), date_or_none(m.get("valid_to")),
                date_or_none(m.get("info_base_date")), int(m["pages_total"]) if m.get("pages_total") else None,
                e["status"] if e else "not_extracted", x.get("notes") or None))
        cur.executemany(
            "INSERT INTO document (doc_id, title, file_name, file_hash, doc_type, n_products, summary, target_customer, join_channels,"
            " deposit_protected, risk_grade, detail, review_no, valid_from, valid_to, info_base_date, page_count,"
            " extract_status, extract_notes) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", doc_rows)

        cur.executemany("INSERT INTO product_document (product_id, doc_id, role) VALUES (%s,%s,%s)",
                        [(r["product_id"], r["doc_id"], r["role"]) for r in rows(META / "product_docs.csv")])

        rate, fee, ben = [], [], []
        for d, e in ext.items():
            x = e["extraction"]
            rate += [(d, i["rate_kind"], i["rate_min"], i["rate_max"], i["term_min_m"], i["term_max_m"], i["amount_min"],
                      i["amount_max"], i["currency"], i["rate_type"], i["condition"], i["pre_tax"], i["section_seq"],
                      i["source_text"]) for i in x["rates"]]
            fee += [(d, i["fee_kind"], i["amount"], i["rate_pct"], i["condition"], i["section_seq"], i["source_text"])
                    for i in x["fees"]]
            ben += [(d, i["benefit_kind"], i["requirement"], i["benefit"], i["value_num"], i["value_unit"], i["section_seq"],
                     i["source_text"]) for i in x["benefits"]]
        cur.executemany("INSERT INTO doc_rate (doc_id, rate_kind, rate_min, rate_max, term_min_m, term_max_m, amount_min, amount_max,"
                        " currency, rate_type, condition, pre_tax, section_seq, source_text)"
                        " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", rate)
        cur.executemany("INSERT INTO doc_fee (doc_id, fee_kind, amount, rate_pct, condition, section_seq, source_text)"
                        " VALUES (%s,%s,%s,%s,%s,%s,%s)", fee)
        cur.executemany("INSERT INTO doc_benefit (doc_id, benefit_kind, requirement, benefit, value_num, value_unit, section_seq,"
                        " source_text) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)", ben)

        clauses = rows(META / "common_clause.csv")
        cur.executemany("INSERT INTO common_clause (clause_id, title, content, content_hash) VALUES (%s,%s,%s,%s)",
                        [(int(c["clause_id"]), c["title"], c["content"],
                          hashlib.sha256(re.sub(r"\s", "", c["content"]).encode()).hexdigest()) for c in clauses])
        cur.executemany("INSERT INTO document_clause (doc_id, clause_id) VALUES (%s,%s)",
                        [(r["doc_id"], int(r["clause_id"])) for r in rows(META / "document_clause.csv")])
        cur.executemany("INSERT INTO glossary_term (term, definition) VALUES (%s,%s)",
                        [(g["term"], g["definition"]) for g in rows(META / "glossary.csv")])

        status = {(r["group"], r["page_product"]): r["sale_status"] for r in rows(RATES / "rate_product_map.csv")}
        cur.executemany(
            "INSERT INTO rate_notice (as_of, kind, page_group, page_product, pay_method, term, amount_from, base_index, base_rate,"
            " online_rate, spread, pref, min_rate, max_rate, assumption, sale_status, source)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            [(r["as_of"], r["kind"], r["group"], r["page_product"], r["pay_method"] or None, r["term"] or None,
              num(r["amount_from"]), r["base_index"] or None, num(r["base_rate"]), num(r["online_rate"]), num(r["spread"]),
              num(r["pref"]), num(r["min_rate"]), num(r["max_rate"]), r["assumption"] or None,
              status.get((r["group"], r["page_product"])), r["source"]) for r in rows(RATES / "rate_notice.csv")])
        rnp = {(r["group"], r["page_product"], pid, r["match"])
               for r in rows(RATES / "rate_product_map.csv") for pid in r["product_ids"].split("|") if pid}
        cur.executemany("INSERT INTO rate_notice_product (page_group, page_product, product_id, match_method) VALUES (%s,%s,%s,%s)",
                        sorted(rnp))

        cur.executemany("INSERT INTO discontinued_product (category, sub_category, site_no, product_name, also_on_sale, snapshot_date)"
                        " VALUES (%s,%s,%s,%s,%s,%s)",
                        [(r["category"], r["subcategory"], int(r["site_no"]) if r["site_no"] else None, r["name"],
                          r["also_on_sale"] == "1", COLLECT_DATE) for r in rows(META / "discontinued.csv")])
        con.commit()

        print("적재 건수")
        for t in ("product", "document", "product_document", "doc_rate", "doc_fee", "doc_benefit", "common_clause",
                  "document_clause", "glossary_term", "rate_notice", "rate_notice_product", "discontinued_product"):
            cur.execute(f"SELECT count(*) FROM {t}")
            print(f"  {t:22} {cur.fetchone()[0]:,}")


if __name__ == "__main__":
    main()
