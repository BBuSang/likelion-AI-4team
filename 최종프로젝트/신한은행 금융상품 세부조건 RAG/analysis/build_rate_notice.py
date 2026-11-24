"""고시 금리(raw TSV) → data/rates/rate_notice.csv + 상품 연결 (LLM 없음)

입력: data/rates/raw/deposit_YYYY-MM-DD.tsv, loan_YYYY-MM-DD.tsv, loan_assumptions_YYYY-MM-DD.tsv
      data/meta/products.csv
출력: data/rates/rate_notice.csv   (schema.sql rate_notice 테이블 형식, 한 행 = 상품 × 조건 1개)
      data/rates/rate_product_map.csv (금리 페이지 상품명 → product_id, 연결 방법)
연결 규칙(순서대로): ① 정규화 이름 완전 일치 ② 끝 괄호 제거 후 일치
                    ③ 금리 페이지 이름이 상품명 안에 포함되고 그런 상품이 하나뿐일 때
정규화: NFC, 공백·괄호·『』「」·전각/반각 쉼표·가운뎃점 제거, 소문자
"""
import csv
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT, META = ROOT / "data" / "rates" / "raw", ROOT / "data" / "rates", ROOT / "data" / "meta"
DATE = sys.argv[1] if len(sys.argv) > 1 else "2026-11-09"

# 금리 페이지 그룹 → products.csv 구분
DEP_SUB = {"목돈만들기": {"적립식예금", "주택청약관련예금"}, "목돈굴리기": {"거치식예금"},
           "입출금자유": {"입출금자유예금"}}
LOAN_SUB = {"신용대출": {"신용대출"}, "부동산담보대출": {"주택담보대출", "기타담보대출"},
            "MY CAR 대출": {"기타담보대출", "신용대출"}, "기타대출": {"기타담보대출", "신용대출"}}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s).lower()
    return re.sub(r"[\s()（）\[\]『』「」,，·ㆍ.]", "", s)


def strip_tail(s: str) -> str:
    return re.sub(r"\s*[(（][^()（）]*[)）]\s*$", "", s).strip()


def read_tsv(path: Path, ncol: int):
    rows = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        c = line.split("\t")
        rows.append(c + [""] * (ncol - len(c)))
    return rows[0], rows[1:]


def pct(s: str):
    s = s.replace("%", "").strip()
    return float(s) if s else None


def main() -> None:
    products = list(csv.DictReader(open(META / "products.csv")))
    by_norm = defaultdict(list)
    for p in products:
        by_norm[norm(p["name"])].append(p)

    def match(name: str, subs: set):
        cands = lambda ps: [p for p in ps if p["subcategory"] in subs] or ps
        for how, key in (("exact", norm(name)), ("strip_paren", norm(strip_tail(name)))):
            if by_norm.get(key):
                return how, cands(by_norm[key])
        k = norm(name)
        # 금리 페이지 이름이 상품명 안에 들어 있을 때만 (반대 방향은 '정기예금'처럼 일반 이름에 잘못 붙음)
        hit = [p for p in products if p["subcategory"] in subs and k in norm(p["name"])]
        names = {p["name"] for p in hit}
        if len(names) == 1:
            return "contains", hit
        return "none", []

    rows, mapping = [], {}
    _, dep = read_tsv(RAW / f"deposit_{DATE}.tsv", 7)
    for g, prod, pay, term, amt, base, onl in dep:
        rows.append({"kind": "deposit", "group": g, "page_product": prod, "pay_method": pay,
                     "term": term, "amount_from": re.sub(r"\D", "", amt), "base_index": "",
                     "base_rate": pct(base), "spread": None, "pref": None,
                     "min_rate": None, "max_rate": None, "online_rate": pct(onl), "assumption": ""})
        mapping.setdefault(("deposit", g, prod), DEP_SUB[g])
    _, asm_rows = read_tsv(RAW / f"loan_assumptions_{DATE}.tsv", 2)
    asm = {a: t for a, t in asm_rows}
    _, loan = read_tsv(RAW / f"loan_{DATE}.tsv", 9)
    seen = set()
    for sub, prod, idx, base, spread, pref, mn, mx, aid in loan:
        key = (sub, prod, idx, base, spread, pref, mn, mx)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"kind": "loan", "group": sub, "page_product": prod, "pay_method": "",
                     "term": "", "amount_from": "", "base_index": idx, "base_rate": pct(base),
                     "spread": pct(spread), "pref": pct(pref), "min_rate": pct(mn),
                     "max_rate": pct(mx), "online_rate": None, "assumption": asm.get(aid, "")})
        mapping.setdefault(("loan", sub, prod), LOAN_SUB[sub])

    # 연결 안 된 상품의 상품공시 확인 결과 (판매 중지 여부) — 사람이 확인해 기록한 파일
    status = {}
    st_path = OUT / f"unlinked_status_{DATE}.tsv"
    if st_path.exists():
        _, st = read_tsv(st_path, 2)
        status = {r[0]: r[1] for r in st}

    map_rows, pid_of = [], {}
    for (kind, g, prod), subs in mapping.items():
        how, ps = match(prod, subs)
        pids = ";".join(p["product_id"] for p in ps)
        pid_of[(kind, g, prod)] = pids
        sale = "on_sale" if ps else status.get(prod, "unknown")
        map_rows.append({"kind": kind, "group": g, "page_product": prod, "match": how,
                         "product_ids": pids, "product_name": ps[0]["name"] if ps else "",
                         "sale_status": sale})

    for r in rows:
        r["product_ids"] = pid_of[(r["kind"], r["group"], r["page_product"])]
        r["as_of"] = DATE
        r["source"] = "shinhan.com 금리안내(예금금리 > 바로가기)"
    fields = ["kind", "group", "page_product", "product_ids", "pay_method", "term", "amount_from",
              "base_index", "base_rate", "spread", "pref", "min_rate", "max_rate", "online_rate",
              "assumption", "as_of", "source"]
    with open(OUT / "rate_notice.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "rate_product_map.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(map_rows[0]))
        w.writeheader()
        w.writerows(map_rows)

    from collections import Counter
    print("금리 행:", Counter(r["kind"] for r in rows))
    print("상품 연결:", Counter((m["kind"], m["match"]) for m in map_rows))
    print("연결 안 된 상품:", [m["page_product"] for m in map_rows if m["match"] == "none"])
    print("판매 상태:", Counter(m["sale_status"] for m in map_rows))


if __name__ == "__main__":
    main()
