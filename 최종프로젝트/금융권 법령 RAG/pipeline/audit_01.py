"""기획안 5-1 판단 기준 1~8 점검 → data/work/01_fetch/audit.json, article_counts.csv

    python pipeline/audit_01.py
"""
import csv
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
from pypdf import PdfReader

from common import ENV, SOURCE, TODAY, WORK

OUT = WORK / "01_fetch"
csv.field_size_limit(2**31 - 1)  # 분리 전 행정규칙 덩어리가 13만 자를 넘음


def load():
    sel = json.loads((OUT / "selected.json").read_text(encoding="utf-8"))
    cur = json.loads((OUT / "api_current.json").read_text(encoding="utf-8"))
    targets = json.loads((SOURCE / "target_laws.json").read_text(encoding="utf-8"))
    a_names = [v[0] for g, d in targets.items() if not g.startswith("_") for v in d.values()]
    rows = list(csv.DictReader(open(OUT / "articles_raw.csv", encoding="utf-8-sig")))
    return sel, cur, a_names, rows


def site_labels(mst, ef):
    """국가법령정보센터 화면(본문 HTML)의 본문 조문 번호 목록 (부칙 제외). 화면은 시행 전 조문도 함께 보여 준다"""
    h = requests.get(f"https://www.law.go.kr/LSW/lsInfoR.do?lsiSeq={mst}&efYd={ef}", timeout=60,
                     headers={"User-Agent": "Mozilla/5.0"}).text
    time.sleep(1.0)
    labs = re.findall(r'<label for="Y\d+">\s*(제\d+조(?:의\d+)?)', h)
    return list(dict.fromkeys(labs))


def pdf_future(path, label):
    """PDF 에서 그 조문 뒤에 붙은 '[시행일: YYYY. M. D.]' 가 기준일보다 뒤인지"""
    t = "\n".join(pg.extract_text() or "" for pg in PdfReader(path).pages)
    m = re.search(r"(?m)^\s*" + re.escape(label) + r"\(", t)
    if not m:
        return False
    d = re.search(r"\[시행일\s*:\s*(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?\]", t[m.start():m.start() + 4000])
    return bool(d) and f"{d.group(1)}{int(d.group(2)):02d}{int(d.group(3)):02d}" > TODAY


def pdf_count(path):
    t = "\n".join(pg.extract_text() or "" for pg in PdfReader(path).pages)
    body = re.split(r"\n\s*부\s*칙", t)[0]
    arts = re.findall(r"(?m)^\s*제\s*(\d+(?:-\d+)?)\s*조(?:의\s*(\d+))?\s*(?:\(|삭제)", body)
    return len(dict.fromkeys(a + ("의" + b if b else "") for a, b in arts))


def is_deleted(raw):
    r"""조문 머리(제N조, 제N조의M, 괄호 제목)와 태그를 뗀 본문이 '삭제'뿐이면 삭제 조문
    (처음 규칙 `^제\S+조\s*삭제` 는 '제28조의6 삭제' 같은 가지조문을 놓쳐 1,151개로 셌음 → 1,821개)"""
    body = re.sub(r"<[^<>]*>|\[[^\[\]]*\]", "", raw)
    body = re.sub(r"^제\s*\d+(?:-\d+)?\s*조(?:\s*의\s*\d+)?\s*(?:\([^)]*\))?", "", body.strip()).strip()
    return re.fullmatch(r"삭\s*제\.?", body) is not None


def main():
    sel, cur, a_names, rows = load()
    chosen, dropped = sel["chosen"], sel["dropped"]
    res = {}

    # 1. 목록 수 = 받은 수 = 파싱 성공
    parsed = 0
    for c in chosen:
        try:
            json.loads(Path(c["path"]).read_text(encoding="utf-8"))
            parsed += 1
        except Exception:
            pass
    in_csv = {r["uid"] for r in rows}
    res["1"] = {"최종 목록": len(chosen), "본문 파일": len({c["path"] for c in chosen}), "파싱 성공": parsed,
                "CSV에 있는 법령": len(in_csv), "범위별": dict(Counter(f'{c["scope"]}·{c["level"]}' for c in chosen)),
                "pass": len(chosen) == parsed == len(in_csv)}

    # 2. 핵심 44개
    names = {c["name"] for c in chosen}
    miss = [n for n in a_names if n not in names]
    res["2"] = {"A 법률": f"{len(a_names) - len(miss)}/{len(a_names)}", "빠진 것": miss, "pass": not miss}

    # 3. 하위 법령: API에 현행이 있는 시행령·시행규칙 = 받은 것
    sub = []
    for a in a_names:
        row = {"법률": a}
        for s in ("시행령", "시행규칙"):
            nm = f"{a} {s}"
            live = [h for h in cur["law"].get(nm, []) if (h["effective"] or "0") <= TODAY]
            row[s] = ("있음" if live else "없음") + "/" + ("받음" if nm in names else "-")
        sub.append(row)
    bad3 = [r for r in sub if any(v == "있음/-" for k, v in r.items() if k != "법률")]
    res["3"] = {"표": sub, "API엔 있는데 못 받은 것": bad3, "pass": not bad3}

    # 4. 현행판
    future = [c["name"] for c in chosen if (c["effective"] or "0") > TODAY]
    mismatch = [(c["name"], c["effective"], c.get("api_effective")) for c in chosen if c["effective"] != c.get("api_effective")]
    res["4"] = {"기준일": TODAY, "예정판": future, "API 현행 시행일자와 다름": mismatch, "pass": not future and not mismatch}

    # 5. 조문 수: 원본 조문 수 = CSV 조문 행 수
    csv_n = Counter(r["uid"] for r in rows if r["unit_type"] == "조문")
    csv_del = Counter(r["uid"] for r in rows if r["unit_type"] == "조문" and is_deleted(r["raw_text"]))
    table, diff = [], []
    for c in sorted(chosen, key=lambda c: (c["scope"], c["name"])):
        arts = [r for r in rows if r["uid"] == c["uid"] and r["unit_type"] == "조문"]
        t = {"uid": c["uid"], "scope": c["scope"], "law_name": c["name"], "level": c["level"],
             "src_articles": c["n_articles"], "csv_articles": csv_n[c["uid"]], "deleted": csv_del[c["uid"]],
             "first": arts[0]["article"] if arts else "", "last": arts[-1]["article"] if arts else "",
             "units_other": sum(r["uid"] == c["uid"] and r["unit_type"] != "조문" for r in rows)}
        table.append(t)
        if t["src_articles"] != t["csv_articles"]:
            diff.append(t["law_name"])
    with open(OUT / "article_counts.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    shape = {c["uid"]: c.get("body_shape", "law") for c in chosen}
    unsplit = [t["law_name"] for t in table if shape[t["uid"]] in ("str", "none")]
    zero = [t["law_name"] for t in table if t["csv_articles"] == 0 and shape[t["uid"]] not in ("str", "none")]
    res["5"] = {"대상": "법령 + 조문으로 나뉘어 온 행정규칙", "법령": len(table) - len(unsplit), "차이 있는 법령": diff,
                "조문 0인 법령": zero, "합계 조문": sum(t["csv_articles"] for t in table),
                "삭제 조문": sum(t["deleted"] for t in table),
                "분리 전 행정규칙 (5-2-3에서 셈)": unsplit, "pass": not diff and not zero}

    # 6. 원문 대조: 국가법령정보센터 화면 (A·B 무작위 5개 + PDF 있는 법령). 화면은 시행 예정 조문도 보여 주므로 뺀다
    random.seed(20261113)
    pool = [c for c in chosen if c["target"] == "law" and c["scope"] in ("A", "B")]
    by_name = {c["name"]: c for c in chosen}
    pdf_names = [p.name.split("(")[0].strip() for p in sorted((SOURCE / "raw").glob("*.pdf"))]
    sample = random.sample(pool, 5) + [by_name[n] for n in pdf_names if n in by_name and by_name[n]["target"] == "law"]
    tc = {t["uid"]: t for t in table}
    pdf_path = {p.name.split("(")[0].strip(): p for p in sorted((SOURCE / "raw").glob("*.pdf"))}
    site = []
    for c in sample:
        scr = site_labels(c["mst"], c["effective"])
        ours = [r["article"] for r in rows if r["uid"] == c["uid"] and r["unit_type"] == "조문"]
        only_scr = [x for x in scr if x not in ours]
        only_ours = [x for x in ours if x not in scr]
        explained = bool(only_scr) and not only_ours and c["name"] in pdf_path and             all(pdf_future(pdf_path[c["name"]], x) for x in only_scr)
        site.append({"법령": c["name"], "시행일자": c["effective"], "우리": len(ours), "화면": len(scr),
                     "화면에만": only_scr, "우리에만": only_ours,
                     "처음·끝": f'{tc[c["uid"]]["first"]} ~ {tc[c["uid"]]["last"]}',
                     "판정": "일치" if not only_scr and not only_ours else ("시행 예정 조문 차이 (PDF로 확인)" if explained else "불일치")})
    pdfs = []
    for p in sorted((SOURCE / "raw").glob("*.pdf")):
        nm = p.name.split("(")[0].strip()
        c = by_name.get(nm)
        pdfs.append({"PDF": nm, "PDF 시행일자": re.findall(r"\((\d{8})\)", p.name)[-1],
                     "우리 시행일자": c["effective"] if c else "-", "PDF 조문(목차·예정 포함)": pdf_count(p),
                     "우리": tc[c["uid"]]["csv_articles"] if c else "-"})
    res["6"] = {"화면 대조": site, "PDF 대조 (참고)": pdfs, "pass": all(x["판정"] != "불일치" for x in site)}

    # 7. 원본 목록(index.jsonl)과 파일 차이
    idx = {(json.loads(l)["target"], json.loads(l)["id"]) for l in open(SOURCE / "api" / "index.jsonl", encoding="utf-8")}
    inv = json.loads((OUT / "inventory.json").read_text(encoding="utf-8"))
    extra = [r for r in inv if r["origin"] == "source" and (r["target"], r["mst"]) not in idx]
    chosen_mst = {c["mst"] for c in chosen}
    res["7"] = {"index.jsonl": len(idx), "원본 파일": sum(r["origin"] == "source" for r in inv),
                "목록에 없는 파일": [{"법령": r["name"], "판": r["effective"], "범위": r["scope"],
                                 "최종 선택": r["mst"] in chosen_mst} for r in extra],
                "pass": True}

    # 8. 인증값
    oc = ENV.get("LAW_API_OC") or "@@none@@"
    leak = [str(p) for p in OUT.rglob("*") if p.is_file() and oc in p.read_text(encoding="utf-8", errors="ignore")]
    res["8"] = {"검사 파일": sum(1 for p in OUT.rglob("*") if p.is_file()), "인증값 나온 파일": len(leak), "pass": not leak}

    res["뺀 파일"] = dict(Counter(d["why"].split(" (")[0] for d in dropped))
    res["뺀 파일 목록"] = [{"법령": d.get("name"), "판": d.get("effective"), "범위": d.get("scope"), "이유": d["why"]}
                       for d in dropped if not d["why"].startswith("범위 밖")]
    (OUT / "audit.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in "12345678":
        print(k, "통과" if res[k]["pass"] else "실패", {kk: vv for kk, vv in res[k].items() if kk not in ("표", "pass")})


if __name__ == "__main__":
    main()
