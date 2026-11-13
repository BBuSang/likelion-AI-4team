"""X1 현장건의 과제 시범 수집 (사용자 요청 2026-11-11): 포털 현장건의 5,914건 중 제목에 수신·마케팅·이익 제공 관련어가 있는 건만 상세 페이지를 받아 본문 품질을 본다.
DB에는 넣지 않는다(검색 대상이 바뀌면 설정 지문이 바뀌고 평가를 다시 해야 한다).
대상: data/raw/list_total.json 중 pastreqType = "현장건의 과제"이고 제목에 KEYWORDS 중 하나가 있는 건
요청: 기존 수집과 같은 간격(common.request, 기본 1초)·User-Agent. 받은 HTML은 건너뛴다(이어받기)
출력: data/raw/exmnt/{번호}.html, data/processed/exmnt_pilot.jsonl, exmnt_pilot.csv
전체 수집(사용자 요청 2026-11-11 "일단 수집만"): `python x1_exmnt_pilot.py --all` 은 현장건의 5,914건 전부의 HTML만 받는다(파싱·전처리는 하지 않음).
    받은 파일은 건너뛰므로 끊겨도 이어받는다. 실패 목록: data/raw/exmnt_failed.json
"""
import sys
import json
import re

from bs4 import BeautifulSoup

import common
from common import BASE, OUT, RAW, manifest, request, write_csv, write_jsonl
from s03_parse import cell_text

KEYWORDS = ["예금", "적금", "경품", "이벤트", "캐시백", "사은품", "혜택", "마케팅", "특별이익", "이익 제공", "포인트"]
FIELDS = {"소관부서": "dept", "방문기간": "visit", "과제분류": "category", "과제생성일": "created", "회신일": "reply_date", "건의내용": "proposal",
          "향후계획": "plan", "과제종결일": "closed", "참고자료": "refs"}
ARTICLE = re.compile(r"제\s?\d+조(?:의\d+)?|§\s?\d+")


def url_of(no):
    return f"{BASE}/ExmntTaskDetail.do?muNo=86&stNo=11&checkplaceNo={no}&checkplaceSetIdx=2&actCd=R"


def parse(html, no):
    soup = BeautifulSoup(html, "html.parser")
    rows = [tr for tr in soup.select("table tr") if tr.find(["th", "td"])]
    out = {"no": no, "url": url_of(no), "title": rows[0].get_text(" ", strip=True) if rows else None}
    seen_review = False
    for tr in rows[1:]:
        cells = tr.find_all(["th", "td"], recursive=False)
        label = cells[0].get_text(strip=True) if cells else ""
        vals = [cell_text(c) for c in cells[1:]]
        if label == "검토결과":
            out["result"] = vals[-1].strip() if vals else None
            seen_review = True
        elif label == "사유":
            out["review_reason" if seen_review and "review_reason" not in out else "close_reason"] = (vals[-1] if vals else "").strip()
        elif label in FIELDS:
            out[FIELDS[label]] = (vals[-1] if vals else "").strip() or None
    return out


def download_all():
    lst = json.load(open(RAW / "list_total.json", encoding="utf-8"))
    rows = [r for r in next(v for v in lst.values() if isinstance(v, list)) if r["pastreqType"] == "현장건의 과제"]
    out_dir = RAW / "exmnt"
    out_dir.mkdir(parents=True, exist_ok=True)
    new, skip, failed = 0, 0, []
    for i, r in enumerate(rows, 1):
        path = out_dir / f"{r['dataIdx']}.html"
        if path.exists() and path.stat().st_size > 5000:
            skip += 1
            continue
        try:
            resp = request("GET", url_of(r["dataIdx"]))
            resp.encoding = "utf-8"
            if "건의내용" not in resp.text:
                raise ValueError("상세 표가 없음")
            path.write_text(resp.text, encoding="utf-8")
            new += 1
        except Exception as e:          # 한 건 실패로 전체를 멈추지 않는다
            failed.append({"no": r["dataIdx"], "title": r["title"], "error": str(e)[:200]})
        if i % 500 == 0:
            print(f"{i}/{len(rows)} 새로 {new}, 건너뜀 {skip}, 실패 {len(failed)}", flush=True)
    (RAW / "exmnt_failed.json").write_text(json.dumps(failed, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest("x1_exmnt_download_all", target=len(rows), new=new, skipped=skip, failed=len(failed), requests=common.n_requests)


def main():
    if "--all" in sys.argv:
        return download_all()
    lst = json.load(open(RAW / "list_total.json", encoding="utf-8"))
    rows = next(v for v in lst.values() if isinstance(v, list))
    target = [r for r in rows if r["pastreqType"] == "현장건의 과제" and any(k in r["title"] for k in KEYWORDS)]
    out_dir = RAW / "exmnt"
    out_dir.mkdir(parents=True, exist_ok=True)
    items, failed, new = [], [], 0
    for r in target:
        path = out_dir / f"{r['dataIdx']}.html"
        try:
            if not path.exists():
                resp = request("GET", url_of(r["dataIdx"]))
                resp.encoding = "utf-8"
                if "건의내용" not in resp.text:
                    raise ValueError("상세 표가 없음")
                path.write_text(resp.text, encoding="utf-8")
                new += 1
            x = parse(path.read_text(encoding="utf-8"), r["dataIdx"])
            x["list_title"], x["list_reply_date"] = r["title"], r["replyRegDate"]
            x["articles"] = sorted(set(ARTICLE.findall((x.get("proposal") or "") + (x.get("review_reason") or ""))))
            items.append(x)
        except Exception as e:          # 한 건 실패로 전체를 멈추지 않는다
            failed.append({"no": r["dataIdx"], "error": str(e)[:200]})
    write_jsonl("exmnt_pilot.jsonl", items)
    write_csv("exmnt_pilot.csv", items, ["no", "title", "reply_date", "category", "dept", "result", "proposal", "review_reason", "plan", "close_reason", "articles", "url"])
    (OUT / "exmnt_pilot_failed.json").write_text(json.dumps(failed, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest("x1_exmnt_pilot", target=len(target), parsed=len(items), new=new, failed=len(failed), requests=common.n_requests)


if __name__ == "__main__":
    main()
