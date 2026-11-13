"""03 파싱: 상세 HTML → 항목별 원문. 답변유형(HTML 표기)과 첨부 필요 여부를 판단한다.
입력: processed/case_list.jsonl, raw/detail/*.html   출력: processed/cases_parsed.jsonl
"""
import re
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup, CData, Comment, Declaration, Doctype, ProcessingInstruction

from common import RAW, SECS, manifest, read_jsonl, write_jsonl
from s02_details import url_of
from text_rules import ANSWER_TYPE, answer_type_of

LABEL = {"질의요지": "question", "회답": "answer", "이유": "reason"}
ATT_REF = re.compile(r"(첨부|붙임)[^\n]{0,20}(참조|참고)")


def attachment_check(body, soup, status):
    """본문이 첨부에만 있는가: 처리구분 완료 + 첨부 있음 + (세 칸 합 200자 미만 또는 칸 앞부분에 안내 문구)."""
    secs = [re.sub(r"\s+", " ", body[k].get_text(" ", strip=True)) if body.get(k) is not None else ""
            for k in LABEL]
    links = []
    for a in soup.select("a[href*=displayFile]"):
        q = parse_qs(urlparse(a["href"]).query)
        links.append({k: q.get(k, [""])[0] for k in ("filePath", "orgFileName", "sysFileName")})
    total = sum(map(len, secs))
    ref = any(ATT_REF.search(x[:80]) for x in secs)
    return {"need": status == "완료" and bool(links) and (total < 200 or ref),
            "reason": "안내문구" if ref else "본문짧음", "body_chars": total, "links": links}


def cell_text(td):
    """두 종류의 칸이 있다.
    - 단순 글 (태그 없이 줄바꿈으로 문단 구분, 주로 옛날 건): 원래 줄바꿈을 살린다.
    - 한글·워드 붙여넣기 HTML (<p>·<span>·<br>): 조각 사이 줄바꿈은 소스 정리용이므로
      줄바꿈만 있는 조각은 지우고 나머지 줄바꿈은 공백으로 바꾼다. <br>·문단 끝에서만 줄을 나눈다."""
    # 화면에 안 보이는 노드는 버린다: HTML 주석(<!--rhwp-studio-clipboard:…-->, <!--[if …]-->, [data-hwpjson]),
    # 처리지시문(<?xml:namespace …>), 선언문. 이걸 남기면 편집기 표식이 본문에 섞인다.
    for n in td.find_all(string=lambda x: isinstance(x, (Comment, ProcessingInstruction, Declaration, Doctype, CData))):
        n.extract()
    plain = td.find(["p", "br", "div", "span", "li", "table"]) is None
    for s in td.find_all(string=True):
        t = str(s).replace("\r\n", "\n").replace("\r", "\n")
        if plain:
            s.replace_with(re.sub(r"[ \t]+", " ", t))
        else:
            s.replace_with("" if (not t.strip() and "\n" in t) else re.sub(r"[ \t\n]+", " ", t))
    for br in td.find_all("br"):
        br.replace_with("\n")
    for p in td.find_all(["p", "div", "li", "tr"]):
        p.insert_after("\n")
    return td.get_text()


def parse_detail(html):
    a = html.find('<div class="sub-con">')
    b = html.find("<footer", a)
    soup = BeautifulSoup(html[a:b if b > 0 else None], "html.parser")
    head = {th.get_text(strip=True): th.find_next_sibling("td") for th in soup.select("table.tbl-view th")}
    body = {th.get_text(strip=True): th.find_next_sibling("td") for th in soup.select("table.tbl-write th")}
    g = lambda d, k: d[k].get_text(" ", strip=True) if d.get(k) is not None else None
    rec = {
        "detail_title": (soup.select_one("table.tbl-view td.subject") or soup).get_text(" ", strip=True),
        "reply_title": (soup.select_one("table.tbl-write td.subject") or soup).get_text(" ", strip=True),
        "status": g(body, "처리구분"), "is_public": g(body, "공개여부"),
        "registrant": g(body, "등록자"), "reply_date": g(body, "회신일") or None,
        "depts": [d.strip() for d in (g(head, "소관부서") or "").split(",") if d.strip()],
    }
    rec["_att"] = attachment_check(body, soup, rec["status"])   # cell_text가 표를 바꾸기 전에 판단
    for lab, key in LABEL.items():
        rec[key] = cell_text(body[lab]) if body.get(lab) is not None else ""
    rec["attachments"] = []
    for x in soup.select("a[href*=displayFile]"):
        q = parse_qs(urlparse(x["href"]).query)
        name = q.get("orgFileName", [""])[0]
        rec["attachments"].append({"org_file_name": name, "ext": name.rsplit(".", 1)[-1].lower() if "." in name else ""})
    return rec


def main():
    rows = []
    for r in read_jsonl("case_list.jsonl"):
        d = parse_detail((RAW / "detail" / f"{r['case_key']}.html").read_text(encoding="utf-8"))
        rec = {**r, **d, "reply_date_source": "detail" if d["reply_date"] else None, "url": url_of(r),
               "body_source": "html", "answer_type": None, "answer_type_raw": None, "notes": []}
        tk = r["reply_date_list"]
        if rec["reply_date"] and tk and rec["reply_date"] != tk:
            rec["notes"].append(f"회신일 상세 {rec['reply_date']} ≠ 통합조회 {tk}")
        if m := ANSWER_TYPE.search("\n".join(rec[s] for s in SECS)):
            rec["answer_type_raw"] = m.group(1).strip()
            rec["answer_type"] = answer_type_of(m.group(1))
            if rec["answer_type"] is None:
                rec["notes"].append("답변유형 값이 정해진 목록에 없음")
        rows.append(rec)
    write_jsonl("cases_parsed.jsonl", rows)
    manifest("03_parse", cases=len(rows), need_attachment=sum(r["_att"]["need"] for r in rows))


if __name__ == "__main__":
    main()
