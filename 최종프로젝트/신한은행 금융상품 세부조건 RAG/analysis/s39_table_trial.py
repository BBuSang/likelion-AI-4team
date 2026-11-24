"""표 재추출 1단계: 깨진 쪽 10개를 두 방법으로 md 로 뽑아 비교 (원본 data/text 는 건드리지 않음)
방법
  pymupdf : find_tables(선 기준) → 3열 이상 표는 md 표, 칸 글자는 칸 영역에서 다시 뽑음. 나머지 글은 블록 순서대로
  vision  : 쪽 이미지를 OpenAI 비전 모델(CHAT_MODEL)에 주고 같은 규칙으로 md 를 받음
  grounded: 쪽 이미지 + PyMuPDF 원문 글자를 함께 주고, 원문 글자만 써서 배치만 md 로 바꾸게 함 (vision 이 낱말을 바꿔 써서)
검사: 글자·숫자만 남겨 원문(pymupdf get_text)과 개수 대조 → 빠진 글자 / 새로 생긴 글자
사용: python analysis/s39_table_trial.py pymupdf | vision | grounded | check
출력: data/stage/table_trial/{method}/{doc_id}_p{쪽}.md, data/stage/table_trial/check.csv
"""
import base64
import collections
import csv
import os
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "stage" / "table_trial"

# (doc_id, 쪽, 무엇을 보려고)
PAGES = [
    ("D038", 5, "금리 결정 도식 (선 없는 그림)"),
    ("D038", 6, "상환방법별 원리금 예시표 (색 상자 표)"),
    ("D195", 7, "공유형 모기지 정산표 (24x16, 병합 머리글)"),
    ("D072", 1, "골드바 기준가격 식·마진율 표"),
    ("D206", 3, "약관 특약사항 속 표"),
    ("D066", 1, "대출거래약정서 머리말 표"),
    ("D003", 1, "표형 설명서 중도해지 이자율"),
    ("D009", 1, "표형 설명서 기본·우대이자율 (소수점 깨짐)"),
    ("D235", 4, "청약 예치기준금액 표"),
    ("D045", 1, "투자상품 수익구조 표"),
]

RULES = """규칙
- 원문 글자를 바꾸거나 요약·보충하지 말 것. 보이는 글자만, 보이는 순서대로.
- 3열 이상 표 → 마크다운 표. 병합된 머리글은 위아래 머리글을 합쳐 칸마다 풀어 씀 (예: '담보인정비율 70% 자녀0').
- 칸 안의 여러 줄은 공백으로 이음. ○ ※ • 같은 글머리표로 시작하는 줄은 <br> 로 구분.
- 2열 표('항목 | 내용' 배치)는 표로 만들지 말고 '항목' 한 줄, 그 아래 내용 줄로 씀.
- 표가 아닌 도식(화살표·상자 그림)은 그림 속 글자만 써서 한 줄 식으로 (예: 최종금리 = 기준금리 + 가산금리(…)).
- 쪽 머리·꼬리의 반복 문구(쪽 번호 등)도 그대로 두기. 설명·머리말 없이 md 본문만 출력."""


def pdf_of(doc_id: str) -> Path:
    for r in csv.DictReader(open(DATA / "meta" / "doc_survey.csv")):
        if r["doc_id"] == doc_id:
            return DATA / "docs" / f"{r['title']}.pdf"
    raise KeyError(doc_id)


def cell_text(page, rect) -> str:
    if rect is None:
        return ""
    lines = [ln.strip() for ln in page.get_text("text", clip=pymupdf.Rect(rect)).splitlines() if ln.strip()]
    out = ""
    for ln in lines:
        if out and re.match(r"^[○※•■◼·\-]", ln):
            out += "<br>" + ln
        else:
            out += (" " if out else "") + ln
    return out.replace("|", "\\|")


def table_md(page, tb) -> str:
    rows = [[cell_text(page, c) for c in row.cells] for row in tb.rows]
    raw = [row.cells for row in tb.rows]
    # 머리글 두 줄의 병합 칸(None)만 왼쪽 값을 이어받음 (본문 병합 칸은 빈칸으로 둠: 긴 글이 칸마다 복사되지 않게)
    for r, cells in list(zip(rows, raw))[:2]:
        for j in range(1, len(r)):
            if cells[j] is None:
                r[j] = r[j - 1]
    # 첫 줄에 병합이 있으면 머리글 두 줄을 합침
    head = rows[0]
    body = rows[1:]
    if any(c is None for c in raw[0]) and len(rows) > 1:
        head = [f"{a} {b}".strip() if a != b else a for a, b in zip(rows[0], rows[1])]
        body = rows[2:]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(md)


def by_pymupdf(page) -> str:
    tabs = [t for t in page.find_tables().tables if t.col_count >= 3]
    boxes = [pymupdf.Rect(t.bbox) for t in tabs]
    items = []  # (y, x, text)
    for b in page.get_text("blocks"):
        r = pymupdf.Rect(b[:4])
        if b[6] != 0 or any(r.intersects(bx) and (r & bx).get_area() > 0.5 * r.get_area() for bx in boxes):
            continue
        items.append((r.y0, r.x0, b[4].strip()))
    for t, bx in zip(tabs, boxes):
        items.append((bx.y0, bx.x0, table_md(page, t)))
    items.sort(key=lambda x: (round(x[0]), x[1]))
    return "\n\n".join(t for _, _, t in items if t)


GROUNDED = """아래 [원문 글자]는 이 쪽의 PDF 글자 층을 그대로 뽑은 것입니다. 글자는 정확하지만 표·도식의 배치가 깨져 있습니다.
쪽 이미지를 보고 배치만 바로잡아 마크다운으로 다시 써 주세요.
- 반드시 [원문 글자]에 있는 글자만 쓸 것. 이미지와 원문이 다르게 보여도 원문 글자를 따름 (오타도 그대로).
- 원문 글자를 빠짐없이 한 번씩 쓸 것 (쪽 가장자리 인쇄 코드 포함). 이미지에만 있는 글자(로고 등)는 쓰지 말 것.
- 표 칸의 구분 기호(|, ---, <br>)와 줄바꿈·공백만 더할 수 있음.
- 원문에 없는 표 머리글(예: '항목', '내용')을 만들지 말 것. 머리글 줄이 없는 표는 첫 행을 머리글 자리에 씀.
"""


def by_vision(page, client, model, grounded=False) -> str:
    png = page.get_pixmap(dpi=150).tobytes("png")
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    if grounded:
        prompt = GROUNDED + RULES + "\n\n[원문 글자]\n" + page.get_text("text")
    else:
        prompt = "이 은행 상품 문서 한 쪽을 마크다운으로 옮겨 적어 주세요.\n" + RULES
    res = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": url}},
        ]}],
    )
    u = res.usage
    print(f"    토큰 입력 {u.prompt_tokens} 출력 {u.completion_tokens}")
    return res.choices[0].message.content.strip()


KEEP = re.compile(r"[0-9A-Za-z가-힣%.]")


def chars(s: str) -> collections.Counter:
    s = s.replace("<br>", "")
    return collections.Counter(KEEP.findall(s))


def check():
    rows = []
    for doc_id, pn, what in PAGES:
        doc = pymupdf.open(pdf_of(doc_id)); page = doc[pn - 1]
        src = chars(page.get_text("text"))
        for m in ("pymupdf", "vision", "grounded"):
            f = OUT / m / f"{doc_id}_p{pn}.md"
            if not f.exists():
                continue
            out = chars(f.read_text())
            miss, extra = src - out, out - src
            dig = lambda c: sum(v for k, v in c.items() if k.isdigit() or k in "%.")  # noqa: E731
            rows.append({
                "doc": f"{doc_id}_p{pn}", "what": what, "method": m, "src_chars": sum(src.values()),
                "missing": sum(miss.values()), "extra": sum(extra.values()),
                "missing_num": dig(miss), "extra_num": dig(extra),
                "missing_ex": "".join(sorted(miss.elements()))[:40], "extra_ex": "".join(sorted(extra.elements()))[:40],
            })
    with open(OUT / "check.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['doc']:10} {r['method']:8} 원문 {r['src_chars']:5}  빠짐 {r['missing']:4} (숫자 {r['missing_num']:3})  "
              f"생김 {r['extra']:4} (숫자 {r['extra_num']:3})  {r['missing_ex'][:20]!r} / {r['extra_ex'][:20]!r}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "check":
        check()
        sys.exit()
    (OUT / mode).mkdir(parents=True, exist_ok=True)
    client = model = None
    if mode in ("vision", "grounded"):
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(ROOT / ".env")
        client, model = OpenAI(), os.environ["CHAT_MODEL"]
    for doc_id, pn, what in PAGES:
        print(f"{doc_id} p{pn} {what}")
        doc = pymupdf.open(pdf_of(doc_id)); page = doc[pn - 1]
        md = by_pymupdf(page) if mode == "pymupdf" else by_vision(page, client, model, grounded=mode == "grounded")
        (OUT / mode / f"{doc_id}_p{pn}.md").write_text(md + "\n")
