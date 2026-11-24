"""표 재추출 2단계: 표·도식이 있는 쪽을 '이미지 + 원문 글자' 방식(s39 grounded)으로 md 로 다시 뽑아 data/text_v2 를 만든다
대상 쪽: 3열 이상 표가 잡히는 쪽(PyMuPDF find_tables) ∪ 지금 청크가 깨진 쪽(짧은 줄 8줄 이상 연속). D238(OCR·잠금 시험 문서)은 뺌
자동 통과 기준 (둘 다 만족해야 md 를 씀, 아니면 그 쪽은 data/text 원래 글 유지)
  1) 빠진 글자 0: 글자·숫자·%·. 만 남겨 개수 대조
  2) 지어낸 낱말 0: md 의 낱말(공백·표 기호로 나눔)이 모두 원문(공백 없앤 글)에 들어 있어야 함 → 순서 바꿈·머리글 풀어 쓰기·중복은 허용
사용
  python analysis/s40_table_extract.py run      # API 호출 (이미 뽑은 쪽은 건너뜀)
  python analysis/s40_table_extract.py build    # 검사 + data/text_v2 생성 + 보고서
출력: data/stage/table_v2/pages/{doc_id}_p{쪽}.md, data/stage/table_v2/report.csv, data/text_v2/{파일명}.txt
원본 data/text 는 건드리지 않음
"""
import collections
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s39_table_trial as trial  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGE = DATA / "stage" / "table_v2"
PAGES = STAGE / "pages"
TEXT, TEXT_V2 = DATA / "text", DATA / "text_v2"
SKIP_DOCS = {"D238"}
KEEP = trial.KEEP


def docs():
    return list(csv.DictReader(open(DATA / "meta" / "doc_survey.csv")))


def broken_pages() -> dict[str, set[int]]:
    bad = collections.defaultdict(set)
    for line in open(DATA / "chunks" / "chunks.jsonl"):
        d = json.loads(line)
        run = best = 0
        for x in d["content"].split("\n"):
            run = run + 1 if 0 < len(x.strip()) <= 6 else 0
            best = max(best, run)
        if best >= 8:
            for p in range(d["page"], (d.get("page_end") or d["page"]) + 1):
                bad[d["doc_id"]].add(p)
    return bad


def targets() -> list[tuple[str, str, int]]:
    bad = broken_pages()
    out = []
    for r in docs():
        if r["doc_id"] in SKIP_DOCS:
            continue
        doc = pymupdf.open(DATA / "docs" / f"{r['title']}.pdf")
        for i, page in enumerate(doc):
            if i + 1 in bad.get(r["doc_id"], ()) or any(t.col_count >= 3 for t in page.find_tables().tables):
                out.append((r["doc_id"], r["title"], i + 1))
    return out


def run():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    client, model = OpenAI(), os.environ["CHAT_MODEL"]
    PAGES.mkdir(parents=True, exist_ok=True)
    todo = [t for t in targets() if not (PAGES / f"{t[0]}_p{t[2]}.md").exists()]
    print(f"남은 쪽 {len(todo)}개")

    def one(t):
        doc_id, title, pn = t
        doc = pymupdf.open(DATA / "docs" / f"{title}.pdf")
        for attempt in range(3):
            try:
                md = trial.by_vision(doc[pn - 1], client, model, grounded=True)
                (PAGES / f"{doc_id}_p{pn}.md").write_text(md + "\n")
                return
            except Exception as e:  # noqa: BLE001 — 네트워크·속도 제한은 다시 시도
                print(f"  {doc_id} p{pn} 실패 {attempt + 1}: {e}")
                time.sleep(5 * (attempt + 1))

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(one, todo))


def check(src: str, md: str) -> dict:
    s_chars, o_chars = collections.Counter(KEEP.findall(src)), collections.Counter(KEEP.findall(md.replace("<br>", " ")))
    missing = s_chars - o_chars
    compact = "".join(KEEP.findall(src))
    words = re.split(r"[\s|]+|<br>|-{3,}", md)
    invented = [w for w in ("".join(KEEP.findall(x)) for x in words) if w and w not in compact]
    return {"missing": sum(missing.values()), "missing_ex": "".join(sorted(missing.elements()))[:30],
            "invented": len(invented), "invented_ex": " ".join(invented[:5]),
            "extra": sum((o_chars - s_chars).values())}


def split_pages(text: str) -> dict[int, str]:
    parts = re.split(r"\n\n=== p\.(\d+) ===\n", text)
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}


def build():
    TEXT_V2.mkdir(parents=True, exist_ok=True)
    tg = collections.defaultdict(set)
    for doc_id, _, pn in targets():
        tg[doc_id].add(pn)
    rows = []
    for r in docs():
        src_file = TEXT / f"{r['title']}.txt"
        pages = split_pages(src_file.read_text())
        doc = pymupdf.open(DATA / "docs" / f"{r['title']}.pdf")
        for pn in sorted(tg.get(r["doc_id"], ())):
            f = PAGES / f"{r['doc_id']}_p{pn}.md"
            if not f.exists():
                rows.append({"doc_id": r["doc_id"], "page": pn, "status": "없음"})
                continue
            md = f.read_text()
            c = check(doc[pn - 1].get_text("text"), md)
            ok = c["missing"] == 0 and c["invented"] == 0
            rows.append({"doc_id": r["doc_id"], "page": pn, "status": "통과" if ok else "불통과", **c})
            if ok:
                pages[pn] = md.rstrip("\n") + "\n"
        (TEXT_V2 / src_file.name).write_text("".join(f"\n\n=== p.{n} ===\n{t}" for n, t in sorted(pages.items())))
    keys = ["doc_id", "page", "status", "missing", "invented", "extra", "missing_ex", "invented_ex"]
    with open(STAGE / "report.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    cnt = collections.Counter(x["status"] for x in rows)
    print(f"대상 {len(rows)}쪽: {dict(cnt)}  문서 {len(tg)}개  → {TEXT_V2}")


if __name__ == "__main__":
    {"run": run, "build": build}[sys.argv[1]]()
