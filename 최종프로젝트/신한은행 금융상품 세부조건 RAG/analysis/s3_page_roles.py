"""2-3 쪽 역할 판정: 사본·요약 상품설명서·부록 쪽을 본문에서 분리 — LLM 없음.

입력: data/stage/s2/{doc_id}.txt, data/meta/doc_survey.csv (summary_page, ocr)
출력: data/stage/s3/{doc_id}.txt  — 본문(body) 쪽만, 쪽 표시 유지
      data/meta/s3_pages.csv       — doc_id, page, role, ref_page
role:
  body      본문 (청크·LLM 추출 대상)
  dup_copy  앞쪽 어떤 쪽과 줄 집합 유사도(Jaccard) ≥ 0.9 이고 새 줄이 2개 이하 → 사본 (ref_page = 원본 쪽)
  summary   '○○ 요약 상품설명서' 쪽과, 제목 없이 이어지는 다음 쪽
  appendix  '<부록>'으로 시작하는 쪽 (용어 안내 → 2-7에서 용어집으로)
'<별지>'처럼 요약 뒤에 오는 다른 제목 쪽은 본문으로 둔다. OCR 대상 문서는 판정하지 않고 모두 body.
data/meta/s3_pages_ref.csv 가 있으면 판정하지 않고 그 역할을 그대로 씀: 쪽 역할은 PDF 의 성질이라,
표 재추출(text_v2)처럼 원본 쪽과 사본 쪽의 줄 배치가 따로 바뀌어도 원래 글로 판정한 결과를 유지 (s40)
"""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S2, S3, META = (ROOT / "data" / "stage" / "s2", ROOT / "data" / "stage" / "s3",
                ROOT / "data" / "meta")
DUP_TH = 0.9
MAX_NEW = 2   # 원본 쪽에 없는 줄이 이보다 많으면 사본이 아님 (예: 기금 약정서 p.9 추가약정서 다른 판)
PAGE = re.compile(r"=== p\.(\d+) ===(.*?)(?==== p\.|\Z)", re.S)


def line_set(body: str) -> set[str]:
    return {s for s in (re.sub(r"\s+", "", l) for l in body.split("\n")) if len(s) >= 6}


def first_line(body: str) -> str:
    return next((re.sub(r"\s+", "", l) for l in body.split("\n") if l.strip()), "")


def roles(pages: list[tuple[int, str]], summary_page: int, ocr: bool):
    out, sets = [], [line_set(b) for _, b in pages]
    in_summary = False
    for i, (pno, body) in enumerate(pages):
        role, ref = "body", ""
        head = first_line(body)
        if not ocr:
            best = max(((len(sets[i] & sets[j]) / max(len(sets[i] | sets[j]), 1), pages[j][0])
                        for j in range(i)), default=(0.0, 0))
            if head.startswith("<부록>"):
                role, in_summary = "appendix", False
            elif summary_page and pno == summary_page:
                role, in_summary = "summary", True
            elif in_summary and not head.startswith("<"):
                role = "summary"          # 제목 없이 이어지는 요약 쪽
            elif best[0] >= DUP_TH and sets[i] and len(sets[i] - sets[[p for p, _ in pages].index(best[1])]) <= MAX_NEW:
                role, ref = "dup_copy", best[1]
            else:
                in_summary = False
        out.append((pno, role, ref))
    return out


def main() -> None:
    S3.mkdir(parents=True, exist_ok=True)
    survey = {r["doc_id"]: r for r in csv.DictReader(open(META / "doc_survey.csv"))}
    rows, fixed = [], None
    if (META / "s3_pages_ref.csv").exists():
        fixed = {}
        for r in csv.DictReader(open(META / "s3_pages_ref.csv")):
            fixed.setdefault(r["doc_id"], []).append((int(r["page"]), r["role"], r["ref_page"]))
        print("쪽 역할: s3_pages_ref.csv 를 그대로 씀")
    for did, sv in survey.items():
        text = (S2 / f"{did}.txt").read_text()
        pages = [(int(m.group(1)), m.group(2)) for m in PAGE.finditer(text)]
        rl = fixed[did] if fixed else roles(pages, int(sv["summary_page"]), sv["ocr"] == "1")
        keep = "".join(f"=== p.{pno} ==={body}" for (pno, body), (_, role, _) in zip(pages, rl)
                       if role == "body")
        (S3 / f"{did}.txt").write_text(keep)
        rows += [(did, pno, role, ref) for pno, role, ref in rl]
    with open(META / "s3_pages.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "page", "role", "ref_page"])
        w.writerows(rows)
    from collections import Counter
    print("쪽 역할:", Counter(r[2] for r in rows))
    print("역할별 문서 수:", {k: len({r[0] for r in rows if r[2] == k})
                          for k in ("dup_copy", "summary", "appendix")})


if __name__ == "__main__":
    main()
