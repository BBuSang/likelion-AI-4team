"""2-1 문서 형식 조사: data/text/*.txt 의 구조 단서를 측정해 형식을 분류한다 (LLM 없음).

출력: data/meta/doc_survey.csv (문서별 특징 + format) 와 요약 출력.
format:
  article   조문형   — 약관·특약·약정서: '제N조' 머리 줄이 2개 이상
  table     표형     — 설명서: '구분' 표 / '■ 기본 정보' 같은 항목 표
  numbered  목차형   — 설명서: 단독 번호 줄(1, 2, …) + 제목 (대출 공통 설명서)
  invest    투자설명서형
  other     위에 해당 없음 (눈으로 확인 필요)
예외 표시: dup_copy(한 파일에 사본 2벌), no_space(띄어쓰기 손실), ocr(텍스트 거의 없음),
          summary(문서 끝에 '요약 상품설명서' 쪽이 붙어 있음, summary_page=그 쪽 번호)
"""
import collections
import csv
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT, META = ROOT / "data" / "text", ROOT / "data" / "meta"

ARTICLE = re.compile(r"^\s*제\s*\d+\s*조(?:의\s*\d+)?\s*[(（【「]?\s*[가-힣]", re.M)
GUBUN = re.compile(r"^\s*구\s*분\s*$", re.M)
SQUARE = re.compile(r"^\s*■\s*\S", re.M)
NUM_ONLY = re.compile(r"^\s*\d{1,2}\s*$", re.M)
REVIEW = re.compile(r"(?:준법감시인|심의필|심사필)[^\n]{0,30}?제?\s*([\d-]{6,}[-\d]*\s*호?)")
BASE_DATE = re.compile(r"(\d{4})\s*[.]\s*(\d{1,2})\s*[.]\s*(\d{1,2})\.?\s*\)?\s*(?:현재|기준)")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def features(text: str) -> dict:
    pages = text.split("=== p.")[1:]
    body = "\n".join(p.split("\n", 1)[1] if "\n" in p else "" for p in pages)
    chars = [c for c in body if not c.isspace()]
    hangul = sum("가" <= c <= "힣" for c in chars)
    # 띄어쓰기 손실: 한글 대비 공백 비율 (보통 문서는 0.2 이상)
    space_ratio = body.count(" ") / max(hangul, 1)
    # 사본 중복: 20자 이상 줄 중 2번 이상 나오는 줄의 비율
    long_lines = [re.sub(r"\s+", " ", l).strip() for l in body.split("\n")]
    long_lines = [l for l in long_lines if len(l) >= 20]
    cnt = collections.Counter(long_lines)
    dup_ratio = sum(1 for l in long_lines if cnt[l] >= 2) / max(len(long_lines), 1)
    # '요약 상품설명서'는 문서 끝 쪽에 붙어 있음 → 그 쪽 번호(1부터)
    summary_page = next((i + 1 for i, p in enumerate(pages)
                         if re.search(r"요약\s*상품\s*설명서", p[:300])), 0)
    return {
        "pages": len(pages),
        "chars": len(chars),
        "chars_per_page": round(len(chars) / max(len(pages), 1)),
        "space_ratio": round(space_ratio, 3),
        "dup_ratio": round(dup_ratio, 2),
        "n_article": len(ARTICLE.findall(body)),
        "n_gubun": len(GUBUN.findall(body)),
        "n_square": len(SQUARE.findall(body)),
        "n_numonly": len(NUM_ONLY.findall(body)),
        "summary_page": summary_page,
        "appendix": int("<부록>" in body or "부록" in body[-3000:]),
        "review_no": (REVIEW.search(body).group(1).strip() if REVIEW.search(body) else ""),
        "base_date": ("-".join(f"{int(x):02d}" for x in BASE_DATE.search(body).groups())
                      if BASE_DATE.search(body) else ""),
    }


def classify(f: dict, doc_type: str) -> str:
    if doc_type in ("투자설명서", "간이투자설명서"):
        return "invest"
    if doc_type in ("약관", "특약", "약정서") and f["n_article"] >= 2:
        return "article"
    if doc_type == "상품설명서":
        # 대출 공통 설명서는 '■'와 '구분'도 많지만 단독 번호 목차(1~11)가 뼈대라 먼저 판정
        if f["n_numonly"] >= 10:
            return "numbered"
        if f["n_gubun"] >= 1 or f["n_square"] >= 3:
            return "table"
    if f["n_article"] >= 2:
        return "article"
    return "other"


def main() -> None:
    docs = list(csv.DictReader(open(META / "documents.csv")))
    rows = []
    for d in docs:
        t = nfc((TEXT / f"{Path(d['file']).stem}.txt").read_text())
        f = features(t)
        f["format"] = classify(f, d["doc_type"])
        f["dup_copy"] = int(f["dup_ratio"] >= 0.6)
        f["no_space"] = int(f["space_ratio"] < 0.15 and f["chars"] > 500)
        f["summary"] = int(f["summary_page"] > 0)
        f["ocr"] = int(f["chars_per_page"] < 150)
        rows.append({"doc_id": d["doc_id"], "title": d["title"], "doc_type": d["doc_type"],
                     "n_products": d["n_products"], **f})

    with open(META / "doc_survey.csv", "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("형식 × 문서 종류")
    tab = collections.Counter((r["format"], r["doc_type"]) for r in rows)
    for (fm, dt), n in sorted(tab.items()):
        print(f"  {fm:9} {dt:8} {n}")
    for flag in ("dup_copy", "no_space", "ocr", "summary", "appendix"):
        hit = [r["title"] for r in rows if r[flag]]
        print(f"{flag:9} {len(hit):3}  {hit[:6]}")
    print("심의필 번호 찾음", sum(bool(r["review_no"]) for r in rows),
          "/ 기준일 찾음", sum(bool(r["base_date"]) for r in rows))
    print("other:", [r["title"] for r in rows if r["format"] == "other"])


if __name__ == "__main__":
    main()
