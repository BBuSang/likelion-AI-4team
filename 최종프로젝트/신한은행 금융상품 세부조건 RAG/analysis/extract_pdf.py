"""data/docs/*.pdf 에서 페이지별 텍스트를 뽑아 data/text/{파일명}.txt 로 저장하고 품질 지표를 출력한다.

품질 지표: 페이지 수, 글자 수, 한글 비율(스캔/깨짐 확인), 이미지만 있는 페이지 수.
"""
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "data" / "docs", ROOT / "data" / "text"


def hangul_ratio(s: str) -> float:
    chars = [c for c in s if not c.isspace()]
    return sum("가" <= c <= "힣" for c in chars) / max(len(chars), 1)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{'파일':40} 쪽  글자수  한글비율  빈쪽")
    for pdf in sorted(SRC.glob("*.pdf")):
        doc = pymupdf.open(pdf)
        pages = [p.get_text("text") for p in doc]
        text = "".join(f"\n\n=== p.{i + 1} ===\n{t}" for i, t in enumerate(pages))
        out = OUT / f"{pdf.stem}.txt"
        # 이미지뿐인 PDF(쪽당 글자 150 미만)는 analysis/ocr_pdf.py 결과를 덮어쓰지 않는다
        if out.exists() and len(text) < 150 * len(pages) and len(out.read_text()) > 2 * len(text):
            print(f"  [유지] {pdf.stem}: 텍스트 레이어가 거의 없음 → 기존 OCR 결과 유지")
            continue
        out.write_text(text)
        empty = sum(len(t.strip()) < 20 for t in pages)
        print(f"{pdf.stem[:40]:40} {len(pages):2} {len(text):7} {hangul_ratio(text):8.2f} {empty:5}")
