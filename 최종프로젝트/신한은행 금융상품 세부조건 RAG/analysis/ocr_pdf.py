"""이미지뿐인 PDF를 macOS Vision OCR로 텍스트화 (로컬 실행, 외부 전송 없음)

사용: python analysis/ocr_pdf.py D###  [--dpi 300]
입력: data/docs/{file}.pdf (documents.csv의 doc_id로 찾음)
출력: data/text/{file}.txt (기존 추출본은 data/text_ocr_backup/에 보관 후 덮어씀)
      data/ocr/{doc_id}_ocr.json (줄별 신뢰도·위치 — 품질 검수용)
형식: 다른 문서와 같게 '=== p.N ===' 쪽 표시 + 줄 단위 텍스트 (위→아래, 같은 줄은 왼→오 순서)
"""
import argparse
import csv
import json
import re
import shutil
import unicodedata
from pathlib import Path

import pymupdf
import Quartz
import Vision
from Foundation import NSData

ROOT = Path(__file__).resolve().parents[1]
DOCS, TEXT, META = ROOT / "data" / "docs", ROOT / "data" / "text", ROOT / "data" / "meta"


def ocr_png(png: bytes):
    data = NSData.dataWithBytes_length_(png, len(png))
    src = Quartz.CGImageSourceCreateWithData(data, None)
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLanguages_(["ko-KR", "en-US"])
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(True)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise RuntimeError(err)
    lines = []
    for obs in req.results() or []:
        cand = obs.topCandidates_(1)[0]
        bb = obs.boundingBox()  # 정규화 좌표, 원점 왼쪽 아래
        lines.append({"text": str(cand.string()), "conf": float(cand.confidence()),
                      "x": bb.origin.x, "y": 1 - (bb.origin.y + bb.size.height),
                      "h": bb.size.height})
    return lines


# 내용이 아닌 줄: 쪽 맨 아래 다운로드 워터마크(다운로드한 사람의 가려진 ID·이름·시각 = 개인 식별 정보 조각)와
# 배경 대각선 워터마크 'Shinhan One'. 위치(하단 4%)와 문구로 제거한다.
FOOTER_Y = 0.96
NOISE = re.compile(r"(\*{2,}|\d{6}\s*/\s*\S\*\S|무단\s*반출|모니터링|^\s*(Shinhan|One|Shinhan One)\s*$)")


def is_noise(l) -> bool:
    return l["y"] > FOOTER_Y or bool(NOISE.search(l["text"]))


def reading_order(lines):
    """위→아래, 같은 높이(줄 높이의 절반 이내)면 왼→오"""
    lines = sorted(lines, key=lambda l: (l["y"], l["x"]))
    rows, cur = [], []
    for l in lines:
        if cur and abs(l["y"] - cur[0]["y"]) > max(cur[0]["h"], l["h"]) * 0.5:
            rows.append(sorted(cur, key=lambda c: c["x"]))
            cur = []
        cur.append(l)
    if cur:
        rows.append(sorted(cur, key=lambda c: c["x"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("doc_id")
    ap.add_argument("--dpi", type=int, default=300)
    a = ap.parse_args()
    doc = next(d for d in csv.DictReader(open(META / "documents.csv")) if d["doc_id"] == a.doc_id)
    pdf = next(p for p in DOCS.glob("*.pdf") if unicodedata.normalize("NFC", p.name) == unicodedata.normalize("NFC", doc["file"]))
    txt = TEXT / f"{pdf.stem}.txt"

    pages_out, detail = [], []
    for i, page in enumerate(pymupdf.open(pdf), 1):
        png = page.get_pixmap(dpi=a.dpi).tobytes("png")
        raw = ocr_png(png)
        kept = [l for l in raw if not is_noise(l)]
        rows = reading_order(kept)
        pages_out.append(f"\n\n=== p.{i} ===\n" + "\n".join(" ".join(c["text"] for c in r) for r in rows))
        detail.append({"page": i, "lines": [c for r in rows for c in r]})
        confs = [c["conf"] for r in rows for c in r]
        print(f"p.{i}: 줄 {len(confs)} (제거 {len(raw) - len(kept)}), 평균 신뢰도 "
              f"{sum(confs) / max(len(confs), 1):.3f}, 0.5 미만 {sum(c < 0.5 for c in confs)}")

    backup = ROOT / "data" / "text_ocr_backup"
    backup.mkdir(exist_ok=True)
    if txt.exists() and not (backup / txt.name).exists():
        shutil.copy(txt, backup / txt.name)
    txt.write_text("".join(pages_out))
    (ROOT / "data" / "ocr").mkdir(exist_ok=True)
    (ROOT / "data" / "ocr" / f"{a.doc_id}_ocr.json").write_text(json.dumps(detail, ensure_ascii=False, indent=1))
    print("저장:", txt.relative_to(ROOT), len(txt.read_text()), "자")


if __name__ == "__main__":
    main()
