"""presentation/발표자료.html → presentation/발표자료.pdf (Edge/Chrome 헤드리스, 16:9 슬라이드 1장 = 1쪽).

PowerShell 에서 실행 (Bash 샌드박스에서는 브라우저가 뜨지 않음)
  python -X utf8 tools/build_slides.py
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "presentation" / "발표자료.html"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]


def main():
    browser = next((b for b in BROWSERS if Path(b).exists()), None) or shutil.which("msedge") or shutil.which("chrome")
    if not browser:
        sys.exit("Edge 또는 Chrome 을 찾지 못했습니다")
    pdf = SRC.with_suffix(".pdf")
    # 한글 경로 문제를 피하려고 임시 폴더에서 출력한 뒤 옮긴다 (build_report.py 와 같은 방식)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_html, tmp_pdf = Path(tmp) / "slides.html", Path(tmp) / "slides.pdf"
        shutil.copy(SRC, tmp_html)
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={Path(tmp) / 'profile'}", f"--print-to-pdf={tmp_pdf}", tmp_html.as_uri()],
                       capture_output=True, timeout=120)
        if not tmp_pdf.exists():
            sys.exit("PDF 출력에 실패했습니다")
        shutil.move(str(tmp_pdf), pdf)
    print(f"{pdf.relative_to(ROOT)} ({pdf.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
