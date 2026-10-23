"""report/*.md → report/*.pdf (Markdown → HTML → Edge/Chrome 헤드리스로 PDF 출력).

보고서는 장마다 파일 하나: 0_요약_및_결론, 1_데이터_관찰_리포트, 2_방식별_비교표, 3_오류_유형_분석,
4_탐지기_선정_근거, 5_개선_과정과_데이터_사용_기록. 한글은 맑은 고딕, 표는 페이지 폭에 맞춰 출력한다.

사용법 (PowerShell 에서 실행: Claude Code 의 Bash 실행 환경에서는 브라우저가 뜨지 않음)
  python -X utf8 tools/build_report.py              # 전부
  python -X utf8 tools/build_report.py 2_방식별_비교표   # 하나만
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]

CSS = """
@page { size: A4; margin: 16mm 14mm 16mm 14mm; }
body { font-family: 'Malgun Gothic', '맑은 고딕', sans-serif; font-size: 9.6pt; line-height: 1.55; color: #1f2328; }
h1 { font-size: 17pt; border-bottom: 2px solid #1f2328; padding-bottom: 6px; margin: 0 0 10px; }
h2 { font-size: 12.5pt; margin: 20px 0 8px; padding: 4px 8px; background: #eef2f7; border-left: 4px solid #2f5d9b;
     break-after: avoid; }
h3 { font-size: 11pt; margin: 14px 0 6px; color: #2f5d9b; break-after: avoid; }
p, li { margin: 4px 0; }
blockquote { margin: 6px 0 10px; padding: 6px 10px; background: #f6f8fa; border-left: 3px solid #9aa4b2; color: #444; font-size: 9pt; }
blockquote ul { margin: 0; padding-left: 16px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 8.4pt; }
tr { break-inside: avoid; }
th, td { border: 1px solid #c9d1d9; padding: 3px 5px; vertical-align: top; }
th { background: #eef2f7; font-weight: bold; }
th:first-child, td:first-child { min-width: 5.5em; }
tbody tr:nth-child(even) td { background: #fafbfc; }
code { font-family: Consolas, 'Malgun Gothic', monospace; font-size: 8.4pt; background: #f2f4f7; padding: 0 3px; border-radius: 3px; }
pre { background: #f6f8fa; padding: 8px; border-radius: 4px; font-size: 8pt; white-space: pre-wrap; }
pre code { background: none; padding: 0; }
hr { border: none; border-top: 1px solid #d0d7de; margin: 14px 0; }
strong { color: #0b3d75; }
"""


def build(md_path, browser):
    pdf_path = md_path.with_suffix(".pdf")
    title = md_path.read_text(encoding="utf-8").split("\n", 1)[0].lstrip("# ").strip()
    body = markdown.markdown(md_path.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "sane_lists"])
    html = (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{title}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")
    # 이미 열린 브라우저 창과 섞이지 않게 임시 프로필을 쓰고, 한글 경로 문제를 피하려고 임시 폴더에 출력한 뒤 옮긴다
    with tempfile.TemporaryDirectory() as tmp:
        tmp_html, tmp_pdf = Path(tmp) / "report.html", Path(tmp) / "report.pdf"
        tmp_html.write_text(html, encoding="utf-8")
        subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={Path(tmp) / 'profile'}", f"--print-to-pdf={tmp_pdf}", tmp_html.as_uri()],
                       capture_output=True, timeout=120)
        if not tmp_pdf.exists():
            sys.exit(f"PDF 출력에 실패했습니다: {md_path.name}")
        shutil.move(str(tmp_pdf), pdf_path)
    print(f"{pdf_path.relative_to(ROOT)} ({pdf_path.stat().st_size // 1024} KB)")


def main():
    browser = next((b for b in BROWSERS if Path(b).exists()), None) or shutil.which("msedge") or shutil.which("chrome")
    if not browser:
        sys.exit("Edge 또는 Chrome 을 찾지 못했습니다")
    names = sys.argv[1:]
    paths = [REPORT_DIR / f"{n}.md" for n in names] if names else sorted(REPORT_DIR.glob("*.md"))
    for p in paths:
        build(p, browser)


if __name__ == "__main__":
    main()
