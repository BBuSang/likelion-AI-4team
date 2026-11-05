"""팀 데모 시연을 자동으로 조작하면서 화면을 녹화해 mp4로 저장한다.

준비 (이 폴더에서 한 번만):
    python -m venv .venv
    .venv\\Scripts\\python -m pip install -r requirements.txt
    .venv\\Scripts\\python -m playwright install chromium

실행:
    .venv\\Scripts\\python -X utf8 record_demo.py
"""

import argparse
import re
import subprocess
import tempfile
import time
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

URL = "http://172.31.98.17:8501/"
QUESTIONS = [
    "보이스피싱 피해 접수 기록 알려줘",
    "고객센터 김민준 직원이 처리한 건 있어?",
]
OUTPUT = Path(__file__).resolve().parent.parent / "report" / "실전2_시연영상.mp4"
VIEWPORT = {"width": 1600, "height": 900}
ANSWER_TIMEOUT_MS = 180_000

# 화면마다 머무는 시간(초). 발표 대본에 맞춰 여기만 고치면 된다.
HOLD = {
    "start": 1,
    "answer": 4,
    "real_name": 5,
    "end": 1,
}
# 답변 생성 대기는 앞뒤만 남기고 영상에서 잘라낸다(초).
KEEP_BEFORE_ANSWER = 1.5
KEEP_AFTER_CLICK = 2
# 질문을 연달아 보내면 CLOVA 호출 한도(429)에 걸려 사이를 띄운다. 이 구간도 영상에서 잘라낸다.
QUESTION_GAP = 75

# 녹화 영상에는 마우스 포인터가 찍히지 않아 화면에 직접 그린다.
CURSOR_SCRIPT = """
document.addEventListener('DOMContentLoaded', () => {
  const dot = document.createElement('div');
  dot.style.cssText = 'position:fixed;left:0;top:0;width:22px;height:22px;margin:-11px 0 0 -11px;'
    + 'border-radius:50%;background:rgba(229,57,53,.55);border:2px solid #fff;'
    + 'box-shadow:0 0 4px rgba(0,0,0,.5);pointer-events:none;z-index:2147483647;'
    + 'transition:transform .12s;';
  document.body.appendChild(dot);
  document.addEventListener('mousemove', e => {
    dot.style.left = e.clientX + 'px';
    dot.style.top = e.clientY + 'px';
  }, true);
  document.addEventListener('mousedown', () => dot.style.transform = 'scale(.65)', true);
  document.addEventListener('mouseup', () => dot.style.transform = 'scale(1)', true);
});
"""


class Recorder:
    def __init__(self, page, video_started):
        self.page = page
        self.video_started = video_started
        self.main = page.locator(".stMain")
        self.cuts = []

    def now(self):
        return time.monotonic() - self.video_started

    def mark(self, label):
        print(f"{self.now():7.1f}s  {label}", flush=True)

    def hold(self, key):
        self.page.wait_for_timeout(HOLD[key] * 1000)

    def click(self, locator):
        """포인터를 대상까지 천천히 옮긴 뒤 누른다."""
        locator.wait_for(state="visible")
        locator.scroll_into_view_if_needed()
        self.page.wait_for_timeout(100)
        box = locator.bounding_box()
        self.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=12)
        self.page.wait_for_timeout(150)
        self.page.mouse.down()
        self.page.mouse.up()
        self.page.wait_for_timeout(100)
        self.park()

    def park(self):
        """글자를 가리지 않게 포인터를 오른쪽 빈 곳으로 치운다."""
        self.page.mouse.move(VIEWPORT["width"] - 90, VIEWPORT["height"] * 0.45, steps=8)

    def scroll_to(self, locator):
        """대상이 화면 위쪽에 오도록 본문을 스크롤한다."""
        locator.evaluate(
            """e => {
              const box = e.closest('.stMain');
              const top = e.getBoundingClientRect().top - box.getBoundingClientRect().top;
              box.scrollTo({top: box.scrollTop + top - 24, behavior: 'smooth'});
            }"""
        )
        self.page.wait_for_timeout(700)

    def run_question(self, question):
        self.mark(f"질문 선택: {question}")
        self.click(self.main.get_by_text(question, exact=True).first)
        clicked = self.now()

        process =self.main.locator("summary", has_text="처리 과정 보기").first
        error = self.main.get_by_text(re.compile(r"^\s*오류:")).first
        process.or_(error).first.wait_for(state="visible", timeout=ANSWER_TIMEOUT_MS)
        if error.is_visible():
            raise RuntimeError(f"데모가 오류를 표시함, 잠시 뒤 다시 실행 필요: {error.inner_text()}")
        self.mark("답변 표시")
        cut_from, cut_to = clicked + KEEP_AFTER_CLICK, self.now() - KEEP_BEFORE_ANSWER
        if cut_to > cut_from:
            self.cuts.append((cut_from, cut_to))
        self.scroll_to(self.main.locator('[data-testid="stCheckbox"]').first)
        self.hold("answer")

        self.mark("실명 보기")
        self.click(self.main.locator('[data-testid="stCheckbox"] label').first)
        self.page.wait_for_timeout(400)
        self.hold("real_name")

    def run(self):
        self.main.get_by_text(QUESTIONS[0], exact=True).first.wait_for(state="visible", timeout=60_000)
        self.park()
        started_at = time.monotonic() - self.video_started
        self.mark("시작 화면")
        self.hold("start")

        for i, question in enumerate(QUESTIONS):
            if i:
                self.mark("새 채팅")
                self.click(self.page.get_by_role("button", name="새 채팅"))
                self.main.get_by_text(question, exact=True).first.wait_for(state="visible", timeout=30_000)
                self.page.wait_for_timeout(500)
                gap_from = self.now()
                self.page.wait_for_timeout(QUESTION_GAP * 1000)
                self.cuts.append((gap_from, self.now()))
                self.page.wait_for_timeout(500)
            self.run_question(question)
        self.hold("end")
        self.mark("끝")
        return started_at


def to_mp4(webm, output, start_seconds, cuts):
    output.parent.mkdir(parents=True, exist_ok=True)
    removed = "+".join(
        f"between(t,{a - start_seconds:.2f},{b - start_seconds:.2f})" for a, b in cuts
    ) or "0"
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
         "-ss", f"{start_seconds:.2f}", "-i", str(webm),
         "-vf", f"select='not({removed})',setpts=N/FRAME_RATE/TB",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport=VIEWPORT,
            locale="ko-KR",
            record_video_dir=tmp,
            record_video_size=VIEWPORT,
        )
        context.add_init_script(CURSOR_SCRIPT)
        page = context.new_page()
        video_started = time.monotonic()
        page.goto(URL, wait_until="domcontentloaded")
        recorder = Recorder(page, video_started)
        try:
            start_seconds = recorder.run()
        except Exception:
            page.screenshot(path=str(args.output.with_suffix(".error.png")))
            raise
        finally:
            video = page.video
            context.close()
            browser.close()
        to_mp4(video.path(), args.output, start_seconds, recorder.cuts)
    print(f"저장: {args.output}")


if __name__ == "__main__":
    main()
