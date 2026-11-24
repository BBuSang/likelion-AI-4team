"""2-2 반복 문구(인쇄 장식) 제거 — LLM 없음.

입력: data/text_v2/{파일명}.txt (표 재추출 s40, 없으면 data/text) (쪽 표시 '=== p.N ===' 포함)
출력: data/stage/s2/{doc_id}.txt (쪽 표시 유지)
로그: data/meta/s2_removed.csv  (doc_id, page, rule, line) — 지운 줄을 전부 남겨 되돌리기·메타 추출에 사용

지우는 것 (쪽 머리·꼬리에 반복되는 인쇄 장식만):
  review     준법감시인 사전심사필 제2026-…호            (위치 무관, 줄 전체가 이 형식일 때)
  valid      (2026. 07. 15 ~ 2027. 07. 14) 유효기간 줄     (위치 무관, 줄 전체가 이 형식일 때)
  notice     ※ 이 약관/계약서는 금융소비자 보호에 관한 법률 등 … 제공됩니다.
  form       5-202-0012(21.0×29.7) 중질지… / NCR용지 / 5-xxx-xxxx (2026.06 개정) 서식 번호
  eco        신한은행은 친환경용지를 사용합니다.
  copy_mark  은 행 용 / 고 객 용 (사본 표시; 사본 자체는 2-3에서 처리)
  page_code  (11-3), (5-1), - 3 -, 3/7, (3/7)             (쪽 머리·꼬리 3줄 안에서만)
  page_num   쪽 번호와 같은 숫자 한 줄                     (쪽 마지막 2줄 안에서만)
  bank_head  '신한은행' 한 단어 줄                          (쪽 첫 3줄 안에서만)
제어문자(\x07 등, 화면에 ☒로 보임)는 쪽마다 지움 (글자 수 보존 검사와 무관한 비인쇄 문자)
지우지 않는 것: 특약 시행 문구(부칙), '구 분/내 용', '거래조건(계속)', 목차 번호 줄 → 2-5에서 구조로 사용
"""
import csv
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT, OUT, META = ROOT / "data" / "text", ROOT / "data" / "stage" / "s2", ROOT / "data" / "meta"
if (ROOT / "data" / "text_v2").exists():
    TEXT = ROOT / "data" / "text_v2"   # 표·도식 쪽을 md 로 재추출한 글 (s40). 원래 PyMuPDF 글은 data/text 에 그대로

ANY_POS = [
    ("review", re.compile(r"^준법감시인\s*(?:사전)?\s*(?:심사|심의)\s*필\s*제?\s*[\d\s-]+호$")),
    ("valid", re.compile(r"^\(\s*\d{4}\s*\.\s*\d{1,2}\s*\.\s*\d{1,2}\s*\.?\s*~\s*\d{4}\s*\.\s*\d{1,2}\s*\.\s*\d{1,2}\s*\.?\s*\)$")),
    ("notice", re.compile(r"^※\s*이\s*(?:약관|계약서|특약)[은는]\s*금융소비자\s*보호에\s*관한\s*법률.*제공됩니다\.?$")),
    ("form", re.compile(r"^\d-\d{3}-\d{4}\s*\(.*(?:중질지|NCR용지|개정).*$")),
    ("eco", re.compile(r"^신한은행은\s*친환경용지를\s*사용합니다\.?$")),
    ("copy_mark", re.compile(r"^(?:은\s*행|고\s*객)\s*용$")),
]
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PAGE_CODE = re.compile(r"^(?:\(\s*\d{1,2}\s*-\s*\d{1,2}\s*\)|-\s*\d{1,3}\s*-|\(?\s*\d{1,3}\s*/\s*\d{1,3}\s*\)?)$")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def strip_page(lines: list[str], page_no: int):
    """한 쪽의 줄 목록 → (남길 줄, 지운 (rule, line) 목록)"""
    idx = [i for i, l in enumerate(lines) if l.strip()]
    head, tail = set(idx[:3]), set(idx[-3:])
    last2 = set(idx[-2:])
    keep, removed = [], []
    for i, raw in enumerate(lines):
        s = re.sub(r"\s+", " ", raw).strip()
        rule = None
        if s:
            for name, pat in ANY_POS:
                if pat.match(s):
                    rule = name
                    break
            if rule is None and (i in head or i in tail) and PAGE_CODE.match(s):
                rule = "page_code"
            if rule is None and i in last2 and s == str(page_no):
                rule = "page_num"
            if rule is None and i in head and s == "신한은행":
                rule = "bank_head"
        if rule:
            removed.append((rule, s))
        else:
            keep.append(raw)
    return keep, removed


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log = []
    for d in csv.DictReader(open(META / "documents.csv")):
        text = CONTROL.sub("", nfc((TEXT / f"{Path(d['file']).stem}.txt").read_text()))
        parts = re.split(r"^=== p\.(\d+) ===$", text, flags=re.M)
        out = [parts[0]]
        for k in range(1, len(parts), 2):
            pno, body = int(parts[k]), parts[k + 1]
            keep, removed = strip_page(body.split("\n"), pno)
            out.append(f"=== p.{pno} ===" + "\n".join(keep))
            log += [(d["doc_id"], pno, r, s) for r, s in removed]
        (OUT / f"{d['doc_id']}.txt").write_text("".join(out))
    with open(META / "s2_removed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "page", "rule", "line"])
        w.writerows(log)
    from collections import Counter
    print("지운 줄", len(log), Counter(r for _, _, r, _ in log).most_common())
    print("문서 수", len({x[0] for x in log}), "/", len(list(OUT.glob('*.txt'))))


if __name__ == "__main__":
    main()
