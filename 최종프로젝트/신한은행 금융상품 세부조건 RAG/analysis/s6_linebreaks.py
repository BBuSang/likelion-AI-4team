"""2-6 줄바꿈 정리 (LLM 없음) — data/sections/{doc_id}.jsonl 의 각 섹션에 `text` 필드 추가 + data/clean/{doc_id}.txt

원칙: 글자는 바꾸지 않고 **줄바꿈과 공백만** 조정한다 (검증: 공백을 모두 지우면 원문과 같아야 함).
`content`(원문 줄)는 그대로 두고, 읽기·임베딩용 `text`를 새로 만든다. 2-5를 다시 돌리면 이 스크립트도 다시 돌린다.

1) 줄을 이을지 (PDF가 폭 때문에 끊은 줄인가)
   잇는다: 앞 줄이 15자 이상이고 문장 끝(다./요./함./음./. 등)·쌍점으로 끝나지 않으며,
          다음 줄이 목록·제목 기호(•, -, ※, ①, 1., 가., (1), 제N조, ■ …)나 표의 행 이름(사전)으로 시작하지 않을 때
   안 잇는다: 짧은 줄(표 칸 값·행 이름), 섹션 제목 줄, 위의 경우가 아닌 모든 줄
2) 이을 때 공백을 넣을지 (단어 중간에서 끊겼나)
   띄어쓰기가 사라진 문서(doc_survey no_space=1)는 공백 없이 잇는다.
   그 밖에는 앞 줄 마지막 어절 a와 다음 줄 첫 어절 b에 대해, 다른 문서들에서 'ab'가 한 어절로 나온 횟수가
   a·b 중 덜 나온 쪽의 단독 횟수 이상이면 붙이고(예: '대출원'+'금에' → '대출원금에'),
   그렇지 않아도 한국어 띄어쓰기 교정기 Kiwi가 이음 지점(앞뒤 두 어절 문맥)에 공백을 넣지 않으면 붙인다. 나머지는 공백 하나.
   Kiwi는 이음 지점의 공백 여부만 판단에 쓰고, 다른 곳의 띄어쓰기는 건드리지 않는다.
   (긴 줄을 무작위로 잘라 복원하는 시뮬레이션: 문서 절반으로 고르고 나머지 절반에서 확인 — 어절 횟수만 0.905 → 횟수+Kiwi 0.969)
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s5_row_labels import GENERIC_SUFFIX, VARIANTS  # noqa: E402
from kiwipiepy import Kiwi  # noqa: E402  (pip install kiwipiepy, 0.24.1)

_KIWI = None


def kiwi_glue(prev: str, nxt: str) -> bool:
    """Kiwi 띄어쓰기 교정으로 이음 지점에 공백이 필요 없다고 보면 True"""
    global _KIWI
    _KIWI = _KIWI or Kiwi()
    left, right = " ".join(prev.split()[-2:]), " ".join(nxt.split()[:2])
    j, n = len(left.replace(" ", "")), 0
    out = _KIWI.space(left + right, reset_whitespace=False)
    for i, ch in enumerate(out):
        if ch != " ":
            if n == j:
                return not (i > 0 and out[i - 1] == " ")
            n += 1
    return True

ROOT = Path(__file__).resolve().parents[1]
SECS, CLEAN, META = ROOT / "data" / "sections", ROOT / "data" / "clean", ROOT / "data" / "meta"

MIN_WRAP = 15
SENT_END = re.compile(r"(다|요|함|음|임|됨|것)\s*[.)]?\s*$|[.:：。]\s*$")
MARKER = re.compile(
    r"^(?:[•·▪◦‣\-*※☞■□◆◇◉▣▶⋄￭ㅁ○●◈♦✔☑⇨➜→⊕=]|[-]"
    r"|\d{1,2}\s*[.)](?!\d)|\(\d{1,2}\)|\d\)|[가-하]\s*[.)]|\([가-하]\)|[①-⑳❶-❿➊-➓]"
    r"|Q\d|A\d|\[|【|〈|<|「?제\s*\d+\s*조|부\s*칙|“|\")"
)
TOKEN = re.compile(r"\S+")


def load_survey():
    return {r["doc_id"]: r for r in csv.DictReader(open(META / "doc_survey.csv"))}


def token_counts(docs):
    """문서별 어절 수 — 공백 판단 때 '그 문서 자신'은 빼고 센다 (자기 줄바꿈 조각에 끌려가지 않게)"""
    per = {}
    for d, secs in docs.items():
        c = Counter()
        for s in secs:
            for line in s["content"].split("\n"):
                c.update(TOKEN.findall(line))
        per[d] = c
    total = Counter()
    for c in per.values():
        total.update(c)
    return per, total


def starts_with_row_label(nxt: str) -> bool:
    """다음 줄이 표의 행 이름으로 시작하면 새 행 ('모집기간 모집기간을 두지 않고…', '만기')"""
    toks = nxt.split()
    for n in (3, 2, 1):
        head = "".join(toks[:n])
        if len(toks) >= n and (head in VARIANTS or (n == len(toks) and GENERIC_SUFFIX.match(head))):
            return True
    return False


def should_join(prev: str, nxt: str) -> bool:
    if prev.lstrip().startswith("|") or nxt.lstrip().startswith("|"):
        return False                       # md 표 줄(표 재추출 text_v2)은 한 줄이 한 행이라 잇지 않음
    return len(prev.strip()) >= MIN_WRAP and not SENT_END.search(prev) and not MARKER.match(nxt.strip()) \
        and not starts_with_row_label(nxt)


def glue(prev: str, nxt: str, no_space: bool, cnt) -> str:
    """공백 없이 붙일지(True=붙임)"""
    if no_space:
        return True
    a, b = prev.split()[-1], nxt.split()[0]
    if not (re.search(r"[가-힣]$", a) and re.match(r"^[가-힣]", b)):
        return False                       # 한글-한글 경계가 아니면 원래 띄어 쓴 것으로 봄
    return (cnt[a + b] > 0 and cnt[a + b] >= min(cnt[a], cnt[b])) or kiwi_glue(prev, nxt)


def clean_section(sec, no_space, cnt) -> tuple[str, int]:
    lines = sec["content"].split("\n")
    out, joins = lines[0], 0
    headed = sec["section_path"] != "(머리말)" and not sec["section_path"].startswith("p.")
    for i in range(1, len(lines)):
        prev, nxt = lines[i - 1], lines[i]
        if (i == 1 and headed) or not should_join(prev, nxt):
            out += "\n" + nxt
            continue
        out += ("" if glue(prev, nxt, no_space, cnt) else " ") + nxt
        joins += 1
    return out, joins


def main() -> None:
    survey = load_survey()
    docs = {p.stem: [json.loads(l) for l in open(p)] for p in sorted(SECS.glob("*.jsonl"))}
    per, total = token_counts(docs)
    CLEAN.mkdir(parents=True, exist_ok=True)
    bad, n_lines, n_joins = [], 0, 0
    for d, secs in docs.items():
        cnt = total - per[d]
        no_space = survey[d]["no_space"] == "1"
        for s in secs:
            s["text"], j = clean_section(s, no_space, cnt)
            n_lines += s["content"].count("\n") + 1
            n_joins += j
            if re.sub(r"\s", "", s["text"]) != re.sub(r"\s", "", s["content"]):
                bad.append((d, s["seq"]))
        with open(SECS / f"{d}.jsonl", "w") as f:
            for s in secs:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        (CLEAN / f"{d}.txt").write_text("\n\n".join(f"## {s['section_path']} (p.{s['page_start']})\n{s['text']}" for s in secs) + "\n")
    print("글자 보존 실패:", bad[:5], len(bad))
    print(f"줄 {n_lines:,} → 이은 곳 {n_joins:,} ({n_joins / n_lines:.0%}) → 정리 후 줄 {n_lines - n_joins:,}")


if __name__ == "__main__":
    sys.exit(main())
