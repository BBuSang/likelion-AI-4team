"""데모 화면 표시용 (기획안 5-7 v0.43): 답변 글을 조문이 바뀌는 곳에서 줄을 나눈다. LLM 답변 내용은 바꾸지 않는다."""
import re

_SENT = re.compile(r"(?<=[다요)]\.)\s+")          # '…합니다.' / '…(법 제4조).' 뒤에서 문장을 나눔
_ART = re.compile(r"제\d+조(?:의\d+)?")            # 제19조, 제24조의7


def split_by_article(text):
    """→ 문단 목록. 문장마다 언급한 조문(제N조)을 보고, 다른 조문을 말하는 문장이 나오면 새 문단.
    조문을 말하지 않는 문장은 앞 문단에 붙는다. 맨 앞의 조문 없는 문장(결론)은 따로 둔다."""
    blocks, key = [], None
    for s in (x.strip() for x in _SENT.split(text or "") if x.strip()):
        k = frozenset(_ART.findall(s))
        if blocks and s.startswith("(") and s.rstrip(".").endswith(")"):  # '(법령명 제N조)'만 있는 문장은 앞 문단에 붙임
            blocks[-1].append(s)
            key = k or key
        elif not blocks or (k and k != key):
            blocks.append([s])
            key = k or None
        else:
            blocks[-1].append(s)
    return [" ".join(b) for b in blocks]


_MDCH = set("\\[]*_`$~")  # Streamlit 마크다운에서 뜻이 있는 글자 → 앞에 역슬래시를 붙여 그대로 보이게


def _esc(t):
    return "".join("\\" + c if c in _MDCH else c for c in t)


def mark_keys(paragraph, keys, color="red"):
    """문단 → Streamlit 마크다운. keys(핵심 구절, 5-8-5)와 글자가 똑같은 곳만 :red[…], 나머지는 그대로.
    긴 구절부터 찾고 겹치는 곳은 한 번만 칠함. 글자는 바꾸지 않음"""
    spans = []
    for k in sorted({k for k in keys if k}, key=len, reverse=True):
        start = 0
        while (i := paragraph.find(k, start)) >= 0:
            if all(i >= e or i + len(k) <= s for s, e in spans):
                spans.append((i, i + len(k)))
            start = i + len(k)
    out, pos = [], 0
    for s, e in sorted(spans):
        out.append(_esc(paragraph[pos:s]))
        out.append(f":{color}[{_esc(paragraph[s:e])}]")
        pos = e
    out.append(_esc(paragraph[pos:]))
    return "".join(out)


_BULLET = re.compile(r"^\s*(?:[-•·*]|\d+[.)])\s+")


def render_answer(text, keys=()):
    """답변 글 → Streamlit 마크다운 (기획안 5-8-6): 줄바꿈과 '- ' 목록은 그대로 살리고,
    목록이 아닌 글은 조문별 문단으로 나눈 뒤, 핵심 구절만 빨간색. 글자는 바꾸지 않음"""
    blocks, para, items = [], [], []

    def flush_para():
        if para:
            blocks.extend(mark_keys(p, keys) for p in split_by_article(" ".join(para)))
            para.clear()

    def flush_items():
        if items:
            blocks.append("\n".join(f"- {mark_keys(i, keys)}" for i in items))
            items.clear()

    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            flush_para()
            flush_items()
        elif _BULLET.match(line):
            flush_para()
            items.append(_BULLET.sub("", line, count=1))
        else:
            flush_items()
            para.append(line)
    flush_para()
    flush_items()
    return "\n\n".join(blocks)
