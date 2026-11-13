"""12 검색 단위: 베이스라인 청크(질의요지·회답·이유 칸별)와 형태소 색인을 만든다.
- 900자를 넘는 칸은 맨 위 항목 경계에서 600자 근처까지 묶는다. 항목이 없으면 문장 끝에서 자른다
- 2016 규제 정비 건은 사례 하나를 rule_card 하나로 만든다
- 포털 첨부 오류가 의심되는 사례(body_suspect)는 만들지 않는다. 같은 묶음의 다른 사례에 같은 글이 있다
- 글자가 없거나 '회답과 동일', '이유 참조'처럼 다른 칸을 가리키기만 하는 조각은 만들지 않는다. 사례 원문은 그대로 둔다
- 개인정보 형식이 발견되면 저장하지 않고 멈춘다
입력: processed/cases_enriched.jsonl, case_group_map.jsonl   출력: processed/chunks.jsonl (+csv)
"""
import re
from collections import Counter

from kiwipiepy import Kiwi

from common import manifest, read_jsonl, write_csv, write_jsonl

MAX, TARGET, TAIL = 900, 600, 100
SECTIONS = ("question", "answer", "reason")
LEVELS = (re.compile(r"^\s*[□☐■◇◆]"), re.compile(r"^\s*\d{1,2}\.(?!\d)"), re.compile(r"^\s*[①-⑳]"))
SENT = re.compile(r"(?<=[다요음임함됨]\.)\s+|(?<=[.!?])\s+|\n+")
POINTER = re.compile(r"(?:위|아래|상기|하단|기|旣|본)?(?:회답|회신|답변|판단|이유|질의|요청)*(?:내용|문|사항)?(?:과|와|에|을)?(?:동일|같음|포함|참조)(?:함|합니다|임|하시기바랍니다)?")
KEEP = ("NNG", "NNP", "NNB", "NR", "SN", "SL", "SH", "XR", "VV", "VA", "MM")
PII = {
    "주민등록번호": re.compile(r"(?<!\d)\d{6}\s*-\s*[1-4]\d{6}(?!\d)"),
    "휴대전화": re.compile(r"(?<!\d)01[016789][-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)"),
    "일반전화": re.compile(r"(?<![\d-])0(?:2|[3-6][1-5])[-.)\s]\s?\d{3,4}[-.\s]\d{4}(?!\d)"),
    "이메일": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "카드번호": re.compile(r"(?<!\d)\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}(?!\d)"),
    "계좌번호": re.compile(r"계좌(?:번호)?\s*[:：]?\s*\d{2,6}-\d{2,6}-\d{2,8}"),
}


def split_on(text, rx):
    """항목 기호로 시작하는 줄에서 나눈다. 기호 앞의 글은 첫 조각이 된다."""
    parts, cur = [], []
    for line in text.split("\n"):
        if rx.match(line) and cur:
            parts.append("\n".join(cur))
            cur = []
        cur.append(line)
    if cur:
        parts.append("\n".join(cur))
    return [p for p in parts if p.strip()]


def by_sentence(text):
    out = []
    for s in (x for x in SENT.split(text) if x and x.strip()):
        while len(s) > MAX:          # 문장 끝이 없는 긴 글은 띄어쓰기에서 자른다
            cut = s.rfind(" ", 0, MAX)
            cut = cut if cut > 0 else MAX
            out.append(s[:cut])
            s = s[cut:].lstrip()
        if s:
            out.append(s)
    return out


def pack(pieces):
    """조각을 차례로 이어 TARGET 근처까지 묶는다."""
    out, cur = [], ""
    for p in pieces:
        if cur and len(cur) + 1 + len(p) > TARGET:
            out.append(cur)
            cur = p
        else:
            cur = cur + "\n" + p if cur else p
    if cur:
        out.append(cur)
    return out


def cut(text, level=0):
    if len(text) <= MAX:
        return [text]
    for i in range(level, len(LEVELS)):
        parts = split_on(text, LEVELS[i])
        if len(parts) >= 2:
            return pack([x for p in parts for x in cut(p, i + 1)])
    return pack(by_sentence(text))


def chunk_text(text):
    text = text.strip()
    out = cut(text)
    if len(out) >= 2 and len(out[-1]) < TAIL and len(out[-2]) + 1 + len(out[-1]) <= MAX:   # 너무 짧은 끝 조각은 앞에 붙인다
        out[-2:] = [out[-2] + "\n" + out[-1]]
    return out


def no_info(text):
    """글자가 없거나 다른 칸을 가리키기만 하는 조각인가."""
    t = re.sub(r"[\s\W_]+", "", text)
    return not t or bool(POINTER.fullmatch(t))


def header(r):
    return f"[{r['gubun']} | {r['category_final'] or '미분류'} | {r['reply_date'] or '회신일 미상'}] {r['title'] or ''}".strip()


def main():
    rows = [r for r in read_jsonl("cases_enriched.jsonl") if r["indexable"]]
    maps = {m["case_key"]: m for m in read_jsonl("case_group_map.jsonl")}
    chunks, skipped, dropped = [], 0, Counter()
    for r in rows:
        m = maps[r["case_key"]]
        if m["body_suspect"]:
            skipped += 1
            continue

        def add(kind, no, text, body=None):
            chunks.append({"case_key": r["case_key"], "group_id": m["group_id"], "chunk_type": kind, "chunk_no": no, "text": text,
                           "embed_text": header(r) + "\n" + (body or text), "doc_class": r["doc_class"], "gubun": r["gubun"],
                           "category_final": r["category_final"], "reply_date": r["reply_date"]})

        if r["doc_class"] == "rule_review_2016":
            body = "\n".join(r[s].strip() for s in SECTIONS if r[s] and r[s].strip())
            add("rule_card", 1, "\n".join(x for x in (r["title"], body) if x), body or r["title"])
            continue
        for s in SECTIONS:
            if r[s] and r[s].strip():
                if no_info(r[s]):
                    dropped[s] += 1
                    continue
                for no, t in enumerate(chunk_text(r[s]), 1):
                    add(s, no, t)

    found = Counter(n for c in chunks for n, rx in PII.items() if rx.search(c["text"]))
    if found:
        raise SystemExit(f"개인정보 형식 발견, 저장 중단: {dict(found)}")

    titles = {r["case_key"]: r["title"] or "" for r in rows}
    kiwi = Kiwi()
    for c, toks in zip(chunks, kiwi.tokenize(titles[c["case_key"]] + "\n" + c["text"] for c in chunks)):
        c["bm25_tokens"] = " ".join(t.form.lower() for t in toks if t.tag.split("-")[0] in KEEP)

    write_jsonl("chunks.jsonl", chunks)
    write_csv("chunks.csv", chunks, ["case_key", "group_id", "chunk_type", "chunk_no", "text", "embed_text", "bm25_tokens", "doc_class", "gubun", "category_final", "reply_date"])
    ls = sorted(len(c["text"]) for c in chunks)
    manifest("12_chunks", cases=len(rows), skipped_suspect=skipped, dropped_no_info=dict(dropped), chunks=len(chunks), types=dict(Counter(c["chunk_type"] for c in chunks)),
             length={"median": ls[len(ls) // 2], "p90": ls[int(len(ls) * .9)], "max": ls[-1]}, over_max=sum(x > MAX for x in ls))


if __name__ == "__main__":
    main()
