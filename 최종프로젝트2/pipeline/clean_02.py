"""기획안 5-2 전처리 (+ 파싱)

입력: data/work/01_fetch/selected.json 의 최종 법령 380개 (원문 JSON)
출력: data/work/02_clean/
    articles.csv        조문 (글 세 층 + 표시)
    article_lines.csv   항·호·목 줄 (단계, 번호, 상위 줄)
    admrul_gate.json    행정규칙 나누기 관문 결과 (규정별)
    clean_stats.json    지운 모양별 건수 등

    python pipeline/clean_02.py
"""
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from common import TODAY, WORK

OUT = WORK / "02_clean"
RULE_VERSION = "5-2 v2 (기획안 v0.9)"
STATS = Counter()

CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
GANADA = "가나다라마바사아자차카타파하"

# ---------------------------------------------------------------- 글 손질 (5-2-2)
# 지우는 모양 목록. 목록에 없는 <…>·[…]는 남긴다 (계산식 대괄호 등)
TAG_PATTERNS = [
    ("img", re.compile(r"<img[^>]*>|</img>", re.I)),
    ("개정·신설 태그", re.compile(r"<[^<>]{0,80}?(?:개정|신설|전문개정|제정)[^<>]{0,80}?>")),
    ("이동 태그", re.compile(r"<\s*종전[^<>]{0,80}>|<[^<>]{0,40}에서\s*이동[^<>]{0,40}>")),
    ("날짜만 태그", re.compile(r"<\s*[\d\s.,ㆍ·]+\s*>")),
    ("본조신설 등", re.compile(r"\[\s*본\s*[조항호목장절편관]\s*신\s*설[^\[\]]{0,80}\]")),
    ("전문개정 등", re.compile(r"\[\s*(?:전문|전면|제목)\s*개정[^\[\]]{0,80}\]")),
    ("신설·개정 대괄호", re.compile(r"\[\s*(?:신설|개정)\s*[,\d<][^\[\]]{0,80}\]")),
    ("이동 대괄호", re.compile(r"\[\s*(?:제\s*\d|종전)[^\[\]]{0,120}?이동[^\[\]]{0,80}\]")),
]
FUTURE = re.compile(r"\[\s*시행일\s*:\s*(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?\s*\](?:\s*제\s*\d+(?:-\d+)?\s*조(?:\s*의\s*\d+)?)?")
# 삭제 이력 태그: <삭제 2010. 3. 31.>, <본호 삭제 …>, <…고시 제2009-2호, 2009. 2. 4. 삭제>, <단서삭제 …>
# 줄에 이 태그밖에 없으면 '삭제'로 바꿔 삭제 줄로 처리하고, 내용이 있으면 태그만 지운다
DEL_TAG = re.compile(r"<\s*(?=[^<>]*(?:\d|본\s*[조항호목]|단서))[^<>]{0,80}?삭\s*제[^<>]{0,60}>")
LINE_MARK = re.compile(r"^\s*(?:[①-⑳]|\d+(?:의\d+)?\.|[가-힣](?:\d+)?\.|\d+\)|[가-힣]\)|\(\d+\))?\s*")
ANNEX = re.compile(r"<\s*(별표|별지|별책)([^<>]{0,40})>")
DELETED_MARK = re.compile(r"<\s*삭\s*제\s*>")
BOX = re.compile(r"[┌┐└┘├┤┬┴┼─━┏┓┗┛┣┫┳┻╋═║╔╗╚╝╠╣╦╩╬]")
VBAR = re.compile(r"[│┃]")
HANJA_PAREN = re.compile(r"\s?\(([一-鿿][一-鿿가-힣·ㆍ\s]{0,20})\)")
DELEGATION = re.compile(r"(?:대통령령|총리령|[가-힣]{1,10}부령|고시|규칙|세칙|규정)\s*(?:으)?로\s*정하")
REF = re.compile(r"(?:같은\s*조\s*)?제\s*\d+\s*조(?:\s*의\s*\d+)?(?:\s*제\s*\d+\s*항)?(?:\s*제\s*\d+\s*호)?(?:\s*\([^)]{0,40}\))?"
                 r"(?:\s*(?:ㆍ|·|,|및|부터|까지|와|과|중|의|단서|본문|는|은|를|을|제외한다|\(|\)|\s))*")


def normalize(t):
    t = t.replace("​", "").replace("﻿", "").replace("　", " ").replace("\xa0", " ")
    t = t.replace("｢", "「").replace("｣", "」").replace("ㆍ", "·").replace("･", "·").replace("‧", "·").replace("ⓛ", "①").replace("〈", "<").replace("〉", ">")
    t = re.sub(r"[ \t]+", " ", t)
    return "\n".join(s.strip() for s in t.split("\n")).strip()


def clean_text(t, flags):
    """표시용 글. flags 에 has_image / has_table / not_in_force 를 기록"""
    t = (t or "").replace("〈", "<").replace("〉", ">")  # 전각 꺾쇠 태그 〈…개정〉 → <…개정>
    if DEL_TAG.search(t):
        STATS["삭제 이력 태그"] += len(DEL_TAG.findall(t))
        rest = DEL_TAG.sub("", t)
        for _, pat in TAG_PATTERNS:
            rest = pat.sub("", rest)
        only_tag = not LINE_MARK.sub("", normalize(rest), count=1).strip()
        t = DEL_TAG.sub("삭제", t, count=1) if only_tag else DEL_TAG.sub("", t)
        t = DEL_TAG.sub("", t)
    for name, pat in TAG_PATTERNS:
        n = len(pat.findall(t))
        if n:
            STATS[name] += n
            if name == "img":
                flags["has_image"] = True
            t = pat.sub("", t)
    for m in FUTURE.finditer(t):
        STATS["시행일 대괄호"] += 1
        if f"{m.group(1)}{int(m.group(2)):02d}{int(m.group(3)):02d}" > TODAY:
            flags["not_in_force"] = True
    t = FUTURE.sub("", t)
    n = len(ANNEX.findall(t))
    if n:
        STATS["별표·별지 괄호만 뺌"] += n
        t = ANNEX.sub(lambda m: (m.group(1) + m.group(2)).strip(), t)
    t = DELETED_MARK.sub("삭제", t)
    # 표: 선만 있는 줄은 지우고 칸은 ' | '
    if BOX.search(t) or VBAR.search(t):
        flags["has_table"] = True
        rows = []
        for ln in t.split("\n"):
            if not re.sub(r"[\s|]", "", BOX.sub("", VBAR.sub("|", ln))):
                STATS["표 선 줄"] += 1
                continue
            ln = BOX.sub("", ln)
            ln = " | ".join(c.strip() for c in VBAR.split(ln) if c.strip())
            rows.append(ln)
        t = "\n".join(rows)
    return normalize(t)


def embed_of(text):
    n = len(HANJA_PAREN.findall(text))
    STATS["한자 괄호(임베딩용에서 뺌)"] += n
    return HANJA_PAREN.sub("", text)


def is_delete_body(t):
    return re.fullmatch(r"삭\s*제\.?", (t or "").strip()) is not None


# ---------------------------------------------------------------- 조문 머리
HEAD = re.compile(r"^\s*제\s*(\d+)(?:-(\d+))?\s*조(?:\s*의\s*(\d+))?\s*(?:\(([^()]*(?:\([^()]*\)[^()]*)*)\))?\s*")


def split_head(t):
    m = HEAD.match(t or "")
    if not m:
        return None, t or ""
    return m, (t or "")[m.end():]


def label_of(no, dash, sub):
    return f"제{no}" + (f"-{dash}" if dash else "") + "조" + (f"의{sub}" if sub and str(sub) not in ("0", "") else "")


# ---------------------------------------------------------------- 줄
class Lines:
    def __init__(self, auid):
        self.auid, self.rows = auid, []

    def add(self, level, marker, raw, parent=None, para=None, item=None, sub=None):
        flags = {}
        text = clean_text(raw, flags)
        deleted = bool(re.fullmatch(r"(?:[①-⑳]|\d+(?:의\d+)?\.|[가-힣](?:\d+)?\.|\d+\)|[가-힣]\))?\s*삭\s*제\.?", text))
        if text == "" and level != "항":
            return None
        uid = f"{self.auid}#{len(self.rows) + 1}"
        self.rows.append({"line_uid": uid, "article_uid": self.auid, "seq": len(self.rows) + 1, "level": level,
                          "marker": marker or "", "para_no": para or "", "item_no": item or "", "sub_no": sub or "",
                          "parent_line_uid": parent or "", "text": text, "embed_text": embed_of(text),
                          "is_deleted": deleted, "not_in_force": flags.get("not_in_force", False),
                          "has_image": flags.get("has_image", False), "has_table": flags.get("has_table", False)})
        return uid


def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def flat(x):
    """'목내용' 처럼 리스트 안 리스트로 오는 글 → 줄 목록"""
    if isinstance(x, list):
        return [s for y in x for s in flat(y)]
    return [str(x)] if x is not None and str(x).strip() else []


def marker_of(text, kind):
    t = (text or "").strip()
    if kind == "항":
        m = re.match(r"^([①-⑳])", t)
        return (m.group(1), str(CIRCLED.index(m.group(1)) + 1)) if m else ("", "")
    if kind == "호":
        m = re.match(r"^(\d+)(?:의(\d+))?\.", t)
        return (m.group(0), m.group(1) + (f"의{m.group(2)}" if m.group(2) else "")) if m else ("", "")
    if kind == "목":
        m = re.match(r"^([가-힣])(\d+)?\.", t)
        return (m.group(0), m.group(1) + (m.group(2) or "")) if m else ("", "")
    if kind == "세목":
        m = re.match(r"^(\d+\)|[가-힣]\)|\(\d+\))", t)
        return (m.group(1), m.group(1)) if m else ("", "")
    return "", ""


# ---------------------------------------------------------------- 법령 (법률·시행령·시행규칙)
CHAP = re.compile(r"^\s*제\s*\d+\s*(편|장|절|관)")
CHAP_ORDER = {"편": 0, "장": 1, "절": 2, "관": 3}


def update_chapter(stack, title):
    m = CHAP.match(title)
    if not m:
        return stack
    lv = CHAP_ORDER[m.group(1)]
    stack = {k: v for k, v in stack.items() if k < lv}
    stack[lv] = normalize(re.sub(r"<[^<>]*>", "", title))
    return stack


def law_articles(c):
    L = json.loads(Path(c["path"]).read_text(encoding="utf-8"))["법령"]
    stack, out = {}, []
    for u in as_list((L.get("조문") or {}).get("조문단위")):
        if u.get("조문여부") == "전문":
            stack = update_chapter(stack, " ".join(flat(u.get("조문내용"))))
            continue
        if u.get("조문여부") != "조문":
            continue
        auid = f"{c['uid']}:{u.get('조문키')}"
        lines = Lines(auid)
        body_raw = "\n".join(flat(u.get("조문내용")))
        m, rest = split_head(body_raw)
        n_json = Counter()
        top = lines.add("본문", "", rest) if rest.strip() else None
        for h in as_list(u.get("항")):
            hraw = "\n".join(flat(h.get("항내용")))
            para_uid, para_no = top, ""
            if hraw.strip():
                n_json["항"] += 1
                mk, para_no = marker_of(hraw, "항")
                para_uid = lines.add("항", mk, hraw, parent=None, para=para_no)
            for ho in as_list(h.get("호")):
                n_json["호"] += 1
                horaw = "\n".join(flat(ho.get("호내용")))
                mk, item_no = marker_of(horaw, "호")
                ho_uid = lines.add("호", mk, horaw, parent=para_uid, para=para_no, item=item_no)
                for mo in as_list(ho.get("목")):
                    n_json["목"] += 1
                    parts = flat(mo.get("목내용"))
                    mk, sub_no = marker_of(parts[0] if parts else "", "목")
                    mo_uid = lines.add("목", mk, parts[0] if parts else "", parent=ho_uid, para=para_no, item=item_no, sub=sub_no)
                    for extra in parts[1:]:  # 목 안의 세목 1) 2) …
                        mk2, _ = marker_of(extra, "세목")
                        lines.add("세목" if mk2 else "목(이어짐)", mk2, extra, parent=mo_uid, para=para_no, item=item_no, sub=sub_no)
        out.append(make_article(c, auid, len(out) + 1, m, u.get("조문번호"), u.get("조문가지번호"), u.get("조문제목"),
                                stack, raw_text(u), lines, u.get("조문시행일자"), "api", n_json))
    return out


def raw_text(u):
    parts = flat(u.get("조문내용"))
    for h in as_list(u.get("항")):
        parts += flat(h.get("항내용"))
        for ho in as_list(h.get("호")):
            parts += flat(ho.get("호내용"))
            for mo in as_list(ho.get("목")):
                parts += flat(mo.get("목내용"))
    return "\n".join(p.strip() for p in parts)


# ---------------------------------------------------------------- 행정규칙
ARTICLE_START = re.compile(r"제\s*(\d+)(?:-(\d+))?\s*조(?:\s*의\s*(\d+))?\s*\(")
SENT_END = re.compile(r"(?:[.다:;」>\]\)]|[가-힣]{0,30}[장절편관]\s*[^\s]{0,30})\s*$")


def art_key(m):
    return (int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0))


def split_admrul_block(text):
    """한 덩어리 → 조문 목록. 조문 머리 후보 중 '번호가 앞보다 크고, 바로 앞이 문장 끝 또는 장·절 제목'인 곳만 자른다"""
    cuts, last = [], (0, 0, 0)
    for m in ARTICLE_START.finditer(text):
        k = art_key(m)
        if k <= last:
            continue
        before = text[max(0, m.start() - 40):m.start()]
        # 괄호 제목 뒤가 조문 본문 시작(①, 글)인지: 본문 속 참조 `제2조(정의)에 따른`·`제3조(목적)를` 은 제외
        close = text.find(")", m.end())
        after = text[close + 1:close + 8] if 0 <= close - m.end() <= 60 else ""
        # 조문 머리 = 괄호 제목 바로 뒤가 본문 시작(①, 태그, 띄어 쓴 글). 괄호 바로 뒤에 조사·쉼표가 붙으면 본문 속 참조
        # (`제2-6조(…한한다), 제7-13조`, `제2조(정의)에 따른`) → 참조 하나를 머리로 잘못 잡으면 뒤 조문이 줄줄이 합쳐지므로 반드시 확인
        after = text[close + 1:close + 16] if after else ""
        looks_head = bool(after) and re.match(r"^(?:\s*[①-⑳ⓛ<\[]|\s+\S|[가-힣「\"“]|제\s*\d)", after) and \
            not re.match(r"^(?:에서|에게|에|의|를|을|및|과|와|부터|까지|이나|이|가|은|는|로|으로|중|만|도|,|·|ㆍ|\s*및|\s*또는"
                         r"|제\s*\d+\s*(?:항|호|목))", after)
        # + 바로 앞이 처음·문장 끝·장 제목·글자이거나(호 목록 끝 `…있을 것제1-4조의2(…)①`),
        #   제목 없는 조문 머리(`제2-14조의2제2-15조(…)`, 본문 없이 번호만 있는 조문) 바로 뒤
        if looks_head and (m.start() == 0 or re.search(r"[.다:;」>\]\)]\s*$", before) or
                           re.search(r"제\s*\d+\s*[편장절관][^.]{0,30}$", before) or re.search(r"[가-힣]$", before) or
                           re.search(r"제\s*\d+(?:-\d+)?\s*조(?:\s*의\s*\d+)?\s*$", before) or
                           # 따옴표·숫자로 끝난 글 바로 뒤(`"사무소"제9-33조(…)①`, `100분의 50제4-51조(…)①`)는 ①이 이어질 때만
                           (re.search(r"[\"”'’\d%]$", before) and re.match(r"^\s*[①ⓛ]", after))):
            cuts.append(m.start())
            last = k
    if not cuts:
        return [], text
    pieces = [text[a:b] for a, b in zip(cuts, cuts[1:] + [len(text)])]
    return pieces, text[:cuts[0]]


TAIL_CHAP = re.compile(r"(?<=[.다)>\]」])\s*(제\s*\d+\s*[편장절관]\s*[^.]{0,40})$")


def split_guideline(text):
    """`1. 목 적` / `목적1. 이 기준은` 형식 지침 → 맨 위 번호 항목 [(시작 위치, 번호, 제목)]
    - 번호는 앞보다 크고 20 이내로만 건너뜀 (그림 속 항목 때문에 1 → 4 → 6, 6 → 10 처럼 비는 경우가 있음)
    - 바로 앞이 처음·문장 끝·그림 태그면 그 자리에서, 문장 끝 뒤에 짧은 제목(30자 이하)이 붙어 있으면 제목 앞에서 자름
    - 날짜 '2016. 3.' 안의 숫자는 안 자름"""
    cuts, last = [], 0
    for m in re.finditer(r"(?<![\d.])(\d+)\.\s?", text):
        k = int(m.group(1))
        if not (last < k <= last + 20):
            continue
        before = text[max(0, m.start() - 60):m.start()]
        if re.search(r"\d\.\s*$", before):
            continue
        if m.start() == 0 or re.search(r"(?:[.다)>\]」]|</img>)\s*$", before) or                 re.search(r"제\s*\d+\s*[편장절관][^.\d]{0,30}$", before):  # '제1장 총칙1.' 처럼 장 제목 바로 뒤
            cuts.append((m.start(), k, ""))
            last = k
            continue
        t = re.search(r"(?:^|[.다)>\]」\u3000]|</img>)\s*([^.\d<>\n]{1,30})$", text[max(0, m.start() - 60):m.start()]
                      if m.start() > 60 else text[:m.start()])
        if t:
            title = t.group(1).strip()
            if CHAP.match(title):  # '제1장 총칙1.' → 장 제목은 앞 조각 끝에 남겨 둠
                cuts.append((m.start(), k, ""))
            else:
                cuts.append((m.start() - len(t.group(1)), k, title))
            last = k
    return cuts


def seq_split(text, kind):
    """조문 본문 → (기호, 글) 목록. 항(①…)·호(1. 1의2.)·목(가.)을 차례로 이어질 때만 자른다"""
    if kind == "항":
        pat = re.compile(r"[①-⑳]")
        key = lambda m: (CIRCLED.index(m.group(0)) + 1, 0)
    elif kind == "호":
        pat = re.compile(r"(?<!\d)(?<!\d\.)(\d+)(?:의(\d+))?\.(?!\d)")  # `같다.1.`은 인정, 소수 `3.5`·`2016. 3.` 안 숫자는 제외
        key = lambda m: (int(m.group(1)), int(m.group(2) or 0))
    else:
        # 글자 바로 뒤의 목 기호(`사항가.`)도 후보로. 단 글자 뒤 후보는 아래 first_mok/item_ok(차례 확인)로만 인정
        pat = re.compile(r"(?<!\d)([" + GANADA + r"])\.(?!\d)")
        key = lambda m: (GANADA.index(m.group(1)) + 1, 0)
    cuts, last = [], (0, 0)
    for m in pat.finditer(text):
        k = key(m)
        # 다음 번호(k+1) 또는 가지 번호(1 → 1의2 → 1의3)만 인정
        ok_seq = (k[0] == last[0] + 1 and k[1] == 0) or \
                 (last[0] > 0 and k[0] == last[0] and k[1] == (last[1] + 1 if last[1] else 2))
        # ①이 없이 ②부터 시작하는 조문 (원문에 ①이 빠진 경우)
        if kind == "항" and last == (0, 0) and k == (2, 0) and "①" not in text:
            ok_seq = True
        # 항은 원문에서 번호 하나가 빠진 경우(①→③, ③ 없이 ④)도 문장·항목 끝 뒤면 인정 (+2까지)
        if kind == "항" and not ok_seq and last[0] > 0 and k[0] == last[0] + 2 and k[1] == 0:
            ok_seq = True
        if not ok_seq:
            continue
        before = text[max(0, m.start() - 30):m.start()]
        if kind != "항" and re.search(r"\d\.\s*$", before):  # 날짜 '2016. 3.' 안의 숫자
            continue
        after = text[m.end():m.end() + 6]
        # 항(②…)은 앞이 문장 끝이 아니어도 자른다: 호 목록 끝(`…거래 중지② …`)처럼 마침표 없이 다음 항이 오는 경우가 많음
        # 단, 본문 속 참조(`제①항`, `①부터 ③까지`, `① 및 ②`)는 자르지 않음
        # 태그 안(`<개정 2007. 3. 22, 2015 .1. 15>`, `[본조 신설 2025. 2. .5.]`)에서는 자르지 않음
        wide = text[max(0, m.start() - 60):m.start()]
        if max(wide.rfind("<"), wide.rfind("[")) > max(wide.rfind(">"), wide.rfind("]")):
            continue
        para_ok = kind == "항" and not re.search(r"(?<!삭)(?<!삭 )제\s*$", before) and \
            not re.match(r"\s*(?:부터|까지|및|또는|의|에|·|ㆍ|,|항)", after)
        # 호·목도 첫 번호(1. / 가.)를 찾은 뒤에는, 앞 항목이 마침표 없이 끝나도(`…있을 것2. 여유자금…`) 다음 번호면 자른다
        # 단, 숫자 바로 뒤(`제1항 2.`의 숫자, 금액)·'제' 뒤는 제외
        item_ok = kind != "항" and last[0] > 0 and re.search(r"[가-힣)」\"”]\s*$", before) and \
            not re.search(r"제\s*\d*\s*$", before) and re.match(r"\s*(?:[가-힣「\"“(<\[]|\d+[가-힣])", after)  # `2. 2종 …`
        # 목의 첫 `가.`가 글자 바로 뒤에 붙은 경우(`…다음 각 목에 해당하는 사항가. 조치의…나. …`): 뒤에 `나.`가 이어질 때만
        # (`경우가 …`처럼 조사 '가'를 목 기호로 잘못 잡지 않게)
        first_mok = kind == "목" and last == (0, 0) and k == (1, 0) and re.search(r"[가-힣)」\"”]\s*$", before) and \
            re.match(r"\s*[가-힣「\"“(<\[]", after) and re.search(r"[가-힣)」\"”.]\s*나\.\s?[가-힣「\"“(<\[]", text[m.end():])
        if kind == "목" and re.search(r"[가-힣]$", before) and not (item_ok or first_mok):
            continue  # 글자 바로 뒤 '다.'는 대부분 문장 끝(`한다.`) → 차례가 확인된 목만
        if kind == "목" and m.group(1) == "다" and re.search(r"[한된있없같하이르니되진런았었겠는]$", before):
            continue  # `…하여야 한다.`의 '다.'를 다목으로 보지 않음
        if m.start() == 0 or para_ok or item_ok or first_mok or re.search(r"[.다:;」>\]\)]\s*$", before):
            cuts.append((m.start(), m.group(0)))
            last = k
    if not cuts:
        return [("", text)] if text.strip() else []
    out = []
    if text[:cuts[0][0]].strip():
        out.append(("", text[:cuts[0][0]]))
    for (a, mk), (b, _) in zip(cuts, cuts[1:] + [(len(text), "")]):
        out.append((mk, text[a:b]))
    return out


def admrul_lines(auid, body):
    """행정규칙 조문 본문 → Lines (항 → 호 → 목, 상위 줄 기록)"""
    lines = Lines(auid)
    body = body.replace("ⓛ", "①").replace("〈", "<").replace("〉", ">")  # 원문 오타 ⓛ(원 l) → ①, 전각 꺾쇠 태그
    for pmk, ptxt in seq_split(body, "항"):
        para_no = str(CIRCLED.index(pmk) + 1) if pmk else ""
        hos = seq_split(ptxt, "호")
        lead = hos[0][1] if hos and not hos[0][0] else ""
        items = [h for h in hos if h[0]]
        p_uid = lines.add("항" if pmk else "본문", pmk, lead if items else ptxt, para=para_no)
        for hmk, htxt in items:
            item_no = hmk.rstrip(".")
            moks = seq_split(htxt, "목")
            mlead = moks[0][1] if moks and not moks[0][0] else ""
            mitems = [m for m in moks if m[0]]
            h_uid = lines.add("호", hmk, mlead if mitems else htxt, parent=p_uid, para=para_no, item=item_no)
            for mmk, mtxt in mitems:
                lines.add("목", mmk, mtxt, parent=h_uid, para=para_no, item=item_no, sub=mmk.rstrip("."))
    return lines


def admrul_articles(c, gate):
    A = json.loads(Path(c["path"]).read_text(encoding="utf-8"))["AdmRulService"]
    body = A.get("조문내용")
    if isinstance(body, list):
        units, method = [str(x) for x in body], "admrul_list"
    elif isinstance(body, str):
        pieces, lead = split_admrul_block(body)
        method = "admrul_split"
        g_meta = {}
        if not pieces:
            method = "guideline_items"
            cuts = split_guideline(body)
            if cuts:
                lead = body[:cuts[0][0]]
                pieces = [body[a:b] for (a, _, _), (b, _, _) in zip(cuts, cuts[1:] + [(len(body), 0, "")])]
                g_meta = {i: (k, t) for i, (_, k, t) in enumerate(cuts)}
            else:  # 번호 항목도 없음 → 규정 전체를 한 단위로
                method, lead, pieces = "whole_body", "", [body]
        units = ([lead] if lead.strip() else []) + pieces
        offset = 1 if lead.strip() else 0
        g = {"law": c["name"], "method": method, "n_pieces": len(pieces), "chars": len(body),
             "lossless": re.sub(r"\s", "", "".join(units)) == re.sub(r"\s", "", body)}
        gate.append(g)
    else:
        return []
    out, stack = [], {}
    for i, unit in enumerate(units, 1):
        unit = unit.strip()
        # 조문 뒤에 붙은 다음 장·절 제목 떼기
        tail = TAIL_CHAP.search(unit) if method != "admrul_list" else None
        next_chap = None
        if tail and ARTICLE_START.match(unit):
            next_chap, unit = tail.group(1), unit[:tail.start()].rstrip()
        if method == "whole_body":
            auid = f"{c['uid']}:1"
            lines = admrul_lines(auid, unit)
            out.append(make_article(c, auid, 1, None, "0", None, "", stack, unit, lines, "", method, Counter(),
                                    label="(전체)"))
        elif method == "guideline_items":
            meta = g_meta.get(i - 1 - offset)
            if meta is None:  # 첫 항목 앞 글 (장 제목·그림 등)
                if CHAP.match(unit):
                    stack = update_chapter(stack, unit)
                elif clean_text(unit, {}):
                    STATS["지침 첫 항목 앞 글(버림 아님, 확인용)"] += 1
                    auid = f"{c['uid']}:{i}"
                    out.append(make_article(c, auid, len(out) + 1, None, "0", None, "", stack, unit,
                                            admrul_lines(auid, unit), "", method, Counter(), label="(앞글)"))
                continue
            k, title = meta
            body_part = unit[len(title):].lstrip() if title and unit.startswith(title) else unit
            m = re.match(r"^\s*(\d+)\.\s?", body_part)
            body_part = body_part[m.end():] if m else body_part
            tail = re.search(r"(?<=[.다)>\]」])\s*(제\s*\d+\s*[편장절관]\s*[^.]{0,40})$", body_part)
            if tail:
                next_chap, body_part = tail.group(1), body_part[:tail.start()]
            auid = f"{c['uid']}:{i}"
            out.append(make_article(c, auid, len(out) + 1, None, str(k), None, title, stack, unit,
                                    admrul_lines(auid, body_part), "", method, Counter(), label=f"{k}."))
        else:
            # 맨 앞 장·절 제목 ("제1장 총칙제1-1조(목적)…")
            lead_ch = re.match(r"^(제\s*\d+\s*[편장절관]\s*[^제]{0,40}?)(?=제\s*\d+(?:-\d+)?\s*조)", unit)
            if lead_ch and CHAP.match(unit):
                stack = update_chapter(stack, lead_ch.group(1))
                unit = unit[lead_ch.end():]
            m, rest = split_head(unit)
            if not m:
                if CHAP.match(unit):
                    stack = update_chapter(stack, unit)
                continue
            auid = f"{c['uid']}:{i}"
            lines = admrul_lines(auid, rest)
            out.append(make_article(c, auid, len(out) + 1, m, m.group(1), m.group(3), m.group(4) or "", stack, unit,
                                    lines, "", method, Counter(), dash=m.group(2)))
        if next_chap:
            stack = update_chapter(stack, next_chap)
    if method != "admrul_list":
        gate[-1]["method"] = method
        gate[-1]["n_articles"] = len(out)
        nums = [(a["article_no"], a["article_dash"], a["article_sub"]) for a in out]
        gate[-1]["increasing"] = all(x < y for x, y in zip(nums, nums[1:]))
    return out


# ---------------------------------------------------------------- 조문 한 행
def make_article(c, auid, seq, head, no, sub, title, stack, raw, lines, art_ef, method, n_json, label=None, dash=None):
    rows = lines.rows
    live = [r for r in rows if not r["is_deleted"] and not r["not_in_force"]]
    text = "\n".join(r["text"] for r in live)
    embed = "\n".join(r["embed_text"] for r in live)
    head_no = head.group(1) if head else no
    sub = sub if sub not in (None, "", "0") else (head.group(3) if head else None)
    dash = dash or (head.group(2) if head else None)
    title = (title or (head.group(4) if head and head.group(4) else "") or "").strip()
    is_del = not live and any(r["is_deleted"] for r in rows) or (not rows and is_delete_body(clean_text(raw, {})))
    skip = "삭제" if is_del else ("본문 없음" if not live and not any(r["not_in_force"] for r in rows) else
                                ("시행 전" if not live else ""))
    if skip == "본문 없음":
        STATS["본문 없는 조문 (머리만)"] += 1
    ref_chars = sum(len(m.group(0)) for m in REF.finditer(text))
    return {
        "article_uid": auid, "law_uid": c["uid"], "law_name": c["name"], "law_level": c["level"], "scope": c["scope"],
        "seq": seq, "article_no": int(head_no) if str(head_no).isdigit() else 0,
        "article_dash": int(dash) if dash else 0, "article_sub": int(sub) if sub else 0,
        "article_label": label or label_of(head_no, dash, sub), "title": title,
        "chapter": " > ".join(stack[k] for k in sorted(stack)), "raw_text": raw, "text": text, "embed_body": embed,
        "article_effective_date": art_ef or "", "law_effective_date": c["effective"], "is_deleted": bool(is_del),
        "not_in_force": bool(rows) and all(r["not_in_force"] for r in rows if not r["is_deleted"]) and not live,
        "has_image": any(r["has_image"] for r in rows), "has_table": any(r["has_table"] for r in rows),
        "has_delegation": bool(DELEGATION.search(text)),
        "is_reference_list": len(text) >= 200 and ref_chars / max(len(text), 1) >= 0.5,
        "searchable": skip == "", "skip_reason": skip,
        "parse_method": method, "n_chars": len(text), "n_lines": len(live),
        "n_json_para": n_json["항"], "n_json_item": n_json["호"], "n_json_sub": n_json["목"],
        "_lines": rows,
    }


# ---------------------------------------------------------------- 실행
def main():
    sel = json.loads((WORK / "01_fetch" / "selected.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    arts, gate = [], []
    for c in sorted(sel["chosen"], key=lambda c: (c["scope"], c["name"])):
        arts += law_articles(c) if c["target"] == "law" else admrul_articles(c, gate)
    # 토큰 수 (KURE 토크나이저, 임베딩용 글 기준)
    try:
        from common import offline_if_cached
        offline_if_cached()
        from transformers import AutoTokenizer
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        tok = AutoTokenizer.from_pretrained("nlpai-lab/KURE-v1")
        texts = [a["embed_body"] for a in arts]
        for i in range(0, len(texts), 512):
            enc = tok(texts[i:i + 512], add_special_tokens=False)["input_ids"]
            for a, ids in zip(arts[i:i + 512], enc):
                a["n_tokens"] = len(ids)
    except Exception as e:
        print("토큰 수 못 셈:", e)
    acols = ["article_uid", "law_uid", "law_name", "law_level", "scope", "seq", "article_label", "article_no",
             "article_dash", "article_sub", "title", "chapter", "text", "embed_body", "raw_text",
             "article_effective_date", "law_effective_date", "is_deleted", "not_in_force", "has_image", "has_table",
             "has_delegation", "is_reference_list", "searchable", "skip_reason", "parse_method", "n_chars", "n_tokens", "n_lines",
             "n_json_para", "n_json_item", "n_json_sub"]
    lcols = ["line_uid", "article_uid", "seq", "level", "marker", "para_no", "item_no", "sub_no", "parent_line_uid",
             "text", "embed_text", "is_deleted", "not_in_force", "has_image", "has_table"]
    with open(OUT / "articles.csv", "w", encoding="utf-8-sig", newline="") as f, \
            open(OUT / "article_lines.csv", "w", encoding="utf-8-sig", newline="") as g:
        wa, wl = csv.DictWriter(f, fieldnames=acols, extrasaction="ignore"), csv.DictWriter(g, fieldnames=lcols)
        wa.writeheader()
        wl.writeheader()
        for a in arts:
            wa.writerow(a)
            wl.writerows(a["_lines"])
    (OUT / "admrul_gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "clean_stats.json").write_text(json.dumps({"rule": RULE_VERSION, "checked_on": TODAY, "stats": STATS},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    print("조문", len(arts), "· 검색 대상", sum(a["searchable"] for a in arts), "· 제외", Counter(a["skip_reason"] for a in arts if a["skip_reason"]),
          "· 줄", sum(len(a["_lines"]) for a in arts))
    print(Counter(a["parse_method"] for a in arts))
    print("지운 것:", dict(STATS))
    for g_ in gate:
        print("  관문", g_)


if __name__ == "__main__":
    main()
