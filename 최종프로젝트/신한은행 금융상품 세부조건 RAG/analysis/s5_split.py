"""2-5b 섹션 분할 (LLM 없음) — data/stage/s3 본문 → data/sections/{doc_id}.jsonl

섹션 = 제목 줄부터 다음 제목 줄 직전까지의 원문 줄들 (글자는 바꾸지 않음, 줄바꿈 정리는 2-6에서).
제목 단계: L1(큰 제목) > L2(표의 행 이름·소제목). section_path = "L1 > L2".

형식별 제목 규칙 (2-1 doc_survey.csv의 format 기준)
  table     L1: 'N. 제목' (N이 1,2,3… 순서대로) · 숫자만 있는 줄+제목 · <별지>·<…사항> (개선 3)      L2: 행 이름 사전 (s5_row_labels.py), 같은 이름이 연이어 나오면 합침,
            행 이름이 내용 없이 3개 이상 붙어 있으면 표의 열 이름(모두 무시), 2개면 묶음 행+하위 행(앞의 것만 무시).
            섹션 이름 = 원문 표기(예: '거래중지'), label = 사전의 대표 이름(예: '거래제한')
  article   L1: '제N조 제목' / '제N조(제목)'·〔제목〕·[제목] / '부칙'  (본문 속 '제7조에 따라' 같은 인용은 제외)
  numbered  L1: 숫자만 있는 줄 + 다음 줄 제목 (순서대로) / 'N. 제목' (순서대로) / '▣ 제목'
  invest    L1: 숫자만 있는 줄 + 다음 줄 제목 / 'N. 제목' (순서대로, '제N부'·'【…】'에서 1부터 다시)   L2: '■항목명'
  other     L1: 'N. 제목'·'N 제목' (순서대로) / '◉ 제목'      L2: '□ 소제목' + 행 이름 사전
보정: table인데 행 이름이 3개 미만이면 numbered 규칙 / 섹션이 2개 이하면 쪽 단위로 나눔(fallback=page)
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s5_row_labels import VARIANTS, drop_header_runs, find_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
S3, OUT, META = ROOT / "data" / "stage" / "s3", ROOT / "data" / "sections", ROOT / "data" / "meta"

NUM_TITLE = re.compile(r"^(\d{1,2})\s*[.)]?\s+([가-힣][^:：]{0,28})$")   # '1. 상품개요 및 특징', '2 거래 조건' (제목은 한글로 시작: OCR 표 행 '1 0 0 X …' 제외)
NUM_TITLE_DOT = re.compile(r"^(\d{1,2})\s*\.\s*([가-힣A-Za-z(][^:：]{0,28})$")
NUM_ONLY = re.compile(r"^(\d{1,2})$")
# 투자설명서: '제1부 …', '【 본 문 】' 에서 번호가 1부터 다시 시작
RESET = re.compile(r"^(제\s*\d+\s*부\s*\S.{0,30}|【[^】]{1,20}】)$")
SQ_ITEM = re.compile(r"^■\s*([^:：■]{1,20})(?:[:：].*)?$")   # 간이투자설명서 '■상품명', '■위험등급 : …'
BR_OPEN, BR_CLOSE = "(（【「〔[", ")）】」〕]"
ARTICLE = re.compile(r"^제\s*(\d+)\s*조(?:의\s*\d+)?\s*(?:[" + re.escape(BR_OPEN) + r"]\s*[^" + re.escape(BR_CLOSE) + r"]{1,30}[" + re.escape(BR_CLOSE) + r"]|[가-힣][가-힣\s·및의,]{0,25}(?<![에의를을와과]))\s*$")
ARTICLE_LOOSE = re.compile(r"^제\s*(\d+)\s*조(?:의\s*\d+)?\s*[" + re.escape(BR_OPEN) + r"]")
BUCHIK = re.compile(r"^부\s*칙\s*(?:<[^>]*>)?\s*$")
SQ_TITLE = re.compile(r"^[▣◉]\s*\S.{0,30}$")
BOX_TITLE = re.compile(r"^□\s*\S.{0,30}$")
SENT_END = re.compile(r"(다|[해세어예네]요|음|함)\.?$")   # 개선 4: '상품개요·공모개요'의 '요'는 문장 끝이 아님
MAX_CHARS = 4000   # 이보다 긴 섹션은 쪽 경계에서 다시 나눔
# 개선 3: '<별지> 연계·제휴서비스', '<혁신금융서비스 승인에 관한 사항>' — 그림 제목('<금리 보장형>', '<… 예시>')은 제외
ANGLE_TITLE = re.compile(r"^<\s*(?:(?:별지|붙임|부록|첨부)[^<>]{0,30}>.{0,30}|[가-힣][^<>]{0,36}(?:사항|안내)\s*>)$")
IMPROVE3 = "--no-improve3" not in sys.argv              # 비교용: 개선 3 끄기
IMPROVE5 = "--no-improve5" not in sys.argv              # 비교용: 개선 5 끄기 (쪼개진 행 이름·서식 칸·이름 붙은 안내)
IMPROVE4 = "--no-improve4" not in sys.argv              # 비교용: 개선 4 끄기 (목차형·투자설명서·기타)
ROMAN = re.compile(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]\s*\.\s*\S.{0,30}$")              # 'Ⅰ. 모집 또는 매출에 관한 일반사항' → 번호 1부터 다시
PUA_TITLE = re.compile(r"^[\ue000-\uf8ff]\s*((?!Q\d)[가-힣][^:：(\ue000-\uf8ff]{1,28})$")   # PDF 전용 기호 '\uf000 상품개요 및 특성'
ARTICLE_BR = re.compile(r"^제\s*\d+\s*조(?:의\s*\d+)?\s*[〔\[(（][^〕\])）]{1,20}[〕\])）]")
ANNEX = re.compile(r"^별표\s*\d+$")
BRACKET_TITLE = re.compile(r"^(?:※\s*)?\[[^\]]{2,20}\]\s*[^\s].{0,38}$|^\[[^\]]{2,20}\]$")      # '[부록] 대출상품 …', '[설명의무이행확인]'
SUB_GANA = re.compile(r"^[가나다라마바사아자차]\.\s*[가-힣].{0,28}$")                 # '가. 청약철회권'
SUB_PAREN = re.compile(r"^(?:\(\d{1,2}\)|\d\))\s*[가-힣].{0,28}$")                  # '(1) 공모개요', '1) 매입시 가격결정방법'
SUB_SYMBOL = re.compile(r"^[■◉◾◆☑⋄▶￭※][\s\x00-\x1f]*[가-힣][^:：]{1,28}$")                    # '■ 상환방법별 특징' (문장은 SENT_END로 제외)


def load(doc_id: str):
    """[(page, line)] — 빈 줄 제외, 쪽 표시 제외"""
    out, page = [], 0
    for raw in (S3 / f"{doc_id}.txt").read_text().split("\n"):
        m = re.match(r"^=== p\.(\d+) ===", raw)
        if m:
            page = int(m.group(1))
            continue
        if raw.strip():
            out.append((page, raw.strip()))
    return out


def seq_headings(lines, patterns, allow_num_only=False):
    """순서대로 증가하는 번호 제목만 받아들인다 → {줄번호: 제목}
    개선 4: 숫자만 있는 줄은 다음 줄이 한글로 시작하는 제목일 때만 (표의 행 번호 '15' / '38%' 제외),
            두 단 배치로 순서가 섞인 경우(1·2·3·4·7·6·5·8)를 위해 아직 안 나온 번호면 expect+3까지 허용"""
    heads, expect, used, i = {}, 1, set(), 0
    while i < len(lines):
        s = lines[i][1]
        hit = None
        if allow_num_only and (RESET.match(s) or (IMPROVE4 and ROMAN.match(s))):
            heads[i] = re.sub(r"\s+", " ", s)
            expect, used = 1, set()
            i += 1
            continue
        for p in patterns:
            m = p.match(s)
            if m and int(m.group(1)) == expect and not SENT_END.search(m.group(2)):
                hit = (i, expect, f"{expect}. {m.group(2).strip()}")
                break
        if not hit and allow_num_only:
            m = NUM_ONLY.match(s)
            if m and i + 1 < len(lines):
                n, nxt = int(m.group(1)), lines[i + 1][1]
                ok_n = n == expect or (IMPROVE4 and n not in used and n <= expect + 3)
                ok_t = len(nxt) <= 30 and not SENT_END.search(nxt) and not NUM_ONLY.match(nxt)
                if IMPROVE4:
                    ok_t = ok_t and re.match(r"^[가-힣A-Za-z]", nxt) is not None
                if ok_n and ok_t:
                    hit = (i, n, f"{n}. {nxt}")
        if hit:
            heads[hit[0]] = hit[2]
            used.add(hit[1])
            expect = max(used) + 1
        i += 1
    return heads


def is_list_or_sentence(texts, k):
    """기호로 시작하는 줄이 문장이거나, 다음 줄이 같은 기호로 시작하는 목록의 한 항목인지"""
    s = texts[k]
    return bool(SENT_END.search(s)) or (k + 1 < len(texts) and texts[k + 1][:1] == s[:1])


def glossary_terms(texts, H):
    """개선 4: '용어 안내' 부록 — 짧은 용어 줄 다음 줄이 •·▪·- 로 시작하는 설명이면 용어 하나 = 섹션"""
    l1 = sorted(k for k in H if H[k][0] == 1)
    for i, k in enumerate(l1):
        if "용어" not in H[k][1]:
            continue
        end = l1[i + 1] if i + 1 < len(l1) else len(texts)
        for j in range(k + 1, end - 1):
            t = texts[j]
            if j in H or len(t) > 25 or re.match(r"^[•▪\-*※(A-Za-z]", t) or SENT_END.search(t) or \
                    t.rstrip().endswith((".", ",")) or t in ("용어", "설명"):
                continue
            n = j + 1   # 용어가 여러 줄이면 '(LTV, …)' 같은 영문·괄호 줄을 건너뜀
            while n < end and n - j <= 3 and re.match(r"^[(A-Za-z]", texts[n]):
                n += 1
            if n < end and re.match(r"^[•▪\-]", texts[n]):
                H[j] = (2, re.sub(r"\s+", " ", t))


# 개선 5: 표 칸에서 여러 줄로 쪼개진 행 이름 ('증권의 / 종류', '처분손익 / 또는 / 평가손익의 / 확정')의 끝말
LABEL_END = re.compile(r"(종류|유무|여부|기준|가격|방법|구조|시점|시기|기간|사유|비용|수수료|소득세|서비스|사항|의무|청구권|확정|정산|검증|한도|대상|고객|구분|개요|이율|이자율|금액|통화|제한|요건|자격|조건|주택)$")
SHORT_PIECE = re.compile(r"^[가-힣][가-힣·/ ]{0,6}$")
FORM_FIELD = re.compile(r"^[\ue000-\uf8ff☑□][\s\x00-\x1f]*([가-힣][^:：]{1,25}?)\s*[:：]")   # '□ 중도상환해약금(률) : (   )'
NAMED_NOTE = re.compile(r"^[※\-•]\s*([가-힣][^:：]{1,25}?)\s*[:：]")                   # '※ 위법계약해지권: …', '- 자료열람요구권 : …'


COL_HEADS = {"구분", "내용", "항목", "비고", "용어", "설명"}


def split_labels(texts, H):
    """개선 5: 짧은 조각 2~4줄을 이으면 행 이름 끝말로 끝나고, 바로 다음 줄이 조각이 아니면 그 첫 줄이 행 이름.
    표 머리(구분·내용·항목·비고)가 섞이면 제외, 앞에서 이미 나온 행 이름이 다음 쪽에 되풀이되면(표가 쪽을 넘김) 제외"""
    seen_labels = set()
    i = 0
    while i < len(texts):
        hit = 0
        for n in (4, 3, 2):
            seg = texts[i:i + n]
            if len(seg) < n or i + n >= len(texts) or not all(SHORT_PIECE.match(t) for t in seg):
                continue
            joined = re.sub(r"\s", "", "".join(seg))
            nxt = texts[i + n]
            prev_title = H[i - 1][1] if i - 1 in H else ""
            content_next = (len(nxt) >= 10 or re.match(r"^[￭•\-■※☞➊-➓❶-❿]", nxt)) and \
                not re.match(r"^[\ue000-\uf8ff(]", nxt)
            if len(joined) <= 18 and (LABEL_END.search(joined) or joined in VARIANTS) and content_next and \
                    len(set(seg)) == n and re.sub(r"\s", "", seg[0]) not in re.sub(r"\s", "", prev_title) and \
                    not any(re.sub(r"\s", "", t) in COL_HEADS for t in seg) and i not in H:
                if joined not in seen_labels:
                    H[i] = (2, " ".join(t.strip() for t in seg))
                seen_labels.add(joined)
                hit = n
                break
        i += hit or 1


def form_and_notes(texts, H):
    """개선 5: 설명이 따라오는 서식 칸 제목, 이름 붙은 안내(사전의 행 이름 또는 '…사항')"""
    for k, s in enumerate(texts):
        if k in H:
            continue
        nxt = texts[k + 1] if k + 1 < len(texts) else ""
        if (m := FORM_FIELD.match(s)) and re.match(r"^[☞※•*]|^-(?!-)|^\(예시\)", nxt):
            H[k] = (2, m.group(1).strip())
        elif (m := NAMED_NOTE.match(s)) and (re.sub(r"\s", "", m.group(1)) in VARIANTS or
                                             re.sub(r"\s", "", m.group(1)).endswith("사항")):
            H[k] = (2, m.group(1).strip())


def extra_headings(texts, H):
    """개선 4 (목차형·투자설명서·기타 공통): 기호·괄호 제목, 소제목, 조문 → H에 추가, 내용 없는 연속 제목 정리"""
    articles = set()
    boxes = {c for t in texts for c in set(t) if "\ue000" <= c <= "\uf8ff" and t.count(c) >= 3}   # '□ 변동 □ 고정 □ 혼합' 같은 선택 칸
    seen = {}   # 같은 [괄호 글]이 문서에 여러 번 나오면 확인 문구 (예: '[주요내용 및 고객부담비용]')
    for t in texts:
        for br in re.findall(r"\[[^\]]{2,20}\]", t):
            seen[br] = seen.get(br, 0) + 1
    for k, s in enumerate(texts):
        if k in H:
            continue
        if ARTICLE.match(s) or (ARTICLE_LOOSE.match(s) and (len(s) <= 40 or ARTICLE_BR.match(s))):   # 뒤에 붙은 약관
            H[k] = (1, re.sub(r"\s+", " ", s))
            articles.add(k)
        elif (m := PUA_TITLE.match(s)) and s[0] not in boxes and not SENT_END.search(s):          # PDF 전용 기호 제목
            H[k] = (1, m.group(1).strip())
        elif ANNEX.match(s) or (BRACKET_TITLE.match(s) and seen.get(re.match(r"(?:※\s*)?(\[[^\]]*\])", s).group(1), 0) == 1):                      # 별표1, [부록] …, [설명의무이행확인]
            H[k] = (1, re.sub(r"\s+", " ", s.lstrip("※ ")))
        elif (SUB_GANA.match(s) or SUB_PAREN.match(s) or SUB_SYMBOL.match(s)) and \
                not SENT_END.search(s) and not s.rstrip().endswith((",", "·", "및")):
            if SUB_SYMBOL.match(s) and k + 1 < len(texts) and texts[k + 1][:1] == s[:1]:
                continue   # 같은 기호 목록의 한 항목
            H[k] = (2, re.sub(r"\s+", " ", s))
    glossary_terms(texts, H)
    if IMPROVE5:
        split_labels(texts, H)
        form_and_notes(texts, H)
    drop_adjacent(texts, H, articles)


def drop_adjacent(texts, H, articles):
    """개선 4: 같은 글자의 제목이 내용 없이 되풀이되면 뒤의 것은 경계로 쓰지 않음.
    뒤에 붙은 약관(조문) 안의 '3. …' 번호 항목은 큰 제목이 아님"""
    if articles:
        first = min(articles)
        for k in [k for k in H if k > first and k not in articles and re.match(r"^\d+\. ", H[k][1])]:
            del H[k]
    ks = sorted(H)
    for a, b in zip(ks, ks[1:]):
        if a in H and b in H and re.sub(r"\s|^\d+\.", "", H[b][1]) == re.sub(r"\s|^\d+\.", "", H[a][1]) and \
                all(re.sub(r"\s", "", texts[j]) == re.sub(r"\s|^\d+\.", "", H[a][1]) for j in range(a + 1, b)):
            del H[b]


def headings(fmt, lines):
    """→ {줄번호: (level, 제목)}"""
    H = {}
    texts = [s for _, s in lines]
    if fmt == "table":
        # 개선 3: 숫자만 있는 줄 + 다음 줄 제목도 큰 제목 (예: '1' / '상품개요')
        for k, v in seq_headings(lines, [NUM_TITLE_DOT], allow_num_only=IMPROVE3).items():
            H[k] = (1, v)
        if IMPROVE3:   # '<별지>', '<혁신금융서비스 …>' 같은 꺾쇠 제목 줄
            for k, s in enumerate(texts):
                if k not in H and ANGLE_TITLE.match(s):
                    H[k] = (1, s)
        prev, prev_key = None, None
        for k, lab, n, raw in drop_header_runs(find_labels(texts, with_raw=True, generic=IMPROVE3)):
            if k in H:
                continue
            l1_before = [j for j in H if H[j][0] == 1 and j < k]
            between_l1 = any(prev is not None and prev < j for j in l1_before)
            # 개선 3: 같은 대표 이름이어도 원문 표기가 다르면 다른 행 (저축한도 ≠ 가입금액, 거래중지 ≠ 거래제한)
            key = re.sub(r"\s", "", raw) if IMPROVE3 else lab
            if key == prev_key and not between_l1:
                continue  # 같은 행 이름이 연이어 나오면 (표 안의 열 이름 '우대/이자율' 등) 나누지 않음
            # 개선 3: 큰 제목 바로 다음 줄이 같은 말이면 (예: '3' / '유의사항') 따로 나누지 않음
            if IMPROVE3 and l1_before and k - max(l1_before) <= 1 and \
                    re.sub(r"\s", "", raw) in re.sub(r"\s", "", H[max(l1_before)][1]):
                continue
            H[k] = (2, raw, lab)   # 섹션 이름은 원문 표기, 대표 이름은 label로
            prev, prev_key = k, key
    elif fmt == "article":
        n_buchik = 0
        for k, s in enumerate(texts):
            if ARTICLE.match(s) or (ARTICLE_LOOSE.match(s) and len(s) <= 40):
                H[k] = (1, re.sub(r"\s+", " ", s))
            elif BUCHIK.match(s):
                n_buchik += 1
                H[k] = (1, "부칙" if n_buchik == 1 else f"부칙 ({n_buchik})")
    elif fmt in ("numbered", "invest"):
        for k, v in seq_headings(lines, [NUM_TITLE_DOT], allow_num_only=True).items():
            H[k] = (1, v)
        for k, s in enumerate(texts):
            if SQ_TITLE.match(s) and k not in H and not (IMPROVE4 and is_list_or_sentence(texts, k)):
                H[k] = (1, s)
            elif fmt == "invest" and k not in H and (m := SQ_ITEM.match(s)) and \
                    not (IMPROVE4 and SENT_END.search(m.group(1).strip())):
                H[k] = (2, m.group(1).strip())
        if IMPROVE4:
            extra_headings(texts, H)
    elif fmt == "other":
        for k, v in seq_headings(lines, [NUM_TITLE]).items():
            H[k] = (1, v)
        for k, s in enumerate(texts):
            if k in H:
                continue
            if SQ_TITLE.match(s) and not (IMPROVE4 and is_list_or_sentence(texts, k)):
                H[k] = (1, s)
            elif BOX_TITLE.match(s):
                H[k] = (2, s)
        if IMPROVE4:
            extra_headings(texts, H)
        for k, lab, n, raw in drop_header_runs(find_labels(texts, with_raw=True)):
            if k not in H:
                H[k] = (2, raw, lab)
    return H


def build(doc_id, fmt, lines, H, fallback=False):
    secs, l1, cur = [], "(머리말)", None
    starts = sorted(H)
    bounds = ([0] if not starts or starts[0] != 0 else []) + starts
    for bi, st in enumerate(bounds):
        en = bounds[bi + 1] if bi + 1 < len(bounds) else len(lines)
        label = ""
        if st in H:
            lv, title, *rest = H[st]
            label = rest[0] if rest else ""
            if lv == 1:
                l1, path = title, title
            else:
                path = title if l1 == "(머리말)" else f"{l1} > {title}"
        else:
            path = "(머리말)"
        chunk = lines[st:en]
        if not chunk:
            continue
        # 원문에 소제목이 없어 너무 긴 섹션은 쪽 경계에서 한 번 더 나눔 (예: 수수료표가 든 조문, 투자설명서 본문)
        pieces = [chunk]
        if sum(len(s) for _, s in chunk) > MAX_CHARS and chunk[0][0] != chunk[-1][0]:
            pieces = [[x for x in chunk if x[0] == p] for p in sorted({x[0] for x in chunk})]
        for pi, piece in enumerate(pieces):
            p_path = path if len(pieces) == 1 else f"{path} (p.{piece[0][0]})"
            secs.append({"doc_id": doc_id, "seq": len(secs) + 1, "section": p_path.split(" > ")[-1],
                         "section_path": p_path, "label": label, "page_start": piece[0][0], "page_end": piece[-1][0],
                         "n_lines": len(piece), "content": "\n".join(s for _, s in piece),
                         "split_rule": "page" if fallback else (fmt if len(pieces) == 1 else f"{fmt}+page")})
    return secs


def by_page(doc_id, lines):
    H, last = {}, None
    for k, (p, _) in enumerate(lines):
        if p != last:
            H[k] = (1, f"p.{p}")
            last = p
    return build(doc_id, "page", lines, H, fallback=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    survey = {r["doc_id"]: r for r in csv.DictReader(open(META / "doc_survey.csv"))}
    summary, bad = [], []
    for did, sv in survey.items():
        lines = load(did)
        fmt = sv["format"]
        H = headings(fmt, lines)
        if fmt == "table" and sum(1 for v in H.values() if v[0] == 2) < 3:
            fmt = "numbered"
            H = headings(fmt, lines)
            if len(H) < 3:
                fmt = "other"
                H = headings(fmt, lines)
        secs = build(did, fmt, lines, H)
        if len(secs) <= 2:
            secs = by_page(did, lines)
            fmt = "page"
        # 글자 보존: 섹션 내용을 이으면 본문 줄과 같아야 함
        joined = "\n".join(s["content"] for s in secs)
        if joined != "\n".join(s for _, s in lines):
            bad.append(did)
        with open(OUT / f"{did}.jsonl", "w") as f:
            for s in secs:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        L = [len(s["content"]) for s in secs]
        summary.append({"doc_id": did, "title": sv["title"], "format": sv["format"], "rule": fmt,
                        "n_sections": len(secs), "max_chars": max(L), "median_chars": sorted(L)[len(L) // 2],
                        "preamble_chars": sum(len(s["content"]) for s in secs if s["section_path"] == "(머리말)")})
    with open(META / "s5_sections.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print("글자 보존 실패:", bad)
    print("적용 규칙:", Counter(s["rule"] for s in summary))
    print("섹션 합계:", sum(s["n_sections"] for s in summary))
    for fm in ("table", "article", "numbered", "invest", "other"):
        ss = [s for s in summary if s["format"] == fm]
        n = sorted(s["n_sections"] for s in ss)
        print(f"  {fm:9} 문서 {len(ss):3}  섹션 수 최소 {n[0]} / 중앙 {n[len(n)//2]} / 최대 {n[-1]}")


if __name__ == "__main__":
    main()
