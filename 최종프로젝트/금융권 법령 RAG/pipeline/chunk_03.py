"""기획안 5-3 청크

입력: data/work/02_clean/articles.csv (검색 대상 조문), article_lines.csv (항·호·목 줄)
출력: data/work/04_chunks/chunks.csv, chunks_input.jsonl (코랩 임베딩 입력: chunk_id, embed_text, input_hash)

규칙 (기획안 5-3-2)
1. 조문 하나 = 기본 한 조각 (본문 900자 이하)
2. 900자를 넘으면 가장 위 단계 줄 경계(항 → 호 → 목 → 세목)에서 나누고, 줄 경계가 없으면 문장 끝에서
3. 나눈 형제 줄을 600자 근처까지 묶음. 하위 줄은 상위 줄과 같은 조각에
4. 200자 미만 조각은 앞 조각에 붙임 (합쳐 900자 이하일 때)
5. 머리말 포함 512토큰을 넘으면 문장 끝에서 한 번 더 나눔
6. 삭제·시행 전 줄은 넣지 않음 (5-2에서 이미 뺀 줄만 씀)

    python pipeline/chunk_03.py
"""
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

from common import TODAY, WORK

csv.field_size_limit(2**31 - 1)
OUT = WORK / "04_chunks"
RULE_VERSION = "5-3 v3 (기획안 v0.15)"
WHOLE, PACK, SHORT, MAX_TOK = 900, 600, 200, 512
STATS = Counter()

os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

hf_logging.set_verbosity_error()
TOK = AutoTokenizer.from_pretrained("nlpai-lab/KURE-v1")


def n_tok(t):
    return len(TOK(t, add_special_tokens=True)["input_ids"])  # 모델이 실제로 읽는 길이 (<s> </s> 포함)


# ---------------------------------------------------------------- 줄 나무
class Node:
    def __init__(self, ln):
        self.ln, self.kids = ln, []

    def size(self):
        return len(self.ln["text"]) + sum(k.size() + 1 for k in self.kids)

    def flat(self):
        return [self.ln] + [x for k in self.kids for x in k.flat()]


def build_tree(lines):
    live = [ln for ln in lines if ln["is_deleted"] != "True" and ln["not_in_force"] != "True"]
    nodes = {ln["line_uid"]: Node(ln) for ln in live}
    roots = []
    for ln in live:
        p = nodes.get(ln["parent_line_uid"])
        (p.kids if p else roots).append(nodes[ln["line_uid"]])
    return roots


# ---------------------------------------------------------------- 문장 나누기
# 나누는 자리: 줄바꿈, '다.' 뒤, 숫자가 아닌 글자 뒤의 '.;:' 뒤, 한 줄 안 번호 목록의 번호 앞 (' 5. ')
# 번호 바로 뒤('1.' 다음)에서는 자르지 않는다 (번호와 내용이 떨어짐)
SENT = re.compile(r"\n|(?<=다\.)\s+|(?<=다\.)(?=[^\s\d)」\]])|(?<=[^\d\s][.;:])\s+(?=\S)|\s+(?=\d{1,2}(?:의\d+)?\.\s)")


def sentence_pieces(text, limit):
    """글을 줄·문장 끝에서 limit 글자 근처까지 묶어 나눔.
    한 문장이 limit 보다 길어도 900자 이하면 자르지 않고 한 조각으로 둠 (문장 중간 잘림 방지).
    900자를 넘는 한 문장만 쉼표 → 공백에서 자름 (문장 끝이 없는 긴 준용 문장 등, 보고서 예외 목록)"""
    sents = [x.strip() for x in SENT.split(text) if x and x.strip()]
    out, cur = [], ""
    for x in sents:
        if len(x) > WHOLE:
            if cur:
                out.append(cur)
                cur = ""
            STATS["900자 넘는 한 문장 자름"] += 1
            out += hard_cut(x, limit)
            continue
        if cur and len(cur) + 1 + len(x) > limit:
            out.append(cur)
            cur = x
        else:
            cur = (cur + " " + x) if cur else x
    if cur:
        out.append(cur)
    # 50자 미만 문장 조각은 앞(안 되면 뒤) 조각에 붙임
    merged = []
    for x in out:
        if merged and (len(x) < 50 or len(merged[-1]) < 50) and len(merged[-1]) + 1 + len(x) <= WHOLE:
            joined = merged[-1] + " " + x
            if x in HARD:  # 뒤 조각이 문장 중간에서 끝났다면 합친 조각도 그렇게 표시
                HARD.add(joined)
            merged[-1] = joined
            STATS["짧은 문장 조각 붙임"] += 1
        else:
            merged.append(x)
    return merged


HARD = set()  # 문장 끝이 아닌 곳에서 자른 조각 글 (보고서 예외 목록)


def hard_cut(x, limit):
    """문장 끝이 없는 긴 글: 쉼표 뒤 → 공백에서 limit 근처로"""
    out = []
    while len(x) > limit:
        cut = max(x.rfind(", ", 0, limit), x.rfind(",", 0, limit))
        if cut < limit // 2:
            cut = x.rfind(" ", 0, limit)
        cut = cut + 1 if cut >= limit // 2 else limit
        out.append(x[:cut].strip())
        x = x[cut:].strip()
    if x:
        out.append(x)
    HARD.update(out[:-1])  # 마지막 조각은 원래 문장 끝으로 끝남
    return out


# ---------------------------------------------------------------- 묶기
def pack(nodes, ctx):
    """형제 줄 목록 → 조각 목록. 조각 = {'lines': [줄], 'ctx': [상위 줄], 'split_text': 문장으로 나눈 글 또는 None}"""
    chunks, cur, cur_len = [], [], 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            chunks.append({"lines": [x for n in cur for x in n.flat()], "ctx": ctx, "piece": None})
        cur, cur_len = [], 0

    for n in nodes:
        sz = n.size()
        if sz > WHOLE and n.kids:
            flush()
            # 상위 줄은 첫 묶음과 함께, 뒤 묶음에는 '이어서: 상위 줄'
            first = Node(n.ln)
            sub = pack([first] + n.kids, ctx + [n.ln])
            sub[0]["ctx"] = ctx
            chunks += sub
            STATS["하위 단계로 내려가 나눔"] += 1
        elif sz > WHOLE:
            flush()
            for i, piece in enumerate(sentence_pieces(n.ln["text"], PACK)):
                chunks.append({"lines": [n.ln], "ctx": ctx if i == 0 else ctx + [n.ln], "piece": piece})
            STATS["한 줄을 문장으로 나눔"] += 1
        else:
            if cur and cur_len + 1 + sz > PACK:
                flush()
            cur.append(n)
            cur_len += sz + (1 if cur_len else 0)
    flush()
    return chunks


def chunk_text(c, key):
    if c["piece"] is not None:
        if key == "text":
            return c["piece"]
        # 임베딩용: 같은 자리의 한자 괄호 뺀 글 (문장 조각은 표시용 글 기준으로 잘렸으므로 조각에서 직접 뺌)
        from clean_02 import embed_of
        return embed_of(c["piece"])
    return "\n".join(ln[key] for ln in c["lines"])


def merge_short(chunks, a=None, check_tok=False):
    """200자 미만 조각은 앞 조각에 붙임 (합쳐 900자 이하). 앞에 붙일 수 없으면(맨 앞, 앞이 문장 조각, 합치면 900자 초과) 뒤 조각에.
    50자 미만은 900자를 넘어도 머리말 포함 512토큰 안이면 붙임"""
    def can(x, y):
        if x["piece"] is not None or y["piece"] is not None or x.get("def") or y.get("def"):
            return False  # 문장 조각·정의 호 조각(규칙 7)은 붙이지 않음
        tx, ty = chunk_text(x, "text"), chunk_text(y, "text")
        if len(tx) + 1 + len(ty) <= WHOLE and not check_tok:
            return True
        if (len(tx) + 1 + len(ty) <= WHOLE or min(len(tx), len(ty)) < 50) and a is not None:
            joined = {"lines": x["lines"] + y["lines"], "ctx": x["ctx"], "piece": None}
            return n_tok(header(a, joined) + "\n" + chunk_text(joined, "embed_text")) <= MAX_TOK
        return False

    out = []
    for c in chunks:  # 앞에 붙이기
        if out and len(chunk_text(c, "text")) < SHORT and can(out[-1], c):
            out[-1] = {"lines": out[-1]["lines"] + c["lines"], "ctx": out[-1]["ctx"], "piece": None}
            STATS["짧은 조각 앞에 붙임"] += 1
        else:
            out.append(c)
    i, res = 0, []
    while i < len(out):  # 뒤에 붙이기
        c = out[i]
        if i + 1 < len(out) and len(chunk_text(c, "text")) < SHORT and can(c, out[i + 1]):
            out[i + 1] = {"lines": c["lines"] + out[i + 1]["lines"], "ctx": c["ctx"], "piece": None}
            STATS["짧은 조각 뒤에 붙임"] += 1
        else:
            res.append(c)
        i += 1
    return res


# ---------------------------------------------------------------- 머리말·출처
def head_title(a):
    if a["article_label"] == "(전체)":
        return ""
    return a["article_label"] + (f"({a['title']})" if a["title"] else "")


def ctx_text(ctx):
    parts = []
    for ln in ctx:
        t = ln["text"].replace("\n", " ")
        parts.append(t if len(t) <= 40 else t[:40] + "…")
    return " > ".join(parts)


def header(a, c):
    h = [a["law_name"], a["law_level"]]
    if head_title(a):
        h.append(head_title(a))
    if c["ctx"]:
        h.append("이어서: " + ctx_text(c["ctx"]))
    return "[" + " | ".join(h) + "]"


def cite(a, c, whole):
    base = f"{a['law_name']} {head_title(a)}".strip()
    if whole:
        return base
    lines = c["lines"] + [ln for ln in c["ctx"]]
    paras = sorted({int(ln["para_no"]) for ln in c["lines"] if ln["para_no"].isdigit()})
    items = [ln["item_no"] for ln in c["lines"] if ln["item_no"] and ln["level"] in ("호",)]
    s = base
    if paras:
        s += f" 제{paras[0]}항" if len(paras) == 1 else f" 제{paras[0]}항~제{paras[-1]}항"
    has_para_line = any(ln["level"] == "항" for ln in c["lines"])
    if items and (not has_para_line or not paras):
        s += f" 제{items[0]}호" if len(items) == 1 else f" 제{items[0]}호~제{items[-1]}호"
    return s


def markers(c):
    ms = [ln["marker"] for ln in c["lines"] if ln["marker"]]
    return (ms[0], ms[-1]) if ms else ("", "")


def fits(a, c):
    return n_tok(header(a, c) + "\n" + chunk_text(c, "embed_text")) <= MAX_TOK


def split_by_tokens(a, c):
    """512토큰을 넘는 조각 → (1) 줄 단위로 다시 묶고, (2) 한 줄이 혼자 넘을 때만 그 줄을 문장으로.
    줄을 알고 있어야 출처(제O항 제O호)를 정확히 만들 수 있다 (처음엔 이어 붙인 글을 문장으로 잘라 출처가 넓게 나왔음)"""
    if c["piece"] is not None:  # 이미 문장 조각이면 그 조각 글만 다시 나눔 (줄 전체를 다시 나누면 글이 겹침)
        return pieces_of_line(a, c["lines"][0], c["ctx"], c["piece"])
    by_uid = {ln["line_uid"]: ln for ln in c["lines"]}

    def ctx_for(first):
        # 상위 줄을 끝까지 (목 → 호 → 항): '이어서: ③ … > 3. …'
        chain, p = [], by_uid.get(first["parent_line_uid"])
        while p is not None:
            chain.insert(0, p)
            p = by_uid.get(p["parent_line_uid"])
        # 원래 조각의 '이어서'는 새 조각의 맨 위 줄이 그 아래에 있을 때만 물려받음 (② 안에서 시작하면 ①은 빼야 함)
        anchor = chain[0] if chain else first
        have = {ln["line_uid"] for ln in c["ctx"]}
        base = c["ctx"] if anchor["parent_line_uid"] in have else []
        return base + chain

    out, cur = [], []
    for ln in c["lines"]:
        trial = {"lines": cur + [ln], "ctx": ctx_for((cur + [ln])[0]) if cur else (c["ctx"] if not out else ctx_for(ln)), "piece": None}
        if fits(a, trial):
            cur = trial["lines"]
            continue
        if cur:
            out.append({"lines": cur, "ctx": c["ctx"] if not out else ctx_for(cur[0]), "piece": None})
            cur = []
        single = {"lines": [ln], "ctx": c["ctx"] if not out else ctx_for(ln), "piece": None}
        if fits(a, single):
            cur = [ln]
        else:  # 한 줄이 혼자 512토큰을 넘음 → 그 줄만 문장으로
            out += pieces_of_line(a, ln, single["ctx"])
    if cur:
        out.append({"lines": cur, "ctx": c["ctx"] if not out else ctx_for(cur[0]), "piece": None})
    # 마지막 조각이 짧으면(100자 미만) 앞 조각의 끝 줄을 넘겨 받아 고르게 (출처는 줄로 계산되므로 그대로 정확)
    while len(out) >= 2 and out[-1]["piece"] is None and out[-2]["piece"] is None and len(out[-2]["lines"]) > 1             and len(chunk_text(out[-1], "text")) < 100:
        moved = {"lines": out[-2]["lines"][-1:] + out[-1]["lines"], "ctx": ctx_for(out[-2]["lines"][-1]), "piece": None}
        if not fits(a, moved):
            break
        out[-2] = {**out[-2], "lines": out[-2]["lines"][:-1]}
        out[-1] = moved
        STATS["512토큰 나눔 끝 조각 고르게"] += 1
    return out


def pieces_of_line(a, ln, ctx, text=None):
    out = []
    for i, pc in enumerate(sentence_pieces(text if text is not None else ln["text"], 450)):
        probe = {"lines": [ln], "ctx": ctx if i == 0 else ctx + [ln], "piece": pc}
        if fits(a, probe):
            out.append(probe)
            continue
        STATS["512토큰 문장도 넘어 글자로 자름"] += 1
        lim = 300
        sub = hard_cut(pc, lim)
        while any(n_tok(header(a, probe) + "\n" + x) > MAX_TOK for x in sub) and lim > 80:
            lim -= 40
            sub = hard_cut(pc, lim)
        out += [{"lines": [ln], "ctx": ctx if (i == 0 and j == 0) else ctx + [ln], "piece": x} for j, x in enumerate(sub)]
    return out


# ---------------------------------------------------------------- 조문 → 조각
DEF_TITLE = re.compile(r"^(?:용어의\s*)?정의(?:\s*등)?$")  # 규칙 7: 제목만으로 정함 (법령·조문 이름은 쓰지 않음)


def is_definition(a):
    return bool(DEF_TITLE.match((a["title"] or "").strip()))


# 용어를 정의하는 줄: (기호) + 따옴표 속 용어 + ~란 / ~이라 함은 / ~라 함은 / ~이라 한다
DEF_LINE = re.compile(r'^(?:[①-⑳]|\d+(?:의\d+)?\.|[가-힣]\.)?\s*[“"「『][^“"」』\n]{1,60}[”"」』]\s*'
                      r'(?:\([^()]{1,40}\)\s*)?(?:이?란|(?:이)?라\s*함은|이?라고?\s*한다|이?라\s*하며)')  # `"체약상대국"(締約相對國)이란`
LEAD = re.compile(r"(?:다음과\s*같다|다음\s*각\s*호와\s*같다)\.?\s*$")


def definition_chunks(roots):
    """규칙 7 (v0.15): 정의 조문은 '정의 줄'(과 그 아래 목·세목·목록) 하나 = 조각 하나.
    - 정의를 거느린 줄(첫 줄 '이 법에서 사용하는 용어의 뜻은 다음과 같다.' 등)은 첫 정의 조각 앞에 붙이고, 뒤 조각에는 '이어서'로
    - 정의 줄도 정의를 거느리지도 않는 줄은 일반 규칙(pack)대로"""
    out, pending = [], []

    def emit(c):
        nonlocal pending
        if pending:
            ids = {x["line_uid"] for x in pending}
            c = {**c, "lines": pending + c["lines"], "ctx": [x for x in c["ctx"] if x["line_uid"] not in ids]}
            pending = []
        out.append(c)

    def is_def(n):
        return bool(DEF_LINE.match(n.ln["text"]))

    def has_def(n):
        return is_def(n) or any(has_def(k) for k in n.kids)

    def visit(n, anc):
        if is_def(n):
            emit({"lines": n.flat(), "ctx": anc, "piece": None, "def": True})
            return
        if not has_def(n):
            if not n.kids and LEAD.search(n.ln["text"]):
                pending.append(n.ln)  # '다음과 같다.'로 끝나는 첫 줄 → 다음 정의 조각에
                return
            for c in pack([n], anc):
                emit(c)
            return
        pending.append(n.ln)  # 정의를 거느린 줄 → 첫 정의 조각에 붙이고, 그 뒤는 '이어서'
        for k in n.kids:
            visit(k, anc + [n.ln])

    for r in roots:
        visit(r, [])
    if pending:  # 뒤에 정의가 없어 남은 줄 (잃지 않게)
        out.append({"lines": pending, "ctx": [], "piece": None})
    return out


def chunk_article(a, lines):
    roots = build_tree(lines)
    if is_definition(a) and any(DEF_LINE.match(ln["text"]) and ln["is_deleted"] != "True" for ln in lines):
        STATS["정의 조문 정의 단위(규칙 7)"] += 1
        chunks = definition_chunks(roots)
    elif int(a["n_chars"]) <= WHOLE:
        chunks = [{"lines": [x for r in roots for x in r.flat()], "ctx": [], "piece": None}]
    else:
        chunks = merge_short(pack(roots, []), a)
    # 5. 512토큰 안전 상한
    final = []
    for c in chunks:
        emb = header(a, c) + "\n" + chunk_text(c, "embed_text")
        if n_tok(emb) <= MAX_TOK:
            final.append(c)
            continue
        STATS["512토큰 넘어 다시 나눔"] += 1
        final += [{**x, "def": c.get("def", False)} for x in split_by_tokens(a, c)]
    if len(final) > 1:  # 512토큰으로 다시 나눈 뒤 생긴 짧은 조각도 이웃에 붙임 (이번엔 토큰 확인)
        final = merge_short(final, a, check_tok=True)
    whole = len(final) == 1
    out = []
    for i, c in enumerate(final, 1):
        uids = {ln["line_uid"] for ln in c["lines"]}
        if c["piece"] is not None:  # 문장 조각: 같은 줄의 앞 조각이 있으면 이어지는 조각
            prev = final[i - 2] if i > 1 else None
            starts_child = bool(prev and prev["piece"] is not None and prev["lines"][0]["line_uid"] == c["lines"][0]["line_uid"])
        else:
            first = c["lines"][0] if c["lines"] else {"parent_line_uid": ""}
            starts_child = bool(first["parent_line_uid"]) and first["parent_line_uid"] not in uids
        text, emb_body = chunk_text(c, "text"), chunk_text(c, "embed_text")
        emb = header(a, c) + "\n" + emb_body
        f, t = markers(c)
        out.append({
            "chunk_id": f"{a['article_uid']}#{i}", "article_uid": a["article_uid"], "law_uid": a["law_uid"],
            "chunk_no": i, "n_chunks": len(final), "item_from": f, "item_to": t, "cite": cite(a, c, whole),
            "text": text, "embed_text": emb, "n_chars": len(text), "n_tokens": n_tok(emb),
            "law_name": a["law_name"], "law_level": a["law_level"], "scope": a["scope"],
            "article_label": a["article_label"], "title": a["title"], "effective_date": a["law_effective_date"],
            "has_ctx": bool(c["ctx"]), "by_sentence": c["piece"] is not None, "starts_child": starts_child,
            "long_sentence": c["piece"] in HARD, "def_item": bool(c.get("def")),
            "has_image": a["has_image"], "has_table": a["has_table"], "has_delegation": a["has_delegation"],
            "is_reference_list": a["is_reference_list"], "embed_model": "nlpai-lab/KURE-v1",
            "input_hash": hashlib.sha256(emb.encode("utf-8")).hexdigest(),
        })
    return out


def main():
    arts = [a for a in csv.DictReader(open(WORK / "02_clean" / "articles.csv", encoding="utf-8-sig"))
            if a["searchable"] == "True"]
    by_art = defaultdict(list)
    for ln in csv.DictReader(open(WORK / "02_clean" / "article_lines.csv", encoding="utf-8-sig")):
        by_art[ln["article_uid"]].append(ln)
    chunks = []
    for i, a in enumerate(arts, 1):
        chunks += chunk_article(a, by_art[a["article_uid"]])
        if i % 5000 == 0:
            print(f"  조문 {i}/{len(arts)}")
    OUT.mkdir(parents=True, exist_ok=True)
    cols = list(chunks[0])
    with open(OUT / "chunks.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(chunks)
    with open(OUT / "chunks_input.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"chunk_id": c["chunk_id"], "embed_text": c["embed_text"], "input_hash": c["input_hash"]},
                               ensure_ascii=False) + "\n")
    (OUT / "chunk_stats.json").write_text(json.dumps({"rule": RULE_VERSION, "checked_on": TODAY, "stats": STATS},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    split = sum(1 for c in chunks if c["chunk_no"] == 1 and c["n_chunks"] > 1)
    print("조문", len(arts), "→ 조각", len(chunks), "· 나뉜 조문", split, f"({split / len(arts):.1%})")
    print("통계", dict(STATS))


if __name__ == "__main__":
    main()
