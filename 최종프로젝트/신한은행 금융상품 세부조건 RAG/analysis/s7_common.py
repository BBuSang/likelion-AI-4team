"""2-7 공통 조항·정형 문구·용어집 (LLM 없음) — 2-6 다음에 실행

문제: 같은 문구가 여러 문서에 그대로 들어 있음 (자료열람요구권 58개 문서, 위법계약해지권 48개, 예금자보호 45개 …).
      청크로 그대로 넣으면 검색 상위가 같은 문구로 채워져 상품 고유 내용이 밀려남.
처리: 섹션을 지우지 않고, 여러 문서에 공통인 섹션을 '공통 조항'으로 묶어 표시만 한다 (청크 단계에서 한 번만 색인 + 문서 연결).

1) 공통 조항 묶기 (섹션 단위)
   - 비교 글: 2-6의 text에서 공백·가운뎃점·글머리 기호를 뺀 것 (첫 줄 제목 포함)
   - 같은 묶음: 정확히 같거나, 같은 제목 안에서 글자 4-gram 자카드 ≥ 0.9 + **숫자 목록이 완전히 같음** + 다른 글자 4자 이하
     (문장 끝·기호 정도만 다른 문구는 묶고, 숫자가 다르거나 '양도 및 담보제공 불가'처럼 내용이 더 붙은 문구는 묶지 않음
      — 섹션 분할이 못 나눈 상품 사실이 공통 조항 속에 묻히지 않게)
   - 공통 조항: 3개 이상 문서에 나오고 본문(제목 줄 제외)이 40자 이상 → clause_id 부여
     (짧은 값 '재예치 / 불가', '상품유형 / 자유적립식'은 상품마다 답해야 하는 사실이라 묶지 않음)
2) 정형 문구(is_boilerplate): 공통 조항 중 상품과 무관한 법정 안내·권리·일반 규정
   - 제목이 권리·민원·휴면·압류·거래중지·일반 유의사항·용어 안내 계열이거나 '금리 등 아래의 내용은…' 안내 문구,
     또는 약관 문서(doc_type=약관)에 있는 조문이 다른 문서에 붙어 있는 사본 (투자설명서 뒤의 예금거래기본약관 등) — 사본 쪽만 표시
   - 예금자보호여부·세제혜택·일부해지처럼 상품 조건을 답하는 항목은 공통 조항이어도 정형 문구가 아님
3) 용어집: '용어' 부록 아래 용어 섹션 → term, definition (문서 간 같은 용어는 가장 많이 나온 정의, 다른 정의 수 기록)

출력: data/sections/*.jsonl 에 clause_id, clause_docs, is_boilerplate 필드 추가
      data/meta/common_clause.csv, data/meta/document_clause.csv, data/meta/glossary.csv
"""
import csv
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s5_row_labels import VARIANTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# 정형 문구 안에 표의 다른 행 이름이 줄 끝에 붙어 있으면(섹션 분할이 못 나눈 상품 사실) 정형 문구로 보지 않음
ROW_TAIL = re.compile(r"(" + "|".join(sorted((re.escape(v) for v in VARIANTS if "+" not in v and len(v) >= 4), key=len, reverse=True)) + r")\S{0,12}$")
SECS, META = ROOT / "data" / "sections", ROOT / "data" / "meta"

MIN_DOCS, MIN_BODY, JACCARD, MAX_DIFF = 3, 40, 0.9, 4
NORM = re.compile(r"[\s·ㆍ•\-‧∙⋅|]")   # | 와 <br>(norm 안에서)은 md 표 기호 — 문서마다 표 배치가 달라도 같은 글로 묶이게 (s40)
GENERIC = re.compile(r"위법계약|자료열람|민원|분쟁|휴면예금|원금(및|또는)이자지급(관련)?제한|압류|유의사항|금융소비자의권리|"
                     r"개인신용평가대응권|채무조정요청권|우선변제채무지정권|청약철회권|금리인하요구권|금리등아래의내용|"
                     r"용어|설명의무|대출계약이신용|연체에따른불이익|그밖의불이익|기한의이익|상계|채무변제충당|거래중지")
FACT = re.compile(r"예금자보호|세제혜택|일부해지|가입대상|가입자격|계약기간|가입금액|이자율|이율|이자지급|중도해지|만기|수수료|우대")


ROW_KEYS = sorted((v for v in VARIANTS if "+" not in v and len(v) >= 4), key=len, reverse=True)


def has_row_line(text: str, title: str) -> bool:
    """정형 문구 섹션 안에 다른 표 행(사전의 행 이름으로 시작하는 줄)이 섞여 있는지 — 섹션 분할이 못 나눈 상품 사실
    (예: '휴면예금 및 출연' 섹션 속 '예금자보호여부 해당: 이 예금은 …'). '계약기간 등 변경에…'처럼 조사가 붙은 문장은 제외"""
    t = norm(title)
    for line in text.split("\n")[1:]:
        n = norm(line)
        k = next((k for k in ROW_KEYS if n.startswith(k) and k not in t), None)
        if k and not re.match(r"(등|의|을|를|이|가|은|는|에|과|와|도)", n[len(k):]):
            return True
    return False


def norm(t: str) -> str:
    return NORM.sub("", t.replace("<br>", ""))


def grams(t: str, n=4):
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def numbers(t: str):
    return re.findall(r"\d+(?:[.,]\d+)*", t)


def small_diff(a: str, b: str) -> bool:
    """다른 글자가 양쪽 합해 MAX_DIFF자 이하 (문장 끝 '습니다.'↔'음', 기호 '○' 정도) — 정보가 더 붙은 문구는 묶지 않음"""
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in ops if t != "equal") <= MAX_DIFF


def cluster(secs):
    """같은 제목(첫 줄) 안에서 정확 일치 → 근사 일치 순으로 묶음. 반환: {섹션 키: 묶음 번호}"""
    by_title = defaultdict(list)
    for s in secs:
        by_title[norm(s["text"].split("\n")[0])].append(s)
    exact = defaultdict(list)
    for s in secs:
        exact[norm(s["text"])].append(s)
    rep_of = {}
    cid = 0
    for title, group in by_title.items():
        reps = []   # (묶음 번호, 대표 글, 4-gram, 숫자)
        for txt in sorted({norm(s["text"]) for s in group}, key=lambda t: -len(exact[t])):
            g, nums = grams(txt), numbers(txt)
            hit = None
            for c, rtxt, rg, rnums in reps:
                if nums == rnums and len(g & rg) / len(g | rg) >= JACCARD and small_diff(txt, rtxt):
                    hit = c
                    break
            if hit is None:
                cid += 1
                hit = cid
                reps.append((cid, txt, g, nums))
            rep_of[txt] = hit
    return {(s["doc_id"], s["seq"]): rep_of[norm(s["text"])] for s in secs}


def main() -> None:
    docs = {p.stem: [json.loads(l) for l in open(p)] for p in sorted(SECS.glob("*.jsonl"))}
    secs = [s for v in docs.values() for s in v]
    cl = cluster(secs)
    members = defaultdict(list)
    for s in secs:
        members[cl[(s["doc_id"], s["seq"])]].append(s)

    # 약관 문서의 조문이 다른 문서에 그대로 붙은 사본 (예: 투자설명서 뒤의 예금거래기본약관)
    dtype = {r["doc_id"]: r["doc_type"] for r in csv.DictReader(open(META / "documents.csv"))}
    appended = set()
    for c, ms in members.items():
        ds = {m["doc_id"] for m in ms}
        if any(dtype.get(d) == "약관" for d in ds) and len(ds) >= 2:
            appended |= {(m["doc_id"], m["seq"]) for m in ms
                         if dtype.get(m["doc_id"]) != "약관" and not has_row_line(m["text"], m["text"].split("\n")[0])}

    clauses, doc_clause = [], []
    keep = {}
    for c, ms in members.items():
        ds = sorted({m["doc_id"] for m in ms})
        body = norm("\n".join(ms[0]["text"].split("\n")[1:]))
        if len(ds) < MIN_DOCS or len(body) < MIN_BODY:
            continue
        rep = Counter(m["text"] for m in ms).most_common(1)[0][0]
        title = rep.split("\n")[0].strip()
        # 상위 제목(예: '세금 등 기타 유의사항')은 보지 않음 — 하위 항목은 상품 사실일 수 있음. 단 '용어 안내' 부록 아래는 일반 정의
        t = norm(title) + body[:40] + ("용어" if "용어" in ms[0]["section_path"].split(" > ")[0] else "")
        generic = bool(GENERIC.search(t)) and not FACT.search(norm(title)) and not ROW_TAIL.search(norm(rep)) \
            and not has_row_line(rep, title)
        keep[c] = generic
        terms_doc = next((d for d in ds if dtype.get(d) == "약관"), "")
        clauses.append({"clause_id": c, "title": title, "n_docs": len(ds), "n_sections": len(ms),
                        "is_boilerplate": int(generic), "terms_doc": terms_doc, "content": rep})
        doc_clause += [{"doc_id": d, "clause_id": c} for d in ds]

    n_bp = 0
    for d, v in docs.items():
        for s in v:
            c = cl[(d, s["seq"])]
            s["clause_id"] = c if c in keep else None
            s["clause_docs"] = len({m["doc_id"] for m in members[c]}) if c in keep else 1
            s["is_boilerplate"] = bool(keep.get(c)) or (d, s["seq"]) in appended
            n_bp += s["is_boilerplate"]
        with open(SECS / f"{d}.jsonl", "w") as f:
            for s in v:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 용어집
    defs = defaultdict(Counter)
    srcs = defaultdict(set)
    for s in secs:
        parts = s["section_path"].split(" > ")
        if len(parts) == 2 and "용어" in parts[0] and "\n" in s["text"]:
            term = s["text"].split("\n")[0].strip()
            body = "\n".join(s["text"].split("\n")[1:]).strip()
            if term.startswith(("※", "[")) or not re.match(r"^[•▪\-(]", body):
                continue   # 용어 부록 뒤에 붙은 다른 표의 행('대상', '주택' …)·서식은 용어가 아님
            defs[term][body] += 1
            srcs[term].add(s["doc_id"])
    gloss = [{"term": t, "definition": c.most_common(1)[0][0], "n_docs": len(srcs[t]), "n_variants": len(c),
              "docs": "|".join(sorted(srcs[t]))} for t, c in sorted(defs.items())]

    for name, rows in (("common_clause", sorted(clauses, key=lambda r: -r["n_docs"])),
                       ("document_clause", doc_clause), ("glossary", gloss)):
        with open(META / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    shared = sum(1 for s in secs if s["clause_id"])
    print(f"섹션 {len(secs):,} 중 공통 조항에 속한 섹션 {shared:,} → 공통 조항 {len(clauses)}개 "
          f"(정형 문구 {sum(r['is_boilerplate'] for r in clauses)}개), 정형 문구 표시 섹션 {n_bp:,} (붙은 약관 {len(appended)} 포함)")
    print(f"용어집 {len(gloss)}개 용어 (정의가 문서마다 다른 용어 {sum(1 for g in gloss if g['n_variants'] > 1)}개)")


if __name__ == "__main__":
    main()
