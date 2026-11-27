"""Holdout2 EDA 발표 자료 (Holdout1 비교) — HTML 슬라이드를 만들고 Edge 로 16:9 PDF 로 인쇄.
숫자는 모두 데이터 파일에서 바로 셉니다 (모델 점수 없음).
실행: python -X utf8 results/holdout2/eda/build_slides.py  (저장소 루트의 holdout2/, 실전프로젝트2/data/ 필요)
"""
import collections
import html
import json
import re
import statistics as st
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # likelion-AI-4team


def rd(p):
    return [json.loads(l) for l in open(ROOT / p, encoding="utf-8") if l.strip()]


H1 = rd("실전프로젝트2/data/holdout_labeled.jsonl")
H2 = rd("holdout2/holdout2_labeled.jsonl")
Q1 = rd("실전프로젝트2/data/holdout_retrieval_queries_gold.jsonl")
Q2 = rd("holdout2/holdout2_retrieval_queries_gold.jsonl")
B2 = {d["id"]: d for d in H2}
LABEL_KO = {"person": "이름", "phone_number": "전화번호", "email": "이메일", "address": "주소",
            "resident_registration_number": "주민등록번호", "account_number": "계좌번호", "card_number": "카드번호"}


def stats(D, Q):
    L = [len(d["text"]) for d in D]
    E = [len(d["entities"]) for d in D]
    dt = collections.Counter(d["doc_type"] for d in D)
    names = collections.Counter()
    for d in D:
        for n in {e["text"].replace(" ", "") for e in d["entities"] if e["label"] == "person" and e.get("form") == "full"}:
            names[n] += 1
    by = collections.defaultdict(list)
    for d in D:
        by[d["doc_type"]].append(len(d["entities"]))
    return {
        "docs": len(D), "types": len(dt), "dt": dt, "len_mean": st.mean(L), "len_median": st.median(L),
        "len_max": max(L), "len_min": min(L), "ents": sum(E), "per_doc": st.mean(E),
        "nopii": sum(e == 0 for e in E), "empty": sum(not d["text"].strip() for d in D),
        "labels": collections.Counter(e["label"] for d in D for e in d["entities"]),
        "form": collections.Counter(e.get("form") for d in D for e in d["entities"] if e["label"] == "person"),
        "hard": collections.Counter(h for d in D for h in d.get("hard_cases", [])),
        "names": names, "density": {t: st.mean(v) for t, v in by.items()},
        "buckets": collections.Counter("빈 문서" if not d["text"].strip() else "200자 미만" if x < 200 else "200~400자" if x < 400
                                       else "400~1,000자" if x < 1000 else "1,000자 이상" for d, x in zip(D, L)),
        "phones": collections.OrderedDict(),
        "queries": len(Q), "qtype": collections.Counter(q["type"] for q in Q),
        "rel_mean": st.mean(len(q["relevant_doc_ids"]) for q in Q),
        "rel10": sum(len(q["relevant_doc_ids"]) >= 10 for q in Q),
        "rel1": sum(len(q["relevant_doc_ids"]) == 1 for q in Q),
        "conf": sum(1 for q in Q if q.get("confusable_doc_ids")),
        "qpii": sum(1 for q in Q if q.get("query_entities")),
    }


S1, S2 = stats(H1, Q1), stats(H2, Q2)
for S, D in ((S1, H1), (S2, H2)):
    for d in D:
        for e in d["entities"]:
            if e["label"] == "phone_number":
                shape = re.sub(r"\d", "9", e["text"])
                if re.search(r"[가-힣]", shape):
                    shape = "한글로 읽은 번호"
                S["phones"].setdefault(shape, e["text"])

esc = html.escape


def n(x, d=0):
    return f"{x:,.{d}f}"


def table(head, rows, cls=""):
    h = "".join(f"<th>{esc(str(c))}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def doc(did, start=0, length=260, mark=()):
    """실제 문서 일부. mark 의 문자열은 강조"""
    if isinstance(start, str):
        start = B2[did]["text"].find(start)
    t = B2[did]["text"][start:start + length]
    s = esc(t)
    for m in mark:
        s = s.replace(esc(m), f"<mark>{esc(m)}</mark>")
    return f'<div class="doc"><div class="doc-h">{did} · {esc(B2[did]["doc_type"])}</div><pre>{s}{"…" if len(B2[did]["text"]) > start + length else ""}</pre></div>'


# ---------------------------------------------------------------- 추가 분석
def skeleton(d):
    """양식 뼈대: 개인정보를 라벨로, 숫자를 9로 바꾼 첫 4줄 (줄마다 앞 25자)"""
    t = d["text"]
    for e in sorted(d["entities"], key=lambda e: -e["start"]):
        t = t[:e["start"]] + "<" + e["label"] + ">" + t[e["end"]:]
    t = re.sub(r"\d+", "9", t)
    return "|".join([l[:25] for l in t.split("\n") if l.strip()][:4])


def context_kind(d, e):
    pre = d["text"][max(0, e["start"] - 8):e["start"]]
    if re.search(r"[가-힣A-Za-z ]{1,6}\s*[:：]\s*$", pre):
        return "항목: 값"
    if pre.rstrip().endswith("|"):
        return "표 칸"
    return "문장 속"


def ctx_counts(D):
    return collections.Counter(context_kind(d, e) for d in D for e in d["entities"])


C1, C2 = ctx_counts(H1), ctx_counts(H2)


def profile(D):
    by = collections.defaultdict(list)
    for d in D:
        by[d["doc_type"]].append(d)
    rows = []
    for t, ds in sorted(by.items(), key=lambda x: -len(x[1])):
        labs = collections.Counter(e["label"] for d in ds for e in d["entities"])
        top = " · ".join(LABEL_KO[l] for l, _ in labs.most_common(2)) or "-"
        table_ratio = sum(1 for d in ds if re.search(r"^\s*\|", d["text"], re.M)) / len(ds)
        rows.append([esc(t), len(ds), n(st.mean(len(d["text"]) for d in ds)), n(st.mean(len(d["entities"]) for d in ds), 1),
                     top, "표" if table_ratio >= .5 else "", len({skeleton(d) for d in ds})])
    return rows


# 이름의 역할(고객·직원·그 외): 정답 파일에 없어서, 이름이 나온 자리(앞 글자·표 열 제목)를 모두 모아
# 사람(Claude)이 한 가지씩 원문을 읽고 나눈 표 name_roles.json 을 씀 (Holdout1 148가지, Holdout2 88가지 — 양식이 반복돼 가짓수가 적음)
ROLE_TABLE = {(r["set"], r["doc_type"], r["where"]): r["role"] for r in json.load(open(HERE / "name_roles.json", encoding="utf-8"))}


def where(d, e):
    """이름이 나온 자리: 표면 열 제목 (+ 줄 전체), 아니면 같은 줄 앞 14자 + 뒤 6자 (숫자는 9)"""
    t = d["text"]
    ls = t.rfind("\n", 0, e["start"]) + 1
    le = t.find("\n", e["end"])
    le = len(t) if le < 0 else le
    line, pre, post = t[ls:le], t[ls:e["start"]], t[e["end"]:le]
    if line.lstrip().startswith("|"):
        col, hdr = pre.count("|") - 1, None
        for l in t[:ls].split("\n")[::-1]:
            if l.strip().startswith("|") and not re.match(r"^\s*\|[\s\-:|]+\|\s*$", l):
                hdr = l
        cells = [c.strip() for c in (hdr or "").strip().strip("|").split("|")]
        rank = re.match(r"\s*(과장|팀장|대리|사원|차장|부장|주임|책임|선임)", post)
        return "TABLE[" + (cells[col] if 0 <= col < len(cells) else "?") + "]" + (" +직급" if rank else ""), line
    return f'{re.sub(r"\d", "9", pre[-14:])}@{re.sub(r"\d", "9", post[:6])}', line


def role_of(d, e):
    k, line = where(d, e)
    r = ROLE_TABLE[("holdout2" if d["id"].startswith("H2") else "holdout1", d["doc_type"], k)]
    if r == "소속으로":  # 출장보고 참석자 표: 소속이 우리 팀이면 직원, 협력사면 그 외
        return "직원" if re.search(r"\|\s*[^|]*팀\s*\|", line) else "그 외"
    return r


def role_split(D):
    """(이름, 역할) 하나를 한 사람으로 셈 — 고객 도재호와 직원 도재호는 둘"""
    docs = collections.defaultdict(set)
    for d in D:
        for e in d["entities"]:
            if e["label"] == "person" and e.get("form") == "full":
                docs[(e["text"].replace(" ", ""), role_of(d, e))].add(d["id"])
    out = collections.defaultdict(list)
    for (nm, r), ds in docs.items():
        out[r].append(len(ds))
    return out, docs


R1, _ = role_split(H1)
R2, DOCS2 = role_split(H2)
DATE = re.compile(r"(20\d\d)[-./년]\s?(\d{1,2})[-./월]")


def months(D):
    c = collections.Counter()
    for d in D:
        for y, m in DATE.findall(d["text"]):
            if y in ("2025", "2026") and 1 <= int(m) <= 12:
                c[f"{y}-{int(m):02d}"] += 1
                break
    return c


M1, M2 = months(H1), months(H2)


def surface(Q, B):
    """질의 속 개인정보가 정답 문서에 같은 글자로 있는지 + 다를 때 문서 쪽 값"""
    tot = same = 0
    pairs = []
    for q in Q:
        for e in q.get("query_entities") or []:
            tot += 1
            hit = any(e["text"] in B[r]["text"] for r in q["relevant_doc_ids"] if r in B)
            same += hit
            if not hit and q["relevant_doc_ids"] and q["relevant_doc_ids"][0] in B:
                d = B[q["relevant_doc_ids"][0]]
                dg = re.sub(r"\D", "", e["text"])
                m = [x["text"] for x in d["entities"] if x["label"] == e["label"]
                     and ((dg and re.sub(r"\D", "", x["text"]) == dg) or x["text"].replace(" ", "") == e["text"].replace(" ", ""))]
                if m:
                    pairs.append((q["type"], LABEL_KO[e["label"]], e["text"], m[0], d["id"]))
    return tot, same, pairs


slides = []


def slide(title, body, kicker=""):
    k = f'<div class="kicker">{esc(kicker)}</div>' if kicker else ""
    slides.append(f'<section class="slide">{k}<h1>{esc(title)}</h1><div class="body"><div class="inner">{body}</div></div>'
                  f'<div class="foot">Holdout2 EDA · 사공삼 403</div></section>')


# 1. 표지
slides.append('''<section class="slide cover"><div class="cv-logo"></div><div class="cv-text">
<div class="cv-kicker">Holdout2 데이터 탐색 · 2026.11</div>
<h1 class="cv-title">Holdout2 데이터<br>EDA</h1>
<p class="cv-sub">Holdout1 과 비교해 무엇이 달라졌나<br>— 문서 · 개인정보 · 사람 · 검색 질의</p>
<p class="cv-team">나호성<span>·</span>백송이<span>·</span>이한별<span>·</span>박지훈</p></div></section>''')

# 2. 데이터 구성
slide("받은 데이터", table(["파일", "내용", "건수"], [
    ["<code>holdout2_public.jsonl</code>", "문서 원문 (id · doc_type · text)", f"{S2['docs']}건"],
    ["<code>holdout2_labeled.jsonl</code>", "같은 문서 + 개인정보 정답 위치(entities)", f"{S2['docs']}건"],
    ["<code>holdout2_retrieval_queries_gold.jsonl</code>", "검색 질의 + 정답 문서 + 헷갈리는 문서 + 질의 속 개인정보", f"{S2['queries']}개"],
]) + '<p class="note">개인정보 라벨 7종: 이름 · 전화번호 · 이메일 · 주소 · 주민등록번호 · 계좌번호 · 카드번호 (Holdout1 과 같음)</p>',
    kicker="1. 데이터 소개")

# 3. 규모
slide("규모 한눈에", table(["", "Holdout1", "Holdout2", "변화"], [
    ["문서", n(S1["docs"]), n(S2["docs"]), "<b>약 4배</b>"],
    ["문서 종류", n(S1["types"]), n(S2["types"]), "<b>겹치는 종류 0개</b>"],
    ["개인정보 개체", n(S1["ents"]), n(S2["ents"]), "약 6배"],
    ["문서당 개인정보", n(S1["per_doc"], 1), n(S2["per_doc"], 1), "더 빽빽함"],
    ["고유 실명 (성+이름)", n(len(S1["names"])), n(len(S2["names"])), "약 3.4배"],
    ["개인정보 없는 문서", n(S1["nopii"]), n(S2["nopii"]), ""],
    ["빈 문서", n(S1["empty"]), n(S2["empty"]), "<b>새로 생김</b>"],
    ["검색 질의", n(S1["queries"]), n(S2["queries"]), ""],
], "wide"), kicker="2. 규모")

# 4. 문서 종류
def type_list(dt):
    return "".join(f"<li><span>{esc(t)}</span><b>{c}</b></li>" for t, c in dt.most_common())
slide("문서 종류: 전부 새로운 종류", f'''<div class="cols"><div><h3>Holdout1 ({S1["types"]}종, 종류당 7~14건)</h3><ul class="types">{type_list(S1["dt"])}</ul></div>
<div><h3>Holdout2 ({S2["types"]}종, 종류당 {min(S2["dt"].values())}~{max(S2["dt"].values())}건)</h3><ul class="types">{type_list(S2["dt"])}</ul></div></div>''',
      kicker="3. 문서 종류")

# 5. 양식 반복
slide("같은 양식이 수십 번 반복된다", f'''<p>종류마다 같은 양식에 값만 바꿔 넣은 문서가 많습니다 (채무조정상담 63건, 중도상환문의 50건 …). 양식 속 일반 단어가 데이터 전체에 대량으로 나옵니다.</p>
<div class="cols">{doc("H20033", 0, 150, mark=("기안", "검토", "승인"))}{doc("H20007", 0, 150, mark=("[양도인]", "[양수인]", "양도인"))}</div>
<p class="note">결재란 '기안 · 검토 · 승인' 40건, 포인트 이관 신청서 '양도인 · 양수인' 30건이 같은 모양으로 반복</p>''', kicker="3. 문서 종류")

# 5-1. 종류별 프로필
prof = profile(H2)
half_p = (len(prof) + 1) // 2
head_p = ["종류", "건수", "평균 길이", "개인정보/문서", "주요 라벨", "", "양식 수"]
slide("Holdout2 문서 종류별 프로필", '<div class="cols">' + table(head_p, prof[:half_p], "xs") + table(head_p, prof[half_p:], "xs") + "</div>"
      + '<p class="note">양식 수 = 개인정보·숫자를 지운 첫 4줄 모양의 가짓수. "표" = 그 종류 문서 절반 이상에 표가 있음</p>', kicker="3. 문서 종류")

# 5-2. 양식 수 비교 + 같은 양식 예
sk1 = sorted(((len({skeleton(d) for d in H1 if d["doc_type"] == t}), t) for t in S1["dt"]), reverse=True)
sk2 = sorted(((len({skeleton(d) for d in H2 if d["doc_type"] == t}), t) for t in S2["dt"]), reverse=True)
few = lambda sk: sum(1 for k, _ in sk if k <= 3)
slide("양식이 몇 가지인가", f"""<p>양식 3가지 이하인 종류: Holdout1 {len(sk1)}종 중 {few(sk1)}종 → Holdout2 {len(sk2)}종 중 <b>{few(sk2)}종</b>.
Holdout2 는 같은 틀에 이름·번호만 바꾼 문서가 대부분입니다 (예외: 지점업무일지 46건은 모두 다름 — 하루치 기록 나열).</p>
<div class="cols">{doc("H20022", 0, 260)}{doc("H20024", 0, 260)}</div>
<p class="note">채무조정상담 63건 중 두 건 — 날짜·상담원·고객만 다르고 문장 틀은 같음</p>""", kicker="3. 문서 종류")

# 6. 길이
order = ["빈 문서", "200자 미만", "200~400자", "400~1,000자", "1,000자 이상"]
slide("문서 길이: 더 길고 더 들쭉날쭉", '<div class="cols">' + table(["길이", "Holdout1", "Holdout2"], [[o, S1["buckets"].get(o, 0), S2["buckets"].get(o, 0)] for o in order]
      + [["평균 / 중앙값", f'{n(S1["len_mean"])} / {n(S1["len_median"])}자', f'{n(S2["len_mean"])} / {n(S2["len_median"])}자'],
         ["최소 ~ 최대", f'{S1["len_min"]} ~ {S1["len_max"]}자', f'{S2["len_min"]} ~ {n(S2["len_max"])}자']])
      + f'''<div><h3>빈 문서 2건 (종류: 빈문서)</h3>
<div class="doc"><div class="doc-h">H20345</div><pre>{esc(repr(B2["H20345"]["text"]))}  ← 공백·줄바꿈뿐</pre></div>
<div class="doc"><div class="doc-h">H20437</div><pre>{esc(repr(B2["H20437"]["text"]))}  ← 빈 문자열</pre></div>
<h3>가장 긴 종류: 지점업무일지</h3>{doc("H20140", 0, 190)}
<p class="note">1,000자 넘는 문서 6건은 모두 지점업무일지 (하루치 고객 응대를 시간순으로 나열)</p></div></div>''', kicker="4. 길이")

# 8. 표 문서 — 한 문서에 사람이 여럿
def table_stats(D):
    has_tab = [d for d in D if re.search(r"^\s*\|", d["text"], re.M)]
    ppl = lambda d: len({e["text"].replace(" ", "") for e in d["entities"] if e["label"] == "person"})
    return [len(has_tab), sum(ppl(d) >= 5 for d in D), max(ppl(d) for d in D)]


T1, T2 = table_stats(H1), table_stats(H2)
slide("표로 된 문서: 한 문서에 사람이 여럿", '<div class="cols"><div style="flex:.8">'
      + table(["", "Holdout1", "Holdout2"], [["표가 있는 문서", T1[0], f"<b>{T2[0]}</b>"],
                                            ["이름이 5명 이상 나오는 문서", T1[1], f"<b>{T2[1]}</b>"],
                                            ["한 문서 최대 인원", f"{T1[2]}명", f"<b>{T2[2]}명</b>"]])
      + '<p>Holdout1 은 문서 하나에 고객 한두 명이었지만, Holdout2 의 콜백요청목록·업무일지는 <b>한 문서에 고객 수십 명</b>이 줄마다 나옵니다.</p>'
      + '<p class="note">같은 문서 안에서도 줄마다 다른 사람의 번호 → 어느 이름의 번호인지 줄(행) 단위로 봐야 함</p></div>'
      + doc("H20589", 0, 420) + "</div>", kicker="5. 개인정보")

# 9. 표기 변형 + 개인정보처럼 보이는 것 — 한 화면에 실제 예까지
def ctx(did, w, before=14, after=10):
    """문서에서 w 가 있는 줄의 앞뒤 몇 글자 (w 는 강조, 줄은 넘지 않음)"""
    t = B2[did]["text"]
    i = t.find(w)
    a = t[max(0, i - before):i].split("\n")[-1].lstrip()
    b = t[i + len(w):i + len(w) + after].split("\n")[0].rstrip()
    more = "…" if b and i + len(w) + after < len(t) and "\n" not in t[i + len(w):i + len(w) + after] else ""
    return f'<span class="snip">{esc(a)}<mark>{esc(w)}</mark>{esc(b)}{more}</span>'


ko_phone = sum(1 for d in H2 for e in d["entities"] if e["label"] == "phone_number" and re.search(r"[가-힣]", e["text"]))
num_shapes = [v for k, v in S2["phones"].items() if k != "한글로 읽은 번호"]
same_rows = [
    ["전화번호 숫자 모양", " · ".join(f"<code>{esc(v)}</code>" for v in num_shapes), f"<span class=nw>H1 {len(S1['phones'])}가지</span><br><span class=nw>→ H2 {len(num_shapes)}가지</span>"],
    ["전화번호를 한글로 읽음", ctx("H20006", "공일공 팔사칠칠 일이공구", 5, 4), f"<b class=nw>새로 등장</b><br><b class=nw>{ko_phone}건</b>"],
    ["이름을 한 글자씩 띄움", ctx("H20006", "정 수 혁", 6, 6), "<span class=nw>새로 등장</span>"],
    ["영문 이름", ctx("H20009", "Seongyul Kim", 9, 6), "<span class=nw>새로 등장</span>"],
    ["카드번호 띄어 씀", ctx("H20033", "4092 4076 2658 6158", 12, 2), ""],
    ["번호 붙여 씀", ctx("H20001", "01071146387", 6, 4), ""],
]
fake_rows = [
    ["법인등록번호<br><span class=note>주민번호와 같은 모양</span>", ctx("H20005", "118023-3131332", 8, 2)],
    ["서버 IP", ctx("H20042", "10.78.215.206", 8, 2)],
    ["장애 번호", ctx("H20042", "INC-841039", 8, 2)],
    ["회사 대표번호", ctx("H20025", "1588-1255", 9, 2)],
    ["송장번호<br><span class=note>12자리 숫자</span>", ctx("H20025", "742665443660", 6, 2)],
    ["카드 끝자리 4자리", ctx("H20002", "1612", 10, 3)],
    ["회사 이름<br><span class=note>영문 이름처럼 보임</span>", ctx("H20009", "Saebom Card", 5, 2)],
]
slide("표기 변형: 같은 값을 여러 모양으로, 개인정보처럼 보이는 가짜도",
      '<div class="cols"><div style="flex:1.25"><h3>같은 값, 다른 모양 (정답 라벨 있음)</h3>'
      + table(["종류", "문서에 적힌 그대로", ""], same_rows, "mid")
      + '</div><div><h3>개인정보처럼 보이지만 아닌 것 (정답 라벨 없음)</h3>'
      + table(["종류", "문서에 적힌 그대로"], fake_rows, "mid") + "</div></div>"
      + '<p class="note">왼쪽은 숫자 규칙 하나로는 다 못 잡는 모양, 오른쪽은 숫자 모양만 보면 개인정보로 오인하기 쉬운 값. 모두 Holdout2 실제 문서</p>',
      kicker="6. 표기 변형")

# 9-1. 특이한 원문 예 — 설명 없이 원문만
slide("원문으로 보는 특이한 문서", f"""<div class="grid3">
{doc("H20009", "Dear", 165, mark=("Seongyul Kim", "Saebom Card"))}
{doc("H20006", "사용자: 카드", 120, mark=("공일공 팔사칠칠 일이공구", "정 수 혁"))}
{doc("H20005", "법인명", 105, mark=("118023-3131332",))}
{doc("H20042", "원인", 80, mark=("10.78.215.206", "INC-841039"))}
{doc("H20025", "연락처", 80, mark=("742665443660", "1588-1255"))}
{doc("H20002", "고객 > 감사", 70, mark=("끝자리가 1612",))}
</div>""", kicker="6. 표기 변형")

# 13. 사람
top = ", ".join(f"{esc(k)}({v})" for k, v in S2["names"].most_common(3))
spread = lambda S: [sum(v == 1 for v in S["names"].values()), sum(2 <= v <= 5 for v in S["names"].values()), sum(v >= 6 for v in S["names"].values())]
a, b = spread(S1), spread(S2)
def homonyms(D):
    """동명이인: 고객으로도 직원으로도 나오는 이름 / 한 문서 안에서 두 역할"""
    roles, same_doc = collections.defaultdict(set), set()
    for d in D:
        per = collections.defaultdict(set)
        for e in d["entities"]:
            if e["label"] == "person" and e.get("form") == "full" and (r := role_of(d, e)) in ("고객", "직원"):
                per[e["text"].replace(" ", "")].add(r)
        for nm, rs in per.items():
            roles[nm] |= rs
            if len(rs) > 1:
                same_doc.add(nm)
    return sum(len(v) > 1 for v in roles.values()), len(same_doc)


HM1, HM2 = homonyms(H1), homonyms(H2)
slide("동명이인: 고객과 직원이 같은 이름인 경우가 섞임", '<div class="cols"><div style="flex:.85">'
      + table(["", "Holdout1", "Holdout2"], [
          ["고유 실명 (성+이름)", len(S1["names"]), len(S2["names"])],
          ["2개 이상 문서에 나오는 실명", a[1] + a[2], f"<b>{b[1] + b[2]}</b>"],
          ["고객으로도 직원으로도 나오는 이름", HM1[0], f"<b>{HM2[0]}</b>"],
          ["한 문서 안에 두 역할로 나오는 이름", HM1[1], f"<b>{HM2[1]}</b>"]])
      + f'<p>같은 이름이 고객으로도 직원으로도 나오는 경우가 <b>{HM1[0]} → {HM2[0]}개</b>로 늘었습니다. Holdout2 에는 <b>고객 ↔ 직원</b>, <b>한 문서 안의 상담원과 고객</b>처럼 역할이 다른 동명이인이 섞여 있습니다.</p>'
      + '<p class="note">역할은 정답 파일에 없어, 이름이 나온 자리를 모두 모아 원문을 읽고 직접 나눔 (양식이 반복돼 Holdout2 는 88가지 자리뿐)</p></div><div>'
      + f'''<h3>① 고객 ↔ 직원 (다른 문서)</h3><div class="cols">{doc("H20001", "청구인", 40, mark=("청구인: 도재호",))}{doc("H20022", "상담원", 40, mark=("상담원: 도재호",))}</div>
<h3>② 한 문서 안: 상담원과 고객</h3>{doc("H20150", "상담원: 박형윤", 230, mark=("상담원: 박형윤", "네, 박형윤입니다"))}</div></div>''', kicker="7. 사람")

# 13-1. 고객과 직원
def role_row(R, r):
    v = R.get(r, [])
    return [r, len(v), n(st.mean(v), 1) if v else "-", sum(1 for x in v if x >= 6)]


top_staff = sorted(((len(v), nm) for (nm, r), v in DOCS2.items() if r == "직원"), reverse=True)[:6]
top_cust = sorted(((len(v), nm) for (nm, r), v in DOCS2.items() if r == "고객"), reverse=True)[:6]
head_r = ["역할", "사람 수", "평균 문서 수", "6개+ 문서"]
slide("고객과 직원: 누가 여러 문서에 나오나", '<div class="cols"><div><h3>Holdout1</h3>' + table(head_r, [role_row(R1, r) for r in ("직원", "고객", "그 외")])
      + "<h3>Holdout2</h3>" + table(head_r, [role_row(R2, r) for r in ("직원", "고객", "그 외")]) + "</div><div>"
      + table(["Holdout2 직원 (많이 나온 순)", "문서"], [[esc(nm), c] for c, nm in top_staff], "tight")
      + table(["Holdout2 고객 (많이 나온 순)", "문서"], [[esc(nm), c] for c, nm in top_cust], "tight") + "</div></div>"
      + '<p class="note">사람 수 = (실명, 역할) 가짓수 — 고객 도재호와 직원 도재호는 둘. 그 외 = 협력사 담당자 · 대리인 · 상속인 · 직원 가족 · 채용 지원자</p>',
      kicker="7. 사람")

# 14. 질의 — 요약 + 예
QTYPE_KO = {"name_topic": "이름 + 주제", "name_variant": "이름을 줄이거나 띄어 씀", "staff_owner": "직원이 담당한 문서",
            "value_exact": "번호·이메일 그대로", "value_format_variant": "번호를 다른 모양으로", "address_partial": "주소 일부만",
            "address_exact": "주소 그대로", "semantic_no_pii": "개인정보 없이 내용만", "hybrid_paraphrase": "이름 + 다른 말로 바꿔 말함",
            "homonym_role": "동명이인 중 고객/직원만", "long_doc_person": "긴 문서 속 사람"}
qex = []
for t in ["name_topic", "name_variant", "staff_owner", "value_format_variant", "address_partial", "semantic_no_pii",
          "hybrid_paraphrase", "homonym_role", "long_doc_person"]:
    q = next(x for x in Q2 if x["type"] == t)
    qex.append([QTYPE_KO[t], esc(q["query"]), len(q["relevant_doc_ids"])])
slide("검색 질의: 정답 문서가 많고, 묻는 방식이 다양해짐", '<div class="cols"><div style="flex:.62">' + table(["", "Holdout1", "Holdout2"], [
          ["질의 수", S1["queries"], S2["queries"]],
          ["질의 속에 개인정보", f'{S1["qpii"]}', f'{S2["qpii"]}'],
          ["정답 문서 수 평균", n(S1["rel_mean"], 1), f'<b>{n(S2["rel_mean"], 1)}</b>'],
          ["정답 1개인 질의", S1["rel1"], S2["rel1"]],
          ["정답 10개 이상", S1["rel10"], f'<b>{S2["rel10"]}</b>'],
          ["헷갈리는 문서 지정", S1["conf"], S2["conf"]],
      ]) + '<p class="note">새로 생긴 질문: 직원이 담당한 문서 찾기, 긴 문서 속 사람 찾기</p></div><div>'
      + "<h3>Holdout2 실제 질의</h3>" + table(["묻는 방식", "질의", "정답 문서"], qex, "tight") + "</div></div>",
      kicker="8. 검색 질의")

# 15-1. 질의 속 값과 정답 문서 속 값
B1 = {d["id"]: d for d in H1}
t1, s1, _ = surface(Q1, B1)
t2, s2, pr = surface(Q2, B2)
seen, ex_rows = set(), []
for ty, lab, a, b, did in pr:
    if (ty, lab) not in seen:
        seen.add((ty, lab))
        ex_rows.append([QTYPE_KO.get(ty, ty), lab, f"<code>{esc(a)}</code>", f"<code>{esc(b)}</code>", did])
slide("질의 속 값과 정답 문서 속 값", table(["", "Holdout1", "Holdout2"], [
    ["질의 속 개인정보", t1, t2], ["정답 문서에 같은 글자로 있음", f"{s1} ({s1 / t1:.0%})", f"{s2} ({s2 / t2:.0%})"],
    ["표기가 다름", t1 - s1, t2 - s2]]) + table(["묻는 방식", "라벨", "질의에 쓴 값", "문서에 적힌 값", "문서"], ex_rows, "tight")
      + '<p class="note">이름 띄어쓰기, 번호 붙여 쓰기·하이픈 위치가 달라도 같은 값으로 찾아야 함</p>', kicker="8. 검색 질의")

# 15-2. 데이터 품질
by_text = collections.defaultdict(list)
for d in H2:
    by_text[d["text"]].append(d["id"])
dups = [v for v in by_text.values() if len(v) > 1 and B2[v[0]]["text"].strip()]
dup = sum(len(v) - 1 for v in dups)
nopii = collections.Counter(d["doc_type"] for d in H2 if not d["entities"])
slide("데이터 품질: 지저분한 입력 · 중복 문서", f"""<div class="cols">{doc("H20310", 0, 260, mark=("■□■ [스캔 변환본] ■□■", "▶▶"))}<div>
{doc(dups[0][0], 0, 150)}<p class="note">↑ {dups[0][0]} 와 {dups[0][1]} 은 원문이 글자 하나까지 같음</p></div></div>
<p class="note">원문이 완전히 같은 문서 {dup}건 (예: {", ".join(" = ".join(v) for v in dups[:3])}). 개인정보 없는 문서 {sum(nopii.values())}건: {esc(", ".join(f"{t} {c}" for t, c in nopii.most_common()))}</p>""", kicker="9. 품질")

# 16. 정리
slide("정리: Holdout1 과 달라진 점", f'''<ol class="sum">
<li><b>문서 종류가 전부 새것</b> — 23종 중 Holdout1 과 겹치는 종류 0개, 종류마다 같은 양식 반복</li>
<li><b>더 길고 다양한 형태</b> — 표 · 영문 메일 · 챗봇 로그 · 하루치 업무일지, 빈 문서 2건</li>
<li><b>개인정보가 더 빽빽</b> — 문서당 4.9 → 7.1개, 표 문서는 한 문서에 고객 수십 명</li>
<li><b>표기 변형 확대</b> — 한글로 읽은 전화번호, 한 글자씩 띄운 이름, 영문 이름이 새로 등장</li>
<li><b>개인정보처럼 보이는 것</b> — 법인등록번호, 서버 IP, 회사 대표번호, 송장번호, 카드 끝자리</li>
<li><b>사람이 여러 문서에 걸침</b> — 2개 이상 문서에 나오는 실명 48 → 418명, 고객이자 직원인 이름(동명이인) {HM1[0]} → {HM2[0]}개</li>
<li><b>질의는 정답 문서가 많아짐</b> — 평균 3.0 → 7.3개, 10개 이상 31개</li>
<li><b>틀이 정해진 문서</b> — 대부분 종류가 양식 몇 가지뿐 (지점업무일지만 예외)</li>
</ol>''', kicker="10. 정리")

CSS = """
@font-face { font-family: 'Pretendard'; src: url('PretendardVariable.woff2') format('woff2'); font-weight: 100 900; }
@page { size: 1280px 720px; margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; font-family: 'Pretendard', 'Malgun Gothic', sans-serif; color: #1b2433; background: #fff; }
.slide { width: 1280px; height: 720px; padding: 40px 64px 44px; position: relative; page-break-after: always; overflow: hidden;
         display: flex; flex-direction: column; }
.slide:last-child { page-break-after: auto; }
.kicker { color: #2f5bd3; font-weight: 700; font-size: 16px; letter-spacing: .02em; margin-bottom: 6px; }
h1 { font-size: 36px; margin: 0 0 14px; font-weight: 800; letter-spacing: -0.02em; }
h3 { font-size: 19px; margin: 0 0 8px; color: #33405a; }
.body { flex: 1; min-height: 0; display: flex; flex-direction: column; font-size: 20px; line-height: 1.55; }
.inner { margin: auto; width: fit-content; max-width: 100%; } .inner:has(> .cols, > .grid3) { width: 100%; }
table.xs { font-size: 14px; line-height: 1.3; } table.xs td, table.xs th { padding: 4px 8px; }
.body p { margin: 0 0 12px; }
.foot { position: absolute; bottom: 16px; right: 64px; font-size: 12px; color: #9aa3b5; }
table { border-collapse: collapse; font-size: 18px; margin-bottom: 12px; }
.inner > table { min-width: 760px; width: 100%; }
.cols table { width: 100%; }
th { background: #eef2fb; color: #2b3a5c; font-weight: 700; text-align: left; padding: 8px 14px; border-bottom: 2px solid #c9d3ea; }
td { padding: 8px 14px; border-bottom: 1px solid #e6e9f0; vertical-align: top; }
table.wide { width: 78%; font-size: 21px; } table.wide td, table.wide th { padding: 10px 18px; }
table.mid { font-size: 16.5px; } table.mid td, table.mid th { padding: 8px 10px; } table.mid td:first-child { white-space: nowrap; }
table.tight { font-size: 15.5px; } table.tight td, table.tight th { padding: 5px 10px; }
code { white-space: nowrap; font-family: Consolas, monospace; font-size: .92em; background: #f3f5f9; padding: 1px 5px; border-radius: 4px; }
.cols { display: flex; gap: 28px; align-items: flex-start; } .cols > * { flex: 1; min-width: 0; }
.grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; align-items: start; }
.doc { border: 1px solid #d8deea; border-radius: 8px; margin-bottom: 10px; background: #fafbfd; }
.doc-h { font-size: 14px; font-weight: 700; color: #5a6782; padding: 6px 12px; border-bottom: 1px solid #e3e7ef; }
.doc pre { margin: 0; padding: 10px 12px; font-family: 'Pretendard', 'Malgun Gothic', sans-serif; font-size: 15.5px; line-height: 1.5; white-space: pre-wrap; }
mark { background: #ffe58a; padding: 0 2px; border-radius: 3px; }
.snip mark, .nw { white-space: nowrap; }
.note { color: #5a6782; font-size: 16px; }
ul.types { list-style: none; padding: 0; margin: 0; columns: 2; font-size: 16px; }
ul.types li { display: flex; justify-content: space-between; padding: 3px 10px 3px 0; break-inside: avoid; border-bottom: 1px dotted #e1e5ee; margin-right: 14px; }
ul.types b { color: #2f5bd3; }
.cover { flex-direction: row; padding: 0; background: #0046ff; color: #fff; }
.cv-logo { width: 589px; height: 720px; background: #003be0 url('cover_logo.png') center / cover no-repeat; flex: none; }
.cv-text { flex: 1; display: flex; flex-direction: column; justify-content: center; padding: 0 60px 0 54px; }
.cv-kicker { font-size: 17px; font-weight: 700; color: rgba(255,255,255,.72); margin-bottom: 20px; }
.cv-title { font-size: 50px; line-height: 1.25; font-weight: 800; margin: 0 0 22px; color: #fff; letter-spacing: -0.02em; }
.cv-sub { font-size: 21px; line-height: 1.65; color: rgba(255,255,255,.88); margin: 0 0 44px; }
.cv-team { font-size: 16px; font-weight: 700; margin: 0; } .cv-team span { opacity: .55; margin: 0 9px; font-weight: 400; }
ol.sum { font-size: 23px; line-height: 1.8; padding-left: 26px; } ol.sum b { color: #2f5bd3; }
"""

out = HERE / "holdout2_eda.html"
out.write_text(f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>Holdout2 EDA</title><style>{CSS}</style></head>"
               f"<body>{''.join(slides)}</body></html>", encoding="utf-8")
pdf = HERE / "holdout2_eda.pdf"
edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
subprocess.run([edge, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", out.as_uri()],
               check=True, timeout=120)
print(f"{len(slides)} slides → {pdf}")
