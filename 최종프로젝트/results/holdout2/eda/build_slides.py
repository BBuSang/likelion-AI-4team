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


ROLE_AFTER_CUST = re.compile(r"^\s?(?:고객|님)")
ROLE_STAFF_BEFORE = re.compile(r"(?:상담원|상담사|상담 직원|작성|처리|담당|심사역|접수|배정|평가자|대상 상담원|기안|직원)\s*[:：]?\s*$")
ROLE_STAFF_AFTER = re.compile(r"^\s?(?:과장|팀장|대리|사원|차장|부장|주임|본부장|이사|대표|\()")


def role_of(d, e):
    after = d["text"][e["end"]:e["end"] + 4]
    before = d["text"][d["text"].rfind("\n", 0, e["start"]) + 1:e["start"]]
    if ROLE_AFTER_CUST.match(after):
        return "고객"
    if ROLE_STAFF_AFTER.match(after) or ROLE_STAFF_BEFORE.search(before):
        return "직원"
    return None


def role_split(D):
    docs, roles = collections.defaultdict(set), collections.defaultdict(collections.Counter)
    for d in D:
        for e in d["entities"]:
            if e["label"] == "person" and e.get("form") == "full":
                nm = e["text"].replace(" ", "")
                docs[nm].add(d["id"])
                r = role_of(d, e)
                if r:
                    roles[nm][r] += 1
    out = collections.defaultdict(list)
    for nm, ds in docs.items():
        out[roles[nm].most_common(1)[0][0] if roles[nm] else "모름"].append(len(ds))
    return out, docs, roles


R1, _, _ = role_split(H1)
R2, DOCS2, ROLES2 = role_split(H2)
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
    slides.append(f'<section class="slide">{k}<h1>{esc(title)}</h1><div class="body">{body}</div>'
                  f'<div class="foot">Holdout2 EDA · 사공삼 403</div></section>')


# 1. 표지
slides.append(f'''<section class="slide cover"><div class="kicker">최종 프로젝트 · 데이터 탐색</div>
<h1 class="big">Holdout2 데이터 EDA</h1><p class="sub">Holdout1 과 비교해 무엇이 달라졌나</p>
<p class="meta">사공삼 403 · 2026-11-27</p></section>''')

# 2. 데이터 구성
slide("받은 데이터", table(["파일", "내용", "건수"], [
    ["<code>holdout2_public.jsonl</code>", "문서 원문 (id · doc_type · text)", f"{S2['docs']}건"],
    ["<code>holdout2_labeled.jsonl</code>", "같은 문서 + 개인정보 정답 위치(entities) + 어려운 사례 태그(hard_cases)", f"{S2['docs']}건"],
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
slide("Holdout2 문서 종류별 프로필", '<div class="cols">' + table(head_p, prof[:half_p], "tight") + table(head_p, prof[half_p:], "tight") + "</div>"
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
slide("문서 길이: 더 길고 더 들쭉날쭉", table(["길이", "Holdout1", "Holdout2"], [[o, S1["buckets"].get(o, 0), S2["buckets"].get(o, 0)] for o in order]
      + [["평균 / 중앙값", f'{n(S1["len_mean"])} / {n(S1["len_median"])}자', f'{n(S2["len_mean"])} / {n(S2["len_median"])}자'],
         ["최소 ~ 최대", f'{S1["len_min"]} ~ {S1["len_max"]}자', f'{S2["len_min"]} ~ {n(S2["len_max"])}자']])
      + f'''<div class="cols">{doc("H20140", 0, 330)}<div><h3>빈 문서 2건 (종류: 빈문서)</h3>
<div class="doc"><div class="doc-h">H20345</div><pre>{esc(repr(B2["H20345"]["text"]))}  ← 공백·줄바꿈뿐</pre></div>
<div class="doc"><div class="doc-h">H20437</div><pre>{esc(repr(B2["H20437"]["text"]))}  ← 빈 문자열</pre></div>
<p class="note">1,000자 넘는 문서 6건은 모두 지점업무일지 (하루치 고객 응대를 시간순으로 나열)</p></div></div>''', kicker="4. 길이")

# 7. 라벨
lab_rows = [[LABEL_KO[l], n(S1["labels"][l]), f'{S1["labels"][l] / S1["ents"]:.0%}', n(S2["labels"][l]), f'{S2["labels"][l] / S2["ents"]:.0%}']
            for l, _ in S2["labels"].most_common()]
dens = sorted(S2["density"].items(), key=lambda x: -x[1])
slide("개인정보 라벨 분포와 밀도", '<div class="cols">' + table(["라벨", "Holdout1", "비율", "Holdout2", "비율"], lab_rows)
      + "<div><h3>Holdout2 문서당 개인정보 수</h3>" + table(["많은 종류", "개/문서"], [[esc(t), n(v, 1)] for t, v in dens[:5]])
      + table(["적은 종류", "개/문서"], [[esc(t), n(v, 1)] for t, v in dens[-4:]]) + "</div></div>"
      + '<p class="note">카드번호·계좌번호 비중이 커지고(H1 3%·4% → H2 6%·5%), 표 형태 문서에 개인정보가 몰림</p>', kicker="5. 개인정보")

# 8. 표 문서
slide("표로 된 문서: 한 문서에 여러 사람", f'''<div class="cols">{doc("H20589", 0, 330)}{doc("H20005", 0, 330, mark=("법인등록번호: 118023-3131332",))}</div>
<p class="note">콜백요청목록(문서당 34.3개)·법인카드신규발급 표에는 고객 이름·번호가 줄마다 나옴. 법인등록번호는 주민번호와 같은 6-7자리 모양이지만 정답 라벨이 아님</p>''',
      kicker="5. 개인정보")

# 8-1. 개인정보가 놓인 자리
ctx_rows = [[k, n(C1[k]), f"{C1[k] / sum(C1.values()):.0%}", n(C2[k]), f"{C2[k] / sum(C2.values()):.0%}"] for k in ("문장 속", "항목: 값", "표 칸")]
slide("개인정보가 놓인 자리: 문장에서 항목·표로", table(["자리", "Holdout1", "비율", "Holdout2", "비율"], ctx_rows)
      + f"""<div class="cols">{doc("H20001", "청구인이", 150, mark=("연락처: ", "이메일: ", "결제 계좌: "))}{doc("H20004", 0, 150, mark=("고객 이유수 요청으로",))}</div>
<p class="note">'항목: 값' = 개인정보 바로 앞이 '연락처:' 같은 항목 이름, '표 칸' = 바로 앞이 표 구분선 '|', 나머지는 문장 속</p>""",
      kicker="5. 개인정보")

# 9. 전화번호 표기
ph_rows = []
for shape in dict.fromkeys(list(S1["phones"]) + list(S2["phones"])):
    ph_rows.append([f"<code>{esc(shape)}</code>", "○" if shape in S1["phones"] else "", "○" if shape in S2["phones"] else "",
                    esc(S2["phones"].get(shape) or S1["phones"].get(shape))])
slide(f"전화번호 표기: 한글로 읽은 번호가 새로 등장", table(["모양 (숫자=9)", "H1", "H2", "실제 예"], ph_rows, "tight")
      + f'<p class="note">숫자 모양은 Holdout1 {len(S1["phones"])}가지 · Holdout2 {len(S2["phones"]) - 1}가지로 비슷 (Holdout2 는 지역번호 02 없음, +82- 꼴 추가). 대신 번호를 한글로 읽어 쓴 꼴이 15건 새로 등장 (챗봇 상담 로그) — 숫자 정규식으로는 안 잡히는 모양</p>', kicker="6. 표기 변형")

# 10. 한글 번호·카드 표기
slide("표기 변형 실제 예", f'''<div class="cols">{doc("H20006", 0, 250, mark=("공일공 팔사칠칠 일이공구", "정 수 혁"))}
<div>{doc("H20001", 60, 200, mark=("01071146387", "982-4606-5999-99"))}{doc("H20033", "1. 대상", 70, mark=("4092 4076 2658 6158",))}</div></div>
<p class="note">카드 표기 변형 문서 H1 {S1["hard"]["card_format_variant"]} → H2 {S2["hard"]["card_format_variant"]}, 계좌 {S1["hard"]["account_format_variant"]} → {S2["hard"]["account_format_variant"]}, 전화 {S1["hard"]["phone_format_variant"]} → {S2["hard"]["phone_format_variant"]}</p>''',
      kicker="6. 표기 변형")

# 11. 어려운 사례 태그
tags = list(dict.fromkeys([t for t, _ in S2["hard"].most_common()] + [t for t, _ in S1["hard"].most_common()]))
hard_rows = [[f"<code>{t}</code>", {
    "phone_format_variant": "전화 표기 변형", "card_format_variant": "카드 표기 변형", "account_format_variant": "계좌 표기 변형",
    "single_char_given_name": "한 글자 이름", "decoy_numbers": "개인정보처럼 보이는 번호", "address_format_variant": "주소 표기 변형",
    "given_only": "성 없는 이름", "homonym": "동명이인", "id_format_variant": "주민번호 표기 변형", "decoy_company_phone": "회사 대표번호",
    "compound_surname": "복성 (독고·황보 …)", "josa_attached": "조사 붙은 이름", "spaced_name": "띄어 쓴 이름", "english_name": "영문 이름",
    "surname_title_unlabeled": "성+직함 (라벨 아님)", "char_spaced_name": "한 글자씩 띄운 이름", "korean_numeral_phone": "한글로 읽은 전화번호",
    "long_doc": "긴 문서", "messy_input": "지저분한 입력", "duplicate_doc": "중복 문서", "same_doc_homonym": "한 문서 안 동명이인", "decoy_words": "이름처럼 보이는 말", "no_pii": "개인정보 없음"}.get(t, ""),
    S1["hard"].get(t, "") or "-", S2["hard"].get(t, "") or "-"] for t in tags]
half = (len(hard_rows) + 1) // 2
slide("어려운 사례 태그 (문서 수)", '<div class="cols">' + table(["태그", "뜻", "H1", "H2"], hard_rows[:half], "tight")
      + table(["태그", "뜻", "H1", "H2"], hard_rows[half:], "tight") + "</div>"
      + f'<p class="note">태그는 데이터에 붙어 온 것. Holdout2 에만 있는 태그: {" · ".join(t for t in tags if t not in S1["hard"])}</p>',
      kicker="7. 어려운 사례")

# 12. 새 함정 예
slide("Holdout2 에 새로 생긴 함정", f'''<div class="cols">{doc("H20009", "Dear", 200, mark=("Seongyul Kim", "Saebom Card"))}{doc("H20042", "장애 번호", 120, mark=("10.78.215.206", "INC-841039"))}</div>
<div class="cols">{doc("H20025", 0, 200, mark=("1588-1255", "742665443660"))}{doc("H20002", "상담사 > 확인", 150, mark=("끝자리가 1612", "+82 10-3861-3808"))}</div>''',
      kicker="7. 어려운 사례")

# 13. 사람
top = ", ".join(f"{esc(k)}({v})" for k, v in S2["names"].most_common(3))
spread = lambda S: [sum(v == 1 for v in S["names"].values()), sum(2 <= v <= 5 for v in S["names"].values()), sum(v >= 6 for v in S["names"].values())]
a, b = spread(S1), spread(S2)
slide("사람: 여러 문서에 걸쳐 나온다", table(["", "Holdout1", "Holdout2"], [
    ["이름 형태: 성+이름 / 성 없음 / 띄어 씀", f'{S1["form"]["full"]} / {S1["form"]["given"]} / {S1["form"]["spaced"]}', f'{n(S2["form"]["full"])} / {S2["form"]["given"]} / {S2["form"]["spaced"]}'],
    ["한 문서에만 나오는 실명", a[0], b[0]], ["2~5개 문서에 나오는 실명", a[1], b[1]], ["6개 이상 문서", a[2], b[2]],
    ["가장 많이 나온 실명", ", ".join(f"{esc(k)}({v})" for k, v in S1["names"].most_common(3)), top],
]) + f'''<div class="cols">{doc("H20001", 0, 70, mark=("청구인: 도재호",))}{doc("H20022", 0, 90, mark=("상담원: 도재호",))}</div>
<p class="note">같은 이름 '도재호' — 개인정보 열람을 청구한 고객과 채권관리팀 상담원은 다른 사람 (동명이인 태그 {S2["hard"]["homonym"]}건)</p>''', kicker="8. 사람")

# 13-1. 고객과 직원
def role_row(R, r):
    v = R.get(r, [])
    return [r, len(v), n(st.mean(v), 1) if v else "-", sum(1 for x in v if x >= 6)]


top_staff = sorted(((len(DOCS2[nm]), nm) for nm in DOCS2 if ROLES2[nm] and ROLES2[nm].most_common(1)[0][0] == "직원"), reverse=True)[:6]
top_cust = sorted(((len(DOCS2[nm]), nm) for nm in DOCS2 if ROLES2[nm] and ROLES2[nm].most_common(1)[0][0] == "고객"), reverse=True)[:6]
head_r = ["역할(추정)", "실명 수", "평균 문서 수", "6개+ 문서"]
slide("고객과 직원: 누가 여러 문서에 나오나", '<div class="cols"><div><h3>Holdout1</h3>' + table(head_r, [role_row(R1, r) for r in ("직원", "고객", "모름")])
      + "<h3>Holdout2</h3>" + table(head_r, [role_row(R2, r) for r in ("직원", "고객", "모름")]) + "</div><div>"
      + table(["Holdout2 직원 (많이 나온 순)", "문서"], [[esc(nm), c] for c, nm in top_staff], "tight")
      + table(["Holdout2 고객 (많이 나온 순)", "문서"], [[esc(nm), c] for c, nm in top_cust], "tight") + "</div></div>"
      + '<p class="note">역할은 정답에 없어 문맥으로 추정: 이름 뒤 "고객·님" → 고객 / 이름 뒤 직급·괄호, 앞에 "상담원:·작성:·처리:·담당" → 직원. 동명이인은 한 줄로 셈</p>',
      kicker="8. 사람")

# 14. 질의
types = list(dict.fromkeys(list(S2["qtype"]) + list(S1["qtype"])))
slide("검색 질의", '<div class="cols">' + table(["질의 유형", "H1", "H2"], [[f"<code>{t}</code>", S1["qtype"].get(t, "-"), S2["qtype"].get(t, "-")] for t in types], "tight")
      + "<div>" + table(["", "Holdout1", "Holdout2"], [
          ["질의 수", S1["queries"], S2["queries"]],
          ["질의 속에 개인정보", f'{S1["qpii"]}', f'{S2["qpii"]}'],
          ["정답 문서 수 평균", n(S1["rel_mean"], 1), n(S2["rel_mean"], 1)],
          ["정답 1개인 질의", S1["rel1"], S2["rel1"]],
          ["정답 10개 이상", S1["rel10"], S2["rel10"]],
          ["헷갈리는 문서 지정", S1["conf"], S2["conf"]],
      ]) + '<p class="note">새 유형: <code>staff_owner</code>(직원이 담당한 문서) · <code>long_doc_person</code>(긴 문서 속 사람)</p></div></div>',
      kicker="9. 검색 질의")

# 15. 질의 예
qex = []
for t in ["name_topic", "name_variant", "staff_owner", "value_format_variant", "address_partial", "semantic_no_pii",
          "hybrid_paraphrase", "homonym_role", "long_doc_person"]:
    q = next(x for x in Q2 if x["type"] == t)
    qex.append([f"<code>{t}</code>", esc(q["query"]), esc(", ".join(q["relevant_doc_ids"][:3]) + (" …" if len(q["relevant_doc_ids"]) > 3 else ""))])
slide("질의 유형별 예", table(["유형", "질의", "정답 문서"], qex, "tight"), kicker="9. 검색 질의")

# 15-1. 질의 속 값과 정답 문서 속 값
B1 = {d["id"]: d for d in H1}
t1, s1, _ = surface(Q1, B1)
t2, s2, pr = surface(Q2, B2)
seen, ex_rows = set(), []
for ty, lab, a, b, did in pr:
    if (ty, lab) not in seen:
        seen.add((ty, lab))
        ex_rows.append([f"<code>{ty}</code>", lab, f"<code>{esc(a)}</code>", f"<code>{esc(b)}</code>", did])
slide("질의 속 값과 정답 문서 속 값", table(["", "Holdout1", "Holdout2"], [
    ["질의 속 개인정보", t1, t2], ["정답 문서에 같은 글자로 있음", f"{s1} ({s1 / t1:.0%})", f"{s2} ({s2 / t2:.0%})"],
    ["표기가 다름", t1 - s1, t2 - s2]]) + table(["질의 유형", "라벨", "질의에 쓴 값", "문서에 적힌 값", "문서"], ex_rows, "tight")
      + '<p class="note">이름 띄어쓰기, 번호 붙여 쓰기·하이픈 위치가 달라도 같은 값으로 찾아야 함</p>', kicker="9. 검색 질의")

# 15-2. 데이터 품질
qrows = []
for tag, ko in (("messy_input", "지저분한 입력"), ("duplicate_doc", "중복 문서"), ("same_doc_homonym", "한 문서 안 동명이인"), ("no_pii", "개인정보 없음")):
    ds = [d for d in H2 if tag in d.get("hard_cases", [])]
    qrows.append([f"<code>{tag}</code>", ko, len(ds), esc(", ".join(f"{t} {c}" for t, c in collections.Counter(d["doc_type"] for d in ds).most_common(3)))])
dup = sum(v - 1 for v in collections.Counter(d["text"] for d in H2).values() if v > 1)
slide("데이터 품질: 지저분한 입력 · 중복 · 같은 문서 속 동명이인", table(["태그", "뜻", "문서", "종류"], qrows, "tight")
      + f"""<div class="cols">{doc("H20310", 0, 170, mark=("■□■ [스캔 변환본] ■□■", "▶▶"))}{doc("H20150", 0, 190, mark=("상담원: 박형윤", "네, 박형윤입니다"))}</div>
<p class="note">원문이 완전히 같은 문서 {dup}건. 개인정보 없는 문서 17건은 시스템장애보고 15 + 빈문서 2</p>""", kicker="10. 품질")

# 15-3. 날짜
allm = sorted(set(M1) | set(M2))
slide("문서 날짜 (문서의 첫 날짜, 월별)", table(["월"] + allm, [["Holdout1"] + [M1.get(m, "") for m in allm], ["Holdout2"] + [M2.get(m, "") for m in allm]], "tight")
      + f'<p class="note">날짜를 찾은 문서: Holdout1 {sum(M1.values())}/{len(H1)}, Holdout2 {sum(M2.values())}/{len(H2)}. 2025년 날짜는 연말정산 대상 기간처럼 지난해를 가리키는 경우</p>',
      kicker="10. 품질")

# 16. 정리
slide("정리: Holdout1 과 달라진 점", '''<ol class="sum">
<li><b>문서 종류가 전부 새것</b> — 23종 중 Holdout1 과 겹치는 종류 0개, 종류마다 같은 양식 반복</li>
<li><b>더 길고 다양한 형태</b> — 표 · 영문 메일 · 챗봇 로그 · 하루치 업무일지, 빈 문서 2건</li>
<li><b>개인정보가 더 빽빽</b> — 문서당 4.9 → 7.1개, 카드·계좌 비중 증가</li>
<li><b>표기 변형 확대</b> — 한글로 읽은 전화번호 새로 등장, 카드·계좌 표기 변형 문서 약 10배</li>
<li><b>개인정보처럼 보이는 것</b> — 법인등록번호, 서버 IP, 회사 대표번호, 송장번호, 카드 끝자리</li>
<li><b>사람이 여러 문서에 걸침</b> — 2개 이상 문서에 나오는 실명 48 → 418명, 동명이인 110건</li>
<li><b>질의는 정답 문서가 많아짐</b> — 평균 3.0 → 7.3개, 10개 이상 31개</li>
<li><b>개인정보가 놓인 자리</b> — 문장 속이 줄고 '항목: 값'·표 칸이 늘어남</li>
<li><b>틀이 정해진 문서</b> — 대부분 종류가 양식 몇 가지뿐 (지점업무일지만 예외)</li>
</ol>''', kicker="10. 정리")

CSS = """
@font-face { font-family: 'Pretendard'; src: url('PretendardVariable.woff2') format('woff2'); font-weight: 100 900; }
@page { size: 1280px 720px; margin: 0; }
* { box-sizing: border-box; }
body { margin: 0; font-family: 'Pretendard', 'Malgun Gothic', sans-serif; color: #1b2433; background: #fff; }
.slide { width: 1280px; height: 720px; padding: 46px 64px 40px; position: relative; page-break-after: always; overflow: hidden; }
.slide:last-child { page-break-after: auto; }
.kicker { color: #2f5bd3; font-weight: 700; font-size: 15px; letter-spacing: .02em; margin-bottom: 6px; }
h1 { font-size: 34px; margin: 0 0 22px; font-weight: 800; letter-spacing: -0.02em; }
h3 { font-size: 16px; margin: 0 0 8px; color: #33405a; }
.body { font-size: 17px; line-height: 1.55; }
.body p { margin: 0 0 12px; }
.foot { position: absolute; bottom: 18px; right: 64px; font-size: 12px; color: #9aa3b5; }
table { border-collapse: collapse; font-size: 15px; margin-bottom: 10px; }
th { background: #eef2fb; color: #2b3a5c; font-weight: 700; text-align: left; padding: 7px 12px; border-bottom: 2px solid #c9d3ea; }
td { padding: 6px 12px; border-bottom: 1px solid #e6e9f0; vertical-align: top; }
table.wide { width: 80%; font-size: 18px; } table.wide td, table.wide th { padding: 9px 16px; }
table.tight { font-size: 13px; } table.tight td, table.tight th { padding: 4px 9px; }
code { font-family: Consolas, monospace; font-size: .92em; background: #f3f5f9; padding: 1px 5px; border-radius: 4px; }
.cols { display: flex; gap: 24px; align-items: flex-start; } .cols > * { flex: 1; min-width: 0; }
.doc { border: 1px solid #d8deea; border-radius: 8px; margin-bottom: 10px; background: #fafbfd; }
.doc-h { font-size: 12px; font-weight: 700; color: #5a6782; padding: 5px 10px; border-bottom: 1px solid #e3e7ef; }
.doc pre { margin: 0; padding: 8px 10px; font-family: 'Pretendard', 'Malgun Gothic', sans-serif; font-size: 13px; line-height: 1.5; white-space: pre-wrap; }
mark { background: #ffe58a; padding: 0 2px; border-radius: 3px; }
.note { color: #5a6782; font-size: 14px; }
ul.types { list-style: none; padding: 0; margin: 0; columns: 2; font-size: 14px; }
ul.types li { display: flex; justify-content: space-between; padding: 2px 10px 2px 0; break-inside: avoid; border-bottom: 1px dotted #e1e5ee; margin-right: 14px; }
ul.types b { color: #2f5bd3; }
.cover { display: flex; flex-direction: column; justify-content: center; background: linear-gradient(135deg, #eef3ff, #ffffff 60%); }
.cover .big { font-size: 58px; margin: 6px 0 10px; } .cover .sub { font-size: 24px; color: #46557a; margin: 0 0 40px; } .cover .meta { color: #7a869f; font-size: 16px; }
ol.sum { font-size: 20px; line-height: 1.75; padding-left: 26px; } ol.sum b { color: #2f5bd3; }
"""

out = HERE / "holdout2_eda.html"
out.write_text(f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>Holdout2 EDA</title><style>{CSS}</style></head>"
               f"<body>{''.join(slides)}</body></html>", encoding="utf-8")
pdf = HERE / "holdout2_eda.pdf"
edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
subprocess.run([edge, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", out.as_uri()],
               check=True, timeout=120)
print(f"{len(slides)} slides → {pdf}")
