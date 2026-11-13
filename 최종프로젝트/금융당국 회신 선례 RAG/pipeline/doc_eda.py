"""문서 생성: EDA_데이터_탐색.md. 실행: python pipeline/doc_eda.py (DB 읽기 전용). 생성 후 일부 고정 서술은 사람이 DB 값과 맞춰 고쳤다(worklog 68번) — 다시 돌리면 그 수정은 되돌아간다.
(원래 임시 폴더에서 만든 스크립트를 보관용으로 옮긴 것이다. 경로가 이 PC 기준 절대 경로로 들어 있다.)"""
import sys

sys.path.insert(0, r"C:\Users\USER\Desktop\likelion-AI-4team\최종프로젝트\금융당국 회신 선례 RAG\pipeline")
sys.stdout.reconfigure(encoding="utf-8")
import rag

ROOT = r"C:\Users\USER\Desktop\likelion-AI-4team\최종프로젝트\금융당국 회신 선례 RAG"
c = rag.connect()
q = lambda s: c.execute(s).fetchall()
one = lambda s: c.execute(s).fetchone()
L = []
add = L.append
pct = lambda a, b: f"{a / b:.1%}" if b else "-"
bar = lambda v, vmax, w=22: "█" * max(1, round(v / vmax * w)) if v else ""


def table(head, rows):
    add("| " + " | ".join(head) + " |")
    add("|" + "|".join("---" for _ in head) + "|")
    for r in rows:
        add("| " + " | ".join(str(x) for x in r) + " |")
    add("")


# ---------- 기본 수치
total = one("select count(*) n from interp.cases")["n"]
idx = one("select count(*) n from interp.cases where indexable")["n"]
chunked = one("select count(distinct case_key) n from interp.chunks")["n"]
groups = one("select count(*) n from interp.case_groups")["n"]
n_chunks = one("select count(*) n from interp.chunks")["n"]
n_art = one("select count(*) n from interp.case_articles")["n"]
n_laws = one("select count(*) n from interp.laws")["n"]
g = {r["gubun"]: r for r in q("select gubun, count(*) n, count(*) filter (where indexable) idx from interp.cases group by 1")}
today = "2026-11-10"

add("# 데이터 탐색(EDA): 금융당국 회신 선례 데이터")
add("")
add(f"작성일 {today}. 이 문서는 RAG를 만들기 전·중에 데이터가 어떻게 생겼는지 확인한 내용을 한곳에 모은 것이다. 분포 수치는 **현재 DB(`interp` 스키마)를 읽기 전용으로 조회해 계산**했고(생성 스크립트는 저장소 밖), 수집·전처리 단계에서 찾은 문제와 처리는 [worklog.md](worklog.md)의 번호로 출처를 달았다.")
add("")
add("**EDA(탐색적 데이터 분석)** 는 모델이나 검색을 만들기 전에 데이터의 구조, 분포, 이상값, 반복을 살펴 설계 결정의 근거를 얻는 단계다. 정답을 내는 게 아니라 \"무엇을 조심해야 하는가\"를 찾는 작업이다.")
add("")
add("---")
add("")
add("## 1. 한눈에 보기")
add("")
table(["단계", "건수", "설명"], [
    ["수집한 회신", f"{total:,}", f"법령해석 {g['법령해석']['n']:,} + 비조치의견서 {g['비조치의견서']['n']:,} (2014년 이후, 포털 상세 페이지)"],
    ["검색 대상", f"{idx:,}", f"법령해석 {g['법령해석']['idx']:,} + 비조치 {g['비조치의견서']['idx']:,}. {total - idx}건은 반려·철회 등으로 제외"],
    ["청크가 있는 사례", f"{chunked:,}", "포털 첨부 오류가 의심되는 1건을 제외"],
    ["같은 내용 묶음", f"{groups:,}", "평가의 정답 단위"],
    ["청크(검색 단위)", f"{n_chunks:,}", "질의 카드, 판단 본문, 규제 정비 카드"],
    ["인용 조문", f"{n_art:,}", f"인용된 법령 {n_laws}종"],
])
add("핵심 발견 5가지")
add("")
add("1. **비조치의견서의 절반 이상이 사실상 같은 문구다.** 2016년 규제 정비 건(제목만 다르고 회답이 거의 같음)이 비조치의 54%다. → 별도 유형(`rule_review_2016`)과 별도 검색 단위(규제 정비 카드)로 분리했다.")
add("2. **본문이 짧고 칸이 비어 있는 사례가 많다.** 회답이 \"이유 참조\"뿐인 사례, 질의요지가 없는 사례가 있다. → 빈 칸·참조 문구는 청크로 만들지 않는다.")
add("3. **같은 내용의 회신이 반복된다.** 연장 건, 같은 본문 사례. → 사례 묶음을 만들고 평가 정답을 묶음 단위로 정했다.")
add("4. **포털이 준 분야 값을 믿을 수 없다.** 구체적 분야도 틀린 예가 많고 2023년부터는 대부분 비어 있다. → 인용 법령으로 분야를 추정해 점수 가산용으로만 쓴다.")
add("5. **오래된 회신일수록 인용 조문이 이후 개정된 비율이 높다.** → 회신일 이후 개정 여부를 표시하는 시점 검증(③)을 필수 기능으로 넣었다.")
add("")
add("---")
add("")

# ---------- 2. 구분·제외
add("## 2. 구분과 검색 제외")
add("")
rows = []
for k in ("법령해석", "비조치의견서"):
    rows.append([k, f"{g[k]['n']:,}", f"{g[k]['idx']:,}", f"{g[k]['n'] - g[k]['idx']:,}"])
rows.append(["합계", f"{total:,}", f"{idx:,}", f"{total - idx:,}"])
table(["구분", "수집", "검색 대상", "제외"], rows)
add("제외 사유 (모두 비조치의견서 쪽에서 나왔다):")
add("")
table(["사유", "건수", "설명"], [[r["skip_reason"], r["n"], {"반려": "포털 처리구분이 반려 (본문 없음)", "철회": "신청인이 철회 (본문이 \"철회에 따른 반려\")", "본문 없음": "본문이 30자 미만", "다른 회신으로 처리": "\"유권해석으로 처리완료\"만 있음"}.get(r["skip_reason"], "")]
                                   for r in q("select skip_reason, count(*) n from interp.cases where not indexable group by 1 order by 2 desc")])
add("- 처리구분이 \"완료\"인데 실제로는 철회·반려인 건이 있어 본문 내용으로 판정했다(worklog 9번). 철회 30건은 전부 눈으로 확인했다.")
add("- 목록 두 개(최근회신사례·통합조회)의 날짜가 3,293건 달랐다. 최근회신사례의 날짜는 등록일이고, 회신일은 상세 페이지 값(= 통합조회 값)이라 회신일을 상세 값으로 정했다(worklog 3번).")
add("")
body = {r["body_source"]: r["n"] for r in q("select body_source, count(*) n from interp.cases where indexable group by 1")}
add(f"**본문 출처**: HTML 본문 {body['html']:,}건, 첨부 회신문(hwp·hwpx) {body['attachment']:,}건. 첨부는 거의 전부 비조치의견서이고(첨부 필요 327건 중 비조치 322건), 본문이 첨부에만 있거나 \"첨부 참조\"로 대체된 경우다. 첨부 중 pdf 2·png 1건은 신청인의 철회 신청서였다(worklog 6번, 9번).")
add("")
add("---")
add("")

# ---------- 3. 연도
add("## 3. 회신 연도 분포")
add("")
ys = q("select extract(year from reply_date)::int y, gubun, count(*) n from interp.cases where indexable and reply_date is not null group by 1,2 order by 1")
by = {}
for r in ys:
    by.setdefault(r["y"], {})[r["gubun"]] = r["n"]
vmax = max(sum(v.values()) for v in by.values())
rows = []
for y in sorted(by):
    a, b = by[y].get("법령해석", 0), by[y].get("비조치의견서", 0)
    rows.append([y, a, b, a + b, bar(a + b, vmax)])
table(["회신 연도", "법령해석", "비조치", "합계", "분포"], rows)
nodate = one("select count(*) n from interp.cases where indexable and reply_date is null")["n"]
add(f"- 회신일이 없는 검색 대상은 {nodate}건이다(O2120: 상세 값이 비었고 첨부 파일명과 경로의 날짜가 달라 추정하지 않음, worklog 9번).")
add("- **2016년이 압도적으로 많다.** 비조치의견서의 대부분이 2016년의 행정지도·모범규준 정비 회신이다(다음 절). 그 외 연도는 대체로 연 100~300건 수준이다.")
add(f"- 회신일 범위: {one('select min(reply_date)::text a, max(reply_date)::text b from interp.cases where indexable')['a']} ~ {one('select min(reply_date)::text a, max(reply_date)::text b from interp.cases where indexable')['b']}.")
add("")
add("---")
add("")

# ---------- 4. 사례 유형
add("## 4. 사례 유형과 답변유형")
add("")
add("**사례 유형(`doc_class`)**")
add("")
rows = []
for r in q("select doc_class, count(*) n, count(*) filter (where gubun='비조치의견서') nb from interp.cases where indexable group by 1 order by 2 desc"):
    rows.append([r["doc_class"], f"{r['n']:,}", pct(r["n"], idx), {"general": "일반 회신", "rule_review_2016": "2016년 행정지도·규제 정비 건 (회신일이 2016-02-01 또는 2016-07-08인 비조치)", "extension": "기한 연장 건"}.get(r["doc_class"], "")])
table(["유형", "건수", "비율", "설명"], rows)
rr = one("select count(*) n from interp.cases where indexable and doc_class='rule_review_2016'")["n"]
add(f"- 규제 정비 건 {rr}건은 비조치 검색 대상 {g['비조치의견서']['idx']:,}건의 {pct(rr, g['비조치의견서']['idx'])}다. 회답이 \"비조치(무효) 동 사안은 현재 효력이 없으며…\"로 거의 같아서, 사례를 구분하는 정보는 제목(행정지도·규제 이름)과 답변유형뿐이다(worklog 7번, 21번).")
add("- 이 유형은 회신일 두 날짜(2016-02-01, 2016-07-08)에 몰려 있다. 같은 해 다른 날짜의 비조치는 일반 요청이다. 처음에는 내용 패턴으로 판정했다가 31건 미표시·2건 과표시가 나와 **회신일 기준**으로 바꿨다(worklog 10번).")
add("")
add("**답변유형(`answer_type`, 비조치의견서 쪽에서 의미가 있음)**")
add("")
rows = []
for r in q("select coalesce(answer_type, '(없음)') a, count(*) n from interp.cases where indexable group by 1 order by 2 desc"):
    rows.append([r["a"], f"{r['n']:,}", pct(r["n"], idx)])
table(["답변유형", "건수", "비율"], rows)
add("- 정해진 값(비조치·비조치(무효)·조치·일부조치·기타·행정지도·감독행정·협회자율규제)만 쓰고 원문 값은 따로 보관했다. 법령해석은 대부분 답변유형이 없다.")
add("- `비조치(무효)`는 \"현재 효력 없음\"으로 답변에 알린다(답변 규칙 4).")
add("")
add("---")
add("")

# ---------- 5. 분야
add("## 5. 분야")
add("")
rows = []
cats = q("select coalesce(category_final, '(없음)') a, count(*) n from interp.cases where indexable group by 1 order by 2 desc")
vmax = cats[0]["n"]
for r in cats:
    rows.append([r["a"], f"{r['n']:,}", pct(r["n"], idx), bar(r["n"], vmax)])
table(["분야(최종)", "건수", "비율", "분포"], rows)
src = {r["a"]: r["n"] for r in q("select coalesce(category_final_source, '없음') a, count(*) n from interp.cases where indexable group by 1")}
portal_blank = one("select count(*) n from interp.cases where indexable and (category is null or category = '')")["n"]
add(f"- 최종 분야의 출처: 법령으로 추정 {src.get('inferred', 0):,}건, 포털 값 {src.get('portal', 0):,}건, 없음 {src.get('없음', 0):,}건. 포털 분야가 빈 사례는 {portal_blank:,}건이다.")
add("- **포털 분야 값은 부정확하다.** 2016~2022년에는 '공통'이 기본값처럼 쓰이고, 2023년부터는 대부분 비어 있다. 구체적인 분야도 \"전자금융거래법 관련 문의\"가 보험으로 되어 있는 식의 오류가 있다. 수기 판정 표본(공통 제외 37건)에서 포털 값 65%(15/23), 법령 기반 추정 95%(35/37)였다(worklog 18번).")
add("- 그래서 추정 분야는 필터가 아니라 **점수 가산용**으로만 쓰도록 설계했다(틀린 값으로 거르면 정답이 통째로 빠지기 때문).")
add("")
add("---")
add("")

# ---------- 6. 글 길이
add("## 6. 글 길이와 빈 칸")
add("")
add("검색 대상 사례의 칸별 글자 수(공백 포함, 비어 있지 않은 칸 기준).")
add("")
rows = []
for dc, label in (("general", "일반 회신"), ("rule_review_2016", "2016 규제 정비 건")):
    for sec, name in (("question", "질의요지"), ("answer", "회답"), ("reason", "이유")):
        r = one(f"""select count(*) n, count(*) filter (where coalesce(btrim({sec}), '') = '') empty,
                    percentile_cont(0.5) within group (order by char_length({sec})) filter (where coalesce(btrim({sec}), '') <> '') med,
                    percentile_cont(0.9) within group (order by char_length({sec})) filter (where coalesce(btrim({sec}), '') <> '') p90,
                    max(char_length({sec})) mx from interp.cases where indexable and doc_class in ('{dc}')""")
        rows.append([label, name, f"{r['n']:,}", f"{r['empty']:,} ({pct(r['empty'], r['n'])})", int(r["med"] or 0), int(r["p90"] or 0), f"{r['mx']:,}"])
table(["유형", "칸", "사례 수", "빈 칸", "중앙값", "90%", "최대"], rows)
bc = one("select percentile_cont(0.5) within group (order by body_chars) med, percentile_cont(0.9) within group (order by body_chars) p90, max(body_chars) mx from interp.cases where indexable")
add(f"- 사례 전체 본문 길이(`body_chars`): 중앙값 {int(bc['med']):,}자, 90% {int(bc['p90']):,}자, 최대 {bc['mx']:,}자.")
add("- **일반 회신은 이유 칸이 가장 길고**(중앙값 400자대), 회답은 짧다(150자 안팎). 회답이 \"하단의 이유 참조\" 같은 가리키는 문구뿐인 사례가 있어, 그런 문구를 지운 뒤 회답이 비는 사례가 늘었다(회답 11건 → 269건, worklog 14번). 회답 단독 조각의 검색 가치가 낮아 회답과 이유를 합쳐 \"판단 본문\"으로 만들게 된 배경이다.")
add("- **2016 규제 정비 건은 세 칸이 모두 40자대로 매우 짧다.** 질의요지가 있는 건도 일부뿐이다.")
add("- 가장 긴 사례는 한컴 웹편집기 내부 데이터가 섞여 25만 자였던 L2770이고, 이를 지워 508자가 됐다(worklog 9번).")
add("")
add("---")
add("")

# ---------- 7. 소관부서
add("## 7. 소관부서")
add("")
rows = []
for r in q("select d.dept_name, count(*) n from interp.case_depts d join interp.cases x using (case_key) where x.indexable group by 1 order by 2 desc limit 10"):
    rows.append([r["dept_name"], f"{r['n']:,}"])
table(["부서(사례에 여러 부서가 붙을 수 있음)", "사례 수"], rows)
add("- 부서 이름에는 기관 이름(금융위원회·금융감독원·금융정보분석원)이 섞여 있어 `is_org`로 구분해 두었다.")
add("")
add("---")
add("")

# ---------- 8. 반복
add("## 8. 반복과 중복")
add("")
gr = q("select group_type, count(*) n, sum(n_cases) cases from interp.case_groups where n_cases > 1 group by 1 order by 2 desc")
multi = sum(r["n"] for r in gr)
mc = sum(r["cases"] for r in gr)
add(f"- **같은 내용 묶음**: 검색 대상 {idx:,}건이 {groups:,}개 묶음이 되었고, 2건 이상 묶인 것은 {multi}개({mc}건)다. " + ", ".join(f"{r['group_type']} {r['n']}개" for r in gr) + ". 묶는 규칙은 구분이 같고 질의·회답·이유가 같을 때이며 연장 건은 묶지 않았다(기한이 달라 효력 기간이 다른 선례, worklog 33번).")
dup = q("""select gubun, count(*) filter (where c >= 2) sets, coalesce(sum(c) filter (where c >= 2), 0) cases from
           (select gubun, regexp_replace(title, '\\s+|\\((연장|재연장)[^)]*\\)', '', 'g') t, count(*) c from interp.cases where indexable group by 1,2) s group by 1""")
add("- **같은 제목을 가진 사례**(공백과 \"(연장)\" 표기를 뺀 제목이 같은 사례): " + ", ".join(f"{r['gubun']} {r['sets']}개 제목에 걸쳐 {r['cases']}건" for r in dup) + ". 비조치가 훨씬 많다. 시황·한시 조치가 반복·연장되기 때문이다.")
ext = q("select gubun, count(*) n from interp.cases where indexable and title like '%연장%' group by 1")
add("- 제목에 \"연장\"이 든 사례: " + ", ".join(f"{r['gubun']} {r['n']}건" for r in ext) + ".")
add("- 질의요지가 완전히 같은 일반 회신 160건(61묶음)은 제목에 \"연장\"이 든 경우가 60건이다(worklog 21번).")
add("- 이 반복 때문에 **평가에서 같은 주제의 다른 회신이 1위로 오면 오답으로 계산되는 문제**가 계속 나타났다(worklog 45번, 56번, 57번).")
add("")
add("---")
add("")

# ---------- 9. 인용 조문
add("## 9. 인용 조문과 법령")
add("")
buck = q("""select x.gubun, case when n = 0 then '0' when n = 1 then '1' when n <= 3 then '2~3' when n <= 6 then '4~6' else '7+' end b, count(*) c from
            (select x.case_key, x.gubun, count(a.id) filter (where a.resolve_method in ('explicit','context') and not a.supplementary) n
             from interp.cases x left join interp.case_articles a using (case_key) where x.indexable group by 1,2) x group by 1,2""")
order = ["0", "1", "2~3", "4~6", "7+"]
tab = {}
for r in buck:
    tab.setdefault(r["b"], {})[r["gubun"]] = r["c"]
rows = []
for b in order:
    a, d = tab.get(b, {}).get("법령해석", 0), tab.get(b, {}).get("비조치의견서", 0)
    rows.append([b, f"{a:,}", f"{d:,}", f"{a + d:,}", pct(a + d, idx)])
table(["사례당 인용 조문 수(확실한 연결만)", "법령해석", "비조치", "합계", "비율"], rows)
zero = tab.get("0", {})
add(f"- 조문이 한 개도 없는 검색 대상은 {sum(zero.values()):,}건이다. 그중 약 {rr}건이 2016 규제 정비 건이고, 나머지는 법령은 언급했지만 조문 번호가 없거나 법령 언급이 없는 회신이다(worklog 18번).")
rm = q("select resolve_method, count(*) n from interp.case_articles group by 1 order by 2 desc")
add("- **조문의 법령 연결 방식(`resolve_method`)**: " + ", ".join(f"{r['resolve_method']} {r['n']:,}" for r in rm) + ". `explicit`(법령명 바로 뒤의 조문)와 `context`(\"동법\"·\"같은 조\"를 문맥으로 해석)만 조문 조회·시점 판정에 쓰고, `default`·`unresolved`는 ❔로 둔다(틀리게 붙이느니 비워 둠).")
add("")
add("**가장 많이 언급된 법령 상위 10**")
add("")
rows = []
for r in q("select l.law_name, l.law_kind, count(*) n from interp.case_laws cl join interp.laws l using (law_id) join interp.cases x using (case_key) where x.indexable group by 1,2 order by 3 desc limit 10"):
    rows.append([r["law_name"], r["law_kind"], f"{r['n']:,}"])
table(["법령", "종류", "언급한 사례 수"], rows)
lk = q("select law_kind, count(*) n from interp.laws group by 1 order by 2 desc")
add("- 인용된 법령 " + f"{n_laws}종: " + ", ".join(f"{r['law_kind']} {r['n']}" for r in lk) + ". 법령 이름은 정식 명칭과 약칭이 뒤섞여 나오고(`금소법`, `자본시장법`), 오타·띄어쓰기 변형도 있어 약칭 사전(공식 약칭 + 수작업)과 이름 정규화로 묶었다(worklog 18번, 20번).")
add("- 협회·거래소 규정(표준규정, 금융투자회사의 영업 및 업무에 관한 규정 등)은 법령 API로 이력을 확인할 수 없어 시점 판정에서 ❔다.")
add("")
add("---")
add("")

# ---------- 10. 시점
add("## 10. 인용 조문의 시점 판정")
add("")
add("기준일 2026-11-09 기준으로, 회신일 이후 인용 조문이 개정됐는지를 조문마다 판정했다.")
add("")
st = q("select status, count(*) n from interp.case_article_status group by 1 order by 2 desc")
tot_st = sum(r["n"] for r in st)
names = {"changed": "⚠️ 회신일 이후 개정·폐지", "unchanged": "✅ 이후 개정 없음", "unknown": "❔ 확인 불가", "scheduled": "🕒 시행 예정 개정", "reg_changed": "🔶 규정 개정됨(조문 변경 미확인)", "reg_unchanged": "✅ 이후 규정 개정 없음"}
table(["조문 판정", "건수", "비율"], [[names.get(r["status"], r["status"]), f"{r['n']:,}", pct(r["n"], tot_st)] for r in st])
tc = q("select coalesce(time_check, '(조문 인용 없음)') a, count(*) n from interp.cases where indexable group by 1 order by 2 desc")
nm = {"article_changed": "⚠️ 조문 개정 확인", "unchanged": "✅ 모두 개정 없음", "unknown": "❔ 확인 불가 포함", "scheduled": "🕒 시행 예정", "reg_changed_only": "🔶 규정 개정만"}
add("사례 단위(인용 조문 중 가장 주의가 필요한 판정):")
add("")
table(["사례 판정", "사례 수", "비율"], [[nm.get(r["a"], r["a"]), f"{r['n']:,}", pct(r["n"], idx)] for r in tc])
add("**회신 연도별로 조문이 개정된 비율** (조문 인용이 있는 사례 중 ⚠️):")
add("")
yr = q("""select extract(year from reply_date)::int y, count(*) filter (where time_check is not null) n, count(*) filter (where time_check = 'article_changed') ch
          from interp.cases where indexable and reply_date is not null group by 1 order by 1""")
rows = [[r["y"], r["n"], r["ch"], pct(r["ch"], r["n"]), bar(r["ch"] / r["n"] if r["n"] else 0, 1.0, 20)] for r in yr if r["n"]]
table(["회신 연도", "조문 인용 사례", "⚠️ 사례", "비율", ""], rows)
add("- **오래된 회신일수록 조문이 이후 개정된 비율이 높다**(2015년 약 75% → 2026년 약 13%, worklog 20번). 회신이 당시에는 맞았어도 지금은 조문이 달라졌을 수 있다는 뜻이므로, 답변에 \"현재 적용 여부 확인 필요\"를 붙이는 것이 필수 기능이 되었다.")
add("- 조문이 개정됐다고 회신이 무효가 되는 것은 아니다. 조 단위 판정이라 변별력에 한계가 있고(전체 조문의 약 절반이 ⚠️), 인용한 항·호가 그대로인 경우는 \"인용 부분은 동일\"로 구분했다.")
add("- 폐지 법령, 시행 전 개정, 전부개정으로 조문 번호가 달라진 경우는 ✅가 되지 않도록 규칙을 넣었다(worklog 26번, 28번).")
add("")
add("---")
add("")

# ---------- 11. 청크
add("## 11. 검색 단위(청크)")
add("")
rows = []
for r in q("""select chunk_type, count(*) n, percentile_cont(0.5) within group (order by char_length(text)) med, percentile_cont(0.9) within group (order by char_length(text)) p90, max(char_length(text)) mx
              from interp.chunks group by 1 order by 2 desc"""):
    rows.append([r["chunk_type"], f"{r['n']:,}", int(r["med"]), int(r["p90"]), f"{r['mx']:,}"])
table(["청크 유형", "개수", "글자 수 중앙값", "90%", "최대"], rows)
add("- 칸이 900자를 넘으면 항목 기호(`□`, `1.`, `①`)에서 600자 근처까지 묶어 자른다. 일반 칸의 900자 초과는 0이고, 규제 정비 카드 2개만 설계상 900자를 넘는다.")
add("- 사례의 대부분은 질의 카드 1개 + 판단 본문 1개로 끝나고(판단 본문의 71%가 청크 1개), 길면 최대 5개로 나뉜다.")
add("- 같은 사례의 조각끼리 평균 유사도(0.862)가 다른 사례(0.432)보다 훨씬 높아서, 검색은 넉넉히 찾은 뒤 묶음 단위로 합친다(worklog 41번).")
add("")
add("---")
add("")

# ---------- 12. 품질 이슈
add("## 12. 데이터 품질 문제와 처리")
add("")
table(["발견한 문제", "규모", "처리", "출처"], [
    ["반려·철회·본문 없음·다른 회신으로 처리", "103건", "검색 대상에서 제외(원문은 사유와 함께 보관)", "worklog 9번"],
    ["한글·워드 붙여넣기 HTML의 줄바꿈이 단어를 끊음 (`신용\\n정보법`)", "다수", "조각 사이 줄바꿈 삭제, `<br>`·문단 끝에서만 줄 나눔", "worklog 4번, 9번"],
    ["HTML 주석·처리지시문이 본문으로 읽힘 (`[if !supportEmptyParas]`, `xml:namespace`, `rhwp-studio-clipboard`)", "수천 곳", "파싱 단계에서 주석·선언문 노드를 통째로 버림 (대문자 변형이 정규식에 안 걸리던 것도 발견)", "worklog 12번"],
    ["한컴 웹편집기 내부 데이터가 본문에 섞임", "1건 (25만 자)", "해당 줄 삭제", "worklog 9번"],
    ["특수 글꼴 문자(원문자 ①②③·가운뎃점이 읽을 수 없는 문자로 남음)", "521개, 112건", "문맥으로 확인한 것만 ①②③…·`·`로 변환", "worklog 10번"],
    ["첨부 회신문의 담당자 이름·전화번호", "10건", "전화번호와 \"…로 문의\" 줄 삭제 (판단 내용이 있는 문장은 전화번호만 삭제)", "worklog 9번, 10번"],
    ["비조치 첨부의 체크칸(☑비조치 □조치 □기타) 판독 오류", "153건", "빈 칸 기호가 앞에 없는 선택지 하나만 체크로 인정", "worklog 10번"],
    ["정보 없는 문구(가리키기·정형 면책·서명·자리채움)", "줄 600여 개", "줄 전체가 정해진 문장 틀일 때만 삭제 (조건을 담은 판단 문장은 보존)", "worklog 14번"],
    ["2016 규제 정비 건 판정 기준이 좁았음", "미표시 31 / 과표시 2", "내용 패턴 → 회신일 기준으로 변경", "worklog 10번"],
    ["제목이 비어 있음 (O767, O775)", "2건", "질의요지 첫 줄로 채우고 출처 표시", "worklog 13번"],
    ["회신일이 없음", "20건 (반려 19 + O2120)", "추정하지 않고 비워 둠", "worklog 9번"],
    ["조문이 다른 법령에 잘못 연결 (줄임 표기가 직전 법률에 붙음)", "이력 없는 조문 72쌍 중 9건 확인", "`resolve_method` 도입, 오배정이 쉬운 표기는 따로 처리, 추출 규칙 재작성", "worklog 20번, 26번"],
    ["법령명 첫 글자가 잘림 (`본인서명사실…` → `인서명사실…`)", "여러 법령", "수식어 제거를 뒤에 공백이 있을 때만 하도록 수정", "worklog 20번"],
    ["포털 첨부가 다른 회신의 파일 (제목과 본문이 안 맞음)", "1건 (L2643)", "같은 묶음의 다른 사례에 같은 글이 있어 청크 생성 제외", "worklog 33번, 36번"],
    ["포털 분야 값 부정확", "공통 기본값, 2023년 이후 공란", "인용 법령으로 분야 추정, 점수 가산용으로만 사용", "worklog 18번"],
])
add("처리할 때마다 **전후를 줄 단위·건 단위로 비교**했고, 규칙으로 설명되지 않는 변경이 0인지 확인했다. 한 건을 보고 규칙을 고치지 않고 일반 규칙으로만 고쳤다(과적합 방지).")
add("")
add("---")
add("")

# ---------- 13. 설계 결정
add("## 13. EDA에서 나온 설계 결정")
add("")
table(["관찰", "설계 결정"], [
    ["비조치의 54%가 2016 규제 정비 건, 회답이 거의 같은 문구", "`doc_class` 분리, 사례당 규제 정비 카드 1개, 기본 검색에서 점수를 낮추고 규제 효력 질문에서만 우선(설계)"],
    ["회답은 짧고 \"이유 참조\" 같은 문구가 많으며, 이유가 가장 김", "질의 카드(제목+질의요지)와 판단 본문(회답+이유)으로 검색 단위를 정함 → 평가에서 확정"],
    ["같은 내용·연장 사례가 반복됨", "사례 묶음(`case_groups`)을 만들고 정답을 묶음 단위로, 화면에서는 \"같은 취지 N건\""],
    ["포털 분야 값이 부정확", "분야는 추정값을 점수 가산으로만 사용, 출처 컬럼 분리"],
    ["오래된 회신일수록 조문이 개정됨", "③ 시점 검증을 필수 기능으로, 판정을 ⚠️✅🔶🕒❔로 표시"],
    ["조문이 다른 법령에 연결되는 오류가 실제로 있음", "`resolve_method`로 신뢰도를 기록하고 불확실하면 비움, 품질 관문(표본 대조) 통과 전 다음 단계 금지"],
    ["법령 이름 표기가 제각각", "약칭 사전·이름 정규화로 묶고, 조문 조회(②)는 법령 이름이 없으면 조회하지 않고 후보를 보여줌"],
    ["첨부 회신문에 담당자 이름·연락처", "적재하지 않음, 질문은 모델 호출 전에 개인정보 형식 마스킹"],
    ["포털 저작권정책", "`data/`는 저장소에 올리지 않고 답변에는 요지와 링크만 사용"],
])
add("---")
add("")

# ---------- 14. 한계
add("## 14. 한계")
add("")
add("- 이 문서의 분포 수치는 현재 DB 기준이다. 전처리·판정 규칙을 바꾸면 수치가 달라진다(조문 9,907건 등).")
add("- 일부 서술(분야 추정 정확도, 연도별 조문 개정 비율의 근사값, 칸 길이 변화)은 worklog 작성 시점의 값을 인용했고 이번에 다시 계산하지 않았다. 해당 문장에는 worklog 번호를 달았다.")
add("- 그래프는 문자 막대로만 표현했다. 발표용 시각화는 따로 만들지 않았다.")
add("- 분야 수기 판정, 조문 연결 대조는 표본(100~200건 규모)이고 판정자가 규칙 작성자와 같다.")
add("- 2016 규제 정비 건과 일반 회신의 차이, 연장 건 효력 기간 같은 도메인 판단은 데이터에서 읽은 것이며 법률 자문이 아니다.")
open(ROOT + r"\EDA_데이터_탐색.md", "w", encoding="utf-8", newline="").write("\n".join(L) + "\n")
print("생성 완료", len(L), "줄")
