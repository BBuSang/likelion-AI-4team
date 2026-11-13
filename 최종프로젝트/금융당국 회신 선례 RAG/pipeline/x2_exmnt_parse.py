"""X2 현장건의 과제 파싱·칸 대응·분포 확인 (사용자 요청 2026-11-11). 청크·임베딩·DB 적재는 하지 않는다.
입력: data/raw/exmnt/{번호}.html (x1 --all로 받은 5,914건), data/raw/list_total.json
규칙 (11-11 대화에서 제안한 표의 1~5)
1. 칸 대응: 질의요지(question) ← 건의내용, 회답(answer) ← 검토 사유, 답변유형(answer_type) ← 검토결과(수용/불수용/일부수용/기조치(수용) …), 구분(gubun) = "현장건의"
2. 검색 제외(skip_reason, 사용자 결정 11-11): 타부처 소관(과제분류 "15."), 건의내용 없음, 사유 없음, 사유 전체가 무의미한 문구("수용", "첨부파일 참고" 등)뿐,
   다른 건 답변을 참조하라는 것뿐. 사유 길이 기준은 쓰지 않는다(짧아도 의미 있는 사유가 많음). 제외 건도 사유와 함께 보관한다
3. 제도 변경 표시: 수용·일부수용 → changed_after(이후 제도가 바뀌었을 수 있음). 기조치(수용) → already_in_place(회신 당시 이미 그렇게 운영 중, 현행 해석에 가까움)
개인정보(사용자 결정 11-11): 전화·이메일·카드·계좌·주민번호 형식과 연락처 근처의 이름(직함 앞, 괄호 안 번호 앞, "성명:" 등)을 모두 가린다. 날짜(YYYY-MM-DD)는 계좌 형식과 겹쳐도 남긴다
4. 조문 추출(articles): 건의내용·사유에서 "제N조(의N)" 표기와 인용 규범 종류(법률·시행령·감독규정·시행세칙·모범규준·행정지도·가이드라인)를 뽑아 둔다. 법령 연결·시점 판정은 다음 단계
5. 중복 확인: 정규화한 제목이 같은 건, 정규화한 건의내용 앞 300자가 같은 건을 센다(묶음 확정은 다음 단계)
출력: data/processed/exmnt_cases.jsonl, exmnt_cases.csv, exmnt_parse_summary.json
"""
import json
import re
from collections import Counter, defaultdict
from statistics import median, quantiles

from common import OUT, RAW, manifest, write_csv, write_jsonl
from rag import PII
from x1_exmnt_pilot import parse, url_of

ARTICLE = re.compile(r"제\s?\d+\s?조(?:의\s?\d+)?")
NORM_KIND = {"법률": re.compile(r"[가-힣]+법(?![인률령규])|법률"), "시행령": re.compile(r"시행령"), "감독규정": re.compile(r"감독규정|업무규정"),
             "시행세칙": re.compile(r"시행세칙|세칙"), "모범규준": re.compile(r"모범규준"), "행정지도": re.compile(r"행정지도|지도공문|지도사항"),
             "가이드라인": re.compile(r"가이드라인|가이드|지침"), "협회 규정": re.compile(r"협회\s?(?:규정|기준|자율)|표준약관|업무방법")}
CHANGED = ("수용", "일부수용")          # 이후 제도가 바뀌었을 수 있음. 기조치(수용)은 회신 당시 이미 그렇게 운영 중이라는 뜻이라 따로 표시(사용자 결정 11-11)
MIN_PROPOSAL = 30
# 사유 길이 기준은 쓰지 않는다(짧아도 의미 있는 사유가 많음, 79번). 사유 전체가 아래 무의미 문구이거나 다른 건 답변을 참조하라는 것뿐이면 제외
VAGUE = re.compile(r"^(?:수용|불수용|일부수용|기조치|내용참고|참고|첨부파일참고|첨부파일참조|첨부참조|첨부참고|붙임참조|붙임참고|별첨참조|별첨참고|중장기검토|중장기검토필요|추가검토|추가검토필요|검토중|회사의요청사항인정됨|요청사항인정됨)$")
CROSS_REF = re.compile(r"답변\s*(?:을\s*)?(?:참조|참고)")
# 개인정보: 모두 가린다(사용자 결정 11-11). 계좌번호 형식은 날짜(YYYY-MM-DD)와 겹치므로 날짜는 건너뛴다
DATE = re.compile(r"(?:19|20)\d{2}-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12]\d|3[01])$")
NAME_TAG = re.compile(r"(?:성명|담당자|건의자|문의|연락처)\s*[:：)]\s*([가-힣]{2,4})(?=[\s,(/]|$)")
NAME_OFFICIAL = re.compile(r"(?<![가-힣\]])(?!이름)([가-힣]{2,4})(?=\s*(?:사무관|주무관|서기관|선임조사역|수석조사역)(?![가-힣]))")
NAME_UNIT = re.compile(r"(?<=[부과팀실국원단회터]\s)(?!이름)([가-힣]{2,4})(?=\s*(?:차장|대리|팀장|과장|부장|조사역|검사역|주임|계장|연구원)(?![가-힣]))")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
NAME_PHONE = re.compile(r"([(,/]\s*)([가-힣]{2,4})(\s*[,/]?\s*)(?=\[전화번호\])")
NAME_TITLE = re.compile(r"([가-힣]{2,4})\s*(?=(?:선임|수석|책임)?\s*(?:사무관|주무관|서기관|과장|차장|대리|팀장|부장|실장|국장|조사역|검사역|연구원|주임|계장|팀원)(?![가-힣]))")


def norm(s):
    return re.sub(r"[\s\W_]+", "", s or "")


def mask(text, counts):
    """전화·이메일·카드·계좌·주민번호 형식과, 연락처 근처(같은 줄 40자 이내)의 이름 + 직함, '성명:' 같은 표기의 이름을 가린다."""
    if not text:
        return text
    text, n = EMAIL.subn("[이메일]", text)          # rag.PII의 이메일 형식은 \w가 한글까지 잡아 뒤의 조사("로")를 함께 지우므로 영문 주소만 따로 가린다
    counts["[이메일]"] += n
    for rx, tag in PII:
        if tag == "[이메일]":
            continue
        def sub(m, tag=tag):
            if tag == "[계좌번호]" and DATE.match(m.group(0)):
                return m.group(0)
            counts[tag] += 1
            return tag
        text = rx.sub(sub, text)

    def near_contact(s, e):
        win = text[max(0, s - 40):s] + text[e:e + 40]
        return "[전화번호]" in win or "[이메일]" in win

    def name_title(m):
        if not near_contact(m.start(), m.end()):
            return m.group(0)
        counts["[이름]"] += 1
        return "[이름]" + m.group(0)[len(m.group(1)):]
    text = NAME_TITLE.sub(name_title, text)
    for rx in (NAME_OFFICIAL, NAME_UNIT):                     # 연락처가 없어도: 공무원 직함 앞 이름, "○○부 이름 대리"처럼 부서명 바로 뒤 이름
        text, n = rx.subn("[이름]", text)
        counts["[이름]"] += n
    text, n = NAME_PHONE.subn(r"\1[이름]\3", text)            # "(이름 [전화번호])"처럼 직함 없이 괄호 안 이름 바로 뒤에 번호
    counts["[이름]"] += n

    def tagged(m):
        counts["[이름]"] += 1
        return m.group(0).replace(m.group(1), "[이름]")
    return NAME_TAG.sub(tagged, text)


def skip_of(x):
    reason, prop, cat = x.get("review_reason") or "", x.get("proposal") or "", x.get("category") or ""
    if cat.startswith("15."):
        return "타부처 소관"
    if not prop.strip() or len(norm(prop)) < MIN_PROPOSAL:
        return "건의내용 없음"
    if not reason.strip():
        return "사유 없음"
    if VAGUE.match(norm(reason)):
        return "사유가 무의미한 문구뿐"
    if CROSS_REF.search(reason) and len(norm(reason)) < 80:
        return "다른 건 답변 참조뿐"
    return None


def main():
    lst = json.load(open(RAW / "list_total.json", encoding="utf-8"))
    meta = {r["dataIdx"]: r for r in next(v for v in lst.values() if isinstance(v, list)) if r["pastreqType"] == "현장건의 과제"}
    items, errors, masked_total = [], [], Counter()
    for no, m in meta.items():
        try:
            html = (RAW / "exmnt" / f"{no}.html").read_text(encoding="utf-8")
            a = html.find('<div class="sub-con">')
            b = html.find("<footer", max(a, 0))
            x = parse(html[a if a > 0 else 0:b if b > 0 else None], no)          # 본문 영역만 읽어 파싱 시간을 줄인다
        except Exception as e:
            errors.append({"no": no, "error": str(e)[:200]})
            continue
        counts = Counter()
        for k in ("title", "proposal", "review_reason", "plan", "close_reason", "refs"):
            x[k] = mask(x.get(k), counts)
        masked_total.update(counts)
        text = (x.get("proposal") or "") + "\n" + (x.get("review_reason") or "")
        result = (x.get("result") or "").strip() or None
        row = {"case_key": f"F{no}", "gubun": "현장건의", "idx": no, "title": (x.get("title") or "").strip(), "list_title": m["title"].strip(),
               "reply_date": x.get("reply_date"), "list_reply_date": m["replyRegDate"], "category": x.get("category"),
               "category_top": (x.get("category") or "").split(">")[0].strip() or None, "dept": x.get("dept"), "visit": x.get("visit"),
               "answer_type": result, "question": x.get("proposal"), "answer": x.get("review_reason"), "plan": x.get("plan"), "close_reason": x.get("close_reason"),
               "closed": x.get("closed"), "refs": x.get("refs"), "url": url_of(no),
               "changed_after": bool(result) and result.startswith(CHANGED),
               "already_in_place": bool(result) and result.startswith("기조치"),
               "pii_masked": {k: v for k, v in counts.items() if v},
               "articles": sorted(set(re.sub(r"\s", "", a) for a in ARTICLE.findall(text))),
               "norm_kinds": sorted(k for k, rx in NORM_KIND.items() if rx.search(text)),
               "pii_left": sorted({tag for rx, tag in PII if any(not (tag == "[계좌번호]" and DATE.match(mm.group(0))) for mm in rx.finditer(text))})}   # 가린 뒤 남은 형식 (0이어야 함)
        row["skip_reason"] = skip_of(x)
        items.append(row)
    items.sort(key=lambda r: (r["reply_date"] or "", r["idx"]))

    # 5 중복 확인
    by_title, by_body = defaultdict(list), defaultdict(list)
    for r in items:
        by_title[norm(r["title"])].append(r["case_key"])
        by_body[norm(r["question"])[:300]].append(r["case_key"])
    for r in items:
        r["dup_title_n"] = len(by_title[norm(r["title"])])
        r["dup_body_n"] = len(by_body[norm(r["question"])[:300]])

    write_jsonl("exmnt_cases.jsonl", items)
    write_csv("exmnt_cases.csv", items, ["case_key", "title", "reply_date", "category", "dept", "answer_type", "changed_after", "already_in_place", "skip_reason", "question", "answer", "plan",
                                         "articles", "norm_kinds", "dup_title_n", "dup_body_n", "url"])

    keep = [r for r in items if not r["skip_reason"]]
    L = lambda rs, k: [len(r[k] or "") for r in rs]
    q = lambda v: {"min": min(v), "p10": quantiles(v, n=10)[0], "median": median(v), "p90": quantiles(v, n=10)[-1], "max": max(v)}
    summary = {
        "n": len(items), "parse_errors": len(errors), "title_mismatch": sum(1 for r in items if r["title"] != r["list_title"]),
        "reply_date_mismatch": sum(1 for r in items if r["reply_date"] != r["list_reply_date"]), "reply_date_missing": sum(1 for r in items if not r["reply_date"]),
        "skip": dict(Counter(r["skip_reason"] for r in items if r["skip_reason"])), "keep": len(keep),
        "year_all": dict(sorted(Counter((r["reply_date"] or "없음")[:4] for r in items).items())),
        "year_keep": dict(sorted(Counter((r["reply_date"] or "없음")[:4] for r in keep).items())),
        "answer_type_all": dict(Counter(r["answer_type"] for r in items).most_common()), "answer_type_keep": dict(Counter(r["answer_type"] for r in keep).most_common()),
        "changed_after_keep": sum(r["changed_after"] for r in keep), "already_in_place_keep": sum(r["already_in_place"] for r in keep),
        "pii_masked": dict(masked_total), "pii_masked_cases": sum(1 for r in items if r["pii_masked"]), "pii_left_cases": sum(1 for r in items if r["pii_left"]),
        "category_top_keep": dict(Counter(r["category_top"] for r in keep).most_common()),
        "category_2nd_keep_top20": dict(Counter(" > ".join(x.strip() for x in (r["category"] or "").split(">")[:2]) for r in keep).most_common(20)),
        "dept_keep": dict(Counter(((r["dept"] or "없음").split(">")[0]).strip() for r in keep).most_common()),
        "len_question_keep": q(L(keep, "question")), "len_answer_keep": q(L(keep, "answer")),
        "plan_nonempty": sum(1 for r in items if (r["plan"] or "").strip()),
        "articles_keep": sum(1 for r in keep if r["articles"]), "norm_kinds_keep": dict(Counter(k for r in keep for k in r["norm_kinds"]).most_common()),
        "dup_title_groups": sum(1 for v in by_title.values() if len(v) > 1), "dup_title_cases": sum(len(v) for v in by_title.values() if len(v) > 1),
        "dup_body_groups": sum(1 for v in by_body.values() if len(v) > 1), "dup_body_cases": sum(len(v) for v in by_body.values() if len(v) > 1),
        "skipped_reason_texts": [(r["case_key"], r["skip_reason"], re.sub(r"\s+", " ", r["answer"] or "").strip()[:60]) for r in items if r["skip_reason"] in ("사유가 무의미한 문구뿐", "다른 건 답변 참조뿐", "사유 없음")],
        "rules": {"MIN_PROPOSAL": MIN_PROPOSAL, "VAGUE": VAGUE.pattern, "CROSS_REF": CROSS_REF.pattern},
    }
    (OUT / "exmnt_parse_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest("x2_exmnt_parse", n=len(items), keep=len(keep), errors=len(errors))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
