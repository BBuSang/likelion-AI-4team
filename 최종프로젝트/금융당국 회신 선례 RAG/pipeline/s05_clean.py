"""05 정제: 본문 정제, 정보 없는 줄 제거, 제목·회신일 보정, 검색 제외 판정, 사례 유형(doc_class).
입력: processed/cases_full.jsonl   출력: processed/cases_clean.jsonl, cases_clean.csv, clean_summary.json
"""
import json
import re
from collections import Counter

from common import OUT, SECS, manifest, read_jsonl, write_csv, write_jsonl
from noise import strip_noise
from text_rules import REDIRECT, WITHDRAW, clean, normalize

# 2016 규제 정비: 행정지도·모범규준·협회규정의 효력을 일괄 회신한 묶음. 회신일이 두 날짜에 몰려 있다
RULE_REVIEW_DATES = ("2016-02-01", "2016-07-08")
# 비조치의견서 효력·기한 연장 건. 제목의 '연장'이 대출 만기·청약기간을 뜻하면 해당 없음
EXTENSION_EXCLUDE = re.compile(r"만기|청약기간")
COLS = ["case_key", "gubun", "idx", "serial_no", "title", "title_source", "reply_title", "doc_class", "category",
        "reg_date", "reply_date", "reply_date_source", "status", "registrant", "answer_type", "answer_type_raw", "depts",
        "question", "answer", "reason", "attachments", "body_source", "body_chars", "indexable", "skip_reason", "notes", "url"]


def doc_class(rec):
    if rec["gubun"] == "비조치의견서":
        if rec["reply_date"] in RULE_REVIEW_DATES:
            return "rule_review_2016"
        if "연장" in rec["title"] and not EXTENSION_EXCLUDE.search(rec["title"]):
            return "extension"
    return "general"


def skip_reason(rec, body):
    n = len(body)
    if rec["status"] == "반려":
        return "반려"
    if WITHDRAW.search(body) and ("반려" in body or "운영규칙" in body or "운영 규칙" in body) and n < 400:
        return "철회"
    if REDIRECT.search(body) and n < 100:
        return "다른 회신으로 처리"
    if n < 30:
        return "본문 없음"
    return None


def main():
    rows, noise_stats = read_jsonl("cases_full.jsonl"), {}
    for rec in rows:
        for s in SECS:
            rec[s] = normalize(strip_noise(clean(rec[s]), noise_stats))
        body = "\n".join(rec[s] for s in SECS)

        # 포털에서부터 제목이 빈 건은 질의요지 첫 줄(규제 이름)로 채운다
        rec["title_source"] = "list"
        if not rec["title"]:
            first = next((ln.strip() for ln in rec["question"].split("\n") if ln.strip()), "")
            if first:
                rec["title"], rec["title_source"] = first[:200], "question"
                rec["notes"].append("제목 없음 → 질의요지 첫 줄로 채움")
            else:
                rec["title_source"] = None
                rec["notes"].append("제목 없음")

        # 회신일: 상세 → 통합조회 → 없으면 비워 두고 표시 (추정하지 않음)
        tk = rec.pop("reply_date_list")
        if not rec["reply_date"] and tk:
            rec["reply_date"], rec["reply_date_source"] = tk, "list_total"
        if not rec["reply_date"]:
            rec["notes"].append("회신일 없음")

        rec["skip_reason"] = skip_reason(rec, body)
        rec["indexable"] = rec["skip_reason"] is None
        rec["doc_class"] = doc_class(rec)
        rec["body_chars"] = len(body)
        rec.pop("is_public", None)      # 전부 Y
        rec.pop("detail_title", None)   # 목록 제목과 같음
    write_jsonl("cases_clean.jsonl", rows)
    write_csv("cases_clean.csv", rows, COLS)
    idx = [r for r in rows if r["indexable"]]
    summary = {"전체": len(rows), "검색대상": len(idx),
               "제외사유": dict(Counter(r["skip_reason"] for r in rows if not r["indexable"])),
               "사례유형(전체)": dict(Counter(r["doc_class"] for r in rows)),
               "사례유형(검색대상)": dict(Counter(r["doc_class"] for r in idx)),
               "본문출처": dict(Counter(r["body_source"] for r in rows)),
               "답변유형": dict(Counter(str(r["answer_type"]) for r in rows)),
               "걸러낸줄": noise_stats}
    (OUT / "clean_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest("05_clean", cases=len(rows), indexable=len(idx), doc_class=summary["사례유형(전체)"])


if __name__ == "__main__":
    main()
