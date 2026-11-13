"""X3 현장건의 청크 (사용자 결정 2026-11-11). 기존 D3 검색 단위(질의 카드 + 판단 본문)와 같은 2조각 구조. 임베딩·DB 적재는 하지 않는다.
- 건의 카드(question_card): 머리말 + 건의내용
- 검토 카드(judgment): 머리말 + "검토결과: ○○" + 사유. 향후계획은 넣지 않는다(회신 당시 계획일 뿐 실제 변경 여부를 알 수 없고 "해당 없음"류가 많음, 원본에만 보관)
- 머리말: 기존 header()와 같은 형식 "[현장건의 | 과제분류 | 회신일] 제목" (분야 칸 대신 과제분류)
- 자르기: 기존 s12.chunk_text() (900자 초과 시 항목 경계에서 600자 근처까지 묶기, 짧은 끝 조각은 앞에 붙임)
- 묶음: 정규화한 제목과 사유 앞 100자가 모두 같으면 한 묶음(같은 건의에 같은 회신을 반복한 건). 대표는 가장 최근 회신, 조각은 대표만 만든다.
  group_id는 기존 묶음 번호와 겹치지 않게 EXMNT_GROUP_BASE부터 매긴다
- 검색 제외 건(skip_reason)은 만들지 않는다. 저장 직전 개인정보 형식 검사(s12.PII)에서 발견되면 멈춘다
입력: processed/exmnt_cases.jsonl, case_groups.jsonl   출력: processed/exmnt_chunks.jsonl (+csv), exmnt_groups.jsonl
"""
import re
from collections import Counter, defaultdict

from kiwipiepy import Kiwi

from common import manifest, read_jsonl, write_csv, write_jsonl
from s12_chunks import KEEP, MAX, PII, chunk_text, no_info

EXMNT_GROUP_BASE = 1_000_000


def norm(s):
    return re.sub(r"[\s\W_]+", "", s or "")


def header(r):
    return f"[현장건의 | {r['category'] or '미분류'} | {r['reply_date'] or '회신일 미상'}] {r['title'] or ''}".strip()


def main():
    rows = [r for r in read_jsonl("exmnt_cases.jsonl") if not r["skip_reason"]]
    top_existing = max(g["group_id"] for g in read_jsonl("case_groups.jsonl"))
    if top_existing >= EXMNT_GROUP_BASE:
        raise SystemExit(f"기존 묶음 번호({top_existing})가 현장건의 시작 번호와 겹칩니다.")
    by = defaultdict(list)
    for r in rows:
        by[(norm(r["title"]), norm(r["answer"])[:100])].append(r)
    groups, rep_of = [], {}
    for gid, (_, members) in enumerate(sorted(by.items(), key=lambda kv: min(m["case_key"] for m in kv[1])), EXMNT_GROUP_BASE):
        members.sort(key=lambda m: (m["reply_date"] or "", m["idx"]), reverse=True)
        groups.append({"group_id": gid, "group_type": "same_title_answer" if len(members) > 1 else "single", "representative_case_key": members[0]["case_key"],
                       "n_cases": len(members), "members": [m["case_key"] for m in members]})
        rep_of[members[0]["case_key"]] = gid

    chunks, dropped = [], Counter()
    for r in rows:
        gid = rep_of.get(r["case_key"])
        if gid is None:
            continue

        def add(kind, no, text):
            chunks.append({"case_key": r["case_key"], "group_id": gid, "chunk_type": kind, "chunk_no": no, "text": text, "embed_text": header(r) + "\n" + text,
                           "doc_class": "field_proposal", "gubun": "현장건의", "category_final": r["category"], "reply_date": r["reply_date"],
                           "answer_type": r["answer_type"], "changed_after": r["changed_after"], "already_in_place": r["already_in_place"]})

        q = (r["question"] or "").strip()
        if q and not no_info(q):
            for no, t in enumerate(chunk_text(q), 1):
                add("question_card", no, t)
        else:
            dropped["question"] += 1
        a = (r["answer"] or "").strip()
        if a and not no_info(a):
            for no, t in enumerate(chunk_text(f"검토결과: {r['answer_type']}\n{a}"), 1):
                add("judgment", no, t)
        else:
            dropped["answer"] += 1

    found = Counter(n for c in chunks for n, rx in PII.items() if rx.search(c["text"]))
    if found:
        raise SystemExit(f"개인정보 형식 발견, 저장 중단: {dict(found)}")
    titles = {r["case_key"]: r["title"] or "" for r in rows}
    kiwi = Kiwi()
    for c, toks in zip(chunks, kiwi.tokenize(titles[c["case_key"]] + "\n" + c["text"] for c in chunks)):
        c["bm25_tokens"] = " ".join(t.form.lower() for t in toks if t.tag.split("-")[0] in KEEP)

    write_jsonl("exmnt_chunks.jsonl", chunks)
    write_jsonl("exmnt_groups.jsonl", groups)
    write_csv("exmnt_chunks.csv", chunks, ["case_key", "group_id", "chunk_type", "chunk_no", "text", "embed_text", "gubun", "category_final", "reply_date", "answer_type"])
    lens = {t: sorted(len(c["text"]) for c in chunks if c["chunk_type"] == t) for t in ("question_card", "judgment")}
    per_case = Counter((c["case_key"], c["chunk_type"]) for c in chunks)
    manifest("x3_exmnt_chunks", cases=len(rows), groups=len(groups), grouped_cases=sum(g["n_cases"] for g in groups if g["n_cases"] > 1),
             multi_groups=sum(1 for g in groups if g["n_cases"] > 1), chunks=len(chunks), types=dict(Counter(c["chunk_type"] for c in chunks)), dropped_no_info=dict(dropped),
             length={t: {"median": v[len(v) // 2], "p90": v[int(len(v) * .9)], "max": v[-1]} for t, v in lens.items()},
             over_max=sum(len(c["text"]) > MAX for c in chunks), split_cases={t: sum(1 for (k, tt), n in per_case.items() if tt == t and n > 1) for t in lens})


if __name__ == "__main__":
    main()
