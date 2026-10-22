"""데이터 관찰 스크립트 (eda.py, 표준 라이브러리만 사용).

정답 데이터의 분포를 집계하고, 탐지기가 어려워할 사례를 예시로 보여줍니다.
탐지기 개선은 dev로만 하므로, 예시를 보며 규칙을 만들 때는 --split dev 로 보는 것을 권장합니다.

사용법
    python eda.py                    # dev, test 모두 집계 (예시는 split별 5개)
    python eda.py --split dev --n 10

출력
    화면 + results/eda/report.md (같은 내용)
    results/eda/entities.csv          : 정답 개체 전체 (라벨·표기 형태·형식 패턴·뒤에 붙은 말·문맥)
    results/eda/unlabeled_candidates.csv : 정답에 없는 번호열·'성+호칭' 후보 (오탐 미끼 / 팀 기준 논의용)
"""
import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

from run_compare import LABELS, ctx, load_jsonl, overlap

DATA = {"dev": "data/pii_synth_finance_dev.jsonl", "test": "data/pii_synth_finance_test.jsonl"}

# 이름 뒤 호칭·조사 후보, '김 과장'류 탐색용 (규칙 기반 탐지기에서도 재사용 가능)
TITLES = ["고객", "님", "씨", "과장", "대리", "팀장", "부장", "차장", "사원", "주임", "실장",
          "본부장", "이사", "상무", "전무", "대표", "선임", "책임", "수석", "상담원", "매니저", "담당"]
SURNAMES = set("김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진나지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용")
SURNAME_TITLE = re.compile(r"([가-힣])\s?(" + "|".join(sorted(TITLES, key=len, reverse=True)) + r")")
NUMBER_LIKE = re.compile(r"\d[\d\- ]{5,}\d")


# ------------------------------------------------------------------ 보조 함수
def shape(s):
    """형식 패턴: 숫자→9, 한글→가, 영문→a (예: 010-1234-5678 → 999-9999-9999)."""
    out = []
    for ch in s:
        if ch.isdigit():
            out.append("9")
        elif "가" <= ch <= "힣":
            out.append("가")
        elif ch.isascii() and ch.isalpha():
            out.append("a")
        else:
            out.append(ch)
    return "".join(out)


def following(text, end):
    """개체 바로 뒤에 공백 없이 붙은 한글 (조사 부착 확인용, 예: '윤선영은' → '은')."""
    m = re.match(r"[가-힣]+", text[end:])
    return m.group() if m else ""


def next_word(text, end):
    """개체 뒤 공백 하나 다음 단어 (호칭 확인용, 예: '윤 선영 고객' → '고객')."""
    m = re.match(r"\s([가-힣]+)", text[end:])
    return m.group(1) if m else ""


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return lines


# ------------------------------------------------------------------ 집계
def overview(docs_by_split):
    out = ["## 1. 기본 구성", ""]
    rows = []
    for sp, docs in docs_by_split.items():
        n_ent = sum(len(d["entities"]) for d in docs)
        rows.append([sp, len(docs), n_ent, sum(not d["entities"] for d in docs),
                     len({d["doc_type"] for d in docs}), len({d["template"] for d in docs})])
    out += table(["split", "문서", "개체", "개인정보 없는 문서", "문서 유형", "템플릿"], rows) + [""]

    if {"dev", "test"} <= docs_by_split.keys():
        dev_t = {d["template"] for d in docs_by_split["dev"]}
        test_t = {d["template"] for d in docs_by_split["test"]}
        only = sorted(test_t - dev_t)
        out += [f"- test 템플릿 {len(test_t)}종 중 dev에 없는 템플릿: {len(only)}종 {only}",
                "  (dev와 같은 템플릿을 공유하는 만큼 test 점수는 holdout보다 낙관적일 수 있음)", ""]

    out += ["### 문서 유형별 문서 수 / 개체 수", ""]
    types = sorted({d["doc_type"] for docs in docs_by_split.values() for d in docs})
    rows = []
    for t in types:
        row = [t]
        for docs in docs_by_split.values():
            sel = [d for d in docs if d["doc_type"] == t]
            row.append(f"{len(sel)} / {sum(len(d['entities']) for d in sel)}")
        rows.append(row)
    out += table(["문서 유형"] + list(docs_by_split), rows) + [""]
    return out


def label_dist(docs_by_split):
    out = ["## 2. 라벨 분포", ""]
    for sp, docs in docs_by_split.items():
        ents = [e for d in docs for e in d["entities"]]
        rows = []
        for lab in LABELS:
            es = [e for e in ents if e["label"] == lab]
            n_doc = sum(any(e["label"] == lab for e in d["entities"]) for d in docs)
            avg_len = sum(len(e["text"]) for e in es) / len(es) if es else 0
            rows.append([lab, len(es), f"{len(es) / len(ents):.1%}", n_doc, f"{avg_len:.1f}"])
        other = Counter(e["label"] for e in ents if e["label"] not in LABELS)
        out += [f"### {sp} (개체 {len(ents)}개)", ""]
        out += table(["라벨", "개수", "비율", "등장 문서", "평균 길이"], rows) + [""]
        if other:
            out += [f"- 채점 대상 7종 밖 라벨: {dict(other)}", ""]
    return out


def person_dist(docs_by_split):
    out = ["## 3. person 표기 형태", ""]
    for sp, docs in docs_by_split.items():
        ps = [(d, e) for d in docs for e in d["entities"] if e["label"] == "person"]
        form = Counter(e.get("form") for _, e in ps)
        n_len = Counter(len(e["text"].replace(" ", "")) for _, e in ps)
        josa = Counter(following(d["text"], e["end"]) for d, e in ps)
        nxt = Counter(next_word(d["text"], e["end"]) for d, e in ps)
        by_pid = defaultdict(set)
        for _, e in ps:
            by_pid[e.get("person_id")].add(e["text"])
        multi = sum(len(v) > 1 for v in by_pid.values())
        out += [f"### {sp} (person {len(ps)}개, 인물 {len(by_pid)}명)", "",
                f"- 표기 형태(form): {dict(form.most_common())}",
                f"- 이름 글자 수(공백 제외): {dict(sorted(n_len.items()))}",
                f"- 바로 뒤에 붙은 말(조사 부착 후보, 상위 15): {dict((k or '(없음)', v) for k, v in josa.most_common(15))}",
                f"- 한 칸 뒤 단어(호칭 후보, 상위 15): {dict((k or '(없음)', v) for k, v in nxt.most_common(15))}",
                f"- 표기가 2가지 이상인 인물: {multi}명 / {len(by_pid)}명 (토큰 일관성에 영향)", ""]
    return out


def format_dist(docs_by_split):
    out = ["## 4. 번호·이메일 형식 패턴 (정규식 작성용)", ""]
    for lab in ["phone_number", "resident_registration_number", "account_number", "card_number", "email"]:
        rows = []
        for sp, docs in docs_by_split.items():
            c = Counter(shape(e["text"]) if lab != "email" else re.sub(r"[^@.]", "", e["text"])
                        for d in docs for e in d["entities"] if e["label"] == lab)
            rows += [[sp, f"`{k}`", v] for k, v in c.most_common(8)]
        out += [f"### {lab}" + (" (기호만 표시)" if lab == "email" else ""), ""]
        out += table(["split", "패턴", "개수"], rows) + [""]
    return out


def hard_tags(docs_by_split):
    out = ["## 5. 어려운 사례 태그 (태그가 붙은 문서 수)", ""]
    tags = sorted({t for docs in docs_by_split.values() for d in docs for t in d.get("hard_cases", [])})
    rows = [[t] + [sum(t in d.get("hard_cases", []) for d in docs) for docs in docs_by_split.values()]
            for t in tags]
    return out + table(["태그"] + list(docs_by_split), rows) + [""]


# ------------------------------------------------------------------ 예시
def unlabeled_candidates(d):
    """정답에 없는 번호열과 '성+호칭' 표현. 오탐 미끼이거나 팀 기준이 필요한 표현."""
    gold = d["entities"]
    found = []
    for m in NUMBER_LIKE.finditer(d["text"]):
        span = {"start": m.start(), "end": m.end()}
        if not any(overlap(span, g) for g in gold):
            found.append(("번호열", m.group(), m.start(), m.end()))
    for m in SURNAME_TITLE.finditer(d["text"]):
        span = {"start": m.start(1), "end": m.end(1)}
        before = d["text"][m.start() - 1] if m.start() else " "
        if m.group(1) in SURNAMES and not ("가" <= before <= "힣") and not any(overlap(span, g) for g in gold):
            found.append(("성+호칭", m.group(), m.start(), m.end()))
    return found


def examples(docs_by_split, n):
    out = ["## 6. 어려운 사례 예시", ""]
    for sp, docs in docs_by_split.items():
        ps = [(d, e) for d in docs for e in d["entities"] if e["label"] == "person"]
        cases = {
            "띄어 쓴 이름 (form=spaced)": [(d, e) for d, e in ps if e.get("form") == "spaced"],
            "이름만 (form=given)": [(d, e) for d, e in ps if e.get("form") == "given"],
            "조사 부착 (정답 범위 바로 뒤에 한글)": [(d, e) for d, e in ps if following(d["text"], e["end"])],
            "전화번호 형식 변형 (999-9999-9999 아님)": [
                (d, e) for d in docs for e in d["entities"]
                if e["label"] == "phone_number" and shape(e["text"]) != "999-9999-9999"],
            "주소": [(d, e) for d in docs for e in d["entities"] if e["label"] == "address"],
        }
        out += [f"### {sp}", ""]
        for title, items in cases.items():
            out.append(f"**{title}** — {len(items)}건")
            out += [f"- {d['id']} `{e['text']}` … {ctx(d['text'], e['start'], e['end'])}" for d, e in items[:n]]
            out.append("")

        # 동명이인: 같은 표기, 다른 인물 ID
        by_text = defaultdict(lambda: defaultdict(set))
        for d, e in ps:
            by_text[e["text"].replace(" ", "")][e.get("person_id")].add(d["id"])
        homo = {t: v for t, v in by_text.items() if len(v) > 1}
        out.append(f"**동명이인 (같은 이름, 다른 person_id)** — {len(homo)}건")
        for t, v in list(homo.items())[:n]:
            out.append(f"- `{t}`: " + ", ".join(f"{pid}({len(ids)}문서)" for pid, ids in v.items()))
        out.append("")

        # 정답에 없는 후보 (미끼·팀 기준 논의용)
        cands = [(d, c) for d in docs for c in unlabeled_candidates(d)]
        for kind in ("번호열", "성+호칭"):
            sel = [(d, c) for d, c in cands if c[0] == kind]
            out.append(f"**정답에 없는 {kind}** — {len(sel)}건 (오탐 미끼 / '김 과장' 기준 논의용)")
            out += [f"- {d['id']} `{c[1]}` … {ctx(d['text'], c[2], c[3])}" for d, c in sel[:n]]
            out.append("")

        # decoy_words는 미끼 위치가 정답에 없으므로 문서 앞부분을 직접 확인
        sel = [d for d in docs if "decoy_words" in d.get("hard_cases", [])]
        out.append(f"**decoy_words 문서 (미끼 단어 위치 정보 없음, 원문 확인)** — {len(sel)}건")
        out += [f"- {d['id']} ({d['doc_type']}): {d['text'][:120].replace(chr(10), ' ')}" for d in sel[:n]]
        out.append("")
    return out


# ------------------------------------------------------------------ 저장
def write_csv(path, header, data):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(data)


def save_csvs(docs_by_split, out_dir):
    rows = []
    for sp, docs in docs_by_split.items():
        for d in docs:
            for e in d["entities"]:
                rows.append([sp, d["id"], d["doc_type"], d["template"], e["label"], e["text"],
                             e.get("form", ""), e.get("person_id", ""), shape(e["text"]),
                             following(d["text"], e["end"]), next_word(d["text"], e["end"]),
                             ctx(d["text"], e["start"], e["end"])])
    write_csv(os.path.join(out_dir, "entities.csv"),
              ["split", "doc_id", "doc_type", "template", "label", "text", "form", "person_id",
               "shape", "attached_after", "next_word", "context"], rows)
    rows = [[sp, d["id"], d["doc_type"], kind, s, ctx(d["text"], a, b)]
            for sp, docs in docs_by_split.items() for d in docs
            for kind, s, a, b in unlabeled_candidates(d)]
    write_csv(os.path.join(out_dir, "unlabeled_candidates.csv"),
              ["split", "doc_id", "doc_type", "kind", "text", "context"], rows)


# ------------------------------------------------------------------ 실행
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--n", type=int, default=5, help="사례별 예시 개수")
    ap.add_argument("--out", default="results/eda")
    a = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    splits = ["dev", "test"] if a.split == "all" else [a.split]
    docs_by_split = {sp: load_jsonl(DATA[sp]) for sp in splits}
    os.makedirs(a.out, exist_ok=True)

    lines = ["# 데이터 관찰 리포트", ""]
    for section in (overview, label_dist, person_dist, format_dist, hard_tags):
        lines += section(docs_by_split)
    lines += examples(docs_by_split, a.n)

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    save_csvs(docs_by_split, a.out)
    print(f"\n결과 저장: {a.out}/ (report.md, entities.csv, unlabeled_candidates.csv)")


if __name__ == "__main__":
    main()
