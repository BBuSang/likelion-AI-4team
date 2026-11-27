"""C1: 답변 루브릭 ① 코드 검사 ([답변_평가_루브릭.md](../답변_평가_루브릭.md) 3장, worklog 133번, 사용자 2026-11-24 "코드 검사 만들기").
LLM을 부르지 않는다. 저장된 답변 캐시만 읽는다(새 답변 생성 없음). 잠긴 시험 세트(test*, final_test)는 읽지 않고 건너뛴다.

답변마다 판정하는 항목 (pass/fail)
- cite_valid  인용 번호 유효성: 답변의 [번호]가 모두 이번 문맥(`order`)에 있던 사례이고, 대괄호 하나에 번호 하나 (규칙 2)
              직접성 판정을 쓴 답변은 '간접'으로 판정된 사례를 인용하지 않았는지도 본다
- format_ok   형식 준수 (형식 규칙 V3): 거절이면 제목 없이 그 문장만. 답했으면 제목이 "### 핵심 답변" → "### 근거" (→ "### 유의사항") 순서와 이름,
              근거 목록 2~4개, 표·번호 매기기 없음, 회신 날짜 표기 없음
              (참고로만 세는 것: 근거 항목마다 굵게 한 곳, 사례번호를 굵게 하지 않음, 효력 표시가 없는데 유의사항을 씀)
- assertive_ok 단정 표현 0 (규칙 3, 표현 목록은 `rag.check_output`과 같다)
- effect_ok   효력 표시 전달 (규칙 4): 인용한 사례에 '현재 효력 없음'·'기한 연장 건' 표시가 있으면 유의사항에서 알렸는지. 해당 사례를 인용하지 않았으면 해당없음(None)
- code_pass   위 네 항목이 모두 통과(해당없음은 통과로 본다)

세트별로 내는 것
- 위 항목별 통과율과 실패 세부 사유 수
- 거절 정확도: 일반 세트 = 과잉 거절률, 정답 제외(`*_holdout`) = 거절률, 답변불가 = 거절률. 정답 제외에서 다른 결론으로 답했는지는 LLM 심사가 필요해 여기서 보지 않는다
- 지연·비용: 시간 중앙값·90%, 답변·직접성 판정 입력/출력 토큰 평균
- 참고(합격 판정에 넣지 않음): 번호 없는 문장이 있는 답변 수(인용 정확도의 '번호 없음'은 LLM 심사 4-3에서 본다)

사용: python pipeline/c1_code_checks.py [--answers _f20_answers.jsonl] [--arms gate,gate_holdout]
출력: data/eval/c1_code_checks.json (요약), data/eval/c1_code_checks_by_answer.csv (답변별, 실패 사유 포함)
"""
import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "app"))
import rag  # noqa: E402  (NOT_FOUND만 쓴다)
from prompts import REPLY_DATE  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
EVAL = ROOT / "data" / "eval"
LOCKED = re.compile(r"^(test|final_test)")

CITE = re.compile(r"\[([LOF]\d+)\]")                                            # app/service.py와 같다 (F = 현장건의)
MULTI = re.compile(r"\[\s*[LOF]\d+(?:\s*[,·/]\s*[LOF]?\d+)+\s*\]")              # [L1, O2] 같은 묶음 표기
ASSERT = re.compile(r"(?:가능합니다|불가합니다|불가능합니다|허용됩니다|금지됩니다)(?!고)")   # rag.check_output과 같다
HEAD = re.compile(r"^\s*#{1,6}\s*(.*?)\s*$", re.M)
HEADS_OK = (["핵심 답변", "근거"], ["핵심 답변", "근거", "유의사항"])
CTX_HEAD = re.compile(r"^\[사례 ([LOF]\d+)\]([^\n]*)", re.M)


def sections(text):
    """'### 제목' 기준으로 나눈 {제목: 본문}. 같은 제목이 두 번이면 뒤의 것을 덧붙인다."""
    out, cur = defaultdict(str), None
    for line in text.splitlines():
        m = re.match(r"^\s*#{1,6}\s*(.*?)\s*$", line)
        if m:
            cur = m.group(1)
            continue
        if cur is not None:
            out[cur] += line + "\n"
    return out


def effect_flags(ctx):
    """문맥 머리줄에서 사례별 효력 표시: {'L123': {'void', 'ext'}}"""
    flags = {}
    for k, head in CTX_HEAD.findall(ctx or ""):
        f = set()
        if "현재 효력 없음" in head:
            f.add("void")
        if "기한 연장 건" in head:
            f.add("ext")
        flags[k] = f
    return flags


def check(r):
    text = (r.get("shown") or "").strip()
    refused = bool(r.get("not_found")) or text.rstrip(".") == rag.NOT_FOUND
    fails = []
    cited = list(dict.fromkeys(CITE.findall(text)))

    # 인용 번호 유효성
    given = set(r.get("order") or [])
    bad = [c for c in cited if c not in given]
    if bad:
        fails.append("문맥에 없는 번호 " + ",".join(bad))
    if MULTI.search(text):
        fails.append("대괄호 하나에 번호 여러 개")
    verdicts = ((r.get("gate") or {}).get("verdicts")) or {}
    indirect = [c for c in cited if verdicts.get(c) == "간접"]
    if indirect:
        fails.append("간접 판정 사례 인용 " + ",".join(indirect))
    cite_valid = not any(f.startswith(("문맥에 없는", "대괄호", "간접")) for f in fails)

    # 효력 표시 (인용한 사례 기준)
    flags = effect_flags(r.get("cited_context"))
    need = {c: flags.get(c, set()) for c in cited if flags.get(c)}
    sec = sections(text)
    note = sec.get("유의사항", "")
    effect_ok, effect_fail = None, []
    if need and not refused:
        for c, f in need.items():
            if "void" in f and not re.search(r"효력[이가은는]?\s*(?:이\s*)?없", note):
                effect_fail.append(f"{c} 효력 없음 미표시")
            if "ext" in f and not re.search(r"기간|연장", note):
                effect_fail.append(f"{c} 기한 연장 미안내")
        effect_ok = not effect_fail
        fails += effect_fail

    # 형식 (V3)
    fmt, style = [], []
    heads = HEAD.findall(text)
    if refused:
        if heads or text.rstrip(".") != rag.NOT_FOUND:
            fmt.append("거절인데 다른 글이 있음")
    else:
        if heads not in HEADS_OK:
            fmt.append("제목 순서·이름 " + "/".join(heads))
        items = [ln for ln in sec.get("근거", "").splitlines() if ln.lstrip().startswith("- ")]
        if not 2 <= len(items) <= 4:
            fmt.append(f"근거 목록 {len(items)}개")
        # 참고(합격 판정에 넣지 않음): 루브릭 3장 형식 항목에 없는 세부 형식 규칙
        if any("**" not in ln for ln in items):
            style.append("근거 항목에 굵게 없음")
        if re.search(r"\*\*[^*\n]*\[[LOF]\d+\][^*\n]*\*\*", text):
            style.append("사례번호를 굵게 함")
        if "유의사항" in heads and not need:
            style.append("효력 표시 없는데 유의사항 씀")   # 제목에 '기한연장'이 있는데 doc_class가 extension이 아닌 사례에서도 나온다
    if re.search(r"^\s*\|.*\|\s*$", text, re.M):
        fmt.append("표")
    if re.search(r"^\s*\d+[.)]\s", text, re.M):
        fmt.append("번호 매기기")
    if any(p.search(text) for p in REPLY_DATE):
        fmt.append("회신 날짜 표기")
    fails += ["형식: " + f for f in fmt]

    n_assert = len(ASSERT.findall(text))
    if n_assert:
        fails.append(f"단정 표현 {n_assert}")

    # 참고: 번호 없는 문장 (제목·유의사항 제외)
    body = "\n".join(v for k, v in sec.items() if k != "유의사항") if sec else text
    joined = re.sub(r"\s+(?=\[[LOF]\d+\])", "", body)
    # rag.check_output은 '다'·'요' 뒤 공백에서도 끊어 "이벤트마다 " 같은 곳을 문장 끝으로 잘못 본다. 여기서는 문장부호·닫는 대괄호 뒤에서만 끊는다
    sents = [s for s in re.split(r"(?<=[.?!\]])\s+|\n+", joined) if len(re.sub(r"\W", "", s)) > 10]
    no_cite = 0 if refused else sum(1 for s in sents if not CITE.search(s))

    ok = {"cite_valid": cite_valid, "format_ok": not fmt, "assertive_ok": n_assert == 0, "effect_ok": effect_ok}
    ok["code_pass"] = all(v is not False for v in ok.values())
    gu = (r.get("gate") or {}).get("usage") or {}
    return {"set": r["set"], "arm": r.get("arm") or r.get("mode") or "", "qid": r["qid"], "corpus": r.get("corpus", ""), "refused": refused,
            **ok, "no_cite_sentences": no_cite, "seconds": r.get("seconds"),
            "in_tok": (r.get("usage") or {}).get("input", 0) + gu.get("input", 0), "out_tok": (r.get("usage") or {}).get("output", 0) + gu.get("output", 0),
            "gate_used": bool(verdicts), "fails": " ; ".join(fails), "style_notes": " ; ".join(style)}


def kind(row):
    if row["set"].startswith("unanswerable"):
        return "답변불가"
    return "정답 제외" if row["arm"].endswith("holdout") else "일반"


def summarize(rows):
    n = len(rows)
    rate = lambda key: round(sum(1 for r in rows if r[key]) / n, 4) if n else None
    eff = [r for r in rows if r["effect_ok"] is not None]
    secs = sorted(r["seconds"] for r in rows if r["seconds"] is not None)
    reasons = Counter()
    for r in rows:
        for f in filter(None, r["fails"].split(" ; ")):
            reasons[re.sub(r"\s[LOF]\d+.*$|\s\d+$|\s[LOF]\d+(,[LOF]\d+)*$", "", re.sub(r"^[LOF]\d+\s", "", f))] += 1
    answered = [r for r in rows if not r["refused"]]
    k = kind(rows[0]) if rows else ""
    return {"n": n, "kind": k,
            "refusal": {"refused": sum(r["refused"] for r in rows), "rate": rate("refused"),
                        "meaning": {"일반": "과잉 거절률(낮을수록 좋음)", "정답 제외": "거절률(높을수록 좋음, 다른 결론 여부는 LLM 심사)", "답변불가": "거절률(1.0이 정답)"}[k] if k else ""},
            "code_pass": rate("code_pass"), "cite_valid": rate("cite_valid"), "format_ok": rate("format_ok"), "assertive_ok": rate("assertive_ok"),
            "effect": {"applicable": len(eff), "ok": sum(1 for r in eff if r["effect_ok"])},
            "answered_with_no_cite_sentence": sum(1 for r in answered if r["no_cite_sentences"]),
            "seconds": {"median": statistics.median(secs) if secs else None, "p90": secs[int(0.9 * (len(secs) - 1))] if secs else None},
            "tokens_mean": {"input": round(statistics.mean(r["in_tok"] for r in rows)) if n else None, "output": round(statistics.mean(r["out_tok"] for r in rows)) if n else None},
            "fail_reasons": dict(reasons.most_common()),
            "style_notes": dict(Counter(s for r in rows for s in filter(None, r["style_notes"].split(" ; "))).most_common())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="_f20_answers.jsonl", help="data/eval 안의 답변 캐시 파일 이름")
    ap.add_argument("--arms", default="", help="쉼표로 구분한 arm(또는 mode) 이름만 본다. 비우면 전부")
    a = ap.parse_args()
    arms = set(filter(None, a.arms.split(",")))
    rows, skipped = [], Counter()
    for line in open(EVAL / a.answers, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if LOCKED.match(r["set"]):
            skipped[r["set"]] += 1
            continue
        if arms and (r.get("arm") or r.get("mode")) not in arms:
            continue
        rows.append(check(r))
    if not rows:
        sys.exit("검사할 답변이 없습니다")

    groups = defaultdict(list)
    for r in rows:
        groups[(kind(r), r["set"], r["corpus"])].append(r)
        groups[(kind(r), "합계", "")].append(r)
    out = {"source": a.answers, "arms": sorted(arms) or "all", "skipped_locked": dict(skipped),
           "groups": {" / ".join(x for x in g if x): summarize(v) for g, v in sorted(groups.items())}}
    (EVAL / "c1_code_checks.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(EVAL / "c1_code_checks_by_answer.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{a.answers}: 답변 {len(rows)}개 검사, 잠긴 세트 건너뜀 {dict(skipped) or 0}")
    print(f"{'구분':<28}{'n':>5}{'거절률':>8}{'합격':>8}{'인용':>8}{'형식':>8}{'단정':>8}{'효력(해당/통과)':>16}{'시간 중앙':>10}")
    for g, s in out["groups"].items():
        print(f"{g:<28}{s['n']:>5}{s['refusal']['rate']:>8.3f}{s['code_pass']:>8.3f}{s['cite_valid']:>8.3f}{s['format_ok']:>8.3f}{s['assertive_ok']:>8.3f}"
              f"{str(s['effect']['applicable']) + '/' + str(s['effect']['ok']):>16}{s['seconds']['median'] or 0:>10.1f}")
    for k in ("일반 / 합계", "정답 제외 / 합계", "답변불가 / 합계"):
        if k in out["groups"]:
            print(k, "실패 사유:", out["groups"][k]["fail_reasons"], "| 참고:", out["groups"][k]["style_notes"],
                  "| 번호 없는 문장이 있는 답변:", out["groups"][k]["answered_with_no_cite_sentence"])


if __name__ == "__main__":
    main()
