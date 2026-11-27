"""C2: 답변 루브릭 ② LLM 심사 + ③ 사람 판정 합의도 ([답변_평가_루브릭.md](../답변_평가_루브릭.md) 4~6장, worklog 134번, 사용자 2026-11-24 "LLM 심사 루브릭 구현").
저장된 답변 캐시를 심사한다(새 답변 생성 없음). 잠긴 시험 세트(test*, final_test)는 읽지 않는다. 기존 심사(`f4.JUDGE_SYSTEM`)와 결과 파일은 건드리지 않는다.

심사 항목 (답변 하나에 심사 호출 1회, JSON)
- correctness   정확성(4-1): 정답 묶음(같은 묶음의 사례 최대 3건)의 결론과 같은 방향인가. 조건 누락은 여기서 보지 않는다   일치/부분일치/불일치/판단불가
- grounded      근거 충실도(4-2): 주장이 답변이 인용한 사례 안에 있는가                                               충실/일부추정/근거없음/판단불가
- units         인용 정확도(4-3): 코드가 답변을 단위(핵심 답변 줄, 근거 항목, 유의사항 항목)로 나눠 번호를 매기고, 단위마다 판정   맞음/틀림/번호 없음/해당없음
- completeness  조건 완전성(4-4): 결론을 가르는 조건·단서·예외·대상을 옮겼는가                                          완전/일부 누락/핵심 누락/판단불가
- conflict      상충 선례(4-5): '답변에 주어진 사례' 안에 같은 쟁점에서 결론이 다른 사례가 있으면 나눠 썼는가                처리함/누락/해당없음
실무 유용성(4-6)은 LLM으로 심사하지 않는다(사람 판정 표본에서만 본다).
거절 답변은 심사하지 않는다(모든 항목 해당없음). 답변불가 세트는 심사하지 않는다.
'답변에 주어진 사례' = 직접성 판정이 '직접'으로 본 사례(판정이 없으면 최종 문맥 전체). DB에서 다시 읽어 답변 때와 같은 모양(`rag.case_block`, `exmnt_search.block`)으로 넣는다.
정답 제외(`*_holdout`) 답변도 같은 기준으로 심사한다. 정확성의 '정답 묶음'은 검색에서 뺀 그 묶음이다(다른 사례로 같은 결론이면 일치).

합격(5장) = ① 코드 합격(`c1_code_checks.check`) + 일치 + 충실 + 단위에 틀림·번호 없음 0 + 조건 완전 또는 일부 누락 + 상충 처리함 또는 해당없음.
거절은 세트 종류로 정한다: 일반 = 불합격(과잉 거절), 정답 제외·답변불가 = 합격.

사용
  python pipeline/c2_llm_judge.py run [--answers _f20_answers.jsonl] [--arms gate,gate_holdout] [--limit N] [--model 이름] [--tag 이름]
      → data/eval/_c2_judge_<tag>.jsonl (캐시, 이어서 돈다), c2_llm_judge_<tag>.json (요약), c2_llm_judge_<tag>_by_answer.csv
  python pipeline/c2_llm_judge.py sheet [--tag 이름]     → data/eval/human/c2_judge_sheet.csv (사람이 채운다, LLM 판정 없음), _c2_judge_key.json (열어 보지 않는다)
      · 뽑는 방법(미리 고정, seed 20261124): 거절이 아닌 답변 중 LLM 합격 25 + 불합격 25 (모자라면 다른 쪽에서 채움), 일반·정답 제외 섞음
  python pipeline/c2_llm_judge.py compare                → 항목별 일치율·Cohen's Kappa, 합격 판정의 precision/recall/Kappa와 bootstrap 95% 구간
--model로 다른 모델을 주면 교차 심사(6-2)가 된다(캐시 태그가 달라 섞이지 않는다).
한계: 기본 심사 모델이 답변 모델과 같다. 사람 판정으로 합의도를 확인하기 전에는 이 점수를 품질 지표로 보고하지 않는다(1장 원칙 3).
"""
import argparse
import csv
import hashlib
import json
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "app"))
import c1_code_checks as c1  # noqa: E402
import exmnt_search as es  # noqa: E402
import rag  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
EVAL = ROOT / "data" / "eval"
HUMAN = EVAL / "human"
SHEET, KEYF = HUMAN / "c2_judge_sheet.csv", HUMAN / "_c2_judge_key.json"
RUBRIC = "R1"
WORKERS = 4
GOLD_MAX = 3

CORRECT = ("일치", "부분일치", "불일치", "판단불가")
GROUND = ("충실", "일부추정", "근거없음", "판단불가")
UNIT = ("맞음", "틀림", "번호 없음", "해당없음")
COMPLETE = ("완전", "일부 누락", "핵심 누락", "판단불가")
CONFLICT = ("처리함", "누락", "해당없음")

JUDGE = """당신은 금융당국 회신 선례 검색 시스템의 답변을 채점하는 심사자입니다. 사용자는 은행 직원입니다. 아래 다섯 항목을 서로 따로 판정하고 JSON으로만 답하세요.
판단할 근거가 부족하면 추정하지 말고 '판단불가'를 고르세요(정확성·충실도·조건 완전성만 해당).
시스템이 붙이는 고정 안내 문구와, 사례 머리줄의 '현재 효력 없음'·'기한 연장 건' 표시에 따른 효력 안내는 사례에 없는 주장으로 보지 않습니다.

1) correctness: 답변의 핵심 결론이 [정답 사례](같은 묶음의 회신, 하나 이상)의 결론과 같은 방향인가. 조건·단서 누락은 3)에서 보므로 여기서는 결론 방향만 봅니다.
   - "일치": 질문이 묻는 쟁점에 대해 정답 사례와 같은 결론이다. 표현이 달라도, 다른 사례를 근거로 썼어도 결론이 같으면 일치다.
     단, 답변이 제시한 기한·수치·적용 대상이 정답 사례와 달라 질문의 상황에서 결론이 달라지면 일치가 아니다.
   - "부분일치": 쟁점이 여럿인데 일부에만 같은 결론을 냈다.
   - "불일치": 정답 사례와 반대이거나 다른 쟁점에 답했다. 예: 당국이 허용하지 않은 것을 허용한 것처럼 씀, 비슷한 다른 상품·행위의 회신 결론을 이 질문의 답처럼 씀.
2) grounded: 답변의 주장이 [답변이 인용한 사례] 안에 있는가. 정답 사례와 무관하게 인용 사례만 기준으로 봅니다.
   - "충실": 모든 주장이 인용한 사례에 있다.
   - "일부추정": 일부 문장이 사례에 없거나 사례 범위를 넘는다. 예: 사례의 적용 대상을 더 넓은 대상으로 일반화, 사례에 없는 권고·주의를 덧붙임.
   - "근거없음": 대부분의 주장이 사례에 없다. 예: 사례에 없는 수치·기한·결론을 만들어 냄.
3) completeness: 결론을 가르는 조건·단서·예외·적용 대상을 빠뜨리지 않았는가. [답변이 인용한 사례]에 적힌 조건만 기준으로 봅니다(인용한 사례 중 정답 사례가 있으면 그 사례를 중심으로).
   인용하지 않은 사례(정답 사례 포함)에만 있는 내용이 빠진 것은 1)의 문제이므로 여기서 보지 않습니다.
   - "완전": 결론에 필요한 조건을 모두 옮겼다.
   - "일부 누락": 부차적인 조건을 빠뜨렸다(빠져도 결론은 그대로).
   - "핵심 누락": 빠지면 결론이 달라지는 조건을 빠뜨렸다. 예: "○○인 경우에 한해" 허용인데 조건 없이 허용처럼 씀.
4) units: [답변 단위]의 번호마다 하나씩 판정합니다. 단위에 붙은 [사례번호]가 그 단위의 내용을 실제로 뒷받침하는가.
   - "맞음": 붙은 번호의 사례마다 그 단위 내용이 있다.
   - "틀림": 붙은 번호 중 하나라도 그 내용이 없는 사례다(다른 사례의 내용이거나 어느 사례에도 없음).
   - "번호 없음": 사례 내용을 옮긴 사실 주장인데 번호가 없다.
   - "해당없음": 번호가 없고 사실 주장도 아니다(연결 문장 등).
5) conflict: [답변에 주어진 사례] 안에 질문의 같은 쟁점에 대해 결론이 서로 다른 사례가 있는가(conflict_in_context), 있으면 답변이 다르다고 쓰고 사례번호로 나눴는가.
   - "처리함": 상충이 있고 나눠 썼다. "누락": 상충이 있는데 한쪽 결론만 썼다. "해당없음": 상충이 없다.
출력 형식:
{"correctness": "...", "correctness_reason": "한 문장", "grounded": "...", "grounded_reason": "한 문장",
 "completeness": "...", "completeness_reason": "한 문장", "units": [{"no": 1, "verdict": "..."}, ...],
 "conflict_in_context": true 또는 false, "conflict": "...", "conflict_reason": "한 문장"}"""
PROMPT_HASH = hashlib.sha256(JUDGE.encode()).hexdigest()[:12]


# ---------- 답변 단위
def units(text):
    """핵심 답변 줄, 근거 항목, 유의사항 항목을 단위로 나눈다. 제목 줄·빈 줄은 뺀다. 각 단위의 인용 번호도 함께."""
    out, sec = [], ""
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"^#{1,6}\s*(.*)$", s)
        if m:
            sec = m.group(1)
            continue
        if not s:
            continue
        if s.startswith("- ") or not out or out[-1]["section"] != sec or sec == "핵심 답변":
            out.append({"no": len(out) + 1, "section": sec, "text": s[2:] if s.startswith("- ") else s})
        else:                                     # 목록 항목이 줄바꿈으로 이어진 경우
            out[-1]["text"] += " " + s
    for u in out:
        u["cites"] = list(dict.fromkeys(c1.CITE.findall(u["text"])))
    return out


def units_text(us):
    return "\n".join(f"({u['no']}) [{u['section']}] {u['text']}" for u in us)


# ---------- 사례 글 (DB)
def load_blocks(conn, keys):
    keys = list(dict.fromkeys(keys))
    lo, f = [k for k in keys if k[0] in "LO"], [k for k in keys if k[0] == "F"]
    out = {}
    if lo:
        for r in conn.execute("select * from interp.cases where case_key = any(%s)", (lo,)).fetchall():
            out[r["case_key"]] = rag.case_block(r)
    if f:
        for r in conn.execute("select case_key, answer_type, reply_date, title, question, answer from interp.exmnt_cases where case_key = any(%s)", (f,)).fetchall():
            out[r["case_key"]] = es.block(r)
    return out


def gold_members(conn, row):
    """정답 묶음의 사례 키(출처 사례 먼저, 최대 GOLD_MAX)."""
    g, src = row.get("gold_group_id"), row.get("source_case_key")
    if g is None:
        return []
    table = "interp.exmnt_cases" if row.get("corpus") == "exmnt" else "interp.cases"
    ks = [r["case_key"] for r in conn.execute(f"select case_key from {table} where group_id = %s order by case_key", (g,)).fetchall()]
    return ([src] if src in ks else []) + [k for k in ks if k != src][:GOLD_MAX - (src in ks)]


def given_keys(ans):
    v = ((ans.get("gate") or {}).get("verdicts")) or {}
    order = ans.get("order") or []
    return [k for k in order if v.get(k) == "직접"] if v else order


# ---------- 심사
def kind(set_name, arm):
    if set_name.startswith("unanswerable"):
        return "답변불가"
    return "정답 제외" if (arm or "").endswith("holdout") else "일반"


def call_judge(model, user):
    for attempt in range(7):                      # 속도 제한(RateLimitError)에 걸리면 5·10·20·40·60·60초 기다린다
        try:
            r = rag.client.chat.completions.create(model=model, reasoning_effort="medium", response_format={"type": "json_object"},
                                                   messages=[{"role": "system", "content": JUDGE}, {"role": "user", "content": user}])
            return json.loads(r.choices[0].message.content), r.usage
        except Exception as e:                    # 키·주소가 담길 수 있어 종류만 남긴다
            if attempt == 6:
                raise RuntimeError(f"심사 호출 실패({type(e).__name__}): 다시 실행하면 캐시에서 이어 간다") from None
            time.sleep(min(60, 5 * 2 ** attempt))


def normalize(j, us):
    """형식이 어긋난 값은 '형식오류'로 남긴다(고쳐 넣지 않는다)."""
    pick = lambda v, ok: v if v in ok else "형식오류"
    got = {int(u.get("no", -1)): u.get("verdict") for u in (j.get("units") or []) if isinstance(u, dict)}
    return {"correctness": pick(j.get("correctness"), CORRECT), "grounded": pick(j.get("grounded"), GROUND),
            "completeness": pick(j.get("completeness"), COMPLETE), "conflict": pick(j.get("conflict"), CONFLICT),
            "conflict_in_context": bool(j.get("conflict_in_context")),
            "units": [{"no": u["no"], "section": u["section"], "cites": u["cites"], "verdict": pick(got.get(u["no"]), UNIT)} for u in us],
            "reasons": {k: j.get(f"{k}_reason", "") for k in ("correctness", "grounded", "completeness", "conflict")}}


def judge_one(item, model):
    ans, q, gold_txt, given_txt = item
    base = {"rubric": RUBRIC, "prompt": PROMPT_HASH, "model": model, "set": ans["set"], "arm": ans.get("arm", ""), "qid": ans["qid"],
            "corpus": ans.get("corpus", ""), "kind": kind(ans["set"], ans.get("arm"))}
    code = c1.check(ans)
    base["code_pass"], base["code_fails"] = code["code_pass"], code["fails"]
    if code["refused"]:
        return {**base, "refused": True, "correctness": "해당없음", "grounded": "해당없음", "completeness": "해당없음", "conflict": "해당없음",
                "conflict_in_context": False, "units": [], "reasons": {}}
    text = ans["shown"].replace(rag.CLOSING, "").strip()
    us = units(text)
    user = (f"[질문]\n{q['question']}\n\n[답변 단위]\n{units_text(us)}\n\n[정답 사례]\n{gold_txt or '(없음)'}\n\n"
            f"[답변이 인용한 사례]\n{ans.get('cited_context') or '(없음)'}\n\n[답변에 주어진 사례]\n{given_txt or '(없음)'}")
    j, usage = call_judge(model, user)
    return {**base, "refused": False, **normalize(j, us), "usage": {"input": usage.prompt_tokens, "output": usage.completion_tokens}}


def passed(r):
    if r["refused"]:
        return r["kind"] != "일반"
    bad_units = any(u["verdict"] in ("틀림", "번호 없음", "형식오류") for u in r["units"])
    return (r["code_pass"] and r["correctness"] == "일치" and r["grounded"] == "충실" and not bad_units
            and r["completeness"] in ("완전", "일부 누락") and r["conflict"] in ("처리함", "해당없음"))


def summarize(rows):
    n = len(rows)
    ans = [r for r in rows if not r["refused"]]
    us = [u for r in ans for u in r["units"] if u["cites"]]
    cnt = lambda key: dict(Counter(r[key] for r in ans))
    conf = [r for r in ans if r["conflict_in_context"]]
    return {"n": n, "refused": n - len(ans), "answered": len(ans), "pass": sum(passed(r) for r in rows), "pass_rate": round(sum(passed(r) for r in rows) / n, 4) if n else None,
            "correctness": cnt("correctness"), "grounded": cnt("grounded"), "completeness": cnt("completeness"),
            "units_with_cite": {"n": len(us), "맞음": sum(u["verdict"] == "맞음" for u in us)},
            "answers_with_bad_unit": sum(any(u["verdict"] in ("틀림", "번호 없음") for u in r["units"]) for r in ans),
            "unit_verdicts": dict(Counter(u["verdict"] for r in ans for u in r["units"])),
            "conflict": {"in_context": len(conf), **Counter(r["conflict"] for r in conf)},
            "format_errors": sum(any(v == "형식오류" for v in (r["correctness"], r["grounded"], r["completeness"], r["conflict"])) or any(u["verdict"] == "형식오류" for u in r["units"]) for r in ans)}


def read_jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def tag_of(a):
    return a.tag or (Path(a.answers).stem.strip("_").replace("_answers", "") + ("" if a.model in ("", rag.CHAT_MODEL) else "_x"))


def run(a):
    model = a.model or rag.CHAT_MODEL
    tag = tag_of(a)
    cache, out_json, out_csv = EVAL / f"_c2_judge_{tag}.jsonl", EVAL / f"c2_llm_judge_{tag}.json", EVAL / f"c2_llm_judge_{tag}_by_answer.csv"
    arms = set(filter(None, a.arms.split(",")))
    answers = [r for r in read_jsonl(EVAL / a.answers) if not c1.LOCKED.match(r["set"]) and (not arms or (r.get("arm") or r.get("mode")) in arms)
               and not r["set"].startswith("unanswerable")]
    if a.qids:                                         # 갈래:qid 목록 (시험 실행)
        want = set(a.qids.split(","))
        answers = [r for r in answers if f"{r.get('arm', '')}:{r['qid']}" in want]
    if a.limit:                                        # 세트·갈래마다 앞에서 N개 (시험 실행)
        seen = Counter()
        answers = [r for r in answers if (seen.update([(r["set"], r.get("arm"))]) or seen[(r["set"], r.get("arm"))] <= a.limit)]
    key = lambda r: (r["set"], r.get("arm", ""), r["qid"])
    done = {key(r): r for r in read_jsonl(cache) if r["rubric"] == RUBRIC and r["prompt"] == PROMPT_HASH and r["model"] == model}
    todo = [r for r in answers if key(r) not in done]
    qs = {}
    for s in {r["set"] for r in answers}:
        qs.update({(s, q["qid"]): q for q in read_jsonl(EVAL / f"{s}.jsonl")})
    print(f"{a.answers}: 심사 대상 {len(answers)} (캐시 {len(answers) - len(todo)}, 새로 {len(todo)}), 심사 모델 {model}, 루브릭 {RUBRIC}/{PROMPT_HASH}")

    items = []
    if todo:
        with rag.connect() as conn:
            for r in todo:
                q = qs[(r["set"], r["qid"])]
                gk = gold_members(conn, q)
                gv = given_keys(r)
                blocks = load_blocks(conn, gk + gv)
                items.append((r, q, "\n\n".join(blocks[k] for k in gk if k in blocks), "\n\n".join(blocks[k] for k in gv if k in blocks)))
    lock = threading.Lock()

    def work(it):
        res = judge_one(it, model)
        with lock, cache.open("a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
        return res

    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        for i, res in enumerate(ex.map(work, items), 1):
            done[key(res)] = res
            if i % 20 == 0 or i == len(items):
                print(f"  {i}/{len(items)} ({time.time() - t0:.0f}초)")

    rows = [done[key(r)] for r in answers if key(r) in done]
    groups = defaultdict(list)
    for r in rows:
        groups[f"{r['kind']} / {r['corpus']}"].append(r)
        groups[f"{r['kind']} / 합계"].append(r)
    summ = {"source": a.answers, "model": model, "rubric": RUBRIC, "prompt": PROMPT_HASH, "note": "사람 판정 합의도 확인 전(1장 원칙 3)",
            "groups": {g: summarize(v) for g, v in sorted(groups.items())}}
    out_json.write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "arm", "qid", "kind", "corpus", "합격", "코드 합격", "정확성", "충실도", "조건 완전성", "상충(문맥에 있음)", "상충 처리",
                    "틀린 단위", "번호 없는 단위", "정확성 사유", "충실도 사유", "조건 사유", "상충 사유", "코드 실패"])
        for r in rows:
            w.writerow([r["set"], r["arm"], r["qid"], r["kind"], r["corpus"], passed(r), r["code_pass"], r["correctness"], r["grounded"], r["completeness"],
                        r["conflict_in_context"], r["conflict"], ",".join(str(u["no"]) for u in r["units"] if u["verdict"] == "틀림"),
                        ",".join(str(u["no"]) for u in r["units"] if u["verdict"] == "번호 없음"),
                        *(r["reasons"].get(k, "") for k in ("correctness", "grounded", "completeness", "conflict")), r["code_fails"]])
    for g, s in summ["groups"].items():
        print(g, json.dumps({k: s[k] for k in ("n", "refused", "pass_rate", "correctness", "grounded", "completeness", "units_with_cite", "conflict", "format_errors")}, ensure_ascii=False))


# ---------- 사람 판정 (6-1)
HCOLS = ["정확성", "충실도", "틀린 단위", "번호 없는 단위", "조건 완전성", "상충 처리", "실무 유용성", "실패 이유", "메모"]
REASONS = "결론 반대 / 다른 쟁점 / 조건 누락 / 대상 확장 / 없는 권고 / 잘못된 인용 / 효력 미표시 / 상충 누락 / 과잉 거절 / 과잉 답변"


def sheet(a):
    if SHEET.exists():
        sys.exit(f"이미 있음: {SHEET} (사람이 채운 판정을 지우지 않습니다)")
    tag = a.tag or "f20"
    rows = [r for r in read_jsonl(EVAL / f"_c2_judge_{tag}.jsonl") if r["rubric"] == RUBRIC and r["prompt"] == PROMPT_HASH and not r["refused"]]
    if len(rows) < 50:
        sys.exit(f"심사 결과가 {len(rows)}개뿐입니다. 먼저 run을 끝까지 돌리세요")
    rows.sort(key=lambda r: (r["set"], r["arm"], r["qid"]))
    rnd = random.Random(20261124)
    good, bad = [r for r in rows if passed(r)], [r for r in rows if not passed(r)]
    nb = min(25, len(bad))
    pick = rnd.sample(bad, nb) + rnd.sample(good, min(50 - nb, len(good)))
    rnd.shuffle(pick)
    ans = {(r["set"], r.get("arm", ""), r["qid"]): r for r in read_jsonl(EVAL / a.answers)}
    qs = {}
    for s in {r["set"] for r in pick}:
        qs.update({(s, q["qid"]): q for q in read_jsonl(EVAL / f"{s}.jsonl")})
    HUMAN.mkdir(parents=True, exist_ok=True)
    keyrows = []
    with rag.connect() as conn, open(SHEET, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["no", "set", "arm", "qid", "질문", "답변 단위(번호)", "정답 사례", "답변이 인용한 사례", "답변에 주어진 사례", *HCOLS])
        for i, r in enumerate(pick, 1):
            an, q = ans[(r["set"], r["arm"], r["qid"])], qs[(r["set"], r["qid"])]
            gk, gv = gold_members(conn, q), given_keys(an)
            b = load_blocks(conn, gk + gv)
            w.writerow([i, r["set"], r["arm"], r["qid"], q["question"], units_text(units(an["shown"].replace(rag.CLOSING, "").strip())),
                        "\n\n".join(b[k] for k in gk if k in b), an.get("cited_context", ""), "\n\n".join(b[k] for k in gv if k in b), *[""] * len(HCOLS)])
            keyrows.append({"no": i, **r})
    KEYF.write_text(json.dumps(keyrows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{SHEET} 50행 (LLM 불합격 {nb} + 합격 {50 - nb}, 섞음)")
    print("칸: 정확성 " + "/".join(CORRECT[:3]) + " · 충실도 " + "/".join(GROUND[:3]) + " · 틀린 단위·번호 없는 단위는 단위 번호를 쉼표로(없으면 비움)"
          " · 조건 완전성 " + "/".join(COMPLETE[:3]) + " · 상충 처리 " + "/".join(CONFLICT) + " · 실무 유용성 1~3 · 실패 이유: " + REASONS)


def kappa(a, b):
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def compare(a):
    keyrows = {r["no"]: r for r in json.loads(KEYF.read_text(encoding="utf-8"))}
    human = [r for r in csv.DictReader(open(SHEET, encoding="utf-8-sig")) if r["정확성"].strip()]
    if not human:
        sys.exit("사람 판정이 비어 있습니다")
    nums = lambda s: {int(x) for x in re.findall(r"\d+", s or "")}
    out, pairs = {"n": len(human)}, []
    items = {"correctness": "정확성", "grounded": "충실도", "completeness": "조건 완전성", "conflict": "상충 처리"}
    for k, col in items.items():
        h = [r[col].strip() for r in human]
        l = [keyrows[int(r["no"])][k] for r in human]
        out[k] = {"agree": round(sum(x == y for x, y in zip(h, l)) / len(h), 3), "kappa": kappa(h, l),
                  "disagree": dict(Counter(f"사람 {x} / LLM {y}" for x, y in zip(h, l) if x != y))}
    hb, lb = [], []
    for r in human:
        k = keyrows[int(r["no"])]
        hbad = bool(nums(r["틀린 단위"]) | nums(r["번호 없는 단위"]))
        lbad = any(u["verdict"] in ("틀림", "번호 없음") for u in k["units"])
        hb.append(hbad)
        lb.append(lbad)
        hp = (k["code_pass"] and r["정확성"].strip() == "일치" and r["충실도"].strip() == "충실" and not hbad
              and r["조건 완전성"].strip() in ("완전", "일부 누락") and r["상충 처리"].strip() in ("처리함", "해당없음"))
        pairs.append((hp, passed(k)))
    out["bad_unit_any"] = {"agree": round(sum(x == y for x, y in zip(hb, lb)) / len(hb), 3), "kappa": kappa(hb, lb)}

    def gate(ps):
        tp = sum(h and l for h, l in ps)
        fp = sum((not h) and l for h, l in ps)
        fn = sum(h and (not l) for h, l in ps)
        return {"precision": round(tp / (tp + fp), 3) if tp + fp else None, "recall": round(tp / (tp + fn), 3) if tp + fn else None,
                "kappa": kappa([h for h, _ in ps], [l for _, l in ps]), "false_pass": fp, "false_fail": fn}
    g = gate(pairs)
    rnd, boot = random.Random(20261124), defaultdict(list)
    for _ in range(1000):
        s = gate([rnd.choice(pairs) for _ in pairs])
        for m in ("precision", "kappa"):
            if s[m] is not None:
                boot[m].append(s[m])
    for m, v in boot.items():
        v.sort()
        g[f"{m}_ci95"] = [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]
    out["pass"] = g
    out["fail_reasons_human"] = dict(Counter(x.strip() for r in human for x in re.split(r"[,/;]", r["실패 이유"]) if x.strip()))
    out["usefulness_human"] = dict(Counter(r["실무 유용성"].strip() for r in human if r["실무 유용성"].strip()))
    (EVAL / "c2_judge_agreement.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


# ---------- 심사 재현성 (6-1-1)
ITEMS = ("correctness", "grounded", "completeness", "conflict")


def fleiss(labels):
    """labels: 답변마다 [실행1, 실행2, ...] (실행 수 같음). Fleiss' kappa."""
    labels = [x for x in labels if x]
    if not labels:
        return None
    n, cats = len(labels[0]), sorted({v for x in labels for v in x}, key=str)
    if len(cats) < 2:
        return 1.0
    P = [(sum(c * c for c in Counter(x).values()) - n) / (n * (n - 1)) for x in labels]
    p = [sum(Counter(x)[c] for x in labels) / (len(labels) * n) for c in cats]
    pe = sum(v * v for v in p)
    return round((sum(P) / len(P) - pe) / (1 - pe), 3) if pe < 1 else 1.0


def majority(vals):
    """다수결. 모두 다르면 None(표 갈림)."""
    v, c = Counter(vals).most_common(1)[0]
    return v if c * 2 > len(vals) else None


def repro(a):
    tags = a.tags.split(",")
    runs = []
    for t in tags:
        runs.append({(r["set"], r["arm"], r["qid"]): r for r in read_jsonl(EVAL / f"_c2_judge_{t}.jsonl") if r["rubric"] == RUBRIC and r["prompt"] == PROMPT_HASH})
    keys = sorted(set.intersection(*(set(r) for r in runs)))
    if not keys:
        sys.exit("모든 실행에 공통인 심사 결과가 없습니다")
    rows, out_rows = [], []
    for k in keys:
        rs = [run_[k] for run_ in runs]
        if rs[0]["refused"]:
            continue
        rows.append(rs)
    per = {}
    for it in ITEMS:
        labs = [[r[it] for r in rs] for rs in rows]
        per[it] = {"all_same": round(sum(len(set(x)) == 1 for x in labs) / len(labs), 3), "fleiss_kappa": fleiss(labs),
                   "all_differ": sum(len(set(x)) == len(x) for x in labs)}
    badu = [[any(u["verdict"] in ("틀림", "번호 없음") for u in r["units"]) for r in rs] for rs in rows]
    per["bad_unit_any"] = {"all_same": round(sum(len(set(x)) == 1 for x in badu) / len(badu), 3), "fleiss_kappa": fleiss(badu)}
    ulabs = [[r["units"][i]["verdict"] for r in rs] for rs in rows for i in range(min(len(r["units"]) for r in rs))]
    per["unit_verdict"] = {"n_units": len(ulabs), "all_same": round(sum(len(set(x)) == 1 for x in ulabs) / len(ulabs), 3), "fleiss_kappa": fleiss(ulabs)}
    pas = [[passed(r) for r in rs] for rs in rows]
    per["pass"] = {"all_same": round(sum(len(set(x)) == 1 for x in pas) / len(pas), 3), "fleiss_kappa": fleiss(pas),
                   "always_pass": sum(all(x) for x in pas), "always_fail": sum(not any(x) for x in pas), "flip": sum(len(set(x)) > 1 for x in pas)}

    # 다수결 판정
    maj_rows = []
    for rs in rows:
        m = {**rs[0], **{it: majority([r[it] for r in rs]) or "표 갈림" for it in ITEMS}}
        m["units"] = [{**u, "verdict": majority([r["units"][i]["verdict"] for r in rs]) or "표 갈림"} for i, u in enumerate(rs[0]["units"])] \
            if len({len(r["units"]) for r in rs}) == 1 else rs[0]["units"]
        maj_rows.append(m)
        out_rows.append([rs[0]["set"], rs[0]["arm"], rs[0]["qid"], rs[0]["kind"], rs[0]["corpus"],
                         *["/".join(str(r[it]) for r in rs) for it in ITEMS], "/".join("합" if p else "불" for p in (passed(r) for r in rs)),
                         *[m[it] for it in ITEMS], passed(m)])
    # 실행별 집계 지표가 얼마나 흔들리는지 (개별 판정이 흔들려도 비율은 안정적일 수 있다)
    by_kind = defaultdict(lambda: defaultdict(list))
    for rs in rows:
        for i, r in enumerate(rs):
            by_kind[r["kind"]][i].append(r)
    agg = {}
    for kd, d in sorted(by_kind.items()):
        agg[kd] = {"answered": len(d[0]), "per_run": [{"pass_rate": round(sum(map(passed, d[i])) / len(d[i]), 4),
                                                        "일치": round(sum(r["correctness"] == "일치" for r in d[i]) / len(d[i]), 4),
                                                        "충실": round(sum(r["grounded"] == "충실" for r in d[i]) / len(d[i]), 4),
                                                        "조건 완전": round(sum(r["completeness"] == "완전" for r in d[i]) / len(d[i]), 4),
                                                        "단위 문제 있음": round(sum(any(u["verdict"] in ("틀림", "번호 없음") for u in r["units"]) for r in d[i]) / len(d[i]), 4)}
                                                       for i in range(len(tags))]}
        mk = [m for m in maj_rows if m["kind"] == kd]
        agg[kd]["majority"] = {"pass_rate": round(sum(map(passed, mk)) / len(mk), 4), "split_any": sum(any(m[it] == "표 갈림" for it in ITEMS) for m in mk)}
    res = {"tags": tags, "rubric": RUBRIC, "prompt": PROMPT_HASH, "answered": len(rows), "items": per, "aggregate": agg}
    (EVAL / "c2_repro.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(EVAL / "c2_repro_by_answer.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["set", "arm", "qid", "kind", "corpus", "정확성(실행별)", "충실도(실행별)", "조건 완전성(실행별)", "상충(실행별)", "합격(실행별)",
                    "정확성(다수결)", "충실도(다수결)", "조건 완전성(다수결)", "상충(다수결)", "합격(다수결)"])
        w.writerows(out_rows)
    print(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "sheet", "compare", "repro"])
    ap.add_argument("--tags", default="r1,r2,r3", help="repro: 비교할 심사 실행 태그")
    ap.add_argument("--answers", default="_f20_answers.jsonl")
    ap.add_argument("--arms", default="gate,gate_holdout")
    ap.add_argument("--limit", type=int, default=0, help="세트·갈래마다 앞에서 N개만(시험 실행)")
    ap.add_argument("--qids", default="", help="쉼표로 구분한 '갈래:qid'만 심사(시험 실행). 예: gate:DEV7-001,gate_holdout:DEV9-098")
    ap.add_argument("--model", default="", help="심사 모델(비우면 답변 모델과 같은 rag.CHAT_MODEL). 다르게 주면 교차 심사")
    ap.add_argument("--tag", default="", help="캐시·결과 파일 이름. 비우면 답변 파일 이름에서 만든다")
    a = ap.parse_args()
    {"run": run, "sheet": sheet, "compare": compare, "repro": repro}[a.cmd](a)


if __name__ == "__main__":
    main()
