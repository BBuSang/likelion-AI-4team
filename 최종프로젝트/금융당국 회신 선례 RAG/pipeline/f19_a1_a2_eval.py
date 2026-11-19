"""F19: 118번 개선안 A1·A2 (worklog 119번, 사용자 "A1·A2 진행", 2026-11-17). 개발셋만 쓴다. 시험 세트는 읽지 않는다.

A1 지금 데모 설정 재측정
- 문항: dev7·dev8·dev9·dev10 300문항(회신 210 + 현장건의 90) + 답변불가 20(unanswerable_20).
- 두 갈래를 같은 코드·같은 시점에 돌린다. 프롬프트 말고는 모두 같다.
  V3 = 지금 데모(`service.ask` 기본값: 리랭킹 + 현장건의 + 못 찾을 때만 키워드 완화 + 기간 조건 분리, 답변 검증 끔, 프롬프트 `APP_SYSTEM_V3`)
  V2 = 같은 경로에서 `service.APP_SYSTEM`만 `APP_SYSTEM_V2`로 바꾼 것(110번 이전 프롬프트)
- 심사: f18과 같은 전문 심사(`f4.JUDGE_SYSTEM`, 정답 사례 전문 + 답변이 인용한 사례 전문, reasoning medium). 답변불가 20은 심사 없이 거절 여부만 센다.
- 사전 기준: V3를 유지하는 조건. 돌리기 전에 정했고 결과를 보고 바꾸지 않는다.
  1. 일치 비율(300문항 분모): V3 ≥ V2 − 3%p
  2. 충실 비율(답한 문항 분모): V3 ≥ V2 − 3%p
  3. 잘못된 사례번호(문맥에 없던 번호) V3 0건
  4. 답변불가 20 거절 수: V3 ≥ V2
  하나라도 못 넘으면 V3를 고쳐 맞추지 않고 보고한다. V2로 되돌릴지는 사용자가 정한다.
  같은 문항 짝 비교(좋아짐/나빠짐, 부호 검정)와 단정 표현 수도 함께 보고한다. 재실행만으로 약 ±1.5%p 흔들린다(73번).

A2 정답 제외 시험 (가장 큰 약점 "맞는 회신이 없을 때 다른 사례로 답함"의 기준선)
- 같은 300문항을 V3로, 문항마다 정답 묶음을 검색 대상에서 뺀 채 돌린다: 회신 후보(`rr.candidates`), 기간 후보(`service.candidates_in`),
  키워드 완화(`kr.relax_search`의 제외 묶음), 현장건의(`es.search`). 정답이 빠지면 그다음 후보가 자리를 채운다(후보 수는 그대로).
- 정답이 없으므로 바라는 응답은 "찾지 못했습니다", 또는 다른 회신으로 같은 결론(같은 취지의 다른 묶음이 있을 수 있다).
- 같은 심사자로 판정한다(정답 사례 = 뺀 정답). 보고: 답변없음 / 일치 / 부분일치 / 불일치, 유보형 답(앞부분에서 "찾지 못"이라 하면서 인용),
  회신·현장건의별. 핵심 지표는 "정답 없이 다른 결론" = 불일치 + 부분일치 비율. B1(인접 사례 표시)의 기준선이며 합격 기준은 없다.
- 확인용: 정답 묶음이 최종 문맥에 들어간 문항은 0이어야 한다.

한계: 질문은 정답 회신을 보고 LLM이 만든 합성 질문이고, 생성·리랭킹·답변·심사가 모두 같은 모델이다. 응답시간은 4개씩 동시에 돌려 재지 않는다.
사용: python pipeline/f19_a1_a2_eval.py [--smoke]      --smoke: 세트마다 2문항, 별도 캐시(_f19s_*)
캐시: data/eval/_f19_answers.jsonl, _f19_judge.jsonl (갈래·세트·문항 단위로 이어서 돈다)   결과: data/eval/f19_a1_a2_eval.json
"""
import argparse
import hashlib
import json
import math
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "pipeline"))
import exmnt_search as es  # noqa: E402
import f4_rag_answer_eval as f4  # noqa: E402
import keyword_relax as kr  # noqa: E402
import rag  # noqa: E402
import rerank as rr  # noqa: E402
import service  # noqa: E402
from common import DATA  # noqa: E402
from prompts import APP_SYSTEM_V2, APP_SYSTEM_V3  # noqa: E402

EVAL = DATA / "eval"
SETS = ["dev7", "dev8", "dev9", "dev10"]
UNANS = "unanswerable_20"
ARMS = ["v3", "v2", "holdout"]
CITE = re.compile(r"\[([LOF]\d+)\]")
BLOCK = re.compile(r"\[사례 ([LOF]\d+)\].*?(?=\n\n\[사례 |\Z)", re.S)
WORKERS = 4
ES_MAX = es.MAX_N          # 현장건의를 붙이는 최대 개수(2)

# ---------- 스레드마다 다른 값: 정답 제외 묶음, 마지막 답변 문맥
_tl = threading.local()
_embed_lock = threading.Lock()
_orig = {"embed": rag.embed, "answer": service._answer, "cands": rr.candidates, "cands_in": service.candidates_in,
         "relax": kr.relax_search, "es": es.search}


def _embed(text):
    with _embed_lock:                              # 임베딩 모델 하나를 여러 스레드가 함께 쓴다
        return _orig["embed"](text)


def _answer(*a, **k):
    out = _orig["answer"](*a, **k)
    _tl.ctx, _tl.ids = a[1], a[2]
    return out


def _gold():
    return getattr(_tl, "gold", None)


def _cands(conn, question, n=rr.N_CAND):
    g = _gold()
    if g is None:
        return _orig["cands"](conn, question, n)
    return [c for c in _orig["cands"](conn, question, n + 1) if c["group_id"] != g][:n]


def _cands_in(conn, masked, rng, n=rr.N_CAND):
    g = _gold()
    if g is None:
        return _orig["cands_in"](conn, masked, rng, n)
    return [c for c in _orig["cands_in"](conn, masked, rng, n + 1) if c["group_id"] != g][:n]


def _relax(conn, concepts, qvec, exclude_groups, n_extra=kr.N_EXTRA):
    g = _gold()
    return _orig["relax"](conn, concepts, qvec, set(exclude_groups) | ({g} if g is not None else set()), n_extra)


def _es(conn, qvec, ref_score=None):
    g = _gold()
    if g is None:
        return _orig["es"](conn, qvec, ref_score)
    fields, top1 = _orig["es"](conn, qvec, ref_score)          # 정답 제외 갈래에서는 es.MAX_N을 하나 늘려 두었다(main). 정답을 빼고 원래 개수로 자른다
    return [f for f in fields if f["group_id"] != g][:ES_MAX], top1


def install():
    rag.embed = _embed
    service._answer = _answer
    rr.candidates = _cands
    service.candidates_in = _cands_in
    kr.relax_search = _relax
    es.search = _es


# ---------- 공용
def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


_wlock = threading.Lock()


def append(path, row):
    with _wlock, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def fingerprint():
    paths = sorted((ROOT / "app").glob("*.py")) + [ROOT / "pipeline" / n for n in ("rag.py", "common.py", "time_marks.py", "f4_rag_answer_eval.py", Path(__file__).name)]
    data = {"files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "sets": {n: hashlib.sha256((EVAL / f"{n}.jsonl").read_bytes()).hexdigest() for n in SETS + [UNANS]},
            "model": rag.CHAT_MODEL, "embedding": rag.QUERY_MODEL, "reasoning": rag.REASONING}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16], data


def gold_refs(conn, rows):
    """정답 사례 글: 회신은 제목·질의요지·회답·이유, 현장건의는 dev10 생성 때와 같은 모양(회답 = '검토결과: …' + 검토 사유)."""
    out = {}
    for r in rows:
        k = r["source_case_key"]
        if r["corpus"] == "exmnt":
            c = conn.execute("select title, question, answer, answer_type from interp.exmnt_cases where case_key = %s", (k,)).fetchone()
            out[r["qid"]] = {"title": c["title"] or "", "question": c["question"] or "", "answer": f"검토결과: {c['answer_type']}\n{c['answer'] or ''}", "reason": ""}
        else:
            c = conn.execute("select title, question, answer, reason from interp.cases where case_key = %s", (k,)).fetchone()
            out[r["qid"]] = {x: c[x] or "" for x in ("title", "question", "answer", "reason")}
    return out


def run_one(arm, name, row, groups):
    _tl.gold = row.get("gold_group_id") if arm.endswith("holdout") else None          # f20이 "gate_holdout"으로도 부른다
    _tl.ctx, _tl.ids = "", []
    t0 = time.time()
    try:
        res = service.ask(row["question"])
    except Exception as e:                         # 키·주소가 담길 수 있어 종류만 남긴다
        return {"arm": arm, "set": name, "qid": row["qid"], "error": type(e).__name__}
    finally:
        _tl.gold = None
    not_found = bool(res["check"]["not_found"] or not res["cited"])
    shown = rag.NOT_FOUND if not_found else res["raw"]
    cited = list(dict.fromkeys(CITE.findall(shown)))
    blocks = {m.group(1): m.group(0) for m in BLOCK.finditer(_tl.ctx)}
    ids = res["order"]
    return {"arm": arm, "set": name, "qid": row["qid"], "corpus": row.get("corpus", "unanswerable"), "seconds": round(time.time() - t0, 1),
            "not_found": not_found, "shown": shown, "cited": cited, "cited_context": "\n\n".join(blocks[c] for c in cited if c in blocks),
            "order": ids, "invalid": res.get("invalid", []), "assertive": res["check"]["assertive"], "hedged": bool(service.partly_not_found(res)),
            "gold_in_context": row.get("gold_group_id") in {groups.get(k) for k in ids}, "gold_cited": row.get("gold_group_id") in {groups.get(k) for k in cited},
            "relax_retried": bool((res.get("relax") or {}).get("retried")), "period": res.get("period"), "usage": res["usage"],
            "gate": res.get("gate")}          # f20(직접성 판정)에서만 값이 있다


def judge(item):
    row, ans, ref = item
    base = {"arm": ans["arm"], "set": ans["set"], "qid": ans["qid"]}
    if ans.get("error"):
        return {**base, "correctness": "시스템오류", "grounded": "해당없음"}
    if ans["not_found"]:
        return {**base, "correctness": "답변없음", "grounded": "해당없음"}
    body = ans["shown"].replace(rag.CLOSING, "").strip()
    user = f"[질문]\n{row['question']}\n\n[시스템 답변]\n{body}\n\n[정답 사례]\n" + json.dumps(ref, ensure_ascii=False) + f"\n\n[답변이 인용한 사례 전문]\n{ans['cited_context']}"
    for attempt in range(4):
        try:
            r = rag.client.chat.completions.create(model=rag.CHAT_MODEL, reasoning_effort="medium", response_format={"type": "json_object"},
                                                   messages=[{"role": "system", "content": f4.JUDGE_SYSTEM}, {"role": "user", "content": user}])
            j = json.loads(r.choices[0].message.content)
            assert j["correctness"] in ("일치", "부분일치", "불일치", "답변없음") and j["grounded"] in ("충실", "일부추정", "근거없음", "해당없음")
            return {**base, "correctness": j["correctness"], "grounded": j["grounded"], "reason": j.get("reason", "")}
        except Exception:
            if attempt == 3:
                raise RuntimeError("심사 실패: 다시 실행하면 캐시에서 이어 간다") from None
            time.sleep(3 * (attempt + 1))


# ---------- 집계
def sign_p(b, w):
    """부호 검정 양측 p (좋아짐 b, 나빠짐 w)."""
    n, k = b + w, min(b, w)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def summary(rows, ans, jd):
    n = len(rows)
    a = [ans[r["qid"]] for r in rows]
    j = [jd[r["qid"]] for r in rows]
    c, g = Counter(x["correctness"] for x in j), Counter(x["grounded"] for x in j)
    answered = sum(x["correctness"] not in ("답변없음", "시스템오류") for x in j)
    return {"n": n, "correctness": dict(c), "accuracy": round(c["일치"] / n, 4), "grounded": dict(g),
            "faithful_answered": round(g["충실"] / answered, 4) if answered else None, "answered": answered,
            "other_conclusion": round((c["불일치"] + c["부분일치"]) / n, 4),
            "gold_in_context": sum(x.get("gold_in_context", False) for x in a), "gold_cited": sum(x.get("gold_cited", False) for x in a),
            "invalid": sum(len(x.get("invalid") or []) for x in a), "assertive": sum(x.get("assertive", 0) for x in a),
            "hedged": sum(x.get("hedged", False) for x in a), "relax_retried": sum(x.get("relax_retried", False) for x in a),
            "period_split": sum(bool(x.get("period")) for x in a), "errors": sum(bool(x.get("error")) for x in a)}


def paired(rows, j3, j2, key, good):
    b = sum(1 for r in rows if (j3[r["qid"]][key] == good) and (j2[r["qid"]][key] != good))
    w = sum(1 for r in rows if (j3[r["qid"]][key] != good) and (j2[r["qid"]][key] == good))
    return {"v3_better": b, "v3_worse": w, "p": round(sign_p(b, w), 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()
    tag = "_f19s" if args.smoke else "_f19"
    ANS, JDG, RUN = EVAL / f"{tag}_answers.jsonl", EVAL / f"{tag}_judge.jsonl", EVAL / f"{tag}_run.json"
    OUT = EVAL / ("f19_smoke.json" if args.smoke else "f19_a1_a2_eval.json")
    fp, parts = fingerprint()
    if RUN.exists():
        assert json.loads(RUN.read_text(encoding="utf-8"))["fingerprint"] == fp, "설정이 바뀌었다: 캐시와 섞지 않는다"
    else:
        save(RUN, {"started_at": now(), "fingerprint": fp, "parts": parts})
    install()
    data = {n: [json.loads(x) for x in (EVAL / f"{n}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] for n in SETS + [UNANS]}
    if args.smoke:
        data = {n: v[:2] for n, v in data.items()}
    with rag.connect() as conn:
        groups = {r["case_key"]: r["group_id"] for r in conn.execute("select case_key, group_id from interp.cases union all select case_key, group_id from interp.exmnt_cases").fetchall()}
        refs = gold_refs(conn, [r for n in SETS for r in data[n]])
    rag.embed("준비")
    answers = {(x["arm"], x["set"], x["qid"]): x for x in read(ANS)}
    for arm in args.arms.split(","):
        service.APP_SYSTEM = APP_SYSTEM_V2 if arm == "v2" else APP_SYSTEM_V3          # _ask가 실행할 때 이 전역을 읽는다
        es.MAX_N = ES_MAX + 1 if arm.endswith("holdout") else ES_MAX          # 정답 제외: 하나 더 골라 두고 `_es`에서 정답을 뺀 뒤 ES_MAX개로 자른다
        todo = [(n, r) for n in SETS + ([UNANS] if arm != "holdout" else []) for r in data[n] if (arm, n, r["qid"]) not in answers]
        t0 = time.time()
        with ThreadPoolExecutor(WORKERS) as ex:
            for i, a in enumerate(ex.map(lambda x: run_one(arm, x[0], x[1], groups), todo), 1):
                append(ANS, a)
                answers[(arm, a["set"], a["qid"])] = a
                if i % 20 == 0 or i == len(todo):
                    print(f"[{arm}] 답변 {i}/{len(todo)} ({time.time() - t0:.0f}초)", flush=True)
    service.APP_SYSTEM, es.MAX_N = APP_SYSTEM_V3, ES_MAX
    judged = {(x["arm"], x["set"], x["qid"]): x for x in read(JDG)}
    todo = [(r, answers[(arm, n, r["qid"])], refs[r["qid"]]) for arm in args.arms.split(",") for n in SETS for r in data[n]
            if (arm, n, r["qid"]) in answers and (arm, n, r["qid"]) not in judged]
    with ThreadPoolExecutor(6) as ex:
        for i, j in enumerate(ex.map(judge, todo), 1):
            append(JDG, j)
            judged[(j["arm"], j["set"], j["qid"])] = j
            if i % 50 == 0 or i == len(todo):
                print(f"심사 {i}/{len(todo)}", flush=True)
    # 집계
    rows = [r for n in SETS for r in data[n]]
    res = {"fingerprint": fp, "updated_at": now(), "n_questions": len(rows), "arms": {}}
    for arm in args.arms.split(","):
        A = {r["qid"]: answers[(arm, n, r["qid"])] for n in SETS for r in data[n] if (arm, n, r["qid"]) in answers}
        J = {r["qid"]: judged[(arm, n, r["qid"])] for n in SETS for r in data[n] if (arm, n, r["qid"]) in judged}
        if len(J) < len(rows):
            continue
        res["arms"][arm] = {"all": summary(rows, A, J), "interp": summary([r for r in rows if r["corpus"] == "interp"], A, J),
                            "exmnt": summary([r for r in rows if r["corpus"] == "exmnt"], A, J),
                            "by_set": {n: summary(data[n], A, J) for n in SETS}}
        if arm != "holdout":
            u = [answers[(arm, UNANS, r["qid"])] for r in data[UNANS]]
            res["arms"][arm]["unanswerable"] = {"n": len(u), "refused": sum(x.get("not_found", False) for x in u), "errors": sum(bool(x.get("error")) for x in u)}
    if {"v3", "v2"} <= set(res["arms"]):
        J3 = {r["qid"]: judged[("v3", n, r["qid"])] for n in SETS for r in data[n]}
        J2 = {r["qid"]: judged[("v2", n, r["qid"])] for n in SETS for r in data[n]}
        v3, v2 = res["arms"]["v3"], res["arms"]["v2"]
        crit = {
            "1_accuracy": {"v3": v3["all"]["accuracy"], "v2": v2["all"]["accuracy"], "pass": v3["all"]["accuracy"] >= v2["all"]["accuracy"] - 0.03},
            "2_faithful_answered": {"v3": v3["all"]["faithful_answered"], "v2": v2["all"]["faithful_answered"],
                                    "pass": (v3["all"]["faithful_answered"] or 0) >= (v2["all"]["faithful_answered"] or 0) - 0.03},
            "3_invalid_zero": {"v3": v3["all"]["invalid"], "pass": v3["all"]["invalid"] == 0},
            "4_unanswerable": {"v3": v3["unanswerable"]["refused"], "v2": v2["unanswerable"]["refused"], "pass": v3["unanswerable"]["refused"] >= v2["unanswerable"]["refused"]}}
        res["a1"] = {"criteria": crit, "all_pass": all(c["pass"] for c in crit.values()),
                     "paired_accuracy": paired(rows, J3, J2, "correctness", "일치"), "paired_faithful": paired(rows, J3, J2, "grounded", "충실")}
    save(OUT, res)
    print(json.dumps({k: v for k, v in res.items() if k != "arms"}, ensure_ascii=False, indent=1))
    for arm, v in res["arms"].items():
        print(arm, json.dumps({k: v["all"][k] for k in ("correctness", "accuracy", "faithful_answered", "other_conclusion", "gold_in_context", "invalid", "assertive", "hedged", "errors")}, ensure_ascii=False),
              v.get("unanswerable", ""))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
