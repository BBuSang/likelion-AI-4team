"""기획안 5-6-7 개선 ① 측정: 답변 호출 한 번 안에서 LLM이 근거 조문을 고름 (dev 세트만)

안:  only1 = ①만 (1~10위를 넘김) / both = ①+③ (1~5위 + 연결 조문을 넘김)
한 번에 한 문항씩 보내 시간(검색 + LLM)을 잰다. 결과는 문항마다 바로 jsonl 에 쌓아 중간에 멈춰도 이어서 한다.

    python pipeline/eval_select_06.py --arm only1 [--limit 5]
    python pipeline/eval_select_06.py --arm both
    python pipeline/eval_select_06.py --report        # 두 안 + ③만 비교표
    python pipeline/eval_select_06.py --arm only1 --effort low --same-as only1   # 추론 강도 시험 (기획안 5-6-7, v0.21):
                                                      # 기본 설정으로 이미 잰 문항만 다시 → select_only1_low_dev.jsonl
    python pipeline/eval_select_06.py --compare only1 only1_low   # 문항 단위 비교
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retriever.search import attach_links, embed, search_vec, union_candidates, with_full_text  # noqa: E402
from retriever.answer import SYSTEM, build, call_llm, compose, key_phrases, parse_full  # noqa: E402,F401  (v0.31: 답변 단계와 같은 지시문·입력·호출)

EVAL = ROOT / "eval"
ENV = dotenv_values(ROOT / ".env")
DEV = ["dev1", "dev2", "dev3", "dev4", "dev5", "dev6"]
EFFORT = "low"  # 기획안 5-6-7 (v0.24 채택): 답변 호출의 추론 강도. "default" 를 주면 모델 기본값(요청에 안 넣음)
OUT = {a: EVAL / f"select_{a}_dev.jsonl" for a in ("only1", "both")}
ARMS = ("only1", "both", "union")  # union (v0.26) = 1~10위 + 1~5위에 연결된 조문 중 1~10위에 없는 것


def candidates(arm, arts):
    if arm == "only1":
        return arts[:10]
    if arm == "union":  # 채택된 구성 (v0.30) — 답변 단계와 같은 함수
        return union_candidates(arts)
    top, extra = attach_links(arts)
    return top + extra


def out_path(name, plain=False):
    """dev 만 잰 예전 결과는 select_<이름>_dev.jsonl, test 가 섞인 실행은 select_<이름>.jsonl"""
    legacy, new = EVAL / f"select_{name}_dev.jsonl", EVAL / f"select_{name}.jsonl"
    return legacy if legacy.exists() or not (plain or new.exists()) else new


LOCK = EVAL / "test_lock.json"


def check_lock(sets):
    """기획안 5-6-4: test 세트는 한 번만. 잠긴 세트가 있으면 멈춘다 (푸는 옵션 없음)"""
    lock = json.loads(LOCK.read_text(encoding="utf-8")) if LOCK.exists() else {}
    done = [s for s in sets if s.startswith("test") and s in lock]
    if done:
        raise SystemExit(f"🔒 이미 잰 test 세트: {done} → 다시 재지 않음")
    return lock


def run(arm, limit, effort="", same_as="", tag="", sets=DEV, text="", prompt=None):
    lock = check_lock(sets)  # test 세트는 한 번만: 잠긴 세트가 있으면 멈추고, 다 재면 잠근다
    qs = [q for q in csv.DictReader(open(EVAL / "queries.csv", encoding="utf-8-sig")) if q["set"] in sets]
    gold = defaultdict(set)
    for g in csv.DictReader(open(EVAL / "gold.csv", encoding="utf-8-sig")):
        gold[g["qid"]].add(g["article_uid"])
    out = out_path("_".join(x for x in (arm, "" if effort == "default" else effort, tag) if x), plain=list(sets) != DEV)
    if same_as:  # 다른 실행에서 이미 잰 문항만 (문항 단위 비교용)
        base = {json.loads(l)["qid"] for l in open(out_path(same_as), encoding="utf-8")}
        qs = [q for q in qs if q["qid"] in base]
    done = set()
    if out.exists():
        done = {json.loads(l)["qid"] for l in open(out, encoding="utf-8")}
    todo = [q for q in qs if q["qid"] not in done][: limit or None]
    print(f"{out.name}: 남은 {len(todo)}문항 (이미 {len(done)})")
    search_vec(embed(["모델 미리 올리기"])[0], k=1)  # 서버 시작 때 하는 모델 로드 (시간에서 뺌)
    with open(out, "a", encoding="utf-8") as f:
        for n, q in enumerate(todo, 1):
            t0 = time.perf_counter()
            arts = search_vec(embed([q["question"]])[0], k=0)
            cands = candidates(arm, arts)
            if text == "full":  # 기획안 5-6-8: 여러 조각 조문은 전체(또는 걸린 조각 전부)를 넘김
                cands = with_full_text(cands)
            t1 = time.perf_counter()
            user = build(q["question"], cands)
            err, picks, answer, usage, keys = "", [], "", {}, []
            try:
                txt, usage = call_llm(user, effort, prompt=prompt)
                t2 = time.perf_counter()
                try:
                    picks, answer, explain = parse_full(txt, len(cands))
                    keys = key_phrases(txt)  # 5-8-5 (v4 지시문만 값이 있음)
                    if explain or prompt == "v3":  # 5-8-4: 채점은 결론 + 조문별 설명
                        answer = compose(answer, explain, cands, picks)
                        if picks and len(explain) < len(picks):
                            err = f"조문별 누락 {len(picks) - len(explain)}"
                except (ValueError, TypeError):
                    err = "JSON 형식 오류"
            except RuntimeError as e:
                t2 = time.perf_counter()
                err = str(e)
            g = gold[q["qid"]]
            rec = {"qid": q["qid"], "set": q["set"], "유형": q["q_type"], "난이도": q.get("difficulty", ""),
                   "후보": [a["doc_id"] for a in cands], "후보 제목": [a["title"] for a in cands],
                   "검색 순위": [a["doc_id"] for a in arts],
                   "정답 검색 순위": next((i for i, a in enumerate(arts, 1) if a["doc_id"] in g), 0),
                   "검색 1위 정답": arts[0]["doc_id"] in g, "검색 5위 안 정답": any(a["doc_id"] in g for a in arts[:5]),
                   "묶음 정답": any(a["doc_id"] in g for a in cands),
                   "근거": [cands[i - 1]["doc_id"] for i in picks], "근거 번호": picks,
                   "선택 1위 정답": bool(picks) and cands[picks[0] - 1]["doc_id"] in g,
                   "근거 안 정답": any(cands[i - 1]["doc_id"] in g for i in picks),
                   "검색 초": round(t1 - t0, 3), "LLM 초": round(t2 - t1, 3), "전체 초": round(t2 - t0, 3),
                   "입력 글자": len(SYSTEM) + len(user), "토큰": usage, "답변": answer, "오류": err,
                   "글 방식": [a.get("text_mode", "최고점 조각") for a in cands], "후보 글자": [len(a["text"]) for a in cands],
                   "핵심": keys, "핵심 답변에 있음": [k in answer for k in keys]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}  마지막 {rec['전체 초']}초", flush=True)
    import hashlib
    rows = {r["qid"]: r for r in map(json.loads, open(out, encoding="utf-8"))}
    for s in sets:
        ids = [q for q in qs if q["set"] == s]
        if not s.startswith("test") or any(q["qid"] not in rows for q in ids):
            continue
        items = [(q["qid"], q["question"], sorted(gold[q["qid"]])) for q in ids]
        lock[s] = {"measured_on": time.strftime("%Y%m%d"), "run": out.stem,
                   "Hit@1": round(sum(rows[q["qid"]]["검색 1위 정답"] for q in ids) / len(ids), 3),
                   "Hit@5": round(sum(rows[q["qid"]]["검색 5위 안 정답"] for q in ids) / len(ids), 3), "n": len(ids),
                   "set_sha256": hashlib.sha256(json.dumps(items, ensure_ascii=False).encode()).hexdigest(),
                   "note": "기획안 5-6-4·5-6-7: 세트마다 한 번만. Hit 는 검색만의 값, LLM 근거 선택·답변을 함께 잼. 이 결과로 검색을 고치지 않는다"}
        LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"🔒 {s} 잠금", flush=True)


def report():
    links = {r["qid"]: r for r in csv.DictReader(open(EVAL / "results_links_dev.csv", encoding="utf-8-sig"))}
    arms = {a: [json.loads(l) for l in open(OUT[a], encoding="utf-8")] for a in OUT if OUT[a].exists()}

    def stat(rs):
        n = len(rs)
        ts = sorted(r["전체 초"] for r in rs)
        return {"문항": n, "1위 정답률": round(sum(r["선택 1위 정답"] for r in rs) / n, 3),
                "근거 안 정답률": round(sum(r["근거 안 정답"] for r in rs) / n, 3),
                "묶음 정답률": round(sum(r["묶음 정답"] for r in rs) / n, 3),
                "검색 1위 정답률": round(sum(r["검색 1위 정답"] for r in rs) / n, 3),
                "평균 초": round(sum(ts) / n, 2), "95% 초": ts[int(n * 0.95) - 1] if n >= 20 else ts[-1], "최대 초": ts[-1],
                "10초 넘음": sum(t > 10 for t in ts), "평균 입력 글자": round(sum(r["입력 글자"] for r in rs) / n),
                "오류": sum(bool(r["오류"]) for r in rs),
                "1위 정답 → 틀림": [r["qid"] for r in rs if r["검색 1위 정답"] and not r["선택 1위 정답"]],
                "1위 틀림 → 정답": [r["qid"] for r in rs if not r["검색 1위 정답"] and r["선택 1위 정답"]]}

    res = {}
    for a, rs in arms.items():
        res[a] = {"전체": stat(rs)}
        for key in ["set", "유형", "난이도"]:
            for v in sorted({r[key] for r in rs if r[key]}):
                sub = [r for r in rs if r[key] == v]
                s = stat(sub)
                res[a][f"{key}:{v}"] = {k: s[k] for k in ["문항", "1위 정답률", "검색 1위 정답률", "근거 안 정답률", "묶음 정답률", "최대 초"]}
    l3 = list(links.values())
    res["③만"] = {"1위 정답률": round(sum(r["Hit@1"] == "True" for r in l3) / len(l3), 3),
                  "묶음 정답률": round(sum(r["ⓒ 묶음"] == "True" for r in l3) / len(l3), 3)}
    json.dump(res, open(EVAL / "summary_select_dev.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps({a: v.get("전체", v) for a, v in res.items()}, ensure_ascii=False, indent=1))


def run_pair(arm, effort, sets, tag, limit):
    """기획안 5-6-7 (v0.22): 문항마다 검색 한 번 → 같은 후보로 추론 기본값과 `effort` 를 연달아 호출 (순서는 문항마다 번갈아).
    test 세트는 모든 문항이 끝나면 잠근다."""
    import hashlib
    lock = check_lock(sets)
    qs = [q for q in csv.DictReader(open(EVAL / "queries.csv", encoding="utf-8-sig")) if q["set"] in sets]
    gold = defaultdict(set)
    for g in csv.DictReader(open(EVAL / "gold.csv", encoding="utf-8-sig")):
        gold[g["qid"]].add(g["article_uid"])
    names = {"": "_".join(x for x in (arm, tag) if x), effort: "_".join(x for x in (arm, effort, tag) if x)}
    outs = {e: out_path(n, plain=True) for e, n in names.items()}
    seen = [({json.loads(l)["qid"] for l in open(o, encoding="utf-8")} if o.exists() else set()) for o in outs.values()]
    done = seen[0] & seen[1]
    todo = [q for q in qs if q["qid"] not in done][: limit or None]
    print(f"{[o.name for o in outs.values()]}: 남은 {len(todo)}문항 (이미 {len(done)})", flush=True)
    search_vec(embed(["모델 미리 올리기"])[0], k=1)
    fs = {e: open(o, "a", encoding="utf-8") for e, o in outs.items()}
    for n, q in enumerate(todo, 1):
        t0 = time.perf_counter()
        arts = search_vec(embed([q["question"]])[0], k=0)
        if arm == "only1":
            cands = arts[:10]
        else:
            top, extra = attach_links(arts)
            cands = top + extra
        ts = time.perf_counter() - t0
        user = build(q["question"], cands)
        g = gold[q["qid"]]
        order = ["", effort] if n % 2 else [effort, ""]
        for pos, e in enumerate(order, 1):
            err, picks, answer, usage = "", [], "", {}
            t1 = time.perf_counter()
            try:
                txt, usage = call_llm(user, e)
                try:
                    j = json.loads(txt)
                    picks = [int(x) for x in j.get("근거", []) if str(x).isdigit() and 1 <= int(x) <= len(cands)]
                    answer = j.get("답변", "")
                except (ValueError, TypeError):
                    err = "JSON 형식 오류"
            except RuntimeError as ex:
                err = str(ex)
            tl = time.perf_counter() - t1
            rec = {"qid": q["qid"], "set": q["set"], "유형": q["q_type"], "난이도": q.get("difficulty", ""),
                   "후보": [a["doc_id"] for a in cands], "후보 제목": [a["title"] for a in cands],
                   "검색 1위 정답": arts[0]["doc_id"] in g, "검색 5위 안 정답": any(a["doc_id"] in g for a in arts[:5]),
                   "묶음 정답": any(a["doc_id"] in g for a in cands),
                   "근거": [cands[i - 1]["doc_id"] for i in picks], "근거 번호": picks,
                   "선택 1위 정답": bool(picks) and cands[picks[0] - 1]["doc_id"] in g,
                   "근거 안 정답": any(cands[i - 1]["doc_id"] in g for i in picks),
                   "검색 초": round(ts, 3), "LLM 초": round(tl, 3), "전체 초": round(ts + tl, 3), "호출 순서": pos,
                   "입력 글자": len(SYSTEM) + len(user), "토큰": usage, "답변": answer, "오류": err}
            fs[e].write(json.dumps(rec, ensure_ascii=False) + "\n")
            fs[e].flush()
        if n % 25 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}", flush=True)
    for f in fs.values():
        f.close()
    # 다 잰 test 세트 잠금 (검색 점수 + 질문·정답 지문)
    rows = {r["qid"]: r for r in map(json.loads, open(outs[""], encoding="utf-8"))}
    for s in sets:
        ids = [q for q in qs if q["set"] == s]
        if not s.startswith("test") or any(q["qid"] not in rows for q in ids):
            continue
        items = [(q["qid"], q["question"], sorted(gold[q["qid"]])) for q in ids]
        lock[s] = {"measured_on": time.strftime("%Y%m%d"), "run": f"select_{names['']} + select_{names[effort]}",
                   "Hit@1": round(sum(rows[q["qid"]]["검색 1위 정답"] for q in ids) / len(ids), 3),
                   "Hit@5": round(sum(rows[q["qid"]]["검색 5위 안 정답"] for q in ids) / len(ids), 3), "n": len(ids),
                   "set_sha256": hashlib.sha256(json.dumps(items, ensure_ascii=False).encode()).hexdigest(),
                   "note": "기획안 5-6-4·5-6-7(v0.22): 세트마다 한 번만. LLM 근거 선택(추론 기본·낮춤)을 함께 잼. 이 결과로 검색을 고치지 않는다"}
        LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"🔒 {s} 잠금", flush=True)


def run_arms(effort, sets, tag, limit):
    """기획안 5-6-7 (v0.25): 문항마다 검색 한 번 → ①만(1~10위)과 ①+③(1~5위 + 연결 조문)을 연달아 호출 (순서 번갈아). dev 세트 전용"""
    if any(s.startswith("test") for s in sets):
        raise SystemExit("후보 구성 비교는 dev 세트로만 한다 (test 는 최종 확인용)")
    qs = [q for q in csv.DictReader(open(EVAL / "queries.csv", encoding="utf-8-sig")) if q["set"] in sets]
    gold = defaultdict(set)
    for g in csv.DictReader(open(EVAL / "gold.csv", encoding="utf-8-sig")):
        gold[g["qid"]].add(g["article_uid"])
    eff = "" if effort == "default" else effort
    outs = {a: out_path("_".join(x for x in (a, eff, tag) if x), plain=True) for a in ("only1", "both")}
    seen = [({json.loads(l)["qid"] for l in open(o, encoding="utf-8")} if o.exists() else set()) for o in outs.values()]
    done = seen[0] & seen[1]
    todo = [q for q in qs if q["qid"] not in done][: limit or None]
    print(f"{[o.name for o in outs.values()]}: 남은 {len(todo)}문항 (이미 {len(done)})", flush=True)
    search_vec(embed(["모델 미리 올리기"])[0], k=1)
    fs = {a: open(o, "a", encoding="utf-8") for a, o in outs.items()}
    for n, q in enumerate(todo, 1):
        t0 = time.perf_counter()
        arts = search_vec(embed([q["question"]])[0], k=0)
        ts = time.perf_counter() - t0
        t0 = time.perf_counter()
        top, extra = attach_links(arts)
        tk = time.perf_counter() - t0
        cand = {"only1": arts[:10], "both": top + extra}
        g = gold[q["qid"]]
        rank = next((i for i, a in enumerate(arts, 1) if a["doc_id"] in g), 0)
        for pos, arm in enumerate(["only1", "both"] if n % 2 else ["both", "only1"], 1):
            cands = cand[arm]
            user = build(q["question"], cands)
            err, picks, answer, usage = "", [], "", {}
            t1 = time.perf_counter()
            try:
                txt, usage = call_llm(user, effort)
                try:
                    j = json.loads(txt)
                    picks = [int(x) for x in j.get("근거", []) if str(x).isdigit() and 1 <= int(x) <= len(cands)]
                    answer = j.get("답변", "")
                except (ValueError, TypeError):
                    err = "JSON 형식 오류"
            except RuntimeError as ex:
                err = str(ex)
            tl = time.perf_counter() - t1
            tsearch = ts + (tk if arm == "both" else 0)
            rec = {"qid": q["qid"], "set": q["set"], "유형": q["q_type"], "난이도": q.get("difficulty", ""),
                   "후보": [a["doc_id"] for a in cands], "후보 제목": [a["title"] for a in cands], "후보 수": len(cands),
                   "정답 검색 순위": rank, "검색 1위 정답": arts[0]["doc_id"] in g,
                   "검색 5위 안 정답": any(a["doc_id"] in g for a in arts[:5]), "묶음 정답": any(a["doc_id"] in g for a in cands),
                   "근거": [cands[i - 1]["doc_id"] for i in picks], "근거 번호": picks,
                   "선택 1위 정답": bool(picks) and cands[picks[0] - 1]["doc_id"] in g,
                   "근거 안 정답": any(cands[i - 1]["doc_id"] in g for i in picks),
                   "검색 초": round(tsearch, 3), "LLM 초": round(tl, 3), "전체 초": round(tsearch + tl, 3), "호출 순서": pos,
                   "입력 글자": len(SYSTEM) + len(user), "토큰": usage, "답변": answer, "오류": err}
            fs[arm].write(json.dumps(rec, ensure_ascii=False) + "\n")
            fs[arm].flush()
        if n % 25 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}", flush=True)
    for f in fs.values():
        f.close()


CONFIGS = {"A": ("only1", ""), "B": ("union", ""), "C": ("union", "full")}  # 기획안 5-6-9


def run_configs(effort, sets, tag, limit):
    """기획안 5-6-9 과적합 점검: 문항마다 검색 한 번 → A(1~10위·조각 하나) / B(합친 구성·조각 하나) / C(합친 구성·조문 글)를
    연달아 호출 (순서는 문항마다 돌림). dev 세트 전용"""
    if any(s.startswith("test") for s in sets):
        raise SystemExit("과적합 점검은 dev 세트로만 한다")
    qs = [q for q in csv.DictReader(open(EVAL / "queries.csv", encoding="utf-8-sig")) if q["set"] in sets]
    gold = defaultdict(set)
    for g in csv.DictReader(open(EVAL / "gold.csv", encoding="utf-8-sig")):
        gold[g["qid"]].add(g["article_uid"])
    eff = "" if effort == "default" else effort
    outs = {k: out_path("_".join(x for x in (arm, eff, text, tag) if x), plain=True) for k, (arm, text) in CONFIGS.items()}
    seen = [({json.loads(l)["qid"] for l in open(o, encoding="utf-8")} if o.exists() else set()) for o in outs.values()]
    done = set.intersection(*seen)
    todo = [q for q in qs if q["qid"] not in done][: limit or None]
    print(f"{[o.name for o in outs.values()]}: 남은 {len(todo)}문항 (이미 {len(done)})", flush=True)
    search_vec(embed(["모델 미리 올리기"])[0], k=1)
    fs = {k: open(o, "a", encoding="utf-8") for k, o in outs.items()}
    keys = list(CONFIGS)
    for n, q in enumerate(todo, 1):
        t0 = time.perf_counter()
        arts = search_vec(embed([q["question"]])[0], k=0)
        ts = time.perf_counter() - t0
        g = gold[q["qid"]]
        for pos, k in enumerate(keys[n % 3:] + keys[:n % 3], 1):
            arm, text = CONFIGS[k]
            t0 = time.perf_counter()
            cands = candidates(arm, arts)
            if text == "full":
                cands = with_full_text(cands)
            tc = time.perf_counter() - t0
            user = build(q["question"], cands)
            err, picks, answer, usage = "", [], "", {}
            t1 = time.perf_counter()
            try:
                txt, usage = call_llm(user, effort)
                try:
                    j = json.loads(txt)
                    picks = [int(x) for x in j.get("근거", []) if str(x).isdigit() and 1 <= int(x) <= len(cands)]
                    answer = j.get("답변", "")
                except (ValueError, TypeError):
                    err = "JSON 형식 오류"
            except RuntimeError as ex:
                err = str(ex)
            tl = time.perf_counter() - t1
            rec = {"qid": q["qid"], "set": q["set"], "유형": q["q_type"], "난이도": q.get("difficulty", ""), "구성": k,
                   "후보": [a["doc_id"] for a in cands], "후보 제목": [a["title"] for a in cands],
                   "검색 순위": [a["doc_id"] for a in arts],
                   "정답 검색 순위": next((i for i, a in enumerate(arts, 1) if a["doc_id"] in g), 0),
                   "검색 1위 정답": arts[0]["doc_id"] in g, "검색 5위 안 정답": any(a["doc_id"] in g for a in arts[:5]),
                   "묶음 정답": any(a["doc_id"] in g for a in cands),
                   "근거": [cands[i - 1]["doc_id"] for i in picks], "근거 번호": picks,
                   "선택 1위 정답": bool(picks) and cands[picks[0] - 1]["doc_id"] in g,
                   "근거 안 정답": any(cands[i - 1]["doc_id"] in g for i in picks),
                   "검색 초": round(ts + tc, 3), "LLM 초": round(tl, 3), "전체 초": round(ts + tc + tl, 3), "호출 순서": pos,
                   "입력 글자": len(SYSTEM) + len(user), "토큰": usage, "답변": answer, "오류": err,
                   "글 방식": [a.get("text_mode", "최고점 조각") for a in cands], "후보 글자": [len(a["text"]) for a in cands]}
            fs[k].write(json.dumps(rec, ensure_ascii=False) + "\n")
            fs[k].flush()
        if n % 25 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}", flush=True)
    for f in fs.values():
        f.close()


def compare(a, b):
    """같은 문항을 두 설정으로 잰 결과를 문항 단위로 비교 (a = 기준, b = 바꾼 것)"""
    A = {r["qid"]: r for r in map(json.loads, open(out_path(a), encoding="utf-8"))}
    B = {r["qid"]: r for r in map(json.loads, open(out_path(b), encoding="utf-8"))}
    ids = [q for q in A if q in B]
    reason = lambda r: (r["토큰"].get("completion_tokens_details") or {}).get("reasoning_tokens", 0)

    def stat(rs):
        n, ts = len(rs), sorted(r["전체 초"] for r in rs)
        return {"문항": n, "1위 정답률": round(sum(r["선택 1위 정답"] for r in rs) / n, 3),
                "근거 안 정답률": round(sum(r["근거 안 정답"] for r in rs) / n, 3),
                "평균 초": round(sum(ts) / n, 2), "95% 초": ts[max(int(n * 0.95) - 1, 0)], "최대 초": ts[-1],
                "10초 넘음": sum(t > 10 for t in ts), "평균 추론 토큰": round(sum(map(reason, rs)) / n),
                "평균 출력 토큰": round(sum(r["토큰"].get("completion_tokens", 0) for r in rs) / n),
                "빈 근거": sum(not r["근거"] for r in rs), "오류": sum(bool(r["오류"]) for r in rs)}

    res = {"문항": len(ids), a: stat([A[q] for q in ids]), b: stat([B[q] for q in ids]), "유형별": {}}
    for t in sorted({A[q]["유형"] for q in ids}):
        sub = [q for q in ids if A[q]["유형"] == t]
        res["유형별"][t] = {"문항": len(sub)} | {f"{n} {k}": round(sum(X[q][k] for q in sub) / len(sub), 3)
                                              for n, X in ((a, A), (b, B)) for k in ("선택 1위 정답", "근거 안 정답")}
    for k in ("선택 1위 정답", "근거 안 정답"):
        res[f"{k}: 맞음 → 틀림"] = [q for q in ids if A[q][k] and not B[q][k]]
        res[f"{k}: 틀림 → 맞음"] = [q for q in ids if not A[q][k] and B[q][k]]
    res["첫 근거가 달라진 문항"] = sum(A[q]["근거"][:1] != B[q]["근거"][:1] for q in ids)
    json.dump(res, open(EVAL / f"summary_select_{a}_vs_{b}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--effort", default=EFFORT, help=f"LLM 추론 강도 (기본 {EFFORT} = 채택값, default = 모델 기본값)")
    ap.add_argument("--same-as", default="", help="이 실행 이름에서 이미 잰 문항만 다시 잼")
    ap.add_argument("--tag", default="", help="결과 파일 이름 뒤에 붙일 말 (같은 설정을 다시 잴 때, 예: run2)")
    ap.add_argument("--pair", default="", metavar="강도", help="문항마다 추론 기본값과 이 강도를 연달아 잼 (--sets, --tag 와 함께)")
    ap.add_argument("--pair-arms", action="store_true", help="문항마다 ①만과 ①+③ 두 후보 구성을 연달아 잼 (dev 전용, --sets, --tag 와 함께)")
    ap.add_argument("--configs", action="store_true", help="과적합 점검(기획안 5-6-9): 문항마다 A/B/C 세 구성을 연달아 잼 (dev 전용)")
    ap.add_argument("--sets", nargs="+", default=DEV)
    ap.add_argument("--prompt", default=None, help="지시문 (retriever/answer.py SYSTEMS 의 키, 기본 = 지금 쓰는 것)")
    ap.add_argument("--text", choices=["", "full"], default="", help="full = 여러 조각 조문은 전체/걸린 조각 전부를 넘김 (기획안 5-6-8)")
    ap.add_argument("--compare", nargs=2, metavar=("기준", "바꾼것"))
    a = ap.parse_args()
    if a.configs:
        run_configs(a.effort, a.sets, a.tag, a.limit)
    elif a.pair_arms:
        run_arms(a.effort, a.sets, a.tag, a.limit)
    elif a.pair:
        run_pair(a.arm, a.pair, a.sets, a.tag, a.limit)
    elif a.compare:
        compare(*a.compare)
    elif a.report:
        report()
    else:
        run(a.arm, a.limit, a.effort, a.same_as, a.tag, a.sets, a.text, a.prompt)
