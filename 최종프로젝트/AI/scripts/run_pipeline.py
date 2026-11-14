"""무인 파이프라인 (멘토 안내문 1-5, worklog 28장): 문서 파일 → 새 DB → 탐지·직원 목록·가명처리·적재 → 검색 → 답변 → 결과 파일.
사람 검수 없이 끝까지. 정답 파일이 있으면 채점까지 하고, 없으면 결과만 저장.

    # 채점용 새 DB (개발 DB 5434 와 따로): DB_PORT=5435 CONTAINER_NAME=saebom_eval_db docker compose -p saebom_eval up -d
    DB_PORT=5435 python scripts/run_pipeline.py --docs 문서.jsonl --queries 질의.jsonl --gold 질의정답.jsonl --out outputs/pipeline_<이름>

    --docs     jsonl: 한 줄에 {id, text, doc_type?, entities?}. entities(정답)는 채점에만 쓰고 적재에는 안 씀
    --queries  jsonl: {qid, query}            (없으면 적재까지만)
    --gold     jsonl: {qid, relevant_doc_ids, confusable_doc_ids?, type?, query_entities?}  (없으면 채점 없이 결과만)
    --persons-gold  jsonl: {id, persons:[{start, end, text, person_id, form?}]}  사람 잇기(동명이인) 채점 (없으면 생략)
    --skip-answer   LLM 답변 단계 건너뜀 (비용·시간)
    --reset         DB 가 비어 있지 않아도 지우고 새로 적재

결과 폴더: detect.jsonl, employees.jsonl, doc_meta.jsonl, load_summary.json, search_run.jsonl, answers.jsonl,
          scores.json, report.md
"""
import argparse
import collections
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core import access, config, db  # noqa: E402

DEV_PORT = 5434  # 화면 테스트용 개발 DB — 여기서는 돌리지 않음 (reset 이 데이터를 지우므로)
_CITE = re.compile(r"[\[<]([A-Z]\d{4})[\]>]")
_NOT_FOUND = re.compile(r"찾을\s*수\s*없|찾지\s*못했")
LEAK_LABELS = {"phone_number", "email", "resident_registration_number", "account_number", "card_number", "address"}


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ov(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


# ------------------------------------------------------------------ 1. 탐지 + 적재
def stage_load(docs, out, reset, log):
    from app.pii import pii_pipeline
    from app.pipeline import load
    t0 = time.perf_counter()
    log(f"탐지 {len(docs)}건 (모델 첫 로딩 포함)...")
    spans = {d["id"]: pii_pipeline.detect(d["text"]) for d in docs}
    t_detect = time.perf_counter() - t0
    write_jsonl(out / "detect.jsonl", [{"id": i, "spans": s} for i, s in spans.items()])
    summary, employees, meta = load.load(docs, spans, reset=reset, log=log)
    write_jsonl(out / "employees.jsonl", employees)
    write_jsonl(out / "doc_meta.jsonl", meta)
    (out / "load_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                           encoding="utf-8")
    return spans, {"detect": t_detect, "load": time.perf_counter() - t0 - t_detect}, summary


# ------------------------------------------------------------------ 2. 탐지 채점
def score_detect(gold_docs, spans):
    """라벨별 정확 일치(라벨·시작·끝) P/R/F1 + 가리기 재현율(정답과 겹치는 구간이 있으면 가려진 것)"""
    c = collections.defaultdict(collections.Counter)
    for d in gold_docs:
        g = [{k: e[k] for k in ("label", "start", "end")} for e in d["entities"]]
        p = [{k: s[k] for k in ("label", "start", "end")} for s in spans.get(d["id"], [])]
        for lab in {x["label"] for x in g + p} | {"ALL"}:
            gl = [x for x in g if lab in ("ALL", x["label"])]
            pl = [x for x in p if lab in ("ALL", x["label"])]
            tp = sum(1 for x in pl if x in gl)
            c[lab]["tp"] += tp
            c[lab]["n_pred"] += len(pl)
            c[lab]["n_gold"] += len(gl)
            c[lab]["masked"] += sum(1 for x in gl if any(ov(x, y) for y in p))  # 라벨이 달라도 가려졌으면 가린 것
    res = {}
    for lab, v in c.items():
        P = v["tp"] / v["n_pred"] if v["n_pred"] else None
        R = v["tp"] / v["n_gold"] if v["n_gold"] else None
        res[lab] = {"n_gold": v["n_gold"], "n_pred": v["n_pred"], "precision": P, "recall": R,
                    "f1": 2 * P * R / (P + R) if P and R else 0.0,
                    "mask_recall": v["masked"] / v["n_gold"] if v["n_gold"] else None}
    return res


# ------------------------------------------------------------------ 2b. 사람 잇기 (동명이인) 채점
def score_persons(gold_rows, out):
    """2주차 eval/check_token_consistency.py 와 같은 정의. 예측 = DB 에 실제로 저장된 사람 언급(vault.person_mentions)
    미탐: 정답 이름 구간이 어떤 토큰으로도 안 바뀜 / 쪼개짐: 한 사람이 2개 이상 토큰 / 합쳐짐: 한 토큰에 2명 이상
    (이름이 같으면 동명이인, 다르면 그 외). 사례 파일에는 이름 원문 없이 문서·사람 id·토큰만"""
    with db.vault_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id, start_pos, end_pos, token, basis FROM vault.person_mentions")
        rows = cur.fetchall()
    pred, basis = collections.defaultdict(list), collections.Counter()
    for doc_id, s, e, tok, b in rows:
        pred[doc_id].append({"start": s, "end": e, "token": tok})
        basis[b] += 1
    person_tokens = collections.defaultdict(collections.Counter)
    token_persons = collections.defaultdict(set)
    names, missed, total = {}, [], 0
    for g in gold_rows:
        P = pred.get(g["id"], [])
        for m in g["persons"]:
            total += 1
            names.setdefault(m["person_id"], m["text"].replace(" ", "") if m.get("form") != "given" else None)
            hit = [p for p in P if p["start"] < m["end"] and m["start"] < p["end"]]
            if not hit:
                missed.append((g["id"], m["person_id"]))
            for p in hit:
                person_tokens[m["person_id"]][p["token"]] += 1
                token_persons[p["token"]].add(m["person_id"])
    persons = {m["person_id"] for g in gold_rows for m in g["persons"]}
    split = {p: t for p, t in person_tokens.items() if len(t) > 1}
    merged = {t: ps for t, ps in token_persons.items() if len(ps) > 1}
    homonym = {t: ps for t, ps in merged.items() if len({names.get(p) for p in ps} - {None}) <= 1}
    wrong = {t: ps for t, ps in merged.items() if t not in homonym}
    with open(out / "person_cases.csv", "w", encoding="utf-8-sig") as f:
        f.write("type,doc_id_or_token,person_id,detail\n")
        for doc_id, pid in missed:
            f.write(f"미탐,{doc_id},{pid},\n")
        for pid, toks in split.items():
            f.write(f"쪼개짐,,{pid},{' | '.join(f'{t}x{c}' for t, c in toks.items())}\n")
        for t, ps in merged.items():
            f.write(f"{'합쳐짐(동명이인)' if t in homonym else '합쳐짐(그 외)'},{t},{'|'.join(sorted(ps))},\n")
    return {"persons": len(persons), "mentions": total, "missed": len(missed), "missed_rate": len(missed) / max(total, 1),
            "split": len(split), "split_rate": len(split) / max(len(persons), 1), "merged_homonym": len(homonym),
            "merged_other": len(wrong), "basis": dict(basis.most_common())}


# ------------------------------------------------------------------ 3. 검색
def stage_search(queries, k, log):
    from app.rag.search import search
    run, times = [], []
    conn = db.rag_conn()
    try:
        search(queries[0]["query"], k, conn)  # 첫 로딩은 시간에서 뺌
        for q in queries:
            t0 = time.perf_counter()
            _, _, docs = search(q["query"], k, conn)
            times.append(time.perf_counter() - t0)
            run.append({"qid": q["qid"], "ranked_doc_ids": [d["doc_id"] for d in docs]})
    finally:
        conn.close()
    log(f"검색 {len(queries)}개 완료")
    return run, times


def score_search(gold, run):
    """2주차 채점 함수(실전프로젝트2/eval/eval_holdout_retrieval.score)가 있으면 그것으로, 없으면 같은 지표를 여기서"""
    w2 = ROOT.parents[1] / "실전프로젝트2" / "eval"
    if (w2 / "eval_holdout_retrieval.py").exists():
        sys.path.insert(0, str(w2))
        from eval_holdout_retrieval import score
        rows = score(gold, run)
        return {g: {c: (sum(v) / len(v) if v else None) for c, v in m.items() if c != "n"} | {"n": len(m["n"])}
                for g, m in rows.items()}
    by = {r["qid"]: r["ranked_doc_ids"] for r in run}
    m = collections.defaultdict(list)
    for q in gold:
        ranked, rel = by.get(q["qid"], []), set(q["relevant_doc_ids"])
        m["Hit@1"].append(float(bool(rel & set(ranked[:1]))))
        m["Hit@5"].append(float(bool(rel & set(ranked[:5]))))
        m["Recall@10"].append(len(rel & set(ranked[:10])) / len(rel))
        m["MRR"].append(next((1 / (i + 1) for i, d in enumerate(ranked) if d in rel), 0.0))
    return {"ALL": {c: sum(v) / len(v) for c, v in m.items()} | {"n": len(gold)}}


# ------------------------------------------------------------------ 4. 답변 (검색 → LLM → 출력 재검사)
def stage_answer(queries, k, out, log):
    from app.rag import chat
    path = out / "answers.jsonl"
    done = {r["qid"] for r in read(path)} if path.exists() else set()  # 끊기면 이어서
    with open(path, "a", encoding="utf-8") as f:
        for i, q in enumerate(queries, 1):
            if q["qid"] in done:
                continue
            err, res = None, {}
            for attempt in range(3):
                try:
                    res = chat.answer(q["query"], k=k, user=access.SYSTEM, embed_retries=3)
                    break
                except Exception as e:  # LLM 일시 오류
                    err = f"{type(e).__name__}"
                    time.sleep(5 * (attempt + 1))
            else:
                res = {"error": err}
            f.write(json.dumps({
                "qid": q["qid"], "masked_query": res.get("masked_query"), "query_tokens": res.get("query_tokens"),
                "doc_ids": [d["doc_id"] for d in res.get("docs", [])],
                "llm_input": chat._context(res.get("docs", [])) + "\n\n" + (res.get("masked_query") or ""),
                "raw_answer": res.get("raw_answer"), "answer": res.get("answer"), "check": res.get("check"),
                "ask_back": res.get("ask_back", False), "timing": res.get("timing"), "error": res.get("error"),
                "model": res.get("model")}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                log(f"답변 {i}/{len(queries)}")
    return read(path)


def score_answer(gold, rows, gold_docs):
    """2주차 eval_rag_e2e 와 같은 항목: 근거 적중·인용 적중·인용 정밀도·혼동 인용·못 찾음·되물음 + 유출 검사"""
    g = {q["qid"]: q for q in gold}
    vals = sorted({(e["label"], e["text"]) for d in gold_docs for e in d.get("entities", [])
                   if e["label"] in LEAK_LABELS or (e["label"] == "person" and len(e["text"]) >= 3)})
    m, leaks, times, errors = collections.defaultdict(list), collections.Counter(), collections.defaultdict(list), 0
    output_check = collections.Counter()
    for r in rows:
        if r.get("error"):
            errors += 1
            continue
        for where in ("llm_input", "raw_answer", "answer"):  # 유출: 정답 원문 값이 그대로 나오면 안 됨
            leaks[where] += sum(1 for _, v in vals if v in (r.get(where) or ""))
        chk = r.get("check") or {}
        output_check["unverified_tokens"] += bool(chk.get("unverified_tokens"))
        output_check["raw_pii"] += bool(chk.get("raw_pii"))
        for kk, v in (r.get("timing") or {}).items():
            times[kk].append(v)
        q = g.get(r["qid"])
        if not q:
            continue
        rel, conf = set(q["relevant_doc_ids"]), set(q.get("confusable_doc_ids") or [])
        cited = set(_CITE.findall(r["answer"] or ""))
        m["근거 적중"].append(float(bool(rel & set(r["doc_ids"]))))
        m["되물음"].append(float(r.get("ask_back", False)))
        if r.get("ask_back"):
            continue
        m["인용 적중"].append(float(bool(rel & cited)))
        m["못 찾음"].append(float(bool(_NOT_FOUND.search(r["answer"] or ""))))
        if cited:
            m["인용 정밀도"].append(len(cited & rel) / len(cited))
            if conf:
                m["혼동 인용"].append(len(cited & conf) / len(cited))
    return {"n": len(rows), "errors": errors, **{kk: sum(v) / len(v) for kk, v in m.items() if v},
            "leaks": dict(leaks), "output_check": dict(output_check),
            "avg_time": {kk: sum(v) / len(v) for kk, v in times.items()},
            "model": next((r.get("model") for r in rows if r.get("model")), None)}


# ------------------------------------------------------------------ 리포트
def fmt(v):
    return "-" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)


def report(out, args, times, summary, det, srch, ans, n_docs, n_q, per=None):
    L = [f"# 무인 파이프라인 결과 — {out.name}", "",
         f"- 실행: {time.strftime('%Y-%m-%d %H:%M')} · DB {config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}",
         f"- 문서 {n_docs}건 · 질의 {n_q}개 · 임베딩 {config.EMBED_PROVIDER}/{config.EMBED_MODEL} · 재순위 {config.RERANK}",
         f"- 입력: docs={args.docs} queries={args.queries} gold={args.gold}",
         "- 사람 손: 없음 (검수 없이 탐지 결과 그대로 적재)", "", "## 단계별 시간 (초)", "", "| 단계 | 초 |", "|---|---|"]
    L += [f"| {k} | {v:.1f} |" for k, v in times.items()]
    L += ["", "## 적재", "", "```", json.dumps({k: v for k, v in summary.items() if not isinstance(v, (list, dict))},
                                               ensure_ascii=False, default=str), "```"]
    if det:
        L += ["", "## 탐지 (정답 라벨 대비)", "", "| 라벨 | 정답 | 예측 | 정밀도 | 재현율 | F1 | 가리기 재현율 |", "|---|---|---|---|---|---|---|"]
        for lab in ["ALL"] + sorted(x for x in det if x != "ALL"):
            d = det[lab]
            L.append(f"| {lab} | {d['n_gold']} | {d['n_pred']} | {fmt(d['precision'])} | {fmt(d['recall'])} | "
                     f"{fmt(d['f1'])} | {fmt(d['mask_recall'])} |")
    if per:
        L += ["", "## 사람 잇기 (동명이인, 2주차 토큰 일관성 채점과 같은 정의)", "", "| 항목 | 값 |", "|---|---|",
              f"| 정답 인물 · 이름 등장 | {per['persons']}명 · {per['mentions']}회 |",
              f"| 미탐 (토큰화 안 됨) | {per['missed']}회 ({per['missed_rate']:.1%}) |",
              f"| 쪼개짐 (한 사람이 여러 토큰) | {per['split']}명 ({per['split_rate']:.1%}) |",
              f"| 합쳐짐 - 동명이인 (이름 같은 다른 사람이 한 토큰) | {per['merged_homonym']}개 토큰 |",
              f"| 합쳐짐 - 그 외 (이름 다른 사람이 한 토큰) | {per['merged_other']}개 토큰 |",
              f"| 판단 근거 | {', '.join(f'{k} {v}' for k, v in per['basis'].items())} |"]
    if srch:
        cols = [c for c in ("Hit@1", "Hit@5", "Recall@10", "MRR", "Confuse@10") if c in srch["scores"].get("ALL", {})]
        L += ["", "## 검색", "", "| 유형 | n | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
        for gname in ["ALL"] + sorted(x for x in srch["scores"] if x != "ALL"):
            s = srch["scores"][gname]
            L.append(f"| {gname} | {s['n']} | " + " | ".join(fmt(s.get(c)) for c in cols) + " |")
        L.append(f"\n검색 시간 평균 {srch['latency']['avg']:.3f}s · p95 {srch['latency']['p95']:.3f}s")
    if ans:
        L += ["", f"## 답변 (모델 {ans.get('model')})", "", "| 항목 | 값 |", "|---|---|"]
        for kk in ("근거 적중", "인용 적중", "인용 정밀도", "혼동 인용", "못 찾음", "되물음"):
            L.append(f"| {kk} | {fmt(ans.get(kk))} |")
        L += [f"| 오류 | {ans['errors']}/{ans['n']} |",
              f"| 출력 재검사 (확인 안 된 토큰 / 원문 개인정보 걸린 답변) | {ans['output_check'].get('unverified_tokens', 0)}"
              f" / {ans['output_check'].get('raw_pii', 0)} |",
              f"| **유출** (LLM 입력 / LLM 원 답변 / 최종 답변, 0 이어야 함) | {ans['leaks'].get('llm_input', 0)} / "
              f"{ans['leaks'].get('raw_answer', 0)} / {ans['leaks'].get('answer', 0)} |",
              f"| 평균 시간 | {', '.join(f'{k} {v:.1f}s' for k, v in ans['avg_time'].items())} |"]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--queries")
    ap.add_argument("--gold")
    ap.add_argument("--persons-gold")
    ap.add_argument("--out", required=True)
    ap.add_argument("-k", type=int, default=3, help="답변 LLM 에 넘길 문서 수")
    ap.add_argument("--search-k", type=int, default=10)
    ap.add_argument("--skip-answer", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--allow-dev-db", action="store_true", help="개발 DB(5434)에서도 실행 (데이터가 지워짐)")
    a = ap.parse_args()
    if config.DB_PORT == DEV_PORT and not a.allow_dev_db:
        sys.exit(f"DB_PORT={DEV_PORT} 는 화면 테스트용 개발 DB. 채점용 새 DB 포트(예: DB_PORT=5435)로 실행하세요")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    t_all = time.perf_counter()
    raw = read(a.docs)
    docs = [{"id": r["id"], "text": r["text"], "doc_type": r.get("doc_type")} for r in raw]  # 정답 라벨은 버림
    gold_docs = [r for r in raw if r.get("entities") is not None]
    queries = read(a.queries) if a.queries else []
    gold = read(a.gold) if a.gold else []
    log(f"시작: 문서 {len(docs)} · 질의 {len(queries)} · 정답 {'있음' if gold else '없음'} · DB 포트 {config.DB_PORT}")

    spans, times, summary = stage_load(docs, out, a.reset, log)
    scores = {}
    det = score_detect(gold_docs, spans) if gold_docs else None
    if det:
        scores["detect"] = det
        log(f"탐지: F1 {det['ALL']['f1']:.4f} · 가리기 재현율 {det['ALL']['mask_recall']:.4f}")

    per = None
    if a.persons_gold:
        per = score_persons(read(a.persons_gold), out)
        scores["persons"] = per
        log(f"사람 잇기: 미탐 {per['missed']} · 쪼개짐 {per['split']}명 · 합쳐짐 동명이인 {per['merged_homonym']} · 그 외 {per['merged_other']}")

    srch = ans = None
    if queries:
        t0 = time.perf_counter()
        run, lat = stage_search(queries, a.search_k, log)
        times["search"] = time.perf_counter() - t0
        write_jsonl(out / "search_run.jsonl", run)
        srch = {"latency": {"avg": statistics.mean(lat), "p95": sorted(lat)[max(int(len(lat) * 0.95) - 1, 0)]},
                "scores": score_search(gold, run) if gold else {}}
        scores["search"] = srch
        if gold:
            s = srch["scores"]["ALL"]
            log("검색: " + " · ".join(f"{c} {v:.4f}" for c, v in s.items() if isinstance(v, float)))
        if not a.skip_answer:
            t0 = time.perf_counter()
            rows = stage_answer(queries, a.k, out, log)
            times["answer"] = time.perf_counter() - t0
            ans = score_answer(gold, rows, gold_docs)
            scores["answer"] = ans
            log(f"답변: 인용 적중 {ans.get('인용 적중', 0):.3f} · 못 찾음 {ans.get('못 찾음', 0):.3f} · 유출 {ans['leaks']}")
    times["total"] = time.perf_counter() - t_all
    (out / "scores.json").write_text(json.dumps({"times": times, **scores}, ensure_ascii=False, indent=1), encoding="utf-8")
    report(out, a, times, summary, det, srch if gold else None, ans if gold else None, len(docs), len(queries), per)
    log(f"끝: {times['total']:.0f}초 · 결과 {out}")


if __name__ == "__main__":
    main()
