"""F3 test4를 LLM 포함 RAG(`rag.answer`)로 **한 번만** 실행한다. 검색 점수(Hit@k, MRR)와 답변 관찰을 같은 실행에서 기록한다.
- 설정은 test1~3·final_test 평가 때와 같아야 한다: `f2_test_eval.fingerprint()`가 이전 실행(`f2_test_lock.json`)의 지문과 다르면 거부한다
- 한 번만 실행: `data/eval/f3_test4_lock.json`이 있으면 거부한다 (--show로 저장된 결과를 본다)
- 실행 전 점검: 문항 수 20, ID 중복·빈 질문, 정답 묶음이 DB에 있고 다른 모든 세트(합성, dev1~4, test1~3, final_test)와 겹치지 않음, 질문이 임베딩 캐시·Colab 입력에 들어간 적 없음
- 이 결과를 보고 설정(검색·프롬프트·코드)을 바꾸지 않는다. 바꾸면 이 점수는 무효이고 새 세트가 필요하다
- 결과 파일에는 질문 글을 넣지 않는다 (문항 ID, 순위, 답변, 검사 결과). 질문은 콘솔에만 나온다
사용: python f3_test4_rag.py --dry-run | python f3_test4_rag.py | python f3_test4_rag.py --show
"""
import json
import sqlite3
import sys
import time
from datetime import datetime
from statistics import median

import rag
import f2_test_eval as f2
from common import DATA, OUT, manifest
from d3b_local_embed import key as local_key

EVAL = DATA / "eval"
LOCK, RESULT = EVAL / "f3_test4_lock.json", EVAL / "f3_test4_result.json"
NAME, N_EXPECT, K = "test4", 20, 10
OTHERS = ("rewrite_150", "dev1", "dev2", "dev3", "dev4", "test1", "test2", "test3", "final_test")
sys.stdout.reconfigure(encoding="utf-8")


def preflight(conn):
    import hashlib
    problems = []
    rows = f2.load(NAME)
    valid = {r["group_id"] for r in conn.execute("select group_id from interp.case_groups").fetchall()}
    if len(rows) != N_EXPECT:
        problems.append(f"문항 수 {len(rows)} (기대 {N_EXPECT})")
    if len({r["qid"] for r in rows}) != len(rows):
        problems.append("문항 ID 중복")
    if any(not (r.get("question") or "").strip() for r in rows):
        problems.append("빈 질문")
    gold = {r["gold_group_id"] for r in rows}
    if gold - valid:
        problems.append(f"DB에 없는 정답 묶음 {len(gold - valid)}개")
    for o in OTHERS:
        ov = gold & {x["gold_group_id"] for x in f2.load(o)}
        if ov:
            problems.append(f"{o}와 정답 묶음 {len(ov)}개 겹침")
    lc, oc = sqlite3.connect(OUT / "embedding_cache_local.sqlite"), sqlite3.connect(EVAL / "query_embed_cache.sqlite")
    seen = 0
    for r in rows:
        t = rag.mask_pii(r["question"])
        seen += sum(1 for name in ("nlpai-lab/KURE-v1", "BAAI/bge-m3") if lc.execute("select 1 from cache where hash = ?", (local_key(name, t),)).fetchone())
        seen += bool(oc.execute("select 1 from q where k = ?", (hashlib.sha256((rag.EMBED_MODEL + "\0" + t).encode()).hexdigest(),)).fetchone())
    colab = set(json.load(open(OUT / "colab_input.json", encoding="utf-8"))["queries"])
    seen_colab = sum(1 for r in rows if rag.mask_pii(r["question"]) in colab)
    if seen or seen_colab:
        problems.append(f"테스트 질문이 임베딩 캐시 {seen}건·Colab 입력 {seen_colab}건에 있음")
    return problems, {"questions_found_in_caches": seen, "questions_found_in_colab_input": seen_colab}


def show():
    if not RESULT.exists():
        print("저장된 결과가 없습니다.")
        return
    r = json.load(open(RESULT, encoding="utf-8"))
    print(json.dumps({k: v for k, v in r.items() if k != "items"}, ensure_ascii=False, indent=1))


def main():
    dry = "--dry-run" in sys.argv
    if "--show" in sys.argv:
        return show()
    if LOCK.exists() and not dry:
        sys.exit(f"이미 실행한 세트입니다 ({json.load(open(LOCK, encoding='utf-8'))}). --show로 저장된 결과를 봅니다. 다시 돌리면 테스트셋이 아닙니다.")
    with rag.connect() as conn:
        fp = f2.fingerprint(conn)
        prev = {v["fingerprint"] for v in json.load(open(f2.LOCK, encoding="utf-8")).values()} if f2.LOCK.exists() else set()
        problems, info = preflight(conn)
        if prev and fp["id"] not in prev:
            problems.append(f"설정 지문 {fp['id']}가 test1~3·final_test 평가 때({sorted(prev)})와 다릅니다. 설정이 바뀌었습니다")
        print(f"설정 지문 {fp['id']} (이전 평가 지문 {sorted(prev)}) | DB 청크 {fp['db']['n']} 모델 {fp['db']['model']} {fp['db']['dim']}차원")
        print("사전 점검:", "문제 없음" if not problems else problems, "|", info)
        if dry:
            print("※ --dry-run: 질문을 임베딩하지 않았고 잠금도 만들지 않았습니다.")
            return
        if problems:
            sys.exit("사전 점검 문제로 실행하지 않았습니다.")
        rep = {r["representative_case_key"]: r["group_id"] for r in conn.execute("select group_id, representative_case_key from interp.case_groups").fetchall()}
        grp_rep = {v: k for k, v in rep.items()}
        rag.embed("준비")
        items, t0 = [], time.time()
        LOCK.write_text(json.dumps({"started_at": datetime.now().isoformat(timespec="seconds"), "fingerprint": fp["id"]}, ensure_ascii=False), encoding="utf-8")   # 실행을 시작하면 바로 잠근다
        for r in f2.load(NAME):
            hits = rag.search(r["question"], K, conn)
            groups = [rep[h["doc_id"]] for h in hits]
            rank = groups.index(r["gold_group_id"]) + 1 if r["gold_group_id"] in groups else None
            gold_key = grp_rep.get(r["gold_group_id"])
            t = time.time()
            a = rag.answer(r["question"])
            sec = time.time() - t
            chk = a["check"]
            items.append({"qid": r["qid"], "tag": r["tag"], "gubun": r["gubun"], "rank": rank, "gold_case": gold_key, "top5": [(h["doc_id"], round(h["score"], 3), h["chunk_type"]) for h in a["hits"]],
                          "answer": a["answer"], "check": chk, "gold_cited": gold_key in chk["cited"], "cited_marks": {k: c["time_mark"] for k, c in a["cases"].items()},
                          "usage": a["usage"], "answer_seconds": round(sec, 1)})
            print("=" * 90)
            print(f"[{r['qid']}] ({r['tag']}, {r['gubun']}) 정답 {gold_key}: {'상위 5 안 ' + str(rank) + '위' if rank and rank <= 5 else ('6~10위 ' + str(rank) + '위' if rank else '10위 밖')}")
            print("Q:", a["question"])
            print("A:", a["answer"].replace("\n", "\n   "))
            print("검사:", {k: chk[k] for k in ("invalid", "no_citation_sentences", "assertive", "not_found")}, "| 인용:", chk["cited"], "| 정답 인용:", gold_key in chk["cited"])
        n = len(items)
        hit = lambda k: sum(1 for x in items if x["rank"] and x["rank"] <= k)
        res = {"set": NAME, "ran_at": datetime.now().isoformat(timespec="seconds"), "fingerprint": fp["id"], "n": n,
               "hit1": round(hit(1) / n, 4), "hit3": round(hit(3) / n, 4), "hit5": round(hit(5) / n, 4), "hit10": round(hit(10) / n, 4),
               "mrr10": round(sum(1 / x["rank"] for x in items if x["rank"]) / n, 4), "not_in_top10": sum(1 for x in items if not x["rank"]),
               "hit1_ci95": f2.wilson(hit(1), n), "hit5_ci95": f2.wilson(hit(5), n),
               "answers": {"invalid_citations": sum(len(x["check"]["invalid"]) for x in items), "assertive_total": sum(x["check"]["assertive"] for x in items),
                           "not_found_answers": sum(x["check"]["not_found"] for x in items), "answers_with_uncited_sentences": sum(1 for x in items if x["check"]["no_citation_sentences"] > 0),
                           "gold_in_top5": sum(1 for x in items if x["rank"] and x["rank"] <= 5), "gold_cited_when_in_top5": sum(1 for x in items if x["rank"] and x["rank"] <= 5 and x["gold_cited"]),
                           "seconds_median": median(x["answer_seconds"] for x in items), "tokens_in": sum(x["usage"]["input"] for x in items), "tokens_out": sum(x["usage"]["output"] for x in items)},
               "items": items}
        json.dump(res, open(RESULT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        manifest("f3_test4_rag", n=n, hit1=res["hit1"], hit5=res["hit5"], mrr10=res["mrr10"], fingerprint=fp["id"], seconds=round(time.time() - t0))
        print("=" * 90)
        print(json.dumps({k: v for k, v in res.items() if k != "items"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
