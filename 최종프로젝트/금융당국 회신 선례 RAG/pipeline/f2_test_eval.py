"""F2 테스트셋 1회 평가: test1, test2, test3, final_test를 각각 **한 번만** 평가한다. 실제 시스템 경로(`rag.search`: 로컬 KURE-v1 질문 임베딩 + DB 벡터 검색)를 쓴다.

사전 등록 (결과를 보기 전에 정한 것. 점수를 본 뒤에는 바꾸지 않는다)
- 지표: 사례 묶음 단위 Hit@1, Hit@3, Hit@5, Hit@10, MRR@10 (순위는 상위 10개 묶음까지). 주 지표는 Hit@1·Hit@5·MRR, 기획안 목표는 Hit@5 ≥ 0.8
- 세트마다 따로 보고하고 합치지 않는다. 정답은 원본 사례의 묶음 하나뿐이라 같은 주제의 다른 회신을 찾으면 오답이다(알려진 한계, 지표는 바꾸지 않는다)
- 질문의 구분·기간 조건(`filter`)은 쓰지 않는다. 검색 설정은 D3 청크 + KURE-v1 + 벡터 검색, 상위 5개 묶음을 답변에 쓴다
- 이 평가 뒤 검색 설정을 바꾸면 점수는 무효이고 새 세트가 필요하다

안전장치
- 세트마다 한 번만 실행된다. 실행하면 `data/eval/f2_test_lock.json`에 설정 지문과 시각을 남기고, 이미 있으면 거부한다 (--show로 저장된 결과를 볼 수 있다)
- 실행 전 점검: 문항 수·ID·정답 묶음 존재, 다른 모든 세트(dev1~3, 합성, 서로)와 정답 묶음이 겹치지 않음, 테스트 질문이 어떤 임베딩 캐시에도 들어간 적 없음
- 설정 지문: DB 청크(개수·유형·글·벡터 앞부분), 임베딩 모델, 검색 코드(`rag.py`)·이 스크립트·KURE 벡터 파일의 해시. 두 번째 세트를 돌릴 때 지문이 달라졌으면 거부한다
- final_test는 정답 수기 검수 전이라 `--unverified`를 붙여야 하고, 결과에 '검수 전' 표시가 남는다
- 결과 파일에는 질문 글을 넣지 않는다 (문항 ID와 순위만). 질문 글을 출력하지 않는다

사용: python f2_test_eval.py --dry-run                 (사전 점검만, 질문을 임베딩하지 않고 잠금도 만들지 않는다)
      python f2_test_eval.py [--sets test1 test2 test3]  (기본: test1 test2 test3)
      python f2_test_eval.py --sets final_test --unverified
      python f2_test_eval.py --show
"""
import argparse
import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime
from math import sqrt
from statistics import median

import numpy as np

import rag
from common import DATA, OUT, manifest
from d3b_local_embed import key as local_key

EVAL = DATA / "eval"
LOCK, RESULT = EVAL / "f2_test_lock.json", EVAL / "f2_test_result.json"
EXPECT = {"test1": 60, "test2": 50, "test3": 40, "final_test": 100}
DEV_OF = {"test1": "dev1", "test2": "dev2", "test3": "dev3"}
OTHERS = ("rewrite_150", "dev1", "dev2", "dev3", "test1", "test2", "test3", "final_test")
K, TOP_FOR_ANSWER, TARGET_HIT5 = 10, 5, 0.8
sys.stdout.reconfigure(encoding="utf-8")


def load(name):
    with open(EVAL / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def fingerprint(conn):
    """검색 결과에 영향을 주는 모든 것의 지문."""
    db = conn.execute("""select count(*) n, count(distinct embed_model) models, min(embed_model) model, min(vector_dims(embedding)) dim,
                                md5(string_agg(md5(embed_text) || chunk_type, '' order by chunk_id)) text_md5,
                                md5(string_agg(left(embedding::text, 80), '' order by chunk_id)) vec_md5 from interp.chunks""").fetchone()
    types = {r["chunk_type"]: r["n"] for r in conn.execute("select chunk_type, count(*) n from interp.chunks group by 1").fetchall()}
    parts = {"db": {**{k: db[k] for k in ("n", "models", "model", "dim", "text_md5", "vec_md5")}, "types": types},
             "query_model": rag.QUERY_MODEL, "max_seq_length": 512, "k": K, "top_for_answer": TOP_FOR_ANSWER,
             "files": {"rag.py": sha(rag.__file__), "f2_test_eval.py": sha(__file__), "embeddings_d3_kure-v1.npy": sha(OUT / "embeddings_d3_kure-v1.npy")}}
    parts["id"] = hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]
    return parts


def preflight(conn, sets):
    """실행 전 점검. 문제가 있으면 목록을 돌려준다."""
    problems, info = [], {}
    valid = {r["group_id"] for r in conn.execute("select group_id from interp.case_groups").fetchall()}
    gold = {n: [x["gold_group_id"] for x in load(n)] for n in OTHERS}
    for n in sets:
        rows = load(n)
        qids = [r["qid"] for r in rows]
        if len(rows) != EXPECT[n]:
            problems.append(f"{n}: 문항 수 {len(rows)} (기대 {EXPECT[n]})")
        if len(set(qids)) != len(qids):
            problems.append(f"{n}: 문항 ID 중복")
        if any(not (r.get("question") or "").strip() for r in rows):
            problems.append(f"{n}: 빈 질문")
        bad = [g for g in gold[n] if g not in valid]
        if bad:
            problems.append(f"{n}: DB에 없는 정답 묶음 {len(bad)}개")
        for o in OTHERS:
            if o != n and set(gold[n]) & set(gold[o]):
                problems.append(f"{n}: {o}와 정답 묶음 {len(set(gold[n]) & set(gold[o]))}개 겹침")
    # 테스트 질문이 어떤 임베딩 캐시에도 들어간 적 없는지
    lc, oc = sqlite3.connect(OUT / "embedding_cache_local.sqlite"), sqlite3.connect(EVAL / "query_embed_cache.sqlite")
    seen = 0
    for n in sets:
        for r in load(n):
            t = rag.mask_pii(r["question"])
            seen += sum(1 for name in ("nlpai-lab/KURE-v1", "BAAI/bge-m3") if lc.execute("select 1 from cache where hash = ?", (local_key(name, t),)).fetchone())
            seen += bool(oc.execute("select 1 from q where k = ?", (hashlib.sha256((rag.EMBED_MODEL + "\0" + t).encode()).hexdigest(),)).fetchone())
    info["questions_found_in_embedding_caches"] = seen
    if seen:
        problems.append(f"테스트 질문이 임베딩 캐시에 {seen}건 있음 (이전에 임베딩한 적이 있음)")
    try:
        colab = set(json.load(open(OUT / "colab_input.json", encoding="utf-8"))["queries"])
        in_colab = sum(1 for n in sets for r in load(n) if rag.mask_pii(r["question"]) in colab)
        info["questions_found_in_colab_input"] = in_colab
        if in_colab:
            problems.append(f"테스트 질문이 Colab 입력 파일에 {in_colab}건 있음")
    except FileNotFoundError:
        info["questions_found_in_colab_input"] = "파일 없음"
    return problems, info


def evaluate(conn, name, rep):
    rows = load(name)
    out, secs = [], []
    for r in rows:
        t = time.time()
        hits = rag.search(r["question"], K, conn)
        secs.append(time.time() - t)
        groups = [rep[h["doc_id"]] for h in hits]
        rank = groups.index(r["gold_group_id"]) + 1 if r["gold_group_id"] in groups else None
        out.append({"qid": r["qid"], "rank": rank, "tag": r.get("tag"), "gubun": r.get("gubun")})
    n = len(out)
    hit = lambda k: sum(1 for x in out if x["rank"] and x["rank"] <= k)
    m = {"n": n, "hit1": round(hit(1) / n, 4), "hit3": round(hit(3) / n, 4), "hit5": round(hit(5) / n, 4), "hit10": round(hit(10) / n, 4),
         "mrr10": round(sum(1 / x["rank"] for x in out if x["rank"]) / n, 4), "not_in_top10": sum(1 for x in out if not x["rank"]),
         "hit1_ci95": wilson(hit(1), n), "hit5_ci95": wilson(hit(5), n), "meets_hit5_target": hit(5) / n >= TARGET_HIT5, "search_seconds_median": round(median(secs), 3)}

    def group_by(key):
        g = {}
        for x in out:
            g.setdefault(str(x[key]), []).append(x)
        return {k: {"n": len(v), "hit1": round(sum(1 for x in v if x["rank"] == 1) / len(v), 3), "hit5": round(sum(1 for x in v if x["rank"] and x["rank"] <= 5) / len(v), 3)} for k, v in sorted(g.items())}
    return {**m, "by_tag": group_by("tag"), "by_gubun": group_by("gubun"), "ranks": [{"qid": x["qid"], "rank": x["rank"]} for x in out]}


def dev_reference():
    """짝이 되는 개발셋의 KURE-v1 점수 (이미 잰 값)."""
    ref = {}
    try:
        d = json.load(open(EVAL / "d3b_eval.json", encoding="utf-8"))["summary"]["kure-v1"]
        ref["dev1"], ref["dev2"] = d["dev1"], d["dev2"]
    except (FileNotFoundError, KeyError):
        pass
    try:
        ref["dev3"] = json.load(open(EVAL / "dev3_eval.json", encoding="utf-8"))["summary"]["kure-v1"]
    except (FileNotFoundError, KeyError):
        pass
    return {k: {m: v[m] for m in ("n", "hit1", "hit5", "mrr10")} for k, v in ref.items()}


def show():
    if not RESULT.exists():
        print("저장된 테스트 결과가 없습니다.")
        return
    res = json.load(open(RESULT, encoding="utf-8"))
    dev = res.get("dev_reference", {})
    for n, r in res["runs"].items():
        d = dev.get(DEV_OF.get(n, ""), {})
        tail = f" | 짝 개발셋 {DEV_OF[n]}: Hit@1 {d['hit1']} Hit@5 {d['hit5']} MRR {d['mrr10']}" if d else ""
        flag = " [정답 검수 전]" if r.get("labels_unverified") else ""
        print(f"{n}{flag} n={r['n']} Hit@1 {r['hit1']:.3f} {r['hit1_ci95']} Hit@3 {r['hit3']:.3f} Hit@5 {r['hit5']:.3f} {r['hit5_ci95']} Hit@10 {r['hit10']:.3f} MRR@10 {r['mrr10']:.3f} 10위 밖 {r['not_in_top10']}{tail} | 실행 {r['ran_at']} 지문 {r['fingerprint']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["test1", "test2", "test3"], choices=list(EXPECT))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--unverified", action="store_true", help="final_test: 정답 수기 검수 전임을 인정하고 실행")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.show:
        return show()
    if "final_test" in a.sets and not a.unverified and not a.dry_run:
        sys.exit("final_test는 정답 수기 검수 전입니다. 검수 전 점수로 실행하려면 --unverified를 붙이세요 (결과에 '검수 전' 표시가 남습니다).")
    lock = json.load(open(LOCK, encoding="utf-8")) if LOCK.exists() else {}
    done = [n for n in a.sets if n in lock]
    if done and not a.dry_run:
        sys.exit(f"이미 평가한 세트입니다: {done} (실행 {[lock[n]['ran_at'] for n in done]}). 저장된 결과는 --show로 봅니다. 한 번 더 돌리면 테스트셋이 아닙니다.")
    with rag.connect() as conn:
        fp = fingerprint(conn)
        problems, info = preflight(conn, a.sets)
        prev = {lock[n]["fingerprint"] for n in lock}
        if prev and fp["id"] not in prev:
            problems.append(f"설정 지문이 이전 실행({sorted(prev)})과 다릅니다({fp['id']}). 설정이 바뀐 뒤에는 이어서 평가하지 않습니다")
        print(f"설정 지문 {fp['id']} | DB 청크 {fp['db']['n']} {fp['db']['types']} 모델 {fp['db']['model']} {fp['db']['dim']}차원")
        print("사전 점검:", "문제 없음" if not problems else problems, "|", info)
        if a.dry_run:
            print("※ --dry-run: 질문을 임베딩하지 않았고 잠금도 만들지 않았습니다. 평가 대상:", {n: EXPECT[n] for n in a.sets})
            return
        if problems:
            sys.exit("사전 점검에서 문제가 나와 평가하지 않았습니다.")
        rep = {r["representative_case_key"]: r["group_id"] for r in conn.execute("select group_id, representative_case_key from interp.case_groups").fetchall()}
        rag.embed("준비")        # 모델을 미리 불러온다 (테스트 질문이 아닌 글)
        results = json.load(open(RESULT, encoding="utf-8")) if RESULT.exists() else {"protocol": __doc__.split("안전장치")[0].strip(), "runs": {}}
        for n in a.sets:
            r = evaluate(conn, n, rep)
            r.update(ran_at=datetime.now().isoformat(timespec="seconds"), fingerprint=fp["id"], labels_unverified=(n == "final_test"))
            results["runs"][n] = r
            lock[n] = {"ran_at": r["ran_at"], "fingerprint": fp["id"]}
            results["fingerprint_detail"], results["dev_reference"] = fp, dev_reference()
            json.dump(results, open(RESULT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)     # 세트마다 바로 저장하고 잠근다
            json.dump(lock, open(LOCK, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            manifest("f2_test_eval", set=n, n=r["n"], hit1=r["hit1"], hit5=r["hit5"], mrr10=r["mrr10"], fingerprint=fp["id"])
    show()


if __name__ == "__main__":
    main()
