"""H1 사람이 쓴 질문 세트: 질문 양식 → 정답 후보 표(사람이 라벨) → 여러 정답을 인정하는 채점.
배경(81번 검토): 지금까지의 질문은 모두 LLM이 정답 회신을 보고 쓴 것이고 정답이 묶음 하나뿐이다. 이 세트는 코퍼스를 보지 않은 사람이 쓰고, 정답을 여러 개 인정한다.
파일은 모두 data/eval/human/ 아래에 둔다(회신 원문 발췌가 들어가므로 저장소에 올리지 않는다). 작성 안내는 `평가_사람질문_안내.md`.

1) python h1_human_set.py template           → questions.csv (빈 양식. qid, 작성자, 업무, 질문, 메모)
2) python h1_human_set.py candidates         → labels.csv (질문마다 검색 상위 20개 후보. 사람이 '판정' 칸에 정답/관련/무관을 적는다)
     · 후보는 현재 시스템의 벡터 검색에서만 뽑는다. 그래서 시스템이 못 찾는 정답은 후보에 없다(후보 풀 편향).
       라벨하는 사람이 포털에서 직접 찾은 사례는 questions.csv의 '추가 정답' 칸에 사례번호(L123; O45)로 적는다.
     · 이미 판정을 적은 행은 다시 실행해도 지우지 않는다.
3) python h1_human_set.py score [--rerank]   → 개발용 절반(dev)만 채점. 정답으로 적힌 사례 중 하나라도 맞으면 정답.
   python h1_human_set.py score --final      → 시험용 절반(test)을 한 번만 채점하고 잠근다(human_test_lock.json).
     · dev/test는 qid의 해시로 나눈다(작성 순서·내용과 무관). 판정에 '정답'이 하나도 없는 질문은 답변불가 후보로 따로 센다.
     · 채점 기준(미리 고정): Hit@1·Hit@5·MRR@10, 정답은 묶음 단위(같은 묶음의 다른 사례도 인정). '관련'은 정답으로 세지 않고 따로 보고한다.
"""
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
import rag  # noqa: E402
import rerank as rr  # noqa: E402
from common import DATA  # noqa: E402

H = DATA / "eval" / "human"
Q, L, LOCK = H / "questions.csv", H / "labels.csv", H / "human_test_lock.json"
QCOLS = ["qid", "작성자", "업무", "질문", "추가 정답", "메모"]
LCOLS = ["qid", "질문", "순위", "사례번호", "구분", "회신일", "제목", "질의요지(발췌)", "회답(발췌)", "원문", "판정"]
sys.stdout.reconfigure(encoding="utf-8")


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, cols, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def split_of(qid):
    return "test" if int(hashlib.sha256(qid.encode()).hexdigest(), 16) % 2 else "dev"


def template():
    H.mkdir(parents=True, exist_ok=True)
    if Q.exists():
        sys.exit(f"이미 있음: {Q} (지우지 않습니다)")
    write(Q, QCOLS, [{"qid": f"H{i:03d}", "작성자": "", "업무": "", "질문": "", "추가 정답": "", "메모": ""} for i in range(1, 61)])
    print(f"빈 양식 60행: {Q}")


def candidates():
    qs = [q for q in read(Q) if q["질문"].strip()]
    old = {(r["qid"], r["사례번호"]): r["판정"] for r in read(L)} if L.exists() else {}
    rows = []
    with rag.connect() as conn:
        for q in qs:
            cs = rr.candidates(conn, q["질문"])
            info = {r["case_key"]: r for r in conn.execute("select case_key, gubun, reply_date, title, question, answer, reason, url from interp.cases where case_key = any(%s)", ([c["doc_id"] for c in cs],)).fetchall()}
            for i, c in enumerate(cs, 1):
                x = info[c["doc_id"]]
                rows.append({"qid": q["qid"], "질문": q["질문"].strip(), "순위": i, "사례번호": x["case_key"], "구분": x["gubun"], "회신일": x["reply_date"] or "", "제목": x["title"],
                             "질의요지(발췌)": rr._short(x["question"], 400), "회답(발췌)": rr._short(x["answer"] or x["reason"], 400), "원문": x["url"], "판정": old.get((q["qid"], x["case_key"]), "")})
    write(L, LCOLS, rows)
    print(f"질문 {len(qs)}개, 후보 {len(rows)}행: {L}  (이미 적은 판정 {sum(1 for r in rows if r['판정'])}행 유지)")


def score(final, use_rerank):
    part = "test" if final else "dev"
    if final and LOCK.exists():
        sys.exit("시험용 절반은 이미 채점했습니다(human_test_lock.json). 다시 채점하지 않습니다.")
    labels = read(L)
    qs = [q for q in read(Q) if q["질문"].strip() and split_of(q["qid"]) == part]
    unlabeled = [q["qid"] for q in qs if not any(r["qid"] == q["qid"] and r["판정"].strip() for r in labels)]
    if unlabeled:
        sys.exit(f"판정이 비어 있는 질문 {len(unlabeled)}개: {', '.join(unlabeled[:10])} … 먼저 labels.csv를 채워 주세요.")
    res = []
    with rag.connect() as conn:
        grp = {r["case_key"]: r["group_id"] for r in conn.execute("select case_key, group_id from interp.cases where group_id is not null").fetchall()}
        for q in qs:
            gold_keys = {r["사례번호"] for r in labels if r["qid"] == q["qid"] and r["판정"].strip() == "정답"} | {x.strip() for x in q["추가 정답"].replace(",", ";").split(";") if x.strip()}
            rel = {grp[r["사례번호"]] for r in labels if r["qid"] == q["qid"] and r["판정"].strip() == "관련" and r["사례번호"] in grp}
            gold = {grp[k] for k in gold_keys if k in grp}
            cs = rr.candidates(conn, q["질문"])
            if use_rerank:
                order, _, _ = rr.rerank(q["질문"], [c["doc_id"] for c in cs], rr.load_brief(conn, [c["doc_id"] for c in cs]), k=5)
                ranked = [grp[d] for d in order]
            else:
                ranked = [c["group_id"] for c in cs[:10]]
            rank = next((i + 1 for i, g in enumerate(ranked) if g in gold), None)
            res.append({"qid": q["qid"], "n_gold": len(gold), "rank": rank, "related_in_top5": sum(1 for g in ranked[:5] if g in rel)})
    has = [r for r in res if r["n_gold"]]
    n = len(has) or 1
    out = {"part": part, "rerank": use_rerank, "at": datetime.now().isoformat(timespec="seconds"), "n_questions": len(res), "n_with_gold": len(has), "no_gold(답변불가 후보)": len(res) - len(has),
           "hit1": round(sum(1 for r in has if r["rank"] == 1) / n, 4), "hit5": round(sum(1 for r in has if r["rank"] and r["rank"] <= 5) / n, 4),
           "mrr10": round(sum(1 / r["rank"] for r in has if r["rank"]) / n, 4), "avg_gold_per_question": round(sum(r["n_gold"] for r in has) / n, 2), "items": res}
    json.dump(out, open(H / f"human_{part}_result{'_rerank' if use_rerank else ''}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if final:
        json.dump({"at": out["at"], "rerank": use_rerank, "n": len(res)}, open(LOCK, "w", encoding="utf-8"))
    print(json.dumps({k: v for k, v in out.items() if k != "items"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "template":
        template()
    elif cmd == "candidates":
        candidates()
    elif cmd == "score":
        score("--final" in sys.argv, "--rerank" in sys.argv)
    else:
        print(__doc__)
