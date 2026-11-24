"""5-4 질문별 검색 결과 보고서 — dev 질문마다 '무엇을 물었고, 정답은 어디고, 무엇을 찾았는지'를 md로

사용: python analysis/s14_report_retrieval.py [방식=rerank_score]
  rerank       → docs/5-2_질문별_검색결과_dev.md (목록형 재순위, 5-2 때 만든 것)
  rerank_score → docs/6-4_질문별_검색결과_점수형_dev.md (점수형 재순위 = 6-3 확정 방식) + 1위만으로 만든 답의 판정, 1위를 놓친 이유 분류
- 검색: .env의 임베딩 모델(현재 KURE-v1) + 확정 방식(rerank = 상품명 좁히기 + 결합 + CHAT_MODEL 재순위)
- 재순위는 data/eval/rerank_cache.jsonl에 있는 결과를 그대로 씀(새 호출 없음 — 없으면 호출해서 캐시에 추가)
- test 질문은 잠가 둔 시험 세트라 넣지 않음
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import s12_retrieval as R  # noqa: E402
from s8_extract import load_env  # noqa: E402

META, EVAL = ROOT / "data" / "meta", ROOT / "data" / "eval"


def one_line(t: str, n: int) -> str:
    t = " / ".join(x.strip() for x in t.split("\n") if x.strip())
    return t if len(t) <= n else t[:n] + "…"


LAYER = {"상품설명서": "설명서", "간이투자설명서": "설명서", "투자설명서": "설명서", "특약": "상품 특약", "약관": "공통 약관", "약정서": "약정서"}


def miss_kind(idx, top, q, docs, secs):
    """1위가 정답이 아닐 때 1위가 무엇이었나"""
    g, it = q["gold"], idx.items[top]
    if "section_seq" not in g:
        return "다른 공통 조항·용어" if it["kind"] != "chunk" else "공통 조항 대신 문서 청크"
    if it["kind"] != "chunk":
        return "공통 조항·용어"
    if it["doc_id"] == g["doc_id"]:
        return "같은 문서의 다른 섹션"
    if it["doc_id"] in idx.links.get(q["product_id"], set()):
        return f"같은 상품의 다른 문서 ({LAYER.get(docs[g['doc_id']]['doc_type'], '?')} → {LAYER.get(docs[it['doc_id']]['doc_type'], '?')})"
    return "다른 상품 문서"


def main():
    method = sys.argv[1] if len(sys.argv) > 1 else "rerank_score"
    env = load_env()
    docs = {r["doc_id"]: r for r in csv.DictReader(open(META / "documents.csv"))}
    prods = {r["product_id"]: r["name"] for r in csv.DictReader(open(META / "products.csv"))}
    clauses = {int(r["clause_id"]): r for r in csv.DictReader(open(META / "common_clause.csv"))}
    secs = {}
    for p in (ROOT / "data" / "sections").glob("*.jsonl"):
        for l in open(p):
            s = json.loads(l)
            secs[(s["doc_id"], s["seq"])] = s
    qs = [json.loads(l) for l in open(EVAL / "retrieval_dev.jsonl")]
    idx = R.Index()
    qv = R.embed_questions(qs, idx)

    def describe(i):
        it = idx.items[i]
        if it["kind"] == "chunk":
            head = it["text"].split("\n", 1)[0]
            return f"{it['doc_id']} {head}"
        if it["kind"] == "clause":
            return f"공통 조항 #{it['clause_id']} {one_line(clauses[it['clause_id']]['title'], 40)}"
        return f"용어집 '{it['term']}'"

    rows = []
    for q, v in zip(qs, qv):
        order = idx.rank(method, q["question"], v)[:10]
        rank = next((k + 1 for k, i in enumerate(order) if idx.is_gold(i, q["gold"])), None)
        g = q["gold"]
        if "section_seq" in g:
            s = secs[(g["doc_id"], g["section_seq"])]
            where = f"{g['doc_id']} {docs[g['doc_id']]['title']} › {s['section_path']} (p.{s['page_start']})"
        elif "clause_id" in g:
            where = f"공통 조항 #{g['clause_id']} {one_line(clauses[g['clause_id']]['title'], 40)} ({clauses[g['clause_id']]['n_docs']}개 문서 공통)"
        else:
            where = f"용어집 '{g['term']}'"
        top = idx.items[order[0]]
        kind = None if rank == 1 else miss_kind(idx, order[0], q, docs, secs)
        rows.append({"q": q, "rank": rank, "where": where, "top_doc": top.get("doc_id"), "miss": kind, "order0": int(order[0]),
                     "found": [(describe(i), idx.is_gold(i, g)) for i in order[:3]]})

    # 1위만 근거로 준 답변의 판정 (s15 --version=v4 k=1, 근거 1위가 같은 기록)
    ans1 = {}
    for l in open(EVAL / "answer_eval_dev.jsonl"):
        r = json.loads(l)
        if r.get("version") == "v4" and r["k"] == 1:
            ans1[(r["qid"], r["context"][0])] = r
    for r in rows:
        top_head = idx.items[r["order0"]]["text"].split("\n", 1)[0]
        a = ans1.get((r["q"]["qid"], top_head))
        r["ans1"] = a

    groups = defaultdict(list)
    for r in rows:
        groups["1위 정답" if r["rank"] == 1 else "2~5위 정답" if r["rank"] and r["rank"] <= 5 else
               "6~10위 정답" if r["rank"] else "10위 안에 없음"].append(r)
    n = len(rows)
    title = {"rerank": "목록형 재순위", "rerank_score": "점수형 재순위 (6-3 확정)"}.get(method, method)
    out = [f"# 질문별 검색 결과 — dev {n}개, {title} (2026-11-11)", "",
           f"- 검색: 임베딩 `{env['EMBED_MODEL']}` + 상품명으로 좁히기 + 벡터·키워드 결합 + `{env['CHAT_MODEL']}` {title}",
           "- 질문: 검색 평가셋 dev (`data/eval/retrieval_dev.jsonl`). 상품 문서의 한 섹션·공통 조항·용어를 보고 LLM이 영업점 직원 말투로 만든 질문, "
           "**정답 = 질문을 만든 바로 그 섹션** (같은 내용이 다른 문서에 있어도 검색 지표에서는 정답으로 안 셈)",
           "- **1위만으로 답하기**: 검색 1위 하나만 근거로 줘서 답을 만들고 정답과 비교한 판정 (`s15_answer_eval.py dev 1 --version=v4 --retrieval=" + method + "`) "
           "— 1위가 '정답 섹션'이 아니어도 실제로 답이 되는지 보려고",
           "- test 질문은 잠가 둔 시험 세트라 넣지 않음 · 만든 스크립트: `analysis/s14_report_retrieval.py " + method + "`", "",
           "## 요약", "", "| 검색 결과 | 질문 수 | 비율 | 1위만으로 답했을 때 맞음 |", "|---|---|---|---|"]
    for k in ("1위 정답", "2~5위 정답", "6~10위 정답", "10위 안에 없음"):
        rs = groups[k]
        ok = sum(1 for r in rs if r["ans1"] and r["ans1"]["verdict"] == "correct")
        has = sum(1 for r in rs if r["ans1"])
        out.append(f"| {k} | {len(rs)} | {len(rs) / n:.0%} | {ok}/{has} |" if rs else f"| {k} | 0 | 0% | — |")
    tot_ok = sum(1 for r in rows if r["ans1"] and r["ans1"]["verdict"] == "correct")
    out += ["", f"- **Hit@1(검색 지표) {len(groups['1위 정답']) / n:.0%}** vs **1위만으로 답했을 때 맞은 비율 {tot_ok}/{sum(1 for r in rows if r['ans1'])}**", ""]

    miss = [r for r in rows if r["rank"] != 1]
    kinds = defaultdict(list)
    for r in miss:
        kinds[r["miss"]].append(r)
    out += ["## Hit@1이 낮은 이유 — 1위가 정답 섹션이 아닐 때 1위는 무엇이었나", "",
            "| 1위로 나온 것 | 질문 수 | 그 1위만으로 답했을 때 맞음 |", "|---|---|---|"]
    for k, rs in sorted(kinds.items(), key=lambda x: -len(x[1])):
        ok = sum(1 for r in rs if r["ans1"] and r["ans1"]["verdict"] == "correct")
        out.append(f"| {k} | {len(rs)} | {ok}/{sum(1 for r in rs if r['ans1'])} |")
    out += ["",
            "- 한 상품에 **설명서(요약) · 상품 특약(상품 전용 규정) · 공통 약관(여러 상품 공통)** 이 함께 붙어 있고 같은 주제를 겹쳐 적음 → "
            "질문을 만든 문서가 아닌 **다른 층의 문서가 1위**로 오는 경우가 대부분",
            "- 그런 1위도 대부분 **실제로 답이 됨** (위 표 오른쪽) → 검색 지표의 Hit@1은 '정답 섹션 하나만 정답'으로 엄격하게 잰 값이라 실제 쓸모보다 낮게 나옴",
            "- 진짜 문제는 1위만으로 답했을 때 **틀린 것** → 아래 각 질문의 '1위만으로 답' 줄에서 ❌·△ 표시를 보면 됨", "",
            "읽는 법: **정답 위치** = 질문을 만든 원문 섹션, **찾은 것** = 검색 상위 3개 (✅ 정답 섹션이 든 청크), "
            "**1위만으로 답** = 1위 하나만 근거로 만든 답의 판정 (⭕ 맞음 · △ 일부 · ❌ 틀림 · ∅ 답 못함). 청크는 짧은 섹션을 묶은 것이라 머리말에 여러 섹션 이름이 보일 수 있음", ""]
    mark = {"correct": "⭕", "partial": "△", "wrong": "❌", "no_answer": "∅"}
    for k in ("10위 안에 없음", "6~10위 정답", "2~5위 정답", "1위 정답"):
        if not groups[k]:
            continue
        out += [f"## {k} ({len(groups[k])}개)", ""]
        for r in sorted(groups[k], key=lambda r: (r["ans1"]["verdict"] == "correct") if r["ans1"] else True):
            q = r["q"]
            pn = prods.get(q["product_id"], "") if q["product_id"] else ""
            out.append(f"### {q['qid']} {q['question']}")
            out.append(f"- 유형: {q['type']}" + (f" · 상품: {pn}" if pn else "") + (f" · 정답 순위: {r['rank']}위" if r["rank"] else "")
                       + (f" · 1위로 나온 것: {r['miss']}" if r["miss"] else ""))
            out.append(f"- 정답 위치: {r['where']}")
            out.append(f"- 정답(요약): {one_line(q['answer'], 120)}")
            out.append(f"- 근거 원문: {one_line(q['evidence'], 140)}")
            out.append("- 찾은 것:")
            for m, (d, ok) in enumerate(r["found"], 1):
                out.append(f"  {m}. {'✅ ' if ok else ''}{one_line(d, 120)}")
            if r["ans1"]:
                a = r["ans1"]
                out.append(f"- 1위만으로 답 {mark[a['verdict']]} {a['verdict']}: {one_line(a['answer'], 200)}")
                if a["verdict"] != "correct":
                    out.append(f"  - 채점 이유: {one_line(a['reason'], 160)}")
            out.append("")
    path = ROOT / "docs" / ("5-2_질문별_검색결과_dev.md" if method == "rerank" else "6-4_질문별_검색결과_점수형_dev.md")
    if method == "rerank":
        sys.exit("목록형 보고서(5-2)는 이미 있음 — 다시 만들면 1위만으로 답한 판정이 없어 표가 비므로 만들지 않음")
    path.write_text("\n".join(out))
    print(path.name, {k: len(v) for k, v in groups.items()}, "1위만으로 맞음", tot_ok)


if __name__ == "__main__":
    main()
