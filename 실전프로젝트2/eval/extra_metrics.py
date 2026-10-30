"""연구계획서 5장에 있지만 제공 채점 도구가 내지 않는 지표 4가지 (API 호출 없음).

    python -X utf8 eval/extra_metrics.py        (실전프로젝트2 폴더에서, 결과 파일이 있는 상태)

  1. Hit@k · Recall@k  (k = 1, 3, 5, 10)
  2. split률 · merge률 (인물 기준)
  3. 탐지기 재검사: 외부로 나간 텍스트(마스킹 본문 · 가명처리 질의 · 시연 답변)에 탐지기를 다시 돌려 남은 개인정보 후보 확인
  4. 질의 묶음별 점수: 이름 포함 / 번호·주소 포함 / 개인정보 없음 (질의 유형으로 분류)

결과: outputs/extra_metrics.json
"""
import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from starter import config, pii_pipeline  # noqa: E402

OUT = config.OUT_DIR
KS = (1, 3, 5, 10)
RUNS = {  # 이름: (run 파일, 가명처리 결과 파일 또는 None)
    "B0 라벨 마스킹": ("run_E2_B0.jsonl", "pseudo_holdout_E2_B0.jsonl"),
    "B1 정규화 없음": ("run_E2_B1.jsonl", "pseudo_holdout_E2_B1.jsonl"),
    "의미 검색만 (주소 규칙 전)": ("run_semantic.jsonl", None),
    "토큰 필터만 (주소 규칙 전)": ("run_token.jsonl", None),
    "제안 · 공식 (주소 규칙 전)": ("run_hybrid.jsonl", "pseudo_holdout_norm_run1.jsonl"),
    "제안 · 사후 (주소 규칙 후)": ("run_E2_Proposed.jsonl", "pseudo_holdout_E2_Proposed.jsonl"),
    "상한선 (정답 라벨)": ("run_E2_UB.jsonl", "pseudo_holdout_E2_UB.jsonl"),
}


def jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def group_of(q):
    """질의 유형으로 묶음. 정답 개체로 나누면 주소 일부 질의(address_partial)가 개인정보 없음으로 섞여서 유형 기준으로 함."""
    if q["type"] == "semantic_no_pii":
        return "개인정보 없음"
    if q["type"].startswith(("value_", "address_")):
        return "번호·주소 포함"
    return "이름 포함"


def retrieval(gold, run):
    ranked = {r["qid"]: r["ranked_doc_ids"] for r in run}
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    for q in gold:
        rel, top = set(q["relevant_doc_ids"]), ranked.get(q["qid"], [])
        for g in ("전체", group_of(q)):
            for k in KS:
                rows[g][f"Hit@{k}"].append(1.0 if rel & set(top[:k]) else 0.0)
                rows[g][f"Recall@{k}"].append(len(rel & set(top[:k])) / len(rel))
            rows[g]["n"].append(1)
    return {g: {m: (len(v) if m == "n" else round(sum(v) / len(v), 3)) for m, v in ms.items()} for g, ms in rows.items()}


def consistency(persons_gold, pred):
    """check_token_consistency.py 와 같은 기준(정답 이름 구간과 겹치는 예측 토큰)으로 인물 단위 비율."""
    pred = {p["id"]: p["entities"] for p in pred}
    person_tokens, token_persons = collections.defaultdict(set), collections.defaultdict(set)
    persons = set()
    for g in persons_gold:
        for m in g["persons"]:
            persons.add(m["person_id"])
            for p in pred.get(g["id"], []):
                if p["start"] < m["end"] and m["start"] < p["end"]:
                    person_tokens[m["person_id"]].add(p["token"])
                    token_persons[p["token"]].add(m["person_id"])
    split = {p for p, t in person_tokens.items() if len(t) > 1}
    merged = {p for ps in token_persons.values() if len(ps) > 1 for p in ps}
    n = len(persons)
    return {"인물 수": n, "split 인물": len(split), "split률": round(len(split) / n, 3),
            "merge 인물": len(merged), "merge률": round(len(merged) / n, 3)}


def recheck():
    """외부로 나간 텍스트에서 토큰을 지운 뒤 탐지기를 다시 돌림. 탐지된 구간은 사람이 확인할 후보."""
    def scan(text):
        clean = pii_pipeline.TOKEN_PATTERN.sub(lambda m: " " * len(m.group()), text)  # 위치 유지
        return [(s["label"], clean[s["start"]:s["end"]]) for s in pii_pipeline.detect(clean)]

    with open(OUT / "rag_chunks_export.tsv", encoding="utf-8") as f:
        chunks = [line.rstrip("\n").split("\t", 1) for line in f if "\t" in line]
    queries = [(r["qid"], r["masked_query"]) for r in jsonl(OUT / "run_E2_Proposed.jsonl")]
    demo = [(s["no"], s["answer"]) for s in json.load(open(OUT / "demo_scenarios.json", encoding="utf-8"))]
    res = {}
    for name, items in (("마스킹 본문 (임베딩 전송)", [(d, t.replace("\\n", "\n")) for d, t in chunks]),
                        ("가명처리 질의 (임베딩 전송)", queries), ("시연 답변 (LLM 출력)", demo)):
        found = [(i, lab, txt) for i, t in items for lab, txt in scan(t)]
        res[name] = {"텍스트 수": len(items), "탐지 건수": len(found),
                     "사례": [{"id": i, "label": lab, "text": txt} for i, lab, txt in found]}
    return res


def main():
    gold = jsonl(config.DATA_DIR / "holdout_retrieval_queries_gold.jsonl")
    persons_gold = jsonl(config.DATA_DIR / "holdout_persons_gold.jsonl")
    res = {"질의 묶음 개수": dict(collections.Counter(group_of(q) for q in gold)), "검색": {}, "토큰 일관성": {}}
    for name, (run, pseudo) in RUNS.items():
        if (OUT / run).exists():
            res["검색"][name] = retrieval(gold, jsonl(OUT / run))
        if pseudo and (OUT / pseudo).exists():
            res["토큰 일관성"][name] = consistency(persons_gold, jsonl(OUT / pseudo))
    res["탐지기 재검사"] = recheck()
    (OUT / "extra_metrics.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print("질의 묶음:", res["질의 묶음 개수"])
    for name, groups in res["검색"].items():
        print(f"\n[{name}]")
        for g in ("전체", "이름 포함", "번호·주소 포함", "개인정보 없음"):
            m = groups[g]
            print(f"  {g:8} n={m['n']:<3}" + "".join(f"  Hit@{k} {m[f'Hit@{k}']:.3f}" for k in KS)
                  + "".join(f"  R@{k} {m[f'Recall@{k}']:.3f}" for k in KS))
    print()
    for name, c in res["토큰 일관성"].items():
        print(f"{name:20} split {c['split 인물']}명 ({c['split률']:.1%})  merge {c['merge 인물']}명 ({c['merge률']:.1%})")
    print()
    for name, r in res["탐지기 재검사"].items():
        print(f"{name}: {r['텍스트 수']}건 중 탐지 {r['탐지 건수']}건", [f"{s['id']}:{s['label']}:{s['text']}" for s in r["사례"]])
    print(f"\n저장: {OUT / 'extra_metrics.json'}")


if __name__ == "__main__":
    main()
