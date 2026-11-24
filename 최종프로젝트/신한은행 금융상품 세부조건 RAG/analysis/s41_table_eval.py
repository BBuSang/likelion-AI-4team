"""표 재추출 4단계: 재추출 전후 검색 비교 (dev 세트만 — 잠금 test·final 은 쓰지 않음)
재추출로 섹션 번호·공통 조항 번호가 바뀌어 기존 정답(gold: 섹션 번호·조항 번호)을 새 청크에 그대로 쓸 수 없다.
그래서 전후 모두 같은 '근거 문장 기준 정답'으로 잰다:
  정답 = 질문의 evidence(근거 문장)를 담은 항목 — evidence 4글자 조각의 COVER 이상이 항목 글에 있음 (공백·표 기호 무시)
         상품이 있는 질문은 그 상품에 연결된 문서의 청크와 그 문서들의 공통 조항·용어만 (다른 상품의 같은 문장은 정답 아님)
원래 청크에서는 기존 정답(보완)과 근거 문장 기준 정답이 얼마나 일치하는지도 같이 출력 (판정 방식 점검)
표 질문(s42, q["table"] 있음)은 정답 = 그 칸의 행 이름·열 이름·값(앞 30자)을 모두 담은 항목 (같은 상품 문서 안)
사용: python analysis/s41_table_eval.py [--sets=dev,real_dev] [방식 ...]   — 이 스크립트가 있는 폴더의 data/ 로 잼 (재추출은 data/runs/table_v2 에 복사해 실행)
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s12_retrieval as s12  # noqa: E402

SETS = ["dev", "real_dev", "common_dev"]
COVER = 0.8
STRIP = re.compile(r"[\s|·ㆍ•\-‧∙⋅<>br.,:()「」『』'\"“”‘’]")


def norm(t: str) -> str:
    return STRIP.sub("", t.replace("<br>", ""))


def grams(t: str, n=4):
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def gold_by_evidence(idx, q, item_grams):
    tb = q.get("table")
    keys = [norm(tb["row"]), norm(tb["col"]), norm(tb["value"])[:30]] if tb else None
    ev = grams(norm(q["evidence"]))
    docs = set(idx.links.get(q["product_id"], set())) if q.get("product_id") else set()
    if q["gold"].get("doc_id"):
        docs.add(q["gold"]["doc_id"])
    clauses = set().union(*(idx.doc_clause[d] for d in docs)) if docs else None
    out = set()
    for i, it in enumerate(idx.items):
        if docs and it["kind"] == "chunk" and it["doc_id"] not in docs:
            continue
        if clauses is not None and it["kind"] == "clause" and it["clause_id"] not in clauses:
            continue
        if keys:
            t = norm(it["text"])
            if all(k in t for k in keys):
                out.add(i)
        elif len(ev & item_grams[i]) / len(ev) >= COVER:
            out.add(i)
    return out


def main(methods, sets=SETS):
    idx = s12.Index()
    item_grams = [grams(norm(it["text"])) for it in idx.items]
    print(f"검색 대상 {len(idx.items)}개 ({s12.CH})")
    for sp in sets:
        qs = [json.loads(line) for line in open(s12.EVAL / f"retrieval_{sp}.jsonl")]
        qvs = s12.embed_questions(qs, idx)
        golds = [gold_by_evidence(idx, q, item_grams) for q in qs]
        no_gold = sum(1 for g in golds if not g)
        agree = sum(1 for q, g in zip(qs, golds) if "table" not in q and any(idx.is_gold_any(i, q) for i in g))
        print(f"\n[{sp}] {len(qs)}문항 · 근거 문장을 담은 항목 없음 {no_gold} · 기존 정답(보완)과 겹침 {agree}/{len(qs)}")
        for m in methods:
            per = defaultdict(list)
            for q, qv, g in zip(qs, qvs, golds):
                order = [int(i) for i in idx.rank(m, q["question"], qv)[:10]]
                r = next((k + 1 for k, i in enumerate(order) if i in g), None)
                per[q["type"][0] if q["type"][0] in "AB" else q["type"]].append((q["qid"], r))
                per["전체"].append((q["qid"], r))
            line = []
            for t in [k for k in per if k != "전체"] + ["전체"]:
                rs = [r for _, r in per[t]]
                h1 = sum(r == 1 for r in rs) / len(rs)
                h5 = sum(1 for r in rs if r and r <= 5) / len(rs)
                line.append(f"{t}(n={len(rs)}) H@1 {h1:.3f} H@5 {h5:.3f}")
            print(f"  {m:8} | " + " | ".join(line))
            out = Path(s12.CH).parent / "table_eval"
            out.mkdir(exist_ok=True)
            (out / f"{sp}_{m}.json").write_text(json.dumps(dict(per["전체"]), ensure_ascii=False))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--sets=")]
    sets = next((a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--sets=")), SETS)
    main(args or ["vec", "kw", "hybrid", "routed"], sets)
