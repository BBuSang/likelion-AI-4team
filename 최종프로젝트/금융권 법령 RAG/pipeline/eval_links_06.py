"""기획안 5-6-6 개선 ③ 측정: ⓐ 1~5위 / ⓑ 1~10위 / ⓒ 1~5위 + 연결 (dev 세트만, test는 쓰지 않음)

지표: Hit@1·Hit@5(ⓒ는 순위가 같으므로 ⓐ와 같아야 함), 근거 묶음 Hit(정답이 LLM에 넘길 묶음 안에 있나),
      문항당 조문 수·글자 수(답변 시간 대용), 검색 시간
    python pipeline/eval_links_06.py
"""
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retriever.search import attach_links, embed, search_vec, search_linked  # noqa: E402

EVAL = ROOT / "eval"
DEV = ["dev1", "dev2", "dev3", "dev4", "dev5", "dev6"]
EXAMPLES = ["은행이 고객에게 금융상품을 판매할 때 설명해야 하는 의무는 어떤 법에 있어?",
            "금융회사가 고객 신용정보를 제3자에게 제공하려면 어떤 조건이 필요해?",
            "전자금융사고가 나면 손해배상 책임은 누가 져?", "예금자보호 한도는 1인당 얼마야?",
            "대부업자가 받을 수 있는 최고 이자율은?", "고객 금융거래 정보를 수사기관에 줄 수 있는 경우는?",
            "자금세탁이 의심되는 거래는 어디에 언제까지 보고해야 해?", "보험 청약을 철회할 수 있는 기간은?",
            "개인정보가 유출되면 정보주체에게 언제 알려야 해?", "가상자산사업자는 이용자 예치금을 어떻게 보관해야 해?",
            "은행이 대주주에게 해 줄 수 있는 신용공여 한도는?", "신용카드 회원을 모집할 때 금지되는 행위는?"]


def main():
    qs = [q for q in csv.DictReader(open(EVAL / "queries.csv", encoding="utf-8-sig")) if q["set"] in DEV]
    gold = defaultdict(set)
    for g in csv.DictReader(open(EVAL / "gold.csv", encoding="utf-8-sig")):
        gold[g["qid"]].add(g["article_uid"])
    vs = []
    for i in range(0, len(qs), 32):
        vs.extend(embed([q["question"] for q in qs[i:i + 32]]))
    rows, t_link = [], 0.0
    for q, v in zip(qs, vs):
        arts = search_vec(v, k=0)
        t0 = time.perf_counter()
        top, extra = attach_links(arts)
        t_link += time.perf_counter() - t0
        g = gold[q["qid"]]
        rank = next((a["rank"] for a in arts if a["doc_id"] in g), None)
        a_ids = {a["doc_id"] for a in arts[:5]}
        b_ids = {a["doc_id"] for a in arts[:10]}
        c_ids = a_ids | {e["doc_id"] for e in extra}
        assert [a["doc_id"] for a in top] == [a["doc_id"] for a in arts[:5]]  # 순위 그대로
        chars = lambda ids: sum(len(a["text"]) for a in arts if a["doc_id"] in ids)
        hit_e = [e for e in extra if e["doc_id"] in g]
        rows.append({"qid": q["qid"], "세트": q["set"], "유형": q["q_type"], "난이도": q.get("difficulty", ""),
                     "정답 순위": rank or ">", "Hit@1": rank == 1, "Hit@5": bool(rank and rank <= 5),
                     "ⓐ 묶음": bool(a_ids & g), "ⓑ 묶음": bool(b_ids & g), "ⓒ 묶음": bool(c_ids & g),
                     "ⓐ 조문": len(a_ids), "ⓑ 조문": len(b_ids), "ⓒ 조문": len(c_ids),
                     "ⓐ 글자": chars(a_ids), "ⓑ 글자": chars(b_ids), "ⓒ 글자": chars(c_ids),
                     "붙은 조문": " / ".join(f'{e["title"]}[{e["link_kind"]}]' + (" ✔" if e in hit_e else "") for e in extra)})

    def summ(rs):
        n = len(rs)
        out = {"문항": n}
        for c in ["Hit@1", "Hit@5", "ⓐ 묶음", "ⓑ 묶음", "ⓒ 묶음"]:
            out[c] = round(sum(r[c] for r in rs) / n, 3)
        for c in ["ⓐ 조문", "ⓑ 조문", "ⓒ 조문", "ⓐ 글자", "ⓑ 글자", "ⓒ 글자"]:
            out[c] = round(sum(r[c] for r in rs) / n, 1)
        return out

    res = {"전체": summ(rows)}
    for key in ["세트", "유형", "난이도"]:
        for val in sorted({r[key] for r in rows if r[key]}):
            res[f"{key}:{val}"] = summ([r for r in rows if r[key] == val])
    lost = [r["qid"] for r in rows if r["ⓐ 묶음"] and not r["ⓒ 묶음"]]
    gained = [r["qid"] for r in rows if r["ⓒ 묶음"] and not r["ⓐ 묶음"]]
    res["ⓐ→ⓒ 빠진 문항"] = lost
    res["ⓐ→ⓒ 새로 들어온 문항"] = gained
    res["ⓑ만 잡은 문항"] = [r["qid"] for r in rows if r["ⓑ 묶음"] and not r["ⓒ 묶음"]]
    res["ⓒ만 잡은 문항"] = [r["qid"] for r in rows if r["ⓒ 묶음"] and not r["ⓑ 묶음"]]
    res["연결 붙이기 시간(문항당 ms)"] = round(t_link / len(rows) * 1000, 2)
    res["붙은 조문 0개 문항"] = sum(1 for r in rows if r["ⓒ 조문"] == 5)

    ex = []
    for qx in EXAMPLES:
        top, extra = search_linked(qx)
        ex.append({"질문": qx, "1~5위": [a["title"] for a in top],
                   "함께 볼 조문": [f'{e["title"]} [{e["link_kind"]}, {e["rank"]}위]' for e in extra]})
    res["예시 12문"] = ex

    with open(EVAL / "results_links_dev.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(res, open(EVAL / "summary_links_dev.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k in ["전체", "세트:dev1", "세트:dev2", "세트:dev3", "세트:dev4", "세트:dev5", "세트:dev6"]:
        print(k, res[k])
    for k, v in res.items():
        if k.startswith(("유형", "난이도")):
            print(k, {c: v[c] for c in ["문항", "Hit@5", "ⓐ 묶음", "ⓑ 묶음", "ⓒ 묶음", "ⓐ 글자", "ⓑ 글자", "ⓒ 글자"]})
    for k in ["ⓐ→ⓒ 빠진 문항", "ⓐ→ⓒ 새로 들어온 문항", "ⓑ만 잡은 문항", "ⓒ만 잡은 문항", "연결 붙이기 시간(문항당 ms)", "붙은 조문 0개 문항"]:
        print(k, res[k])


if __name__ == "__main__":
    main()
