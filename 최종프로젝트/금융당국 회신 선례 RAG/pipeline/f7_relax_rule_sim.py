"""F7 키워드 완화 적용 규칙 비교 (시뮬레이션, API 호출 없음): f6에서 만든 답변·심사를 조합해 규칙별 수치를 낸다. 개발셋(합성 150, dev1~4)과 답변불가 20만.
규칙
- A 끔: 대조 실행(`_f6c_*`, 상위 5만) 답변
- B 항상 켬: 키워드 완화 실행(`_f6_*`, 상위 5 + 넓혀 찾은 사례) 답변
- C 못 찾았을 때만: A 답변이 "찾지 못했습니다"(또는 인용 없음)면 B 답변으로 바꾼다. 두 번째 호출의 입력(질문 + 상위 5 + 넓혀 찾은 사례)은 B와 똑같으므로 B 답변을 그대로 쓴다
  (실제로 다시 호출하면 LLM 답변이 조금 다를 수 있다. 이 시뮬레이션의 한계)
- 참고: f4 기존 답변(같은 문맥, 이전 시점)도 함께 보여 준다
사용: python f7_relax_rule_sim.py     결과: data/eval/f7_relax_rule_sim.json
"""
import json
import sys
from collections import Counter
from math import comb

from common import DATA

EVAL = DATA / "eval"
sys.stdout.reconfigure(encoding="utf-8")
DEV = ("rewrite_150", "dev1", "dev2", "dev3", "dev4")


def rd(name):
    return {(r["set"], r["qid"]): r for r in map(json.loads, open(EVAL / name, encoding="utf-8"))}


def sign_p(b, w):
    n = b + w
    return 1.0 if not n else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, w) + 1)) / 2 ** n)


def nf(a):
    return a["check"]["not_found"] or not a["check"]["cited"]


def main():
    s1 = {(r["set"], r["qid"]): r for r in map(json.loads, open(EVAL / "_f4_stage1.jsonl", encoding="utf-8")) if r["set"] in DEV + ("unanswerable_20",)}
    f6 = json.load(open(EVAL / "f6_keyword_relax_eval.json", encoding="utf-8"))
    extra = {(i["set"], i["qid"]): {e["doc_id"] for e in i["extra"]} for i in f6["items"]}
    src = {"f4 기존": (rd("_f4_answers.jsonl"), rd("_f4_judge2.jsonl")), "A 끔": (rd("_f6c_answers.jsonl"), rd("_f6c_judge.jsonl")), "B 항상 켬": (rd("_f6_answers.jsonl"), rd("_f6_judge.jsonl"))}
    aA, jA = src["A 끔"]
    aB, jB = src["B 항상 켬"]
    switched = {k for k in s1 if nf(aA[k])}
    src["C 못 찾을 때만"] = ({k: aB[k] if k in switched else aA[k] for k in s1}, {k: jB[k] if k in switched else jA[k] for k in jA})

    gold = [k for k in s1 if s1[k]["gold_key"]]
    unans = [k for k in s1 if k[0] == "unanswerable_20"]
    in5 = lambda k: bool(s1[k]["rank"] and s1[k]["rank"] <= 5)
    out = {"switched": len(switched), "switched_items": sorted(f"{a}/{b}" for a, b in switched if a != "unanswerable_20"), "rules": {}}
    for name, (a, j) in src.items():
        c = Counter(j[k]["correctness"] for k in gold)
        g = Counter(j[k]["grounded"] for k in gold if j[k]["correctness"] != "답변없음")
        miss = [k for k in gold if not in5(k)]
        calls_kw = 0 if name in ("f4 기존", "A 끔") else (len(s1) if name == "B 항상 켬" else len(switched))
        out["rules"][name] = {
            "정확도 일치": round(c["일치"] / len(gold), 4), "일치+부분일치": round((c["일치"] + c["부분일치"]) / len(gold), 4), "정확도 분포": dict(c),
            "충실도 충실": round(g["충실"] / sum(g.values()), 4), "충실도 분포": dict(g),
            "답변없음(정답 있는 410)": c["답변없음"],
            "상위 5 밖 8문항 일치/부분/답변없음": [sum(1 for k in miss if j[k]["correctness"] == x) for x in ("일치", "부분일치", "답변없음")],
            "답변불가 20 거절": sum(1 for k in unans if nf(a[k])),
            "근거 번호 오류": sum(len(a[k]["check"]["invalid"]) for k in s1), "단정 표현": sum(a[k]["check"]["assertive"] for k in s1),
            "근거 없는 문장이 있는 답변(410)": sum(1 for k in gold if a[k]["check"]["no_citation_sentences"] > 0),
            "넓혀 찾은 사례를 인용한 답변": sum(1 for k in gold if set(a[k]["check"]["cited"]) & extra.get(k, set())),
            "질문 430개당 추가 LLM 호출(키워드 추출 + 재답변)": calls_kw * (1 if name == "B 항상 켬" else 2),
        }
    R = {"일치": 3, "부분일치": 2, "불일치": 1, "답변없음": 0}
    out["paired_vs_A"] = {}
    for name in ("B 항상 켬", "C 못 찾을 때만"):
        j = src[name][1]
        b = sum(1 for k in gold if R[j[k]["correctness"]] > R[jA[k]["correctness"]])
        w = sum(1 for k in gold if R[j[k]["correctness"]] < R[jA[k]["correctness"]])
        gb = sum(1 for k in gold if jA[k]["grounded"] != "충실" and j[k]["grounded"] == "충실")
        gw = sum(1 for k in gold if jA[k]["grounded"] == "충실" and j[k]["grounded"] != "충실")
        out["paired_vs_A"][name] = {"정확도 좋아짐/나빠짐": [b, w], "p": round(sign_p(b, w), 4), "충실도 좋아짐/나빠짐": [gb, gw], "p_충실": round(sign_p(gb, gw), 4)}
    out["C_switched_detail"] = [{"item": f"{k[0]}/{k[1]}", "rank": s1[k]["rank"], "A": jA[k]["correctness"], "C": jB[k]["correctness"], "C_grounded": jB[k]["grounded"],
                                 "gold_in_extra": s1[k]["gold_key"] in extra.get(k, set()) if s1[k]["gold_key"] else None} for k in sorted(switched) if k in jA]
    json.dump(out, open(EVAL / "f7_relax_rule_sim.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
