"""H2 LLM 심사를 사람이 확인하기 위한 표본: 개발셋 답변 50개를 뽑아 사람이 직접 판정하고, LLM 심사와 얼마나 같은지 잰다.
배경(81번 검토): 심사 모델이 답변 모델과 같고, 사람이 확인한 심사는 10건뿐이다. 개발셋(합성, dev1~4)만 쓴다. 시험 세트는 읽지 않는다.
파일은 data/eval/human/ 아래(회신 원문 발췌 포함, 저장소에 올리지 않는다).

1) python h2_judge_sample.py sample    → judge_sheet.csv (사람이 '정확도'·'충실도' 칸을 채운다. LLM 판정은 이 파일에 넣지 않는다), _judge_key.json(LLM 판정, 열어 보지 않는다)
     · 뽑는 방법(미리 고정, seed 20261111): LLM 판정 기준으로 불일치 전부, 부분일치 12, 일부추정(정확도 일치) 13, 나머지는 일치·충실에서 채워 50개
     · 정확도: 일치 / 부분일치 / 불일치 / 답변없음     충실도: 충실 / 일부추정 / 근거없음 / 해당없음  (기준은 f4의 심사 지시문과 같다)
2) python h2_judge_sample.py compare   → 사람 판정과 LLM 판정의 일치율, 어긋난 방향(LLM이 후한지 박한지)
"""
import csv
import json
import random
import sys
from collections import Counter

import f4_rag_answer_eval as f4
import rag
from common import DATA

H = DATA / "eval" / "human"
SHEET, KEYF = H / "judge_sheet.csv", H / "_judge_key.json"
COLS = ["no", "set", "qid", "질문", "시스템 답변", "정답 사례", "정답 사례 내용", "답변이 인용한 사례 내용", "정확도", "충실도", "메모"]
sys.stdout.reconfigure(encoding="utf-8")
key = lambda r: (r["set"], r["qid"])


def sample():
    H.mkdir(parents=True, exist_ok=True)
    if SHEET.exists():
        sys.exit(f"이미 있음: {SHEET} (사람이 채운 판정을 지우지 않습니다)")
    recs = {key(r): r for r in f4.read_jsonl(f4.S1) if r["set"] in f4.DEV_SETS}
    ans = {key(r): r for r in f4.read_jsonl(f4.ANS)}
    jdg = {key(r): r for r in f4.read_jsonl(f4.JDG) if key(r) in recs and r["correctness"] != "답변없음"}
    rnd = random.Random(20261111)
    pool = sorted(jdg)
    pick = [k for k in pool if jdg[k]["correctness"] == "불일치"]
    for cond, n in ((lambda j: j["correctness"] == "부분일치", 12), (lambda j: j["correctness"] == "일치" and j["grounded"] == "일부추정", 13)):
        c = [k for k in pool if cond(jdg[k]) and k not in pick]
        pick += rnd.sample(c, min(n, len(c)))
    rest = [k for k in pool if jdg[k]["correctness"] == "일치" and jdg[k]["grounded"] == "충실"]
    pick += rnd.sample(rest, 50 - len(pick))
    rnd.shuffle(pick)
    questions = {}
    for name in f4.DEV_SETS:
        for l in open(DATA / "eval" / f"{name}.jsonl", encoding="utf-8"):
            x = json.loads(l)
            questions[(name, x["qid"])] = x["question"]
    with rag.connect() as conn:
        golds = {r["case_key"]: r for r in conn.execute("select case_key, title, question, answer, reason from interp.cases where case_key = any(%s)", ([recs[k]["gold_key"] for k in pick],)).fetchall()}
    rows = []
    for i, k in enumerate(pick, 1):
        r, g = recs[k], golds[recs[k]["gold_key"]]
        rows.append({"no": i, "set": k[0], "qid": k[1], "질문": questions[k], "시스템 답변": ans[k]["answer"].replace(rag.CLOSING, "").strip(), "정답 사례": r["gold_key"],
                     "정답 사례 내용": f"제목: {g['title']}\n질의요지: {(g['question'] or '')[:600]}\n회답: {(g['answer'] or '')[:800]}\n이유: {(g['reason'] or '')[:600]}",
                     "답변이 인용한 사례 내용": f4.cited_context(r["context"], ans[k]["check"]["cited"]), "정확도": "", "충실도": "", "메모": ""})
    with open(SHEET, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    json.dump({f"{k[0]}|{k[1]}": {"correctness": jdg[k]["correctness"], "grounded": jdg[k]["grounded"]} for k in pick}, open(KEYF, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"표본 {len(rows)}개: {SHEET}  (LLM 판정은 {KEYF.name}에 따로 저장, 채점 전에는 열지 않는다)")


def compare():
    llm = json.load(open(KEYF, encoding="utf-8"))
    rows = [r for r in csv.DictReader(open(SHEET, encoding="utf-8-sig", newline="")) if r["정확도"].strip()]
    if not rows:
        sys.exit("사람 판정이 아직 없습니다.")
    order = {"correctness": ["답변없음", "불일치", "부분일치", "일치"], "grounded": ["해당없음", "근거없음", "일부추정", "충실"]}
    out = {"n": len(rows)}
    for field, col in (("correctness", "정확도"), ("grounded", "충실도")):
        pairs = [(r[col].strip(), llm[f"{r['set']}|{r['qid']}"][field]) for r in rows if r[col].strip()]
        lv = order[field]
        out[field] = {"n": len(pairs), "agree": sum(1 for h, m in pairs if h == m), "llm_more_lenient": sum(1 for h, m in pairs if h in lv and lv.index(m) > lv.index(h)),
                      "llm_stricter": sum(1 for h, m in pairs if h in lv and lv.index(m) < lv.index(h)), "table(사람→LLM)": {f"{h}→{m}": n for (h, m), n in Counter(pairs).most_common()}}
    json.dump(out, open(H / "judge_agreement.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"sample": sample, "compare": compare}.get(cmd, lambda: print(__doc__))()
