"""기획안 5-6-7 (v0.27) 답변 정확도: 답변을 정답 조문 전체 글 + 출제 메모와 대조해 LLM 이 채점 (측정용, 문항당 1회)

    python pipeline/eval_answer_06.py --run union_low_set9      # eval/select_<run>.jsonl → eval/answer_<run>.jsonl (이어서 실행 가능)

판정: 정확 / 부분 / 틀림 / 답없음. AI 판정이므로 정확이 아닌 문항은 사람이(또는 AI가 원문과 대조해) 다시 본다.
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval"
ENV = dotenv_values(ROOT / ".env")
CHUNKS = ROOT / "data" / "work" / "04_chunks" / "chunks.csv"
MAX_GOLD = 6000  # 정답 조문 하나에서 채점자에게 보여 줄 최대 글자

SYSTEM = """너는 한국 법령 질의응답의 채점자다. [정답 조문]과 [출제 메모]를 기준으로 [답변]을 채점한다.
- 정확: 질문이 묻는 핵심을 정답 조문 내용과 어긋남 없이 답했다. 표현이 달라도 되고, 정답 조문에 없는 덧붙인 내용은 정답 조문과 모순되지 않으면 감점하지 않는다.
- 부분: 핵심의 일부만 답했거나, 핵심 중 일부가 정답 조문과 다르다.
- 틀림: 핵심이 정답 조문과 다르거나 질문과 다른 내용을 답했다.
- 답없음: 근거가 없어 답할 수 없다고 했다.
출력은 JSON 하나만: {"판정": "정확|부분|틀림|답없음", "이유": "한 문장", "빠진 핵심": "없으면 빈 글", "정답 조문 밖 내용": true/false}"""


def call(user):
    url = ENV["LLM_API_BASE"].rstrip("/") + "/chat/completions"
    body = {"model": ENV["LLM_MODEL"], "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"}}
    for attempt in range(3):
        r = requests.post(url, headers={"Authorization": f"Bearer {ENV['LLM_API_KEY']}"}, json=body, timeout=120)
        if r.status_code == 200:
            return json.loads(r.json()["choices"][0]["message"]["content"])
        if r.status_code in (429, 500, 502, 503) and attempt < 2:
            time.sleep(5 * (attempt + 1))
            continue
        raise RuntimeError(f"LLM 오류 {r.status_code}: {r.text[:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    src, out = EVAL / f"select_{a.run}.jsonl", EVAL / f"answer_{a.run}.jsonl"
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    qs = {q["qid"]: q for q in csv.DictReader(open(EVAL / "queries.csv", encoding="utf-8-sig"))}
    gold = defaultdict(list)
    for g in csv.DictReader(open(EVAL / "gold.csv", encoding="utf-8-sig")):
        gold[g["qid"]].append(g)
    need = {g["article_uid"] for r in rows for g in gold[r["qid"]]}
    csv.field_size_limit(10 ** 9)
    text = defaultdict(list)
    for c in csv.DictReader(open(CHUNKS, encoding="utf-8-sig")):
        if c["article_uid"] in need:
            text[c["article_uid"]].append((int(c["chunk_no"]), c["text"]))
    done = {json.loads(l)["qid"] for l in open(out, encoding="utf-8")} if out.exists() else set()
    todo = [r for r in rows if r["qid"] not in done][: a.limit or None]
    print(f"{out.name}: 남은 {len(todo)}문항 (이미 {len(done)})", flush=True)
    with open(out, "a", encoding="utf-8") as f:
        for n, r in enumerate(todo, 1):
            q = qs[r["qid"]]
            parts = [f"[질문]\n{q['question']}", f"[출제 메모]\n{q.get('note', '')}"]
            for g in gold[r["qid"]]:
                body = "\n".join(t for _, t in sorted(text[g["article_uid"]]))[:MAX_GOLD]
                parts.append(f"[정답 조문] {g['law_name']} {g['article_label']}({g['title']})\n{body}")
            parts.append(f"[답변]\n{r['답변']}")
            try:
                j = call("\n\n".join(parts))
            except (RuntimeError, ValueError) as e:
                j = {"판정": "오류", "이유": str(e)[:200]}
            f.write(json.dumps({"qid": r["qid"], **j}, ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
