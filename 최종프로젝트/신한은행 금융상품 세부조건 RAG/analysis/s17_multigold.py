"""5-5 검색 평가셋 보완 — 같은 내용을 담은 '다른 문서·섹션'도 정답으로 표시 (정답 여러 개)

사용: python analysis/s17_multigold.py [split ...]    # 기본 dev·test → data/eval/retrieval_{split}.jsonl 에 gold_alt 추가 (test2도 가능)
문제: 정답이 '질문을 만든 섹션 하나'뿐이라, 같은 상품의 특약·공통 약관·다른 섹션에 같은 답이 있어도 검색 지표에서 오답으로 셈
방법 (정보검색 평가의 pooling 방식)
  1. 후보 모으기: 질문마다 vec·kw·routed 세 방식의 상위 10개를 합쳐(중복 제거, 최대 20개) 후보로 — 재순위 같은 LLM 단계는 쓰지 않음
     (어느 한 방식에 유리하지 않게, 실제로 검색될 만한 것들만 판정)
  2. 판정: CHAT_MODEL이 [질문·정답·정답 근거]와 후보들을 보고 "이 후보 하나만 보고 정답과 같은 답(핵심 숫자·조건 포함)을 할 수 있는" 후보만 고름.
     일부만 있거나 다른 상품·다른 기간 상품이면 제외. 원래 정답 후보는 판정에서 빼고 따로 둠
  3. 저장: gold_alt = [{"doc_id", "section_seqs"} | {"clause_id"} | {"term"}] — 청크 단위
     (청크 = 섹션 몇 개 묶음이라 청크 규칙을 바꾸면 다시 표시해야 함)
- 원래 정답(gold)은 그대로 두고 지표는 '엄격(gold만)'과 '보완(gold + gold_alt)' 둘 다 냄
- test에도 표시만 함 (방식 고르기에는 쓰지 않음, 마지막 한 번 채점용)
결과 판정 기록: data/eval/multigold_judgments.jsonl
"""
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import s12_retrieval as R  # noqa: E402
from s8_extract import load_env  # noqa: E402

EVAL = ROOT / "data" / "eval"
LOG = EVAL / "multigold_judgments.jsonl"
INSTR = """너는 은행 상품 문서 검색 평가셋의 정답 표시자다. [질문]과 [정답](짧은 답과 근거 원문)을 보고,
아래 후보 문서 조각 중 **그 조각 하나만 읽고도 정답과 같은 답을 할 수 있는 것**의 번호만 ids에 넣는다.
- 핵심 숫자·조건·대상이 정답과 같아야 함 (표현은 달라도 됨)
- 정답의 일부만 있거나, 관련 주제지만 답이 없으면 넣지 않음
- 질문한 상품과 다른 상품(이름이 비슷한 다른 기간·회차 상품 포함)의 문서면 넣지 않음. 단 질문 상품에 적용되는 공통 약관·공통 조항·용어 설명은 넣을 수 있음
- 확실하지 않으면 넣지 않음. 해당하는 것이 없으면 빈 목록"""


class Pick(BaseModel):
    ids: list[int]
    note: str


def item_key(it):
    if it["kind"] == "chunk":
        return {"doc_id": it["doc_id"], "section_seqs": sorted(it["seqs"])}
    if it["kind"] == "clause":
        return {"clause_id": it["clause_id"]}
    return {"term": it["term"]}


def main():
    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
    idx = R.Index()
    done = {}
    if LOG.exists():
        for l in open(LOG):
            r = json.loads(l)
            done[r["key"]] = r
    splits = [a for a in sys.argv[1:]] or ["dev", "test"]
    for split in splits:
        path = EVAL / f"retrieval_{split}.jsonl"
        qs = [json.loads(l) for l in open(path)]
        qv = R.embed_questions(qs, idx)

        def job(arg):
            q, v = arg
            pool = []
            for m in ("vec", "kw", "routed"):
                for i in idx.rank(m, q["question"], v)[:10]:
                    i = int(i)
                    if i not in pool and not idx.is_gold(i, q["gold"]):
                        pool.append(i)
            pool = pool[:20]
            key = hashlib.sha256(f"{env['CHAT_MODEL']}|{q['qid']}|{q['question']}|{pool}".encode()).hexdigest()
            if key in done:
                return q, done[key], False
            body = "\n\n".join(f"<{n}>\n{idx.items[i]['text'][:1500]}" for n, i in enumerate(pool))
            r = client.responses.parse(model=env["CHAT_MODEL"], instructions=INSTR, text_format=Pick,
                                       input=f"[질문] {q['question']}\n[정답] {q['answer']}\n[정답 근거 원문] {q['evidence']}\n\n[후보]\n{body}")
            picked = [pool[n] for n in dict.fromkeys(r.output_parsed.ids) if 0 <= n < len(pool)]
            rec = {"key": key, "qid": q["qid"], "split": split, "pool": [item_key(idx.items[i]) for i in pool],
                   "alt": [item_key(idx.items[i]) for i in picked], "alt_heads": [idx.items[i]["text"].split("\n", 1)[0] for i in picked],
                   "note": r.output_parsed.note}
            return q, rec, True

        with ThreadPoolExecutor(6) as ex, open(LOG, "a") as f:
            results = list(ex.map(job, zip(qs, qv)))
            for q, rec, new in results:
                if new:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    done[rec["key"]] = rec
        out = []
        for q, rec, _ in results:
            q["gold_alt"] = rec["alt"]
            out.append(q)
        with open(path, "w") as f:
            for q in out:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")
        n_alt = [len(q["gold_alt"]) for q in out]
        print(f"[{split}] 질문 {len(out)} · 추가 정답이 생긴 질문 {sum(1 for x in n_alt if x)} · 추가 정답 합 {sum(n_alt)} "
              f"(질문당 최대 {max(n_alt)})")


if __name__ == "__main__":
    main()
