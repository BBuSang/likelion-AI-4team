"""10-1 common_test의 잘못된 질문 4개만 교체 — 같은 조항으로, 그 조항이 실제로 적용되는 상품의 질문을 다시 만듦

사용: python analysis/s32_common_fix.py   → data/eval/retrieval_common_fix_test.jsonl (한 번만, 있으면 멈춤)
      이어서 질문 임베딩(s10 embed-questions-local) · 새 질문 4개만 보완 정답(s17 common_fix_new) → 합치기(--merge)
왜: 10-1 사람 검수에서 특정 예금 종류(당좌·정액적립식)에만 적용되는 조항을 다른 상품에 붙인 질문 4개가 발견됨
     (CTc006 외화당좌 조항 → 외화보통예금, CTc009 당좌·가계당좌 조항 → 일반 통장, CTc024·CTc042 정액적립식 조항 → 자유적립식 적금)
방법
  - 정답 섹션(조항)은 그대로. 상품은 사람이 그 상품 전용 문서의 상품유형·저축방법을 읽고 조항이 적용되는 종류로 고름 (PICK)
    (LLM 적용 확인만으로 고르게 했더니 외화보통예금·자유적립식 적금을 다시 '적용됨'으로 골라, 10-1과 같은 잘못이 반복됨)
  - 고른 상품도 LLM 적용 확인을 거쳐 기록. 질문 지시문은 s25와 같음
  - 나머지 26개 질문·보완 정답은 그대로
"""
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from s11_evalset import GEN_PROMPT, GenQ  # noqa: E402
from s25_common_set import NO_DOC, SEED  # noqa: E402
from s31_realistic_set import APPLY_PROMPT, Apply  # noqa: E402
from s8_extract import lines_in_order, load_env, ns  # noqa: E402

META, EVAL = ROOT / "data" / "meta", ROOT / "data" / "eval"
SRC, OUT, NEW = EVAL / "retrieval_common_test.jsonl", EVAL / "retrieval_common_fix_test.jsonl", EVAL / "retrieval_common_fix_new.jsonl"
BAD = ["CTc006", "CTc009", "CTc024", "CTc042"]
PROFILE_KEYS = ("상품유형", "예금과목", "저축방법", "가입대상", "상품개요", "예금종류", "상품의 개요", "적립방법", "상품특징")
PICK = {"CTc006": ("외화당좌예금", ""),                       # 제10조 준용 = 제3장 외화당좌예금
        "CTc009": ("당좌예금", ""),                           # 당좌·가계당좌예금 해지 후 수표·어음
        "CTc024": ("신한 S드림(DREAM)기업적금", "이 상품은 정기적립식·자유적립식 중 고를 수 있음. 질문은 '정기적립식으로 가입한 경우'로 한정해서 만든다."),
        "CTc042": ("신한 사업자 성공기원적금(정기적립식)", "")}   # 매월 약정일 입금 = 정기적립식


def merge():
    alt = {q["qid"]: q.get("gold_alt", []) for q in map(json.loads, open(NEW))}
    rows = [q if q["qid"] not in alt else {**q, "gold_alt": alt[q["qid"]]} for q in map(json.loads, open(OUT))]
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    NEW.unlink()
    print("합침", len(alt), "→", OUT.name, len(rows))


def main():
    if OUT.exists():
        sys.exit("이미 있음 — 다시 만들지 않음")
    prods = {r["product_id"]: r for r in csv.DictReader(open(META / "products.csv"))}
    docs = {r["doc_id"]: r for r in csv.DictReader(open(META / "documents.csv"))}
    links, plinks = defaultdict(set), defaultdict(set)
    for r in csv.DictReader(open(META / "product_docs.csv")):
        links[r["product_id"]].add(r["doc_id"])
        plinks[r["doc_id"]].add(r["product_id"])
    shared = {d for d, ps in plinks.items() if len(ps) >= 10}
    secs = {}
    for p in (ROOT / "data" / "sections").glob("*.jsonl"):
        for l in open(p):
            s = json.loads(l)
            secs[(s["doc_id"], s["seq"])] = s
    by_name = {}
    for pid, p in prods.items():
        by_name.setdefault(p["name"], pid)
    src = [json.loads(l) for l in open(SRC)]

    def split_of(pid):
        return "test" if int(hashlib.sha256(f"{SEED}:{prods[pid]['name']}".encode()).hexdigest(), 16) % 2 else "dev"

    def profile(pid):
        own = [s for (d, _), s in sorted(secs.items()) if d in links[pid] and d not in shared and any(k in s["section_path"] for k in PROFILE_KEYS)]
        txt = "\n".join(s["text"] for s in own)[:1500]
        return f"분류: {prods[pid]['category']} / {prods[pid]['subcategory']}\n" + (txt or "(상품 전용 문서 없음)")

    env = load_env()
    from openai import OpenAI
    client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
    new, log = {}, []
    for q in src:
        if q["qid"] not in BAD:
            continue
        s = secs[(q["gold"]["doc_id"], q["gold"]["section_seq"])]
        name, hint = PICK[q["qid"]]
        pid = by_name[name]
        assert pid in plinks[s["doc_id"]], (name, s["doc_id"])
        app = client.responses.parse(model=env["CHAT_MODEL"], instructions=APPLY_PROMPT, text_format=Apply,
                                     input=f"[상품] {name}\n{profile(pid)}\n\n[약관 조항] ({docs[s['doc_id']]['title']})\n{s['text'][:1500]}").output_parsed
        log.append((q["qid"], name, app.applies, app.note[:100]))
        got = None
        for _ in range(3):
            g = client.responses.parse(model=env["CHAT_MODEL"], instructions=GEN_PROMPT + NO_DOC + (f"\n- {hint}" if hint else ""), text_format=GenQ,
                                       input=f"상품 이름: {name}\n문서: {docs[s['doc_id']]['title']}\n\n{s['text']}").output_parsed
            if g and g.usable and lines_in_order(g.evidence, ns(s["text"])) and name in g.question:
                got = {**q, "qid": q["qid"] + "R", "question": g.question, "product_id": pid, "category": prods[pid]["category"],
                       "answer": g.answer, "evidence": g.evidence, "gold_alt": [], "apply_note": app.note, "replaces": q["qid"],
                       "product_split": split_of(pid)}
                break
        new[q["qid"]] = got
    for l in log:
        print(*l)
    rows = []
    for q in src:
        if q["qid"] in BAD:
            if new[q["qid"]]:
                rows.append(new[q["qid"]])
        else:
            rows.append(q)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    NEW.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in new.values() if r))
    print(OUT.name, len(rows), "| 교체", sum(1 for r in new.values() if r), "| 적용 상품 없음", [k for k, v in new.items() if not v])


if __name__ == "__main__":
    merge() if "--merge" in sys.argv else main()
