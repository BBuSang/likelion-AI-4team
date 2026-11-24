"""15-1 질문 유형별 실제 답변 예시 → docs/답변_예시.txt (답변 형식 정하기용)

사용: python analysis/s36_answer_examples.py
- 최종 검증 세트(13-1)를 지금 방식으로 채점할 때 나온 답변을 그대로 옮김 (LLM을 다시 부르지 않음)
- 15-1: 답변 맨 위에 '대상 상품' 한 줄(s15 product_line), 아래에 '근거 문서'(인용한 근거)·'함께 볼 문서'(인용 안 한 근거 중 재순위 1점 이상, s15 source_lines)를
  코드로 붙여 보여 줌 — 답변 본문은 그대로. 근거는 채점 때와 같은지 확인한 뒤 씀
"""
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import s12_retrieval as R  # noqa: E402
import s15_answer_eval as A  # noqa: E402

EVAL = ROOT / "data" / "eval"
OUT = ROOT / "docs" / "답변_예시.txt"
BAD_NOTE = {"FC122": "검색 실패 — 정답 조항(여신거래 기본약관 회보 의무)이 근거에 없어 '확인되지 않음'으로 답함",
            "FF264": "목록에 없는 '별꽃 S드림적금'을 추정 표시 없이 실제 S드림적금 정보로 답함. 상품명 찾기는 '찾지 못함'이었는데 검색이 S드림적금 설명서를 가져와 본문이 그걸로 답함 → 새 '대상 상품: 찾지 못함' 줄이 본문과 어긋나서, 읽는 사람이 이상함을 알아챌 수 있음",
            "FA013": "출금 불가는 맞지만, 같은 날 입금도 안 된다는 점을 빠뜨림"}


def main():
    Q, ANS = {}, {}
    for f in ("retrieval", "absent", "status", "compare", "gap"):
        for q in map(json.loads, open(EVAL / f"{f}_final.jsonl")):
            Q[q["qid"]] = q
    for f in ("answer_eval_final", "absent_eval_final", "status_compare_eval_final"):
        for r in map(json.loads, open(EVAL / f"{f}.jsonl")):
            if r.get("k", 5) == 5:
                ANS[r["qid"]] = r
    rnd = random.Random(20261117)

    def pick(typ, style=None, verdicts=("correct", "ok", "refused"), n=2, cond=None):
        xs = sorted(q for q, r in ANS.items() if Q[q]["type"] == typ and (style is None or Q[q].get("style") == style)
                    and r["verdict"] in verdicts and (cond is None or cond(r)))
        rnd.shuffle(xs)
        return xs[:n]

    def infer(r):
        return "보고 답" in r["answer"][:160] or "(질문 속 상품 이름을" in r["answer"][:40]

    sections = [
        ("1. 상품 사실 — 단정한 말투", pick("A", "formal", cond=lambda r: not infer(r))),
        ("2. 상품 사실 — 실제 말투 (반말·오타·붙여 쓰기·질문 두 개)", pick("A", "real", n=3, cond=lambda r: not infer(r))),
        ("3. 상품 이름을 추정해서 답한 경우 (줄임말 등) — 앞에 추정 표시", pick("A", None, n=2, cond=infer)),
        ("4. 답이 공통 약관에만 있는 질문", pick("B-common")),
        ("5. 공용 대출 설명서 질문", pick("B-shared")),
        ("6. 문서에 없는 값을 물은 경우 — 지어내지 않고 확인할 곳 안내", pick("A-own", n=3)),
        ("7. 판매 중지 상품", pick("disc")),
        ("8. 판매 중지 상품을 줄임말로 물은 경우", pick("disc_nick")),
        ("9. 판매 중인 상품의 판매 여부", pick("onsale")),
        ("10. 두 상품 비교", pick("compare")),
        ("11. 목록에 없는 가짜 상품 이름", pick("fake_full", n=1) + pick("fake_paren", n=1) + pick("fake_full", verdicts=("partial",), n=1)),
        ("12. 잘 안 된 예 (참고)", list(BAD_NOTE)),
    ]
    idx = R.Index()
    idx._name_index()
    all_ids = [x for _, ids in sections for x in ids]
    qvs = dict(zip(all_ids, R.embed_questions([Q[x] for x in all_ids], idx)))
    L = [len(r["answer"]) for r in ANS.values()]
    out = [f"""최종프로젝트 — 질문 유형별 실제 답변 예시
================================================================

- 가져온 곳: 최종 검증 세트(13-1)를 지금 방식으로 채점할 때 실제로 나온 답변을 그대로 옮김 (답변 원칙 v9, 근거 상위 5개, 비교는 상품마다 3개)
- 답변 맨 위 '대상 상품' 줄은 15-1에서 새로 붙인 것: 어떤 상품으로 답했는지와 판매 상태를 코드가 만들어 늘 같은 자리에 둠
  (판매 중 / 판매 중지 / 판매 중·판매 중지 목록에 같은 이름 / · 추정 = 질문 속 이름을 이 상품으로 추정 / 찾지 못함). 그 아래 본문은 그대로
- 답변 아래 '근거 문서'·'함께 볼 문서'도 15-1에서 새로 붙인 것 (코드로 만듦)
  · 근거 문서: 답변 속 [1][2] 번호로 실제 인용한 근거 (번호는 검색이 넘겨준 근거 순서 그대로)
  · 함께 볼 문서: 인용하지 않았지만 재순위가 '관련 있음'(1점 이상)으로 본 근거. 0점(무관·다른 상품)은 뺌. 같은 조항이 원문과 공통 조항으로 두 번 오면 하나만
    판매 중인 대상 상품을 찾지 못했으면(판매 중지·목록에 없는 이름) 근거가 그 상품의 자료가 아니라서 보이지 않음
- 각 예시: [질문] → [답변] (대상 상품 · 본문 · 근거 문서 · 함께 볼 문서)
- 질문과 답변 본문은 고치지 않음
- 고른 방법: 유형마다 무작위(seed 20261117)로, 1~11은 맞게 답한 것, 12는 잘 안 된 것

지금 답변 형식에서 보이는 공통점 (형식 정할 때 참고)
----------------------------------------------------------------
- 길이: 본문 글자 수 중앙값 {int(st.median(L))}자 (최소 {min(L)} · 최대 {max(L)}), 보통 1~4문장, 따로 머리말·요약 없이 바로 답
- 근거 표시: 본문에는 문장 끝에 [1][2] 번호만 붙고, 문서 이름·섹션·쪽은 아래 '근거 문서'에 있음
- 기준일: 금리·수수료처럼 바뀌는 값은 "(2026.09.03 현재, 세전)"처럼 문서의 기준일을 함께 말함
- 상품 이름을 추정했으면 본문 맨 앞에도 "질문하신 ‘○○’을 ‘△△’으로 보고 답합니다." (모델이 빼먹으면 코드가 붙임)
- 문서에 없으면 "보유 문서에서 확인되지 않습니다" + 확인할 곳(영업점·홈페이지·여신 담당)
- 판매 중지 상품은 판매 중지라는 사실을 먼저 말하고, 세부 조건은 말하지 않음
- 말투: 질문이 반말이어도 답은 존댓말
"""]
    for title, ids in sections:
        out.append("\n" + "=" * 64 + "\n" + title + "\n" + "=" * 64)
        for qid in ids:
            q, r = Q[qid], ANS[qid]
            out.append(f"\n[{qid}] 말투: {'실제 말투' if q.get('style') == 'real' else '단정'}" + (f"   (채점: {r['verdict']})" if qid in BAD_NOTE else ""))
            out.append("[질문] " + q["question"])
            ctx_ids, _ = A.context_ids(idx, q["question"], qvs[qid], "v9", k=5)
            heads = [idx.items[i]["text"].split("\n", 1)[0] for i in ctx_ids]
            assert not r.get("context") or heads == r["context"], f"{qid}: 근거가 채점 때와 다름"
            out.append("[답변]\n" + A.product_line(idx, q["question"]) + "\n" + r["answer"].strip() + "\n"
                       + A.source_lines(idx, q["question"], r["answer"], ctx_ids))
            if qid in BAD_NOTE:
                out.append("[무엇이 문제였나] " + BAD_NOTE[qid])
    OUT.write_text("\n".join(out) + "\n")
    print(OUT.name, sum(len(i) for _, i in sections), "개 예시")


if __name__ == "__main__":
    main()
