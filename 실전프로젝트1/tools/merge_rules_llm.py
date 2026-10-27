"""규칙 v4 + LLM 합치기 (하이브리드).

규칙 v4 결과를 먼저 놓고, LLM 결과 중 규칙과 겹치지 않는 것만 더함 (겹치면 규칙을 믿음).
--filter 를 주면 LLM이 더한 것 중 개인정보가 아닌 것을 거름 (dev 오탐을 보고 만든 규칙):
  회사·부서·창구 이름(…카드, …센터, …팀), 역할 단어(민원인 등), 이름 모양이 아닌 것,
  대표번호(15xx-xxxx), 한글이 섞인 숫자 라벨('민원번호: …')

사용법
    python tools/merge_rules_llm.py <dev|test> <LLM 예측.jsonl> <출력.jsonl> [--filter] [--rules rules]
    (--rules rules: 팀장님 규칙과 합침. 없으면 예전 rules_v4)
"""
import json, re, sys
ORG_TAIL = re.compile(r"(카드|은행|보안|시스템즈|솔루션|센터|팀|본부|지점|법인|회사|그룹|저축은행|캐피탈|보험)$")
NOT_NAME = {"민원인", "고객", "담당자", "상담원", "본인", "자녀", "고객님"}
def keep(e, text, use_filter):
    if not use_filter:
        return True
    v = text[e["start"]:e["end"]]
    if e["label"] == "person":
        if ORG_TAIL.search(v) or v in NOT_NAME:          # 회사·부서·창구 이름, 역할 단어
            return False
        if not re.fullmatch(r"[가-힣]{2,4}|[가-힣] [가-힣]{2}", v):  # 사람 이름 모양이 아니면
            return False
    if e["label"] == "phone_number" and re.fullmatch(r"1[5-8]\d{2}-?\d{4}", v):  # 대표번호
        return False
    if e["label"] in ("phone_number", "resident_registration_number", "account_number", "card_number") \
            and re.search(r"[가-힣]", v):  # 숫자 라벨인데 '민원번호:' 같은 한글이 섞임
        return False
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    split, llm_path, out_path = args[:3]
    use_filter = "--filter" in sys.argv
    # --rules <이름>: 합칠 규칙 예측 (preds/<이름>_<split>.jsonl). 기본은 예전 rules_v4, 팀장님 규칙은 --rules rules
    rules_name = sys.argv[sys.argv.index("--rules") + 1] if "--rules" in sys.argv else "rules_v4"
    gold = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(f"data/pii_synth_finance_{split}.jsonl", encoding="utf-8")}
    rules = {json.loads(l)["id"]: json.loads(l) for l in open(f"preds/{rules_name}_{split}.jsonl", encoding="utf-8")}
    llm = {json.loads(l)["id"]: json.loads(l) for l in open(llm_path, encoding="utf-8")}
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in rules.items():
            ents = list(r["entities"])
            for e in llm[i]["entities"]:
                if keep(e, gold[i], use_filter) and not any(e["start"] < x["end"] and x["start"] < e["end"] for x in ents):
                    ents.append(e)
            f.write(json.dumps({"id": i, "entities": sorted(ents, key=lambda x: x["start"]), "ms": r["ms"] + llm[i]["ms"]},
                               ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
