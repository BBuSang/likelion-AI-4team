"""규칙 v4 + 여러 LLM을 한꺼번에 합치기.

  union: 규칙 결과에, LLM 중 하나라도 찾은 것을 더함 (규칙과 겹치면 규칙을 믿음)
  vote : 규칙 결과에, LLM 중 2개 이상이 같은 위치를 찾은 것만 더함
LLM이 더한 것에는 merge_rules_llm.py의 후처리 규칙(keep)을 똑같이 적용. --no-filter 를 주면 후처리 없이 합침.

사용법
    python tools/merge_rules_multi_llm.py <dev|test> <union|vote> <출력.jsonl> <LLM1.jsonl> <LLM2.jsonl> ... [--no-filter] [--rules rules]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from merge_rules_llm import keep  # noqa: E402


def overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


def main():
    use_filter = "--no-filter" not in sys.argv
    argv = sys.argv[1:]
    rules_name = "rules_v4"
    if "--rules" in argv:  # --rules rules: 팀장님 규칙과 합침
        i = argv.index("--rules"); rules_name = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    split, mode, out_path, *llm_paths = [a for a in argv if not a.startswith("--")]
    text = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(f"data/pii_synth_finance_{split}.jsonl", encoding="utf-8")}
    rules = {json.loads(l)["id"]: json.loads(l) for l in open(f"preds/{rules_name}_{split}.jsonl", encoding="utf-8")}
    llms = [{json.loads(l)["id"]: json.loads(l) for l in open(p, encoding="utf-8")} for p in llm_paths]
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in rules.items():
            ents = list(r["entities"])
            cands = []
            for k, llm in enumerate(llms):
                for e in llm[i]["entities"]:
                    if keep(e, text[i], use_filter):
                        cands.append((k, e))
            for k, e in cands:
                if mode == "vote":
                    voters = {k2 for k2, e2 in cands if e2["label"] == e["label"] and overlap(e, e2)}
                    if len(voters) < 2:
                        continue
                if not any(overlap(e, x) for x in ents):
                    ents.append(e)
            ms = r["ms"] + max(llm[i]["ms"] for llm in llms)  # LLM은 동시에 부른다고 보고 가장 느린 것 기준
            f.write(json.dumps({"id": i, "entities": sorted(ents, key=lambda x: x["start"]), "ms": ms}, ensure_ascii=False) + "\n")
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
