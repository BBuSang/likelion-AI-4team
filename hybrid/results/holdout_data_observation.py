import json, re, sys
from collections import Counter
import os; base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
sets = {"dev": "pii_synth_finance_dev", "test": "pii_synth_finance_test", "holdout": "holdout_labeled"}
L = ["person", "phone_number", "email", "address", "resident_registration_number", "account_number", "card_number"]
def fmt(lab, t):
    if lab == "phone_number":
        if t.startswith("+82"): return "+82 국제"
        if re.match(r"\(0\d+\)", t): return "괄호 지역번호"
        sep = re.sub(r"\d", "", t)
        k = "하이픈" if set(sep) == {"-"} else "점" if set(sep) == {"."} else "공백" if set(sep) == {" "} else "구분자 없음" if not sep else "혼합"
        return ("휴대폰 " if t.startswith("01") else "지역/기타 ") + k
    if lab == "resident_registration_number":
        return "하이픈" if "-" in t and " " not in t else "공백" if " " in t else "붙여씀" if t.isdigit() else "기타"
    if lab == "account_number":
        return "-".join(str(len(g)) for g in re.split(r"[-\s]", t)) if re.search(r"[-\s]", t) else f"붙여씀({len(t)}자리)"
    if lab == "card_number":
        return "-".join(str(len(g)) for g in re.split(r"[-\s]", t)) + (" 공백" if " " in t else "") if re.search(r"[-\s]", t) else "붙여씀"
    if lab == "address":
        k = []
        if re.match(r"\(?\d{5}\)", t): k.append("우편번호")
        k.append("도로명" if re.search(r"(로|길) ?\d", t) else "지번")
        if re.search(r"(읍|면) \d", t): k.append("읍면+번지")
        if re.search(r"\d+(층|호)", t): k.append("동호층")
        return "+".join(k)
    if lab == "person":
        return None
    if lab == "email":
        return t.split("@")[1]
out = {}
for k, f in sets.items():
    docs = [json.loads(l) for l in open(os.path.join(base, f + ".jsonl"), encoding="utf-8") if l.strip()]
    ents = [(e, d) for d in docs for e in d["entities"]]
    r = {"docs": len(docs), "ents": len(ents), "chars/doc": round(sum(len(d["text"]) for d in docs) / len(docs)),
         "ents/doc": round(len(ents) / len(docs), 2),
         "labels": Counter(e["label"] for e, _ in ents),
         "doc_type": Counter(d["doc_type"] for d in docs),
         "hard": Counter(t for d in docs for t in d.get("hard_cases", [])),
         "no_pii_docs": sum(1 for d in docs if not d["entities"]),
         "person_form": Counter(e.get("form") for e, _ in ents if e["label"] == "person"),
         "person_len": Counter(len(e["text"].replace(" ", "")) for e, _ in ents if e["label"] == "person"),
         "compound": sum(1 for e, _ in ents if e["label"] == "person" and e["text"][:2] in ("남궁","제갈","선우","황보","독고","사공","서문")),
         "fmt": {lab: Counter(fmt(lab, e["text"]) for e, _ in ents if e["label"] == lab) for lab in L if lab not in ("person",)},
         "templates": len(set(d.get("template") for d in docs)),
         }
    out[k] = r
for k, r in out.items():
    print("=" * 20, k)
    for kk, v in r.items():
        if isinstance(v, Counter): v = dict(v.most_common())
        if kk == "fmt": 
            for lab, c in v.items(): print(f"  fmt[{lab}]", dict(c.most_common()))
        else: print(" ", kk, v)
# 템플릿 겹침
tp = {k: set(json.loads(l).get("template") for l in open(os.path.join(base, f + ".jsonl"), encoding="utf-8")) for k, f in sets.items()}
dt = {k: set(json.loads(l).get("doc_type") for l in open(os.path.join(base, f + ".jsonl"), encoding="utf-8")) for k, f in sets.items()}
print("templates holdout∩(dev∪test):", len(tp["holdout"] & (tp["dev"] | tp["test"])), "/", len(tp["holdout"]), sorted(tp["holdout"])[:40])
print("doc_type holdout only:", dt["holdout"] - dt["dev"] - dt["test"], "| dev/test only:", (dt["dev"] | dt["test"]) - dt["holdout"])
