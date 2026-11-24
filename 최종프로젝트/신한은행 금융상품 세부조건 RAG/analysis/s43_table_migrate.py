"""표 재추출 적용: 옛 섹션·공통 조항 번호 → 새 번호 이전 (평가 정답·LLM 추출 근거)
옛 = data/backup_pre_table_v2 (재추출 전), 새 = data/runs/table_v2/data (재추출 후, 적용 전 작업 폴더)
섹션 이전 (같은 문서 안, 공백·표 기호 무시)
  exact  옛 섹션 글이 새 섹션 하나에 통째로 들어 있음
  best   새 섹션 하나가 옛 글 4글자 조각의 80% 이상을 담음 (표가 md로 바뀌며 순서만 달라짐)
  multi  여러 새 섹션에 나뉨 → 옛 조각의 20% 이상을 담은 새 섹션들
  weak   위에 안 걸림 → 가장 많이 담은 새 섹션 하나 (사람 확인 대상으로 표시)
조항 이전: 옛 공통 조항 글을 가장 많이 담은 새 공통 조항 (80% 이상), 없으면 섹션 정답으로 바꿈
사용
  python analysis/s43_table_migrate.py map       # 이전표·요약만 (data/meta/table_v2_section_map.csv, table_v2_clause_map.csv)
  python analysis/s43_table_migrate.py eval      # data/eval/*.jsonl 정답 번호 바꿈 (원본은 백업에)
  python analysis/s43_table_migrate.py extract   # data/extract/full/*.json section_seq 바꿈 (원본은 백업에)
  python analysis/s43_table_migrate.py evidence  # eval 뒤 점검: 옮긴 정답이 근거 문장을 잃은 경우 근거 문장을 담은 섹션으로
"""
import collections
import csv
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OLD = DATA / "backup_pre_table_v2"
NEW = DATA / "runs" / "table_v2" / "data"
STRIP = re.compile(r"[\s|·ㆍ•\-‧∙⋅]")


def norm(t):
    return STRIP.sub("", (t or "").replace("<br>", ""))


def grams(t, n=4):
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def sections(root):
    out = {}
    for f in glob.glob(str(root / "sections" / "*.jsonl")):
        for line in open(f):
            s = json.loads(line)
            out[(s["doc_id"], s["seq"])] = norm(s.get("text") or s["content"])
    return out


def clauses(root):
    return {int(r["clause_id"]): norm(r["content"]) for r in csv.DictReader(open(root / "meta" / "common_clause.csv"))}


def build_maps():
    old, new = sections(OLD), sections(NEW)
    by_doc = collections.defaultdict(list)
    for (d, q), t in new.items():
        by_doc[d].append((q, t, grams(t)))
    smap = {}
    for (d, q), t in old.items():
        hit = [nq for nq, nt, _ in by_doc[d] if t and t in nt]
        if hit:
            smap[(d, q)] = ("exact", [min(hit, key=lambda x: len(new[(d, x)]))])
            continue
        g = grams(t)
        cov = sorted(((len(g & ng) / len(g), nq) for nq, _, ng in by_doc[d]), reverse=True)
        if cov and cov[0][0] >= 0.8:
            smap[(d, q)] = ("best", [cov[0][1]])
        elif sum(c for c, _ in cov if c >= 0.2) >= 0.8:
            smap[(d, q)] = ("multi", sorted(nq for c, nq in cov if c >= 0.2))
        else:
            smap[(d, q)] = ("weak", [cov[0][1]] if cov else [])
    oc, nc = clauses(OLD), clauses(NEW)
    ng = {k: grams(v) for k, v in nc.items()}
    cmap = {}
    for k, t in oc.items():
        g = grams(t)
        best = max(((len(g & v) / len(g), nk) for nk, v in ng.items()), default=(0, None))
        cmap[k] = best[1] if best[0] >= 0.8 else None
    return smap, cmap


def load_maps():
    smap = {(r["doc_id"], int(r["old_seq"])): (r["how"], [int(x) for x in r["new_seqs"].split()] if r["new_seqs"] else [])
            for r in csv.DictReader(open(DATA / "meta" / "table_v2_section_map.csv"))}
    cmap = {int(r["old_clause_id"]): (int(r["new_clause_id"]) if r["new_clause_id"] else None)
            for r in csv.DictReader(open(DATA / "meta" / "table_v2_clause_map.csv"))}
    return smap, cmap


def do_map():
    smap, cmap = build_maps()
    with open(DATA / "meta" / "table_v2_section_map.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "old_seq", "how", "new_seqs"])
        for (d, q), (how, ns) in sorted(smap.items()):
            w.writerow([d, q, how, " ".join(map(str, ns))])
    with open(DATA / "meta" / "table_v2_clause_map.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["old_clause_id", "new_clause_id"])
        for k, v in sorted(cmap.items()):
            w.writerow([k, v if v is not None else ""])
    print("섹션", collections.Counter(h for h, _ in smap.values()))
    print("조항", len(cmap), "→ 새 조항 있음", sum(v is not None for v in cmap.values()))


def old_clause_docs():
    m = collections.defaultdict(set)
    for r in csv.DictReader(open(OLD / "meta" / "document_clause.csv")):
        m[int(r["clause_id"])].add(r["doc_id"])
    return m


def clause_to_sections(cid, smap_new_by_doc, old_clause_text):
    """조항이 새 데이터에서 공통 조항이 아니게 된 경우: 그 조항 글을 가장 많이 담은 섹션을 문서마다"""
    out = []
    g = grams(old_clause_text)
    for d, secs in smap_new_by_doc.items():
        best = max(((len(g & sg) / len(g), q) for q, sg in secs), default=(0, None))
        if best[0] >= 0.8:
            out.append({"doc_id": d, "section_any": [best[1]]})
    return out


def do_eval():
    smap, cmap = load_maps()
    oc = clauses(OLD)
    cdocs = old_clause_docs()
    new = sections(NEW)
    new_by_doc = collections.defaultdict(list)
    for (d, q), t in new.items():
        new_by_doc[d].append((q, grams(t)))
    stat = collections.Counter()
    weak = []

    def sec(d, q, qid):
        how, ns = smap[(d, q)]
        stat[how] += 1
        if how == "weak":
            weak.append((qid, d, q, ns))
        return ns

    for f in sorted(glob.glob(str(DATA / "eval" / "*.jsonl"))):
        rows = [json.loads(line) for line in open(f)]
        changed = False
        for q in rows:
            g = q.get("gold")
            if isinstance(g, dict) and "section_seq" in g:
                ns = sec(g["doc_id"], g["section_seq"], q.get("qid"))
                q["gold"] = {"doc_id": g["doc_id"], "section_seq": ns[0]} if len(ns) == 1 else {"doc_id": g["doc_id"], "section_any": ns}
                changed = True
            elif isinstance(g, dict) and "clause_id" in g:
                nk = cmap.get(g["clause_id"])
                if nk is not None:
                    q["gold"] = {"clause_id": nk}
                    stat["clause"] += 1
                else:
                    alts = clause_to_sections(g["clause_id"], {d: new_by_doc[d] for d in cdocs[g["clause_id"]]}, oc[g["clause_id"]])
                    q["gold"] = {"doc_id": alts[0]["doc_id"], "section_any": alts[0]["section_any"]} if alts else g
                    q.setdefault("gold_alt", []).extend(alts[1:])
                    stat["clause→section" if alts else "clause_lost"] += 1
                changed = True
            new_alt = []
            for a in q.get("gold_alt", []) or []:
                if "section_seqs" in a:
                    ns = sorted({n for s in a["section_seqs"] for n in sec(a["doc_id"], s, q.get("qid"))})
                    new_alt.append({"doc_id": a["doc_id"], "section_any": ns})
                elif "clause_id" in a:
                    nk = cmap.get(a["clause_id"])
                    if nk is not None:
                        new_alt.append({"clause_id": nk})
                    else:
                        new_alt += clause_to_sections(a["clause_id"], {d: new_by_doc[d] for d in cdocs[a["clause_id"]]}, oc[a["clause_id"]])
                        stat["alt clause→section"] += 1
                else:
                    new_alt.append(a)
                changed = True
            if "gold_alt" in q:
                q["gold_alt"] = new_alt
        if changed:
            with open(f, "w") as fp:
                fp.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(dict(stat))
    print("weak (사람 확인):", weak[:30])


def do_extract():
    new = sections(NEW)
    smap, _ = load_maps()
    stat = collections.Counter()
    for f in sorted(glob.glob(str(DATA / "extract" / "full" / "*.json"))):
        d = json.load(open(f))
        e = d.get("extraction") or {}
        did = d["doc_id"]
        secs = [(q, t) for (dd, q), t in new.items() if dd == did]
        for k in ("rates", "fees", "benefits"):
            for it in e.get(k) or []:
                st = norm(it.get("source_text"))
                hit = [q for q, t in secs if st and st in t]
                if not hit:
                    lines = [norm(x) for x in (it.get("source_text") or "").split("\n") if norm(x)]
                    cnt = collections.Counter(q for x in lines for q, t in secs if x in t)
                    hit = [cnt.most_common(1)[0][0]] if cnt and all(any(x in t for _, t in secs) for x in lines) else []
                    how = "lines" if hit else "map"
                else:
                    how = "exact"
                if not hit:
                    hit = smap.get((did, it["section_seq"]), ("", []))[1][:1]
                stat[how if hit else "lost"] += 1
                if hit:
                    it["section_seq_v1"] = it["section_seq"]
                    it["section_seq"] = hit[0]
        json.dump(d, open(f, "w"), ensure_ascii=False, indent=1)
    print(dict(stat))


def do_evidence():
    """eval 뒤 점검: 섹션 경계가 움직여, 옮긴 정답 섹션이 근거 문장을 거의 안 담게 된 경우(옛 정답보다 0.3 이상 낮음)
    → 같은 문서에서 근거 문장을 가장 많이 담은 새 섹션으로 바꿈 (바꾼 문항에 gold_fix 표시)"""
    ev_norm = lambda t: STRIP.sub("", (t or "").replace("<br>", "").replace("▶", ""))
    def load(root):
        s = {}
        for f in glob.glob(str(root / "sections" / "D[0-9][0-9][0-9].jsonl")):   # iCloud 충돌 사본("D001 2.jsonl") 제외
            for line in open(f):
                x = json.loads(line)
                s[(x["doc_id"], x["seq"])] = grams(ev_norm(x.get("text") or x["content"]))
        return s
    old, new = load(OLD), load(DATA)
    fixed = []
    for f in sorted(glob.glob(str(DATA / "eval" / "retrieval_*.jsonl"))):
        bf = OLD / "eval" / Path(f).name
        if not bf.exists():
            continue
        rows = [json.loads(line) for line in open(f)]
        changed = False
        for o, n in zip(map(json.loads, open(bf)), rows):
            og, ng = o.get("gold") or {}, n.get("gold") or {}
            if "section_seq" not in og or not o.get("evidence") or n.get("gold_fix"):
                continue
            ev = grams(ev_norm(o["evidence"]))
            oc = len(ev & old[(og["doc_id"], og["section_seq"])]) / len(ev)
            seqs = [ng["section_seq"]] if "section_seq" in ng else ng.get("section_any", [])
            nc = max((len(ev & new[(ng["doc_id"], s)]) / len(ev) for s in seqs), default=0)
            if nc >= oc - 0.3:
                continue
            bc, bq = max((len(ev & g) / len(ev), q) for (d, q), g in new.items() if d == og["doc_id"])
            if bc >= 0.8:
                n["gold_fix"] = {"from": ng, "why": f"근거 문장 {nc:.2f} → {bc:.2f}"}
                n["gold"] = {"doc_id": og["doc_id"], "section_seq": bq}
                fixed.append((Path(f).stem, n["qid"], seqs, bq))
                changed = True
        if changed:
            with open(f, "w") as fp:
                fp.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print("근거 문장 기준으로 바꾼 정답", len(fixed), fixed)


if __name__ == "__main__":
    {"map": do_map, "eval": do_eval, "extract": do_extract, "evidence": do_evidence}[sys.argv[1]]()
