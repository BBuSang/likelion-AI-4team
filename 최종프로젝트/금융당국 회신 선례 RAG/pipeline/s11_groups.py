"""11 묶음: 내용이 같은 회신을 묶고 최신 회신을 대표로 정한다. 다른 선례가 합쳐지지 않도록 보수적으로 묶는다.
- same_body: 구분이 같고 질의·회답·이유가 같음. 질의가 짧거나 2016 규제 정비 건이면 제목도 같아야 한다 (공통 답변 문구가 많다)
- same_body(near): 구분·답변유형·숫자가 같고 글 유사도가 NEAR 이상
- same_question: 구분·답변유형·회답 숫자가 같고 질의가 같으며 회답·이유 유사도가 NEAR 이상 ('판매'와 '대여'처럼 핵심어만 다른 회답을 합치지 않는다)
- 연장 건은 기한이 달라 묶지 않는다. 효력 기간이 다른 선례를 합치면 안 된다
- body_suspect: 첨부에서 본문을 얻었는데, 같은 묶음에 같은 첨부 파일을 쓰고 본문이 완전히 같으며 본문을 포털 화면에서 얻은 사례가 있으면 포털 첨부 오류로 본다. 대표로 뽑지 않는다
- 묶음 유형은 묶인 뒤 글을 비교해 정한다: 전부 같으면 same_body, 질의만 같으면 same_question, 그 외 거의 같으면 same_body
입력: processed/cases_enriched.jsonl   출력: processed/case_groups.jsonl, case_group_map.jsonl (+csv)
"""
import re
import zlib
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from common import manifest, read_jsonl, write_csv, write_jsonl

NEAR = 0.97
MIN_Q, MIN_NEAR = 50, 100
nz = lambda s: re.sub(r"[\s\W_]+", "", s or "")
ratio = lambda a, b: SequenceMatcher(None, a, b, autojunk=False).ratio()
digits = lambda s: Counter(re.findall(r"\d+", s))


class Union:
    def __init__(self, keys):
        self.p = {k: k for k in keys}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def join(self, a, b):
        self.p[self.find(a)] = self.find(b)


def sketch(t, n=30, k=8):
    """글 일부로 만든 지문: 비슷한 글끼리 같은 값이 많이 겹친다."""
    return set(sorted({zlib.crc32(t[i:i + k].encode()) for i in range(0, max(1, len(t) - k))})[:n])


def main():
    rows = [r for r in read_jsonl("cases_enriched.jsonl") if r["indexable"]]
    R = {r["case_key"]: r for r in rows}
    q = {k: nz(r["question"]) for k, r in R.items()}
    ar = {k: nz(r["answer"]) + "|" + nz(r["reason"]) for k, r in R.items()}
    full = {k: q[k] + "|" + ar[k] for k in R}
    title = {k: nz(r["title"]) for k, r in R.items()}
    u, how = Union(R), {}

    def link(keyfn, tag, ok=lambda a, b: True):
        g = defaultdict(list)
        for k, r in R.items():
            v = keyfn(k, r)
            if v is not None:
                g[v].append(k)
        n = 0
        for m in g.values():
            for b in m[1:]:
                if ok(m[0], b):
                    u.join(m[0], b)
                    how.setdefault(m[0], tag)
                    how[b] = tag
                    n += 1
        return n

    def body_key(k, r):
        if r["doc_class"] == "extension" or len(ar[k]) < 21:
            return None
        if len(q[k]) >= 20 and r["doc_class"] == "general":
            return (r["gubun"], full[k])
        return (r["gubun"], title[k], full[k]) if len(title[k]) >= 4 else None    # 짧은 글은 제목까지 같아야 같은 건

    n_exact = link(body_key, "exact")

    def question_key(k, r):
        return (r["gubun"], r["answer_type"], q[k]) if r["doc_class"] == "general" and len(q[k]) >= MIN_Q else None

    n_q = link(question_key, "question", lambda a, b: ar[a] == ar[b] or (digits(ar[a]) == digits(ar[b]) and ratio(ar[a], ar[b]) >= NEAR))

    # 거의 같은 글: 일반 회신만, 글자 지문이 많이 겹치는 쌍만 비교한다
    docs = [k for k, r in R.items() if r["doc_class"] == "general" and len(full[k]) >= MIN_NEAR]
    idx, sk = defaultdict(list), {}
    for k in docs:
        sk[k] = sketch(full[k])
        for h in sk[k]:
            idx[h].append(k)
    pairs = Counter()
    for m in idx.values():
        if len(m) <= 60:
            m = sorted(m)
            for i, a in enumerate(m):
                for b in m[i + 1:]:
                    pairs[(a, b)] += 1
    n_near = 0
    for (a, b), c in pairs.items():
        if c < 10 or u.find(a) == u.find(b):
            continue
        x, y = R[a], R[b]
        if x["gubun"] != y["gubun"] or x["answer_type"] != y["answer_type"] or digits(full[a]) != digits(full[b]):
            continue
        if SequenceMatcher(None, full[a], full[b], autojunk=False).quick_ratio() >= NEAR and ratio(full[a], full[b]) >= NEAR:
            u.join(a, b)
            how.setdefault(a, "near")
            how[b] = "near" if how.get(b) != "exact" else "exact"
            n_near += 1

    clusters = defaultdict(list)
    for k in R:
        clusters[u.find(k)].append(k)

    newest = lambda m: max(m, key=lambda k: (R[k]["reply_date"] or "", k))
    files = lambda k: {a["org_file_name"] for a in R[k]["attachments"] or []}

    def suspects(part):
        return {k for k in part if R[k]["body_source"] == "attachment"
                and any(o != k and R[o]["body_source"] != "attachment" and files(o) & files(k) and full[o] == full[k] for o in part)}

    groups, maps, kicked = [], [], 0
    for m in clusters.values():
        rep = newest(m)
        # 사슬처럼 이어진 묶음은 대표와 직접 비슷한 사례만 남긴다
        keep = [k for k in m if full[k] == full[rep] or how.get(k) == "question" or ratio(full[k], full[rep]) >= NEAR]
        if len(keep) < len(m):
            kicked += len(m) - len(keep)
        parts = [keep] + [[k] for k in m if k not in keep]
        for part in parts:
            bad = suspects(part)
            rep2 = newest([k for k in part if k not in bad])
            same_all = len({full[k] for k in part}) == 1
            typ = "single" if len(part) == 1 else ("same_question" if not same_all and len({q[k] for k in part}) == 1 else "same_body")
            match = lambda k: "single" if len(part) == 1 else ("exact" if full[k] == full[rep2] else "question" if q[k] == q[rep2] else "near")
            gid = len(groups) + 1
            groups.append({"group_id": gid, "group_type": typ, "representative_case_key": rep2, "n_cases": len(part)})
            for k in part:
                maps.append({"case_key": k, "group_id": gid, "match": match(k), "is_representative": k == rep2, "body_suspect": k in bad})

    write_jsonl("case_groups.jsonl", groups)
    write_jsonl("case_group_map.jsonl", maps)
    write_csv("case_groups.csv", groups, ["group_id", "group_type", "representative_case_key", "n_cases"])
    write_csv("case_group_map.csv", maps, ["case_key", "group_id", "match", "is_representative", "body_suspect"])
    multi = [g for g in groups if g["n_cases"] > 1]
    manifest("11_groups", cases=len(rows), groups=len(groups), multi_groups=len(multi), cases_in_multi=sum(g["n_cases"] for g in multi),
             types=dict(Counter(g["group_type"] for g in groups)), edges={"exact": n_exact, "question": n_q, "near": n_near}, kicked=kicked,
             largest=max(g["n_cases"] for g in groups), body_suspect=sum(m["body_suspect"] for m in maps))


if __name__ == "__main__":
    main()
