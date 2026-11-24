"""5-2 검색 방식 비교 (검색 평가셋)

사용: python analysis/s12_retrieval.py dev            # dev로 방식 비교 (방식 고르기)
      python analysis/s12_retrieval.py test METHOD     # 고른 방식 하나만 test로 한 번
검색 대상: 문서별 청크 2,218 + 공통 조항 216 + 용어 23 (data/chunks, 임베딩은 embed_cache에서 .env EMBED_MODEL 것만)
정답 판정: A형 — 정답 문서의 정답 섹션 번호가 든 청크 / B형 — 정답 공통 조항·용어
지표: Hit@1, Hit@5, MRR@10 (질문 유형별·전체) — '엄격'(질문을 만든 섹션만 정답)과 '보완'(gold_alt: 같은 답을 담은 다른 청크도 정답, 5-5) 둘 다

방식
  vec      임베딩 코사인
  kw       BM25 (Kiwi 형태소 — s10의 tsv와 같은 토큰)
  hybrid   vec + kw 순위 결합 (RRF, k=60)
  rerank   routed 상위 10개를 CHAT_MODEL이 '질문에 직접 답하는 순서'로 다시 줄 세움 (결과는 data/eval/rerank_cache.jsonl에 저장)
  routed   질문에서 상품명을 찾아(상품공시 이름 · '신한' 뺀 이름 · 괄호 뺀 이름 중 가장 긴 일치) 그 상품의 문서 청크와
           그 문서에 연결된 공통 조항, 용어집으로 좁힌 뒤 hybrid. 상품명을 못 찾으면 hybrid 그대로
"""
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
CH, EVAL, META = ROOT / "data" / "chunks", ROOT / "data" / "eval", ROOT / "data" / "meta"


def norm(t: str) -> str:
    return re.sub(r"[\s·ㆍ\-_'’‘\"“”「」『』《》<>,.:()（）\[\]&]", "", t).lower()


class Index:
    def __init__(self):
        from s10_chunks import kiwi_tokens
        self.kiwi_tokens = kiwi_tokens
        from s8_extract import load_env
        from s10_chunks import read_cache
        self.model = load_env()["EMBED_MODEL"]
        cache = read_cache(self.model)
        self.cache = cache
        items = []
        for c in (json.loads(l) for l in open(CH / "chunks.jsonl")):
            items.append({"kind": "chunk", "doc_id": c["doc_id"], "seqs": set(c["section_seqs"]), "text": c["embed_text"], "h": c["content_hash"]})
        for c in (json.loads(l) for l in open(CH / "clauses.jsonl")):
            items.append({"kind": "clause", "clause_id": c["clause_id"], "text": c["embed_text"], "h": c["content_hash"]})
        for g in (json.loads(l) for l in open(CH / "glossary.jsonl")):
            items.append({"kind": "term", "term": g["term"], "text": g["embed_text"], "h": g["content_hash"]})
        self.items = items
        E = np.array([cache[i["h"]] for i in items], dtype=np.float32)
        self.E = E / np.linalg.norm(E, axis=1, keepdims=True)
        # BM25
        toks = kiwi_tokens([i["text"] for i in items])
        self.docs_tok = [t.split() for t in toks]
        self.df = Counter(w for d in self.docs_tok for w in set(d))
        self.N, self.avgdl = len(items), sum(map(len, self.docs_tok)) / len(items)
        self.tf = [Counter(d) for d in self.docs_tok]
        # 상품 이름 → 문서, 문서 → 공통 조항
        prods = list(csv.DictReader(open(META / "products.csv")))
        links = defaultdict(set)
        for r in csv.DictReader(open(META / "product_docs.csv")):
            links[r["product_id"]].add(r["doc_id"])
        self.alias = self.build_aliases(prods, links)
        self.links = links
        self.doc_clause = defaultdict(set)
        for r in csv.DictReader(open(META / "document_clause.csv")):
            self.doc_clause[r["doc_id"]].add(int(r["clause_id"]))

    @staticmethod
    def build_aliases(prods, links):
        """상품 이름 별칭: 상품공시 이름, '신한은행'·'신한' 뺀 이름, 괄호 뺀 이름, 괄호 안 이름(헤이영·썸데이),
        연결 문서 제목(종류 말 뺀 것), 이름 뒤쪽 어절들(스타트업 통장·골드테크) — 단 4개 이상 상품에 겹치는 별칭은 버림(정기예금·통장 같은 일반 말)"""
        docs = {r["doc_id"]: r["title"] for r in csv.DictReader(open(META / "documents.csv"))}
        cand = defaultdict(set)
        for p in prods:
            names = {p["name"]}
            names |= {re.sub(r"\s*(상품설명서|간이투자설명서|투자설명서|특약|약관|약정서|추가약정서)$", "", docs[d])
                      for d in links[p["product_id"]] if docs.get(d)}
            for n in list(names):
                base = n.replace("신한은행", "").replace("신한", "").strip()
                vs = {n, base, re.sub(r"\(.*?\)|（.*?）|『|』|「|」", "", base)}
                vs |= set(re.findall(r"\((.*?)\)", n))
                words = re.sub(r"\(.*?\)", " ", base).split()
                vs |= {" ".join(words[i:]) for i in range(1, len(words))}
                for v in vs:
                    a = norm(v)
                    if len(a) >= 4 or (len(a) >= 3 and v == n):
                        cand[a].add(p["product_id"])
        full = {norm(p["name"]) for p in prods}
        # 겹침은 '상품 이름' 수로 셈 — 상품공시에 같은 이름이 여러 번 나오는 상품(신한 문화콘텐츠 보증대출 ×4)은 한 상품
        name_of = {p["product_id"]: p["name"] for p in prods}
        return {a: ps for a, ps in cand.items() if len({name_of[x] for x in ps}) <= 3 or a in full}

    def bm25(self, q, k1=1.2, b=0.75):
        qt = self.kiwi_tokens([q])[0].split()
        s = np.zeros(self.N, dtype=np.float32)
        for w in set(qt):
            if w not in self.df:
                continue
            idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
            for i in range(self.N):
                f = self.tf[i].get(w)
                if f:
                    s[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(self.docs_tok[i]) / self.avgdl))
        return s

    def detect_products(self, q):
        nq = norm(q)
        hits = [(len(a), a) for a in self.alias if a in nq]
        if not hits:
            return set()
        L = max(h[0] for h in hits)
        return set().union(*(self.alias[a] for l, a in hits if l == L))

    def rank(self, method, q, qv):
        vec = self.E @ qv
        if method == "vec":
            return list(np.argsort(-vec))
        if method == "rerank":
            base = self.rank("routed", q, qv)
            return self.rerank(q, [int(i) for i in base[:10]]) + base[10:]
        if method == "rerank_score":
            base = self.rank("routed", q, qv)
            return self.rerank_score(q, [int(i) for i in base[:10]]) + base[10:]
        kw = self.bm25(q)
        if method == "kw":
            return list(np.argsort(-kw))
        allowed = None
        if method == "routed":
            pids = self.detect_products(q)
            if pids:
                docs = set().union(*(self.links[p] for p in pids))
                clauses = set().union(*(self.doc_clause[d] for d in docs))
                allowed = np.array([(i["kind"] == "chunk" and i["doc_id"] in docs) or (i["kind"] == "clause" and i["clause_id"] in clauses)
                                    or i["kind"] == "term" for i in self.items])
        rv, rk = np.argsort(-vec), np.argsort(-kw)
        score = np.zeros(self.N)
        for r, i in enumerate(rv):
            score[i] += 1 / (60 + r)
        for r, i in enumerate(rk):
            if kw[i] > 0:
                score[i] += 1 / (60 + r)
        if allowed is not None:
            score = np.where(allowed, score, -1)
        return list(np.argsort(-score))

    def rerank(self, q, cand):
        """CHAT_MODEL 재순위 — 후보 번호만 받아 순서를 바꿈 (후보 밖 번호는 버리고, 빠진 후보는 원래 순서로 뒤에 붙임)"""
        import hashlib
        from pydantic import BaseModel
        key = hashlib.sha256((q + "|" + ",".join(map(str, cand))).encode()).hexdigest()
        cpath = EVAL / "rerank_cache.jsonl"
        if not hasattr(self, "_rr"):
            self._rr = {}
            if cpath.exists():
                for l in open(cpath):
                    r = json.loads(l)
                    self._rr[r["key"]] = r["order"]
        if key not in self._rr:
            from s8_extract import load_env
            from openai import OpenAI
            env = load_env()
            if not hasattr(self, "_client"):
                self._client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)

            class Order(BaseModel):
                ids: list[int]
            body = "\n\n".join(f"<{n}>\n{self.items[i]['text'][:1500]}" for n, i in enumerate(cand))
            r = self._client.responses.parse(
                model=env["CHAT_MODEL"], text_format=Order,
                instructions="은행 상품 문서 검색 결과를 다시 줄 세운다. 질문에 직접 답하는 근거가 있는 후보 번호부터 순서대로 ids에 넣는다. "
                             "질문의 상품과 다른 상품의 문서는 뒤로. 답이 없는 후보는 빼도 된다.",
                input=f"질문: {q}\n\n후보:\n{body}")
            order = [cand[n] for n in r.output_parsed.ids if 0 <= n < len(cand)]
            self._rr[key] = order
            with open(cpath, "a") as f:
                f.write(json.dumps({"key": key, "order": [int(x) for x in order]}) + "\n")
        order = list(dict.fromkeys(self._rr[key]))
        return order + [i for i in cand if i not in order]

    def rerank_score(self, q, cand):
        """점수형 재순위 (6-3): 후보마다 0~3점, 점수 높은 순, 같으면 원래 순위. 결과는 data/eval/rerank_score_cache.jsonl"""
        import hashlib
        from pydantic import BaseModel
        from s16_rerank_variants import SCORE_INSTR
        key = hashlib.sha256(("score|" + q + "|" + ",".join(map(str, cand))).encode()).hexdigest()
        cpath = EVAL / "rerank_score_cache.jsonl"
        if not hasattr(self, "_rs"):
            self._rs = {}
            if cpath.exists():
                for l in open(cpath):
                    r = json.loads(l)
                    self._rs[r["key"]] = {int(k): v for k, v in r["scores"].items()}
        if key not in self._rs:
            from s8_extract import load_env
            from openai import OpenAI
            env = load_env()
            if not hasattr(self, "_client"):
                self._client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)

            class Item(BaseModel):
                id: int
                score: int

            class Scored(BaseModel):
                items: list[Item]
            body = "\n\n".join(f"<{n}>\n{self.items[i]['text'][:1500]}" for n, i in enumerate(cand))
            r = self._client.responses.parse(model=env["CHAT_MODEL"], instructions=SCORE_INSTR, text_format=Scored,
                                             input=f"질문: {q}\n\n후보:\n{body}")
            sc = {cand[it.id]: it.score for it in r.output_parsed.items if 0 <= it.id < len(cand)}
            self._rs[key] = sc
            with open(cpath, "a") as f:
                f.write(json.dumps({"key": key, "scores": {str(k): v for k, v in sc.items()}}) + "\n")
        sc = self._rs[key]
        base = {i: r for r, i in enumerate(cand)}
        return sorted(cand, key=lambda i: (-sc.get(i, 0), base[i]))

    def is_gold_any(self, i, q):
        """보완 판정 (5-5): 원래 정답 또는 gold_alt(같은 답을 담은 다른 청크·공통 조항·용어)"""
        if self.is_gold(i, q["gold"]):
            return True
        it = self.items[i]
        for a in q.get("gold_alt", []):
            if "section_seqs" in a and it["kind"] == "chunk" and it["doc_id"] == a["doc_id"] and sorted(it["seqs"]) == a["section_seqs"]:
                return True
            if "clause_id" in a and it["kind"] == "clause" and it["clause_id"] == a["clause_id"]:
                return True
            if "term" in a and it["kind"] == "term" and it["term"] == a["term"]:
                return True
        return False

    def is_gold(self, i, gold):
        it = self.items[i]
        if "section_seq" in gold:
            return it["kind"] == "chunk" and it["doc_id"] == gold["doc_id"] and gold["section_seq"] in it["seqs"]
        if "clause_id" in gold:
            return it["kind"] == "clause" and it["clause_id"] == gold["clause_id"]
        return it["kind"] == "term" and it["term"] == gold["term"]


def embed_questions(qs, idx):
    from s8_extract import load_env
    from openai import OpenAI
    import hashlib
    env = load_env()
    need = [q["question"] for q in qs if hashlib.sha256(q["question"].encode()).hexdigest() not in idx.cache]
    if need and not env["EMBED_MODEL"].startswith("text-embedding"):
        sys.exit(f"{env['EMBED_MODEL']} 질문 벡터 {len(need)}개 없음 — export-colab 입력에 평가 질문이 포함되므로 코랩 결과를 import-colab")
    if need:
        client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
        r = client.embeddings.create(model=env["EMBED_MODEL"], input=need)
        with open(CH / "embed_cache.jsonl", "a") as f:
            for t, e in zip(need, r.data):
                h = hashlib.sha256(t.encode()).hexdigest()
                idx.cache[h] = e.embedding
                f.write(json.dumps({"hash": h, "model": env["EMBED_MODEL"], "vec": e.embedding}) + "\n")
    out = []
    for q in qs:
        v = np.array(idx.cache[hashlib.sha256(q["question"].encode()).hexdigest()], dtype=np.float32)
        out.append(v / np.linalg.norm(v))
    return out


def evaluate(split, methods):
    qs = [json.loads(l) for l in open(EVAL / f"retrieval_{split}.jsonl")]
    idx = Index()
    qvs = embed_questions(qs, idx)
    res = {}
    for m in methods:
        per, per_any = defaultdict(list), defaultdict(list)
        for q, qv1 in zip(qs, qvs):
            order = idx.rank(m, q["question"], qv1)[:10]
            r = next((k + 1 for k, i in enumerate(order) if idx.is_gold(i, q["gold"])), None)
            ra = next((k + 1 for k, i in enumerate(order) if idx.is_gold_any(i, q)), None)
            for t in (q["type"][0], "전체"):
                per[t].append(r)
                per_any[t].append(ra)
        res[m] = per
        for label, P in (("엄격", per), ("보완", per_any)):
            line = []
            for t in ("A", "B", "전체"):
                rs = P[t]
                h1 = sum(1 for r in rs if r == 1) / len(rs)
                h5 = sum(1 for r in rs if r and r <= 5) / len(rs)
                mrr = sum(1 / r for r in rs if r) / len(rs)
                line.append(f"{t}(n={len(rs)}) H@1 {h1:.3f} H@5 {h5:.3f} MRR {mrr:.3f}")
            print(f"{m:12} {label} | " + " | ".join(line))
    if split == "dev" and "routed" in methods:
        found = sum(1 for q in qs if q["product_id"] and idx.detect_products(q["question"]))
        right = sum(1 for q in qs if q["product_id"] and q["product_id"] in idx.detect_products(q["question"]))
        print(f"상품명 찾기: 상품이 있는 질문 {sum(1 for q in qs if q['product_id'])}개 중 찾음 {found}, 정답 상품 포함 {right}")
    return res


if __name__ == "__main__":
    sp = sys.argv[1] if len(sys.argv) > 1 else "dev"
    evaluate(sp, sys.argv[2:] or ["vec", "kw", "hybrid", "routed"])
