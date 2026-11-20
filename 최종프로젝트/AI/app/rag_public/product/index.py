"""금융상품 RAG 색인·검색 (worklog 53장).

검색 대상 (메모리 색인, DB 없음): 문서 청크 2,218 + 공통 조항 216 + 용어 23
  데이터 data/public_rag/product/  meta/*.csv (상품 391 · 문서 247 · 상품↔문서 · 문서↔공통 조항 · 판매 중지 1,338)
                                    chunks/*.jsonl (은행 문서 원문, git 제외) · vectors.npz (KURE-v1, scripts/build_product_vectors.py)

검색 순서
  1. 상품명 찾기 — 별칭 정확 일치 → 애매하면 LLM 확인 → 별칭이 없으면 이름 후보(임베딩 + 글자쌍) 중 LLM 이 고름
     (판매 중지 이름도 후보, 판매 중지 목록에만 있는 이름이면 판매 중 상품으로 추정하지 않음)
  2. 좁히기 — 찾은 상품의 문서 청크 + 그 문서의 공통 조항 + 용어
  3. 결합 순위 — 벡터 + 키워드(BM25, Kiwi 명사·동사) 순위 결합 (RRF k=60)
  4. 점수형 재순위 — 후보(결합 상위 7 + 상품 이름을 뺀 주제 순위 상위 3)를 LLM 이 0~3점
LLM 결과(상품 고르기·재순위)는 outputs/public_rag/product/ 에 쌓아 같은 질문은 다시 부르지 않음.
"""
import csv
import hashlib
import json
import math
import re
import threading
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from app.core import config

AI_ROOT = Path(__file__).resolve().parents[3]
DATA = AI_ROOT / "data" / "public_rag" / "product"
META, CHUNKS, VECTORS = DATA / "meta", DATA / "chunks", DATA / "vectors.npz"
CACHE = AI_ROOT / "outputs" / "public_rag" / "product"

RERANK_CHARS = 1500   # 재순위 후보마다 읽는 글자 수
FROM_BASE, FROM_TOPIC = 7, 3   # 재순위 후보 = 결합 순위 상위 7 + 주제 순위 상위 3

# 상품 종류를 뜻하는 일반어 — 별칭 하나가 이 말들로만 되어 있으면 특정 상품을 가리키지 않음
GENERIC_WORDS = {"통장", "적금", "예금", "정기예금", "정기적금", "적립식예금", "거치식예금", "입출금통장", "입출금", "저축", "종합저축",
                 "저축예금", "청약통장", "청약예금", "청약저축", "청약부금", "주택청약", "주택청약종합저축", "대출", "담보대출", "신용대출",
                 "전세대출", "전세자금대출", "주택담보대출", "기업대출", "외화예금", "외화적금", "외화통장", "외화정기예금", "기업통장",
                 "계좌", "저금통", "상품", "예금통장", "자유적금", "보통예금", "기업자유예금", "정기예금특약", "특약"}
NOUN_TAGS = {"NNG", "NNP", "NR", "SL", "SH", "XR"}
PREFIX_OK = {"신한", "신한은행", "쏠", "sol", "쏠편한"}

PICK_INSTR = """너는 신한은행 상품 안내 도우미의 '상품명 찾기' 단계다. 영업점 직원의 [질문]이 신한은행 상품을 줄임말·애칭·다른 표기(영어↔한글, 띄어쓰기, 일부 글자 생략)로 부르고 있을 수 있다.
[후보 상품] 중 질문이 가리키는 상품의 번호를 ids에 넣는다.
- 질문 속 상품 호칭이 후보 이름의 줄임말이나 다른 표기로 자연스럽게 볼 수 있을 때만 고른다
- 호칭이 여러 후보에 똑같이 들어맞으면 그 후보들을 모두 고른다 (최대 3개)
- 질문이 특정 상품이 아니라 공통 규정·용어·일반 절차를 묻거나, 호칭이 어느 후보와도 맞지 않으면(없는 상품·다른 회사 상품일 수 있음) 빈 목록
- **상품을 구별하는 고유한 이름 부분**(브랜드·대상·용도 등)이 후보와 맞아야 한다. 상품 종류를 뜻하는 말(대출·적금·통장·예금·비상금·전세 등)만 겹치면 고르지 않는다
- 반대로 **호칭 안의 고유한 말은 모두** 후보 이름 안의 말(또는 그 줄임말·다른 표기: 영어↔한글, 기관 이름 줄임말)과 대응해야 한다. 후보 이름에 없는 고유한 말(브랜드·수식어)이 호칭에 덧붙어 있으면 다른 상품일 수 있으니 고르지 않는다
  (은행·브랜드 머리말 '신한'·'쏠'·'SOL', 상품 종류 말, 질문 내용의 말(금리·한도 등)은 따지지 않음)
- 글자 몇 개가 겹친다는 이유만으로 고르지 않는다. 확실하지 않으면 빈 목록
- 이름 뒤에 '(판매 중지)'가 붙은 후보는 판매가 끝난 상품이다. 호칭이 그 상품을 가리키면 그것을 고른다 (판매 중인 비슷한 상품으로 바꾸지 않음).
  판매 중 후보와 판매 중지 후보 중 어느 쪽인지 애매하면 둘 다 고른다
note에 한 문장 이유."""

SCORE_INSTR = ("은행 상품 문서 검색 결과를 평가한다. 후보마다 질문에 대한 쓸모를 0~3점으로 매긴다. "
               "3: 질문에 직접 답하는 근거가 있음, 2: 답의 일부가 있음, 1: 관련은 있지만 답이 없음, 0: 무관하거나 다른 상품. "
               "후보끼리 비교하지 말고 각 후보를 따로 판단한다. 모든 후보 번호에 점수를 준다.")


def norm(t):
    return re.sub(r"[\s·ㆍ\-_'’‘\"“”「」『』《》<>,.:()（）\[\]&]", "", t).lower()


GENERIC_N = {norm(w) for w in GENERIC_WORDS} | {"신한", "신한은행", "쏠", "sol", "쏠편한", "상품"}


def bigrams(t):
    s = norm(t)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def text_hash(t):
    return hashlib.sha256(t.encode()).hexdigest()


def _csv(name):
    with open(META / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class _Cache:
    """LLM 결과를 jsonl 로 쌓는 작은 캐시 (key → value)"""

    def __init__(self, name):
        self.path = CACHE / name
        self.data, self.lock = {}, threading.Lock()
        if self.path.exists():
            for r in _jsonl(self.path):
                self.data[r["key"]] = r["value"]

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            CACHE.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


_kiwi, _kiwi_lock = None, threading.Lock()


def kiwi():
    global _kiwi
    with _kiwi_lock:
        if _kiwi is None:
            from kiwipiepy import Kiwi
            _kiwi = Kiwi()
    return _kiwi


def kiwi_tokens(texts):
    """키워드 검색용 토큰: 명사·동사·형용사·외국어·숫자·어근"""
    keep = ("NN", "VV", "VA", "SL", "SN", "XR")
    k = kiwi()
    with _kiwi_lock:  # 한 Kiwi 를 여러 스레드가 동시에 쓰지 않게
        return [" ".join(t.form for t in toks if t.tag.startswith(keep)) for toks in k.tokenize(list(texts))]


def items_texts():
    """색인 항목 글 (벡터 만들기와 색인이 같은 순서·같은 글을 쓰게)"""
    items = []
    for c in _jsonl(CHUNKS / "chunks.jsonl"):
        items.append({"kind": "chunk", "doc_id": c["doc_id"], "seqs": set(json.loads(c["section_seqs"]) if isinstance(c["section_seqs"], str) else c["section_seqs"]),
                      "text": c["embed_text"]})
    for c in _jsonl(CHUNKS / "clauses.jsonl"):
        items.append({"kind": "clause", "clause_id": int(c["clause_id"]), "text": c["embed_text"]})
    for g in _jsonl(CHUNKS / "glossary.jsonl"):
        items.append({"kind": "term", "term": g["term"], "text": g["embed_text"]})
    return items


def name_lists():
    """이름 후보 목록: 판매 중 상품 이름, 판매 중지 이름(판매 중 목록에 없는 것, ELD 회차는 묶음)"""
    prods = _csv("products.csv")
    on = sorted({p["name"] for p in prods})
    disc = {}
    for r in _csv("discontinued.csv"):
        if r["also_on_sale"] == "1" or r["name"] in on:
            continue
        disc.setdefault(re.sub(r"\d{2}-\d{2}호", "○호", r["name"]), r["subcategory"] or r["category"])
    return on, sorted(disc), disc


class ProductIndex:
    def __init__(self):
        self.items = items_texts()
        vec = np.load(VECTORS)
        self.vectors = {h: v for h, v in zip(vec["hashes"], vec["vectors"])}
        E = np.array([self.vectors[text_hash(i["text"])] for i in self.items], dtype=np.float32)
        self.E = E / np.linalg.norm(E, axis=1, keepdims=True)
        # 키워드 (BM25)
        self.docs_tok = [t.split() for t in kiwi_tokens(i["text"] for i in self.items)]
        self.df = Counter(w for d in self.docs_tok for w in set(d))
        self.N = len(self.items)
        self.avgdl = sum(map(len, self.docs_tok)) / self.N
        self.tf = [Counter(d) for d in self.docs_tok]
        self._bm25_memo = {}
        # 상품 ↔ 문서 ↔ 공통 조항
        prods = _csv("products.csv")
        self.links = defaultdict(set)
        for r in _csv("product_docs.csv"):
            self.links[r["product_id"]].add(r["doc_id"])
        self.doc_clause = defaultdict(set)
        for r in _csv("document_clause.csv"):
            self.doc_clause[r["doc_id"]].add(int(r["clause_id"]))
        self.docs = {r["doc_id"]: r for r in _csv("documents.csv")}
        self.pid_name = {p["product_id"]: p["name"] for p in prods}
        self.alias = self._build_aliases(prods)
        self.full_names = set()
        for p in prods:
            b = p["name"].replace("신한은행", "").replace("신한", "").strip()
            self.full_names |= {norm(p["name"]), norm(b), norm(re.sub(r"\(.*?\)|（.*?）", "", b))}
        self.all_names_n = {p["name"]: norm(p["name"]) for p in prods}
        # 이름 후보 (상품명 찾기 2단계)
        self.name_pids = defaultdict(set)
        for p in prods:
            self.name_pids[p["name"]].add(p["product_id"])
        self.name_cat = {p["name"]: p["subcategory"] or p["category"] for p in prods}
        self.names, self.disc_names, self.disc_cat = name_lists()
        self.NE = self._name_matrix(self.names)
        self.DE = self._name_matrix(self.disc_names)
        self.name_bg = [bigrams(n) for n in self.names]
        self.disc_bg = [bigrams(n) for n in self.disc_names]
        self._disc_exact = self._build_disc_exact(prods)
        self.pick_cache, self.score_cache = _Cache("pick.jsonl"), _Cache("rerank_score.jsonl")
        self.q_scores = {}  # 질문 → {항목: 재순위 점수} (근거 문서·함께 볼 문서 고르기)
        self._client = None
        self._lock = threading.Lock()

    # ── 준비 ──

    def _name_matrix(self, names):
        M = np.array([self.vectors[text_hash(n)] for n in names], dtype=np.float32)
        return M / np.linalg.norm(M, axis=1, keepdims=True)

    def _build_aliases(self, prods):
        """상품 이름 별칭: 정식 이름, '신한' 뺀 이름, 괄호 뺀/안 이름, 연결 문서 제목, 앞·뒤 어절.
        4개 이상 상품에 겹치는 별칭과 상품 종류 일반어로만 된 별칭은 버림"""
        titles = {d: r["title"] for d, r in self.docs.items()}
        cand = defaultdict(set)
        for p in prods:
            names = {p["name"]}
            names |= {re.sub(r"\s*(상품설명서|간이투자설명서|투자설명서|특약|약관|약정서|추가약정서)$", "", titles[d])
                      for d in self.links[p["product_id"]] if titles.get(d)}
            for n in list(names):
                base = n.replace("신한은행", "").replace("신한", "").strip()
                vs = {n, base, re.sub(r"\(.*?\)|（.*?）|『|』|「|」", "", base)}
                vs |= set(re.findall(r"\((.*?)\)", n))
                words = re.sub(r"\(.*?\)", " ", base).split()
                vs |= {" ".join(words[i:]) for i in range(1, len(words))}       # 뒤쪽 어절 (스타트업 통장)
                if n == p["name"]:  # 앞쪽 어절은 상품 이름에서만 (문서 제목 앞머리 일반어가 별칭이 되지 않게)
                    vs |= {" ".join(words[:i]) for i in range(1, len(words))}   # 앞쪽 어절 (마이홈플랜)
                vs |= {re.sub(r"\S*\(([^)]*)\)", r"\1", base)}                 # 'Someday(썸데이) 외화적금' → '썸데이 외화적금'
                for v in vs:
                    a = norm(v)
                    if v != n and all(norm(w) in GENERIC_WORDS for w in v.split()):
                        continue
                    if len(a) >= 4 or (len(a) >= 3 and v == n):
                        cand[a].add(p["product_id"])
        full = {norm(p["name"]) for p in prods}
        return {a: ps for a, ps in cand.items() if len({self.pid_name[x] for x in ps}) <= 3 or a in full}

    def _build_disc_exact(self, prods):
        """판매 중지 이름 정확 대조표: 정규화 이름 → (이름, discontinued | both)"""
        onsale = {p["name"] for p in prods}
        onsale_n = [norm(n) for n in onsale]
        out = {}
        for r in _csv("discontinued.csv"):
            for v in {r["name"], r["name"].replace("신한은행", "").replace("신한", "").strip()}:
                k = norm(v)
                # 짧은 이름이 판매 중 상품 이름 여러 개 안에 들어 있으면('보증대출') 흔한 말이라 대조하지 않음
                if len(k) < 6 and sum(k in x for x in onsale_n) >= 2:
                    continue
                if len(k) >= 6 or (v == r["name"] and len(k) >= 4):
                    out[k] = (r["name"], "both" if (r["also_on_sale"] == "1" or r["name"] in onsale) else "discontinued")
        return out

    def client(self):
        with self._lock:
            if self._client is None:
                from openai import OpenAI
                self._client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.OPENAI_API_BASE)
        return self._client

    # ── 벡터·키워드 ──

    def text_vec(self, t):
        v = self.vectors.get(text_hash(t))
        if v is None:
            from app.rag.embed import embed
            v = np.asarray(embed([t])[0], dtype=np.float32)
        return v / np.linalg.norm(v)

    def bm25(self, q, k1=1.2, b=0.75):
        if q not in self._bm25_memo:
            if len(self._bm25_memo) > 512:
                self._bm25_memo.clear()
            s = np.zeros(self.N, dtype=np.float32)
            for w in set(kiwi_tokens([q])[0].split()):
                if w not in self.df:
                    continue
                idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
                for i in range(self.N):
                    f = self.tf[i].get(w)
                    if f:
                        s[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(self.docs_tok[i]) / self.avgdl))
            self._bm25_memo[q] = s
        return self._bm25_memo[q]

    # ── 상품명 찾기 ──

    def detect_exact(self, q):
        """별칭 정확 일치 — 가장 긴 별칭의 상품들과 그 별칭들"""
        nq = norm(q)
        hits = [(len(a), a) for a in self.alias if a in nq]
        if not hits:
            return set(), []
        L = max(h[0] for h in hits)
        als = [a for n, a in hits if n == L]
        return set().union(*(self.alias[a] for a in als)), als

    def sale_status_exact(self, q):
        """질문 속 상품 이름을 판매 중지 목록과 대조 → [{"name", "status": discontinued | both, "len"}] (겹치면 긴 것)"""
        nq = norm(q)
        _, als = self.detect_exact(q)
        # 판매 중 상품의 더 긴 별칭 안에 들어 있는 판매 중지 이름은 무시 ('신한 클린플러스 보증대출' 속 '보증대출')
        hits = sorted(((len(k), k) for k in self._disc_exact if k in nq and not any(k in a and len(a) > len(k) for a in als)), reverse=True)
        out, used = [], []
        for L, k in hits:
            s = nq.find(k)
            if any(s < e and s + L > b for b, e in used):
                continue
            used.append((s, s + L))
            name, st = self._disc_exact[k]
            if name not in [o["name"] for o in out]:
                out.append({"name": name, "status": st, "len": L})
        return out

    def disc_blocks(self, q, alias_len=0):
        """판매 중지 목록에만 있는 이름이 별칭 일치보다 길게 질문에 있으면 판매 중 상품으로 찾거나 추정하지 않음"""
        return any(s["status"] == "discontinued" and s["len"] > alias_len for s in self.sale_status_exact(q))

    def alias_ambiguous(self, als):
        """맞은 별칭이 정식 이름이 아니라 일부에서 만든 것이거나, 그 별칭을 이름 안에 품은 다른 상품이 있으면 애매함"""
        for a in als:
            if a not in self.full_names:
                return True
            owners = {self.pid_name[x] for x in self.alias[a]}
            if any(a in nn for n, nn in self.all_names_n.items() if n not in owners):
                return True
        return False

    def adjacent_extra(self, q, als):
        """맞은 이름 바로 앞에 붙은 명사(띄어 써도), 바로 뒤에 띄어쓰기 없이 붙은 명사가 있으면 애매함
        ('꽃온 Npay biz 신한통장', '올리브영 SOL통장별빛바구니'). 조사·어미·'신한' 같은 머리말은 무시"""
        if not als:
            return False
        keep = [i for i, ch in enumerate(q) if norm(ch)]  # 정규화에서 살아남는 글자의 원문 위치
        nq = norm(q)
        s = nq.find(als[0])
        if s < 0 or len(keep) != len(nq):
            return False
        rs, re_ = keep[s], keep[s + len(als[0]) - 1] + 1
        toks = kiwi().tokenize(q)
        before = [tk for tk in toks if tk.start + tk.len <= rs]
        if before:
            tk = before[-1]
            gap = q[tk.start + tk.len:rs]
            if tk.tag in NOUN_TAGS and not gap.strip(" '‘’\"“”") and norm(tk.form) not in PREFIX_OK:
                return True
        while re_ < len(q) and q[re_] in ")]）』」'\"’”":  # 이름 끝 닫는 괄호·따옴표는 이름의 일부
            re_ += 1
        return any(tk.tag in NOUN_TAGS and (tk.start < re_ < tk.start + tk.len or tk.start == re_) for tk in toks)

    def needs_verify(self, q, als):
        return self.alias_ambiguous(als) or self.adjacent_extra(q, als)

    def _candidates(self, q, M, names, bgs, k):
        """이름 임베딩 상위 k ∪ 글자쌍 겹침 상위 k"""
        v = self.text_vec(q)
        qb = bigrams(q)
        sb = np.array([len(qb & b) / max(1, len(b)) for b in bgs])
        out = []
        for i in list(np.argsort(-(M @ v))[:k]) + list(np.argsort(-sb, kind="stable")[:k]):
            if names[i] not in out:
                out.append(names[i])
        return out

    def pick_names(self, q, keep=None):
        """후보 중 LLM 이 고른 이름 (판매 중지 이름 포함). keep: 별칭으로 찾은 상품 — 후보 맨 앞에 넣어 맞는지 확인"""
        cands = self._candidates(q, self.NE, self.names, self.name_bg, 10)
        if keep:
            first = sorted({n for n, ps in self.name_pids.items() if ps & set(keep)})
            cands = first + [c for c in cands if c not in first]
        cands += self._candidates(q, self.DE, self.disc_names, self.disc_bg, 5)
        key = text_hash(f"{config.LLM_MODEL}|{PICK_INSTR}|{q}|{cands}")
        picked = self.pick_cache.get(key)
        if picked is None:
            class Pick(BaseModel):
                ids: list[int]
                note: str
            body = "\n".join(f"{n}. {c} ({self.name_cat[c]})" if c in self.name_cat else f"{n}. {c} ({self.disc_cat[c]}, 판매 중지)"
                             for n, c in enumerate(cands))
            r = self.client().responses.parse(model=config.LLM_MODEL, instructions=PICK_INSTR, text_format=Pick,
                                              input=f"[질문] {q}\n\n[후보 상품]\n{body}").output_parsed
            picked = [cands[n] for n in dict.fromkeys(r.ids) if 0 <= n < len(cands)][:3]
            self.pick_cache.put(key, picked)
        return list(picked)

    def detect_full(self, q):
        """→ (판매 중 상품 pids, LLM 이 고른 판매 중지 이름들)"""
        pids, als = self.detect_exact(q)
        if self.disc_blocks(q, len(als[0]) if als else 0):
            return set(), []
        if pids and not self.needs_verify(q, als):
            return pids, []
        names = self.pick_names(q, keep=pids or None)
        on = [n for n in names if n in self.name_pids]
        return (set().union(*(self.name_pids[n] for n in on)) if on else set()), [n for n in names if n not in self.name_pids]

    def detect_products(self, q):
        return self.detect_full(q)[0]

    def detect_how(self, q):
        """상품을 어떻게 찾았나: exact(별칭 확실) · verified(별칭이 애매해 LLM 확인) · fallback(LLM 이 고름) · discontinued · none"""
        pids, als = self.detect_exact(q)
        if self.disc_blocks(q, len(als[0]) if als else 0):
            return "discontinued"
        if pids:
            return "verified" if self.needs_verify(q, als) else "exact"
        return "fallback" if self.detect_products(q) else "none"

    def detect_groups(self, q):
        """두 상품 비교: 서로 겹치지 않는 별칭 일치를 긴 것부터, 이름 사이에 연결어(와·과·랑·및·vs…)가 있을 때만. 아니면 [상품 묶음 하나]"""
        nq = norm(q)
        hits = sorted(((len(a), a) for a in self.alias if a in nq), reverse=True)
        spans, groups = [], []
        for L, a in hits:
            s = nq.find(a)
            if any(s < e and s + L > b for b, e in spans):
                continue
            g = set(self.alias[a])
            if any(g <= x for x in groups):
                continue
            spans.append((s, s + L))
            groups.append(g)
        if len(groups) >= 2:
            keep = [i for i, ch in enumerate(q) if norm(ch)]
            if len(keep) == len(nq):
                sp = sorted(zip(spans, groups))
                for (s1, _g1), (s2, _g2) in zip(sp, sp[1:]):
                    if not re.search(r"(와|과|랑|하고|및|,|vs|VS|·)", q[keep[s1[1] - 1] + 1:keep[s2[0]]]):
                        groups = []
                        break
        if len(groups) < 2:
            p = self.detect_products(q)
            return [p] if p else []
        return groups

    def sale_status(self, q):
        """판매 상태: 판매 중지 목록 정확 대조 + 상품명 찾기(LLM)가 고른 판매 중지 이름 (inferred)"""
        out = self.sale_status_exact(q)
        names = {o["name"] for o in out}
        if not any(o["status"] == "discontinued" for o in out):
            for n in self.detect_full(q)[1]:
                if n not in names:
                    out.append({"name": n, "status": "discontinued", "len": 0, "inferred": True})
        return out

    # ── 순위 ──

    def topic_query(self, q, pids):
        """찾은 상품의 이름(어절)을 뺀 주제 — 좁힌 범위 안 순위용 (상품 이름이 붙은 조각만 올라오는 쏠림 방지)"""
        out = q
        for p in pids:
            for w in sorted(set(re.split(r"[\s()（）\[\]·,]+", self.pid_name.get(p, ""))), key=len, reverse=True):
                if len(w) >= 2 and w not in ("신한", "신한은행"):
                    out = out.replace(w, " ")
        out = re.sub(r"(^|\s)(신한은행|신한)(?=\s|$)", " ", out)
        return re.sub(r"\s+", " ", out).strip()

    def routed(self, q, qv, pids=None, topic=False):
        """상품으로 좁힌 결합 순위 (상품을 못 찾으면 전체). pids: 상품명 찾기 대신 이 상품들로 (비교 질문의 상품별 검색)"""
        vec, kw = self.E @ qv, self.bm25(q)
        pids = self.detect_products(q) if pids is None else pids
        allowed = None
        if pids:
            if topic:
                qt = self.topic_query(q, pids)
                if qt != q and len(norm(qt)) >= 4:
                    vec, kw = self.E @ self.text_vec(qt), self.bm25(qt)
            docs = set().union(*(self.links[p] for p in pids))
            clauses = set().union(*(self.doc_clause[d] for d in docs))
            allowed = np.array([(i["kind"] == "chunk" and i["doc_id"] in docs) or (i["kind"] == "clause" and i["clause_id"] in clauses)
                                or i["kind"] == "term" for i in self.items])
        score = np.zeros(self.N)
        for r, i in enumerate(np.argsort(-vec)):
            score[i] += 1 / (60 + r)
        for r, i in enumerate(np.argsort(-kw)):
            if kw[i] > 0:
                score[i] += 1 / (60 + r)
        if allowed is not None:
            score = np.where(allowed, score, -1)
        return [int(i) for i in np.argsort(-score)]

    def ranked(self, q, qv, pids=None):
        """최종 순위: 재순위 후보(결합 상위 7 + 주제 순위 상위 3)를 LLM 점수 순으로, 나머지는 결합 순서대로"""
        base = self.routed(q, qv, pids=pids)
        by_topic = self.routed(q, qv, pids=pids, topic=True)
        head = base[:FROM_BASE]
        cand = head + [i for i in by_topic if i not in set(head)][:FROM_TOPIC]
        return self.rerank_score(q, cand) + [i for i in base if i not in set(cand)]

    def rerank_score(self, q, cand):
        """후보마다 0~3점 (점수 높은 순, 같으면 원래 순위)"""
        key = text_hash("score|" + q + "|" + ",".join(map(str, cand)) + "|" + config.LLM_MODEL)
        sc = self.score_cache.get(key)
        if sc is None:
            class Item(BaseModel):
                id: int
                score: int

            class Scored(BaseModel):
                items: list[Item]
            body = "\n\n".join(f"<{n}>\n{self.items[i]['text'][:RERANK_CHARS]}" for n, i in enumerate(cand))
            r = self.client().responses.parse(model=config.LLM_MODEL, instructions=SCORE_INSTR, text_format=Scored,
                                              input=f"질문: {q}\n\n후보:\n{body}")
            sc = {str(cand[it.id]): it.score for it in r.output_parsed.items if 0 <= it.id < len(cand)}
            self.score_cache.put(key, sc)
        sc = {int(k): v for k, v in sc.items()}
        self.q_scores.setdefault(q, {}).update(sc)
        base = {i: r for r, i in enumerate(cand)}
        return sorted(cand, key=lambda i: (-sc.get(i, 0), base[i]))

    def label(self, i):
        """근거 표시 이름: '[문서 제목] 섹션 (p.쪽)' 머리 → '문서 제목 > 섹션 (p.쪽)'"""
        h = self.items[i]["text"].split("\n", 1)[0].replace("\x07", "")
        m = re.match(r"\[([^\]]+)\]\s*(.*)", h)
        return f"{m.group(1)} > {m.group(2)}" if m and m.group(2) else h

    def display(self, i):
        """화면 근거 문서 글 (worklog 59장): 한 섹션이 조각 여러 개로 나뉜 경우(49개 섹션) 같은 문서에서 섹션이 겹치는 조각을
        파일 순서대로 이어 섹션 전체를 보여 줌. 이어 붙인 조각의 머리 줄('[문서] 섹션 (p.쪽)')은 첫 조각 것만 남김.
        LLM 에 넘기는 글(items[i]['text'])은 그대로"""
        it = self.items[i]
        if it["kind"] != "chunk":
            return it["text"]
        if not hasattr(self, "_by_doc"):
            by = {}
            for j, x in enumerate(self.items):
                if x["kind"] == "chunk":
                    by.setdefault(x["doc_id"], []).append(j)
            self._by_doc = by
        group, seqs, changed = {i}, set(it["seqs"]), True
        while changed:  # 겹치는 섹션을 따라 이어진 조각 모두 (A-B, B-C 처럼 이어진 경우까지)
            changed = False
            for j in self._by_doc[it["doc_id"]]:
                if j not in group and self.items[j]["seqs"] & seqs:
                    group.add(j)
                    seqs |= self.items[j]["seqs"]
                    changed = True
        if len(group) == 1:
            return it["text"]
        parts = [self.items[j]["text"] for j in sorted(group)]
        head, _, body = parts[0].partition("\n")
        head = re.sub(r"\s*\(\d+/\d+\)", "", head)  # 이어 붙였으니 '(1/3)' 같은 조각 번호는 뺌
        return "\n".join([head, body] + [p.split("\n", 1)[1] if "\n" in p else p for p in parts[1:]])
