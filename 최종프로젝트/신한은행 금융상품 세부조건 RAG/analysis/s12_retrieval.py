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


# 상품 종류를 뜻하는 일반어 (별칭 하나가 이 말들로만 되어 있으면 특정 상품을 가리키지 않음) — 7-2
GENERIC_WORDS = {"통장", "적금", "예금", "정기예금", "정기적금", "적립식예금", "거치식예금", "입출금통장", "입출금", "저축", "종합저축",
                 "저축예금", "청약통장", "청약예금", "청약저축", "청약부금", "주택청약", "주택청약종합저축", "대출", "담보대출", "신용대출",
                 "전세대출", "전세자금대출", "주택담보대출", "기업대출", "외화예금", "외화적금", "외화통장", "외화정기예금", "기업통장",
                 "계좌", "저금통", "상품", "예금통장", "자유적금", "보통예금", "기업자유예금", "정기예금특약", "특약"}


def norm(t: str) -> str:
    return re.sub(r"[\s·ㆍ\-_'’‘\"“”「」『』《》<>,.:()（）\[\]&]", "", t).lower()


GENERIC_N = {norm(w) for w in GENERIC_WORDS} | {"신한", "신한은행", "쏠", "sol", "쏠편한", "상품"}


def bigrams(t):
    s = norm(t)
    return {s[i:i + 2] for i in range(len(s) - 1)}


PICK_CACHE = CH / "product_pick_cache.jsonl"
# 9-1: 줄임말마다 규칙을 넣지 않고 일반 원칙만 (과적합 방지)
PICK_INSTR = """너는 신한은행 상품 안내 도우미의 '상품명 찾기' 단계다. 영업점 직원의 [질문]이 신한은행 상품을 줄임말·애칭·다른 표기(영어↔한글, 띄어쓰기, 일부 글자 생략)로 부르고 있을 수 있다.
[후보 상품] 중 질문이 가리키는 상품의 번호를 ids에 넣는다.
- 질문 속 상품 호칭이 후보 이름의 줄임말이나 다른 표기로 자연스럽게 볼 수 있을 때만 고른다
- 호칭이 여러 후보에 똑같이 들어맞으면 그 후보들을 모두 고른다 (최대 3개)
- 질문이 특정 상품이 아니라 공통 규정·용어·일반 절차를 묻거나, 호칭이 어느 후보와도 맞지 않으면(없는 상품·다른 회사 상품일 수 있음) 빈 목록
- **상품을 구별하는 고유한 이름 부분**(브랜드·대상·용도 등)이 후보와 맞아야 한다. 상품 종류를 뜻하는 말(대출·적금·통장·예금·비상금·전세 등)만 겹치면 고르지 않는다
- 반대로 **호칭 안의 고유한 말은 모두** 후보 이름 안의 말(또는 그 줄임말·다른 표기: 영어↔한글, 기관 이름 줄임말)과 대응해야 한다. 후보 이름에 없는 고유한 말(브랜드·수식어)이 호칭에 덧붙어 있으면 다른 상품일 수 있으니 고르지 않는다
  (은행·브랜드 머리말 '신한'·'쏠'·'SOL', 상품 종류 말, 질문 내용의 말(금리·한도 등)은 따지지 않음) — 9-3
- 글자 몇 개가 겹친다는 이유만으로 고르지 않는다. 확실하지 않으면 빈 목록
note에 한 문장 이유."""

PICK_INSTR_DISC = PICK_INSTR.replace("note에 한 문장 이유.", """- 이름 뒤에 '(판매 중지)'가 붙은 후보는 판매가 끝난 상품이다. 호칭이 그 상품을 가리키면 그것을 고른다 (판매 중인 비슷한 상품으로 바꾸지 않음).
  판매 중 후보와 판매 중지 후보 중 어느 쪽인지 애매하면 둘 다 고른다 — 11-5
note에 한 문장 이유.""")
PICK_INSTR_V93A = PICK_INSTR
PICK_INSTR_V93B = PICK_INSTR.replace("note에 한 문장 이유.", """mention에는 질문 속 상품 호칭을 그대로 적는다 (없으면 빈 문자열).
extra_words에는 **고른 후보의 이름에 대응하지 않는, 호칭 안의 고유한 말**을 모두 적는다 (줄임말·다른 표기로 대응하면 넣지 않음, 은행 머리말·상품 종류 말은 넣지 않음). 고른 것이 없으면 빈 목록
note에 한 문장 이유.""")   # 9-3b (dev에서 버림): 판단 근거를 밖으로 꺼내 코드로 검사 — 가짜 7→6, 진짜 줄임말 -3 → 쓰지 않음
PICK_INSTR_V92 = PICK_INSTR_V93A.replace(PICK_INSTR_V93A[PICK_INSTR_V93A.index("- 반대로 **호칭"):PICK_INSTR_V93A.index("- 글자 몇 개가")], "")


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
        연결 문서 제목(종류 말 뺀 것), 이름 뒤쪽 어절들(스타트업 통장·골드테크), 앞쪽 어절들(마이홈플랜) — 단 4개 이상 상품에 겹치는 별칭과
        상품 종류 일반어로만 된 별칭(청약통장·정기예금)은 버림 (7-2: test에서 '마이홈플랜'을 못 찾고 '청약통장'으로 엉뚱한 상품을 고른 오류)"""
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
                vs |= {" ".join(words[i:]) for i in range(1, len(words))}     # 뒤쪽 어절 (스타트업 통장·골드테크)
                if n == p["name"]:   # 앞쪽 어절은 상품 이름에서만 (문서 제목 앞머리 '부가가치세' 같은 일반어가 별칭이 되지 않게) — 7-2
                    vs |= {" ".join(words[:i]) for i in range(1, len(words))}     # 앞쪽 어절 (마이홈플랜·장병내일준비)
                # 괄호 안 이름을 넣은 형태: 'Someday(썸데이) 외화적금' → '썸데이 외화적금' — 7-2
                vs |= {re.sub(r"\S*\(([^)]*)\)", r"\1", base)}
                for v in vs:
                    a = norm(v)
                    if v != n and all(norm(w) in GENERIC_WORDS for w in v.split()):
                        continue   # 상품 종류를 뜻하는 일반어만으로 된 별칭은 버림 (청약통장·정기예금·주택청약 종합저축) — 7-2
                    if len(a) >= 4 or (len(a) >= 3 and v == n):
                        cand[a].add(p["product_id"])
        full = {norm(p["name"]) for p in prods}
        # 겹침은 '상품 이름' 수로 셈 — 상품공시에 같은 이름이 여러 번 나오는 상품(신한 문화콘텐츠 보증대출 ×4)은 한 상품
        name_of = {p["product_id"]: p["name"] for p in prods}
        return {a: ps for a, ps in cand.items() if len({name_of[x] for x in ps}) <= 3 or a in full}

    def bm25(self, q, k1=1.2, b=0.75):
        """14-1: 같은 글에 대한 점수는 한 번만 계산 (한 질문에서 결합 순위·주제 순위·상품별 순위가 같은 글로 여러 번 부름). 결과는 같음"""
        memo = self.__dict__.setdefault("_bm25_memo", {})
        key = (q, k1, b)
        if key not in memo:
            if len(memo) > 512:
                memo.clear()
            memo[key] = self._bm25(q, k1, b)
        return memo[key]

    def _bm25(self, q, k1=1.2, b=0.75):
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

    FALLBACK = __import__("os").environ.get("PRODUCT_FALLBACK", "1") == "1"   # 9-1 (끄기: PRODUCT_FALLBACK=0): 별칭으로 못 찾으면 후보 뽑기(이름 임베딩 + 글자쌍) → CHAT_MODEL 고르기

    TOPIC_MODE = __import__("os").environ.get("ROUTED_TOPIC", "mix")   # 10-1: off · on(주제만, dev에서 H@1 크게 떨어짐) · mix(지금 방식 7 + 주제 3)
    TOPIC = TOPIC_MODE == "on"
    QUOTA = int(__import__("os").environ.get("RERANK_QUOTA", "0"))       # 10-1 비교용: 재순위 후보 10개 중 공통 문서·공통 조항 몫
    VERIFY = __import__("os").environ.get("PRODUCT_VERIFY", "1") == "1"
    RERANK_CHARS = int(__import__("os").environ.get("RERANK_CHARS", "1500"))   # 14-1 시험: 재순위 후보마다 읽는 글자 수
    RERANK_EFFORT = __import__("os").environ.get("RERANK_EFFORT", "")          # 14-1 시험: 재순위 추론 강도 (빈 값 = 모델 기본)

    def topic_query(self, q, pids):
        """질문에서 찾은 상품의 이름(어절 단위)을 뺀 주제 — 범위 안 순위용 (10-1)"""
        if getattr(self, "_pid_name", None) is None:
            self.alias_ambiguous([])
        out = q
        for p in pids:
            name = self._pid_name.get(p, "")
            for w in sorted(set(re.split(r"[\s()（）\[\]·,]+", name)), key=len, reverse=True):
                if len(w) >= 2 and w not in ("신한", "신한은행"):
                    out = out.replace(w, " ")
        out = re.sub(r"(^|\s)(신한은행|신한)(?=\s|$)", " ", out)
        return re.sub(r"\s+", " ", out).strip()

    def text_vec(self, t):
        """글의 벡터 (캐시에 없으면 로컬 모델로 한 번 계산해 캐시에 추가)"""
        import hashlib
        h = hashlib.sha256(t.encode()).hexdigest()
        if h not in self.cache:
            import threading
            from s10_chunks import local_model
            if not hasattr(Index, "_name_lock"):
                Index._name_lock = threading.Lock()
            with Index._name_lock:   # 14-1: 불러오기와 계산 모두 한 번에 하나씩 (상품별 재순위를 동시에 돌릴 때)
                if getattr(self, "_lm", None) is None:
                    self._lm = local_model(self.model)
                v = self._lm.encode([t], normalize_embeddings=True)[0]
            self.cache[h] = [round(float(x), 6) for x in v]
            with open(CH / "embed_cache.jsonl", "a") as f:
                f.write(json.dumps({"hash": h, "model": self.model, "vec": self.cache[h]}) + "\n")
        v = np.array(self.cache[h], dtype=np.float32)
        return v / np.linalg.norm(v)

    def rerank_candidates(self, base, k=10):
        """재순위 후보 (기본: routed 상위 k). QUOTA>0이면 k개 중 그만큼을 공통 문서·공통 조항 몫으로 (10-1 비교용)"""
        if self.QUOTA <= 0:
            return base[:k]
        if getattr(self, "_shared_docs", None) is None:
            n = Counter(d for ds in self.links.values() for d in ds)
            self._shared_docs = {d for d, c in n.items() if c >= 10}
        is_common = lambda i: self.items[i]["kind"] == "clause" or (self.items[i]["kind"] == "chunk" and self.items[i]["doc_id"] in self._shared_docs)
        head = base[:k - self.QUOTA]
        extra = [i for i in base[k - self.QUOTA:] if is_common(i)][:self.QUOTA]
        rest = [i for i in base[k - self.QUOTA:] if i not in set(extra)]
        return head + extra + rest[:k - len(head) - len(extra)]   # 9-2 (끄기: PRODUCT_VERIFY=0): 애매한 별칭 일치는 LLM으로 확인

    def sale_status(self, q):
        """정확 일치 대조 + 상품명 찾기(LLM)가 고른 판매 중지 이름 (11-5, inferred=True)"""
        out = self.sale_status_exact(q)
        names = {o["name"] for o in out}
        if self.DISC_CANDS and not any(o["status"] == "discontinued" for o in out):
            for n in self.detect_full(q)[1]:
                if n not in names:
                    out.append({"name": n, "status": "discontinued", "len": 0, "inferred": True})
        return out

    def sale_status_exact(self, q):
        """11-2 판매 상태: 질문 속 상품 이름을 판매 중 목록(상품공시 연결표)·판매 중지 목록과 대조
        → [{"name", "status": discontinued | both | on_sale, "len"}] (판매 중지 이름은 정확 일치만, 겹치면 긴 것)"""
        if getattr(self, "_disc", None) is None:
            disc = list(csv.DictReader(open(META / "discontinued.csv")))
            onsale = {p["name"] for p in csv.DictReader(open(META / "products.csv"))}
            onsale_n = [norm(n) for n in onsale]
            self._disc = {}
            for r in disc:
                for v in {r["name"], r["name"].replace("신한은행", "").replace("신한", "").strip()}:
                    k = norm(v)
                    # '신한'을 뺀 짧은 이름(6글자 미만, 예: '보증대출')은 줄임말 속 일부로 잘못 잡히므로 원래 이름 그대로일 때만
                    # 짧은 이름이 판매 중 상품 이름 2개 이상 안에 들어 있으면('보증대출') 이름 일부로 흔히 쓰이는 말이라 대조하지 않음
                    if len(k) < 6 and sum(k in x for x in onsale_n) >= 2:
                        continue
                    if len(k) >= 6 or (v == r["name"] and len(k) >= 4):
                        self._disc[k] = (r["name"], "both" if (r["also_on_sale"] == "1" or r["name"] in onsale) else "discontinued")
        nq = norm(q)
        _, als = self.detect_exact_info(q)
        # 판매 중 상품의 더 긴 별칭 안에 들어 있는 판매 중지 이름은 무시 ('신한 클린플러스 보증대출' 속 '보증대출')
        hits = sorted(((len(k), k) for k in self._disc if k in nq and not any(k in a and len(a) > len(k) for a in als)), reverse=True)
        out, used = [], []
        for L, k in hits:
            s = nq.find(k)
            if any(s < e and s + L > b for b, e in used):
                continue
            used.append((s, s + L))
            name, st = self._disc[k]
            if name not in [o["name"] for o in out]:
                out.append({"name": name, "status": st, "len": L})
        return out

    def disc_blocks(self, q, alias_len=0):
        """판매 중지 목록에만 있는 이름이 질문에 있고 별칭 일치보다 길면 True → 판매 중 상품으로 찾거나 추정하지 않음 (11-2, SD013 '신한 희망대출'→새희망드림대출 방지)"""
        return any(s["status"] == "discontinued" and s["len"] > alias_len for s in self.sale_status_exact(q))

    def detect_products(self, q):
        """상품명 찾기 — 1단계 별칭 정확 일치 → 애매하면 LLM 확인(9-2), 못 찾으면 2단계(9-1). 판매 중지 이름이면 찾지 않음(11-2)"""
        return self.detect_full(q)[0]

    def detect_how(self, q):
        """상품을 어떻게 찾았는지 (답변의 '(추정)' 표시용, 9-3): exact(별칭이 확실) · verified(별칭이 애매해 LLM 확인) · fallback(별칭 없음, LLM 고르기) · none"""
        pids, als = self.detect_exact_info(q)
        if self.disc_blocks(q, len(als[0]) if als else 0):
            return "discontinued"
        if pids:
            return "verified" if self.VERIFY and self.FALLBACK and self.needs_verify(q, als) else "exact"
        return "fallback" if self.FALLBACK and self.detect_products(q) else "none"

    def detect_groups(self, q):
        """11-3 여러 상품 찾기: 질문 속 서로 겹치지 않는 별칭 일치를 긴 것부터 모두 → [상품 묶음, ...] (상품 하나면 [detect_products])
        판매 중지 이름에 걸리는 질문은 나누지 않음"""
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
        if len(groups) >= 2 and self.JOIN_ONLY:   # 11-7: 두 이름 사이에 연결어가 있을 때만 '두 상품 비교'로
            keep = [i for i, ch in enumerate(q) if norm(ch)]
            if len(keep) == len(nq):
                sp = sorted(zip(spans, groups))
                ok = True
                for (s1, g1), (s2, g2) in zip(sp, sp[1:]):
                    between = q[keep[s1[1] - 1] + 1:keep[s2[0]]]
                    if not re.search(r"(와|과|랑|하고|및|,|vs|VS|·)", between):
                        ok = False
                if not ok:
                    groups = []
        if len(groups) < 2:
            p = self.detect_products(q)
            return [p] if p else []
        return groups

    def detect_v92(self, q):
        """9-2 방식 (비교용): 9-3 고르기 원칙 추가 전 지시문"""
        global PICK_INSTR
        cur, PICK_INSTR = PICK_INSTR, PICK_INSTR_V92
        try:
            return self.detect_products(q)
        finally:
            PICK_INSTR = cur

    def detect_v93a(self, q):
        """9-3 첫 판 (비교용): 고유한 말 원칙만, extra_words 검사 없음"""
        global PICK_INSTR
        cur, PICK_INSTR = PICK_INSTR, PICK_INSTR_V93A
        try:
            return self.detect_products(q)
        finally:
            PICK_INSTR = cur

    def detect_v91(self, q):
        """9-1 방식 (비교용): 별칭이 맞으면 바로 확정, 못 찾으면 2단계"""
        pids = self.detect_exact(q)
        return pids if pids else self.detect_fallback(q)

    NOUN_TAGS = {"NNG", "NNP", "NR", "SL", "SH", "XR"}
    PREFIX_OK = {"신한", "신한은행", "쏠", "sol", "쏠편한"}

    def adjacent_extra(self, q, als):
        """11-6 애매함 신호 ③: 맞은 이름 바로 앞에 붙은 명사(띄어 써도), 또는 바로 뒤에 띄어쓰기 없이 붙은 명사가 있으면 True
        ('꽃온 Npay biz 신한통장', '올리브영 SOL통장별빛바구니') — 조사·어미·'신한' 같은 머리말은 무시"""
        if not als:
            return False
        keep = [i for i, ch in enumerate(q) if norm(ch)]   # norm에서 살아남는 글자의 원문 위치
        nq = norm(q)
        s = nq.find(als[0])
        if s < 0 or len(keep) != len(nq):
            return False
        rs, re_ = keep[s], keep[s + len(als[0]) - 1] + 1
        if getattr(Index, "_kiwi", None) is None:
            from kiwipiepy import Kiwi
            Index._kiwi = Kiwi()
        toks = Index._kiwi.tokenize(q)
        before = [tk for tk in toks if tk.start + tk.len <= rs]
        if before:
            tk = before[-1]
            gap = q[tk.start + tk.len:rs]
            if tk.tag in self.NOUN_TAGS and not gap.strip(" '‘’\"“”") and norm(tk.form) not in self.PREFIX_OK:
                return True
        while self.PAREN_SKIP and re_ < len(q) and q[re_] in ")]）』」'\"’”":   # 11-7: 이름 끝 닫는 괄호·따옴표는 이름의 일부 ('…(주택도시보증)누리온')
            re_ += 1
        for tk in toks:
            if tk.tag not in self.NOUN_TAGS:
                continue
            if tk.start < re_ < tk.start + tk.len or tk.start == re_:   # 이름 끝을 가로지르거나 바로 붙어 시작
                return True
        return False

    def needs_verify(self, q, als):
        """바로 확정하지 않고 LLM 확인으로 보낼지: 9-2 신호 ①② + 11-6 신호 ③"""
        return self.alias_ambiguous(als) or (self.ADJ and self.adjacent_extra(q, als))

    ADJ = __import__("os").environ.get("ALIAS_ADJ", "1") == "1"
    JOIN_ONLY = __import__("os").environ.get("GROUP_JOIN", "1") == "1"
    PAREN_SKIP = __import__("os").environ.get("ALIAS_PAREN", "1") == "1"   # 11-7 (끄기: ALIAS_PAREN=0)   # 11-7 (끄기: GROUP_JOIN=0)   # 11-6 (끄기: ALIAS_ADJ=0)

    def alias_ambiguous(self, als):
        """9-2: 맞은 별칭이 ① 상품 정식 이름이 아니라 일부에서 만든 별칭이거나 ② 그 별칭을 이름 안에 품은 다른 상품이 있으면 애매함
        (줄임말마다 규칙을 넣지 않는 일반 신호 — dev 오탐 6개가 모두 해당, 바로 확정되는 경우는 모두 맞음)"""
        if getattr(self, "_full_names", None) is None:
            prods = list(csv.DictReader(open(META / "products.csv")))
            self._full_names = set()
            for p in prods:
                n = p["name"]
                b = n.replace("신한은행", "").replace("신한", "").strip()
                self._full_names |= {norm(n), norm(b), norm(re.sub(r"\(.*?\)|（.*?）", "", b))}
            self._pid_name = {p["product_id"]: p["name"] for p in prods}
            self._all_names_n = {p["name"]: norm(p["name"]) for p in prods}
        for a in als:
            if a not in self._full_names:
                return True
            owners = {self._pid_name[x] for x in self.alias[a]}
            if any(a in nn for n, nn in self._all_names_n.items() if n not in owners):
                return True
        return False

    def _name_index(self):
        import threading
        if getattr(self, "_names", None) is not None:
            return self._names
        if not hasattr(Index, "_name_lock"):
            Index._name_lock = threading.Lock()
        with Index._name_lock:   # 여러 질문을 동시에 처리할 때 준비가 끝나기 전에 쓰지 않게 (8-4에서 발견)
            if getattr(self, "_names", None) is None:
                self._build_name_index()
        return self._names

    def _build_name_index(self):
        if True:
            import hashlib
            from s10_chunks import embed_local, read_cache
            prods = list(csv.DictReader(open(META / "products.csv")))
            self._name_pids = defaultdict(set)
            for p in prods:
                self._name_pids[p["name"]].add(p["product_id"])
            self._name_cat = {p["name"]: p["subcategory"] or p["category"] for p in prods}
            names = sorted(self._name_pids)
            hs = {hashlib.sha256(n.encode()).hexdigest(): n for n in names}
            embed_local(hs, self.model)
            c = read_cache(self.model)
            NE = np.array([c[hashlib.sha256(n.encode()).hexdigest()] for n in names], dtype=np.float32)
            self._NE = NE / np.linalg.norm(NE, axis=1, keepdims=True)
            self._name_bg = [bigrams(n) for n in names]
            # 11-5: 판매 중지 이름 (판매 중 목록에 없는 것, ELD는 회차를 묶음)
            dn = {}
            for r in csv.DictReader(open(META / "discontinued.csv")):
                if r["also_on_sale"] == "1" or r["name"] in self._name_pids:
                    continue
                n = re.sub(r"\d{2}-\d{2}호", "○호", r["name"])
                dn.setdefault(n, r["subcategory"] or r["category"])
            self._disc_names, self._disc_cat = sorted(dn), dn
            hs = {hashlib.sha256(n.encode()).hexdigest(): n for n in self._disc_names}
            embed_local(hs, self.model)
            c = read_cache(self.model)
            DE = np.array([c[hashlib.sha256(n.encode()).hexdigest()] for n in self._disc_names], dtype=np.float32)
            self._DE = DE / np.linalg.norm(DE, axis=1, keepdims=True)
            self._disc_bg = [bigrams(n) for n in self._disc_names]
            self._pick_cache = {}
            if PICK_CACHE.exists():
                for l in open(PICK_CACHE):
                    r = json.loads(l)
                    self._pick_cache[r["key"]] = r["picked"]
            self._names = names   # 마지막에 — 이 값이 있으면 준비 끝

    def name_candidates(self, q, k=10):
        """후보: 이름 임베딩 상위 k ∪ 글자쌍 겹침(이름 글자쌍 중 질문에 있는 비율) 상위 k — 최대 2k개 (9-1에서 정해 둔 값)"""
        import hashlib
        names = self._name_index()
        h = hashlib.sha256(q.encode()).hexdigest()
        if h not in self.cache:   # 질문 벡터가 없으면 로컬 모델로 한 번만 (모델은 한 번 불러 두고, 캐시 파일을 다시 읽지 않음 — 9-1 속도)
            from s10_chunks import local_model
            with Index._name_lock:
                if getattr(self, "_lm", None) is None:
                    self._lm = local_model(self.model)
            vec = self._lm.encode([q], normalize_embeddings=True)[0]
            self.cache[h] = [round(float(x), 6) for x in vec]
            with open(CH / "embed_cache.jsonl", "a") as f:
                f.write(json.dumps({"hash": h, "model": self.model, "vec": self.cache[h]}) + "\n")
        v = np.array(self.cache[h], dtype=np.float32)
        v /= np.linalg.norm(v)
        qb = bigrams(q)
        sb = np.array([len(qb & b) / max(1, len(b)) for b in self._name_bg])
        out = []
        for i in list(np.argsort(-(self._NE @ v))[:k]) + list(np.argsort(-sb, kind="stable")[:k]):
            if names[i] not in out:
                out.append(names[i])
        return out

    DISC_CANDS = __import__("os").environ.get("PICK_DISC", "1") == "1"   # 11-5 (끄기: PICK_DISC=0)

    def disc_candidates(self, q, k=5):
        """판매 중지 후보: 이름 임베딩 상위 k ∪ 글자쌍 상위 k (판매 중 후보는 그대로 두고 따로 더함)"""
        import hashlib
        self.name_candidates(q)   # 질문 벡터 준비
        v = np.array(self.cache[hashlib.sha256(q.encode()).hexdigest()], dtype=np.float32)
        v /= np.linalg.norm(v)
        qb = bigrams(q)
        sb = np.array([len(qb & b) / max(1, len(b)) for b in self._disc_bg])
        out = []
        for i in list(np.argsort(-(self._DE @ v))[:k]) + list(np.argsort(-sb, kind="stable")[:k]):
            if self._disc_names[i] not in out:
                out.append(self._disc_names[i])
        return out

    def detect_full(self, q):
        """상품명 찾기 + 판매 중지 추정 (11-5) → (판매 중 상품 pids, LLM이 고른 판매 중지 이름들)"""
        pids, als = self.detect_exact_info(q)
        if self.disc_blocks(q, len(als[0]) if als else 0):
            return set(), []
        if pids:
            if self.VERIFY and self.FALLBACK and self.needs_verify(q, als):
                names = self.pick_names(q, keep=pids)
            else:
                return pids, []
        elif self.FALLBACK:
            names = self.pick_names(q)
        else:
            return set(), []
        on = [n for n in names if n in self._name_pids]
        return (set().union(*(self._name_pids[n] for n in on)) if on else set()), [n for n in names if n not in self._name_pids]

    def detect_fallback(self, q, keep=None):
        names = self.pick_names(q, keep=keep)
        on = [n for n in names if n in self._name_pids]
        return set().union(*(self._name_pids[n] for n in on)) if on else set()

    def pick_names(self, q, keep=None):
        """keep: 1단계 별칭이 찾은 상품(9-2) — 후보 맨 앞에 넣고 LLM이 맞는지 확인 (아니면 다른 후보나 없음)
        11-5: 판매 중지 후보를 뒤에 더하고 '(판매 중지)'로 표시 → 고른 이름 목록(판매 중지 이름 포함)"""
        import hashlib
        from pydantic import BaseModel
        cands = self.name_candidates(q)
        if keep:
            first = sorted({n for n, ps in self._name_pids.items() if ps & set(keep)})
            cands = first + [c for c in cands if c not in first]
        dc = self.disc_candidates(q) if self.DISC_CANDS else []
        instr = PICK_INSTR_DISC if dc and PICK_INSTR == PICK_INSTR_V93A else PICK_INSTR
        cands = cands + dc
        from s8_extract import load_env
        env = load_env()
        key = hashlib.sha256(f"{env['CHAT_MODEL']}|{instr}|{q}|{cands}".encode()).hexdigest()
        if key not in self._pick_cache:
            from openai import OpenAI

            class Pick(BaseModel):
                ids: list[int]
                note: str

            class PickX(Pick):
                mention: str
                extra_words: list[str]
            with_extra = PICK_INSTR == PICK_INSTR_V93B
            client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
            body = "\n".join(f"{n}. {c} ({self._name_cat[c]})" if c in self._name_cat else f"{n}. {c} ({self._disc_cat[c]}, 판매 중지)" for n, c in enumerate(cands))
            r = client.responses.parse(model=env["CHAT_MODEL"], instructions=instr, text_format=PickX if with_extra else Pick,
                                       input=f"[질문] {q}\n\n[후보 상품]\n{body}").output_parsed
            picked = [cands[n] for n in dict.fromkeys(r.ids) if 0 <= n < len(cands)][:3]
            extra = [w for w in getattr(r, "extra_words", []) if norm(w) and norm(w) not in GENERIC_N]
            rec = {"key": key, "question": q, "cands": cands, "picked": [] if extra else picked, "note": r.note}
            if with_extra:
                rec |= {"mention": r.mention, "extra_words": extra, "picked_before_check": picked}
            self._pick_cache[key] = rec["picked"]
            with open(PICK_CACHE, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return list(self._pick_cache[key])

    def detect_exact(self, q):
        return self.detect_exact_info(q)[0]

    def detect_exact_info(self, q):
        """별칭 정확 일치 — 가장 긴 별칭의 상품들과 그 별칭들"""
        nq = norm(q)
        hits = [(len(a), a) for a in self.alias if a in nq]
        if not hits:
            return set(), []
        L = max(h[0] for h in hits)
        als = [a for l, a in hits if l == L]
        return set().union(*(self.alias[a] for a in als)), als

    def rank(self, method, q, qv, topic=None, pids=None):
        """pids: 상품명 찾기 대신 이 상품들로 좁힘 (11-3 비교 질문의 상품별 검색)"""
        vec = self.E @ qv
        if method == "vec":
            return list(np.argsort(-vec))
        if method == "rerank":
            base = self.rank("routed", q, qv)
            return self.rerank(q, [int(i) for i in base[:10]]) + base[10:]
        if method == "rerank_score":
            base = [int(i) for i in self.rank("routed", q, qv, pids=pids)]
            if self.TOPIC_MODE == "mix":   # 10-1: 재순위 후보 = 지금 방식 상위 7 + 주제만으로 순위를 매겼을 때 위에 오는 것 3 (점수가 같으면 지금 방식 순서)
                by_topic = [int(i) for i in self.rank("routed", q, qv, topic=True, pids=pids)]
                head = base[:7]
                cand = head + [i for i in by_topic if i not in set(head)][:3]
            else:
                cand = self.rerank_candidates(base)
            return self.rerank_score(q, cand) + [i for i in base if i not in set(cand)]
        kw = self.bm25(q)
        if method == "kw":
            return list(np.argsort(-kw))
        allowed = None
        if method == "routed":
            pids = self.detect_products(q) if pids is None else pids
            if pids:
                if (self.TOPIC if topic is None else topic):   # 10-1: 범위를 좁힌 뒤에는 상품 이름을 뺀 주제만으로 순위 (상품 이름이 붙은 조각만 올라오는 쏠림 방지)
                    qt = self.topic_query(q, pids)
                    if qt != q and len(norm(qt)) >= 4:
                        vec, kw = self.E @ self.text_vec(qt), self.bm25(qt)
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
        tag = "" if (self.RERANK_CHARS, self.RERANK_EFFORT) == (1500, "") else f"|{self.RERANK_CHARS}|{self.RERANK_EFFORT}"   # 14-1: 기본값이면 예전 캐시 그대로
        key = hashlib.sha256(("score|" + q + "|" + ",".join(map(str, cand)) + tag).encode()).hexdigest()
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
            body = "\n\n".join(f"<{n}>\n{self.items[i]['text'][:self.RERANK_CHARS]}" for n, i in enumerate(cand))
            extra = {"reasoning": {"effort": self.RERANK_EFFORT}} if self.RERANK_EFFORT else {}
            r = self._client.responses.parse(model=env["CHAT_MODEL"], instructions=SCORE_INSTR, text_format=Scored,
                                             input=f"질문: {q}\n\n후보:\n{body}", **extra)
            sc = {cand[it.id]: it.score for it in r.output_parsed.items if 0 <= it.id < len(cand)}
            self._rs[key] = sc
            with open(cpath, "a") as f:
                f.write(json.dumps({"key": key, "scores": {str(k): v for k, v in sc.items()}}) + "\n")
        sc = self._rs[key]
        self.__dict__.setdefault("_q_scores", {}).setdefault(q, {}).update(sc)   # 15-1: 질문별 근거 점수 (출처·함께 볼 문서 고르기용)
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
            if "section_any" in a and it["kind"] == "chunk" and it["doc_id"] == a["doc_id"] and set(a["section_any"]) & it["seqs"]:
                return True   # s43 번호 이전: 옛 청크(섹션 묶음) → 그 글을 담은 새 섹션 중 하나라도 든 청크
            if "clause_id" in a and it["kind"] == "clause" and it["clause_id"] == a["clause_id"]:
                return True
            if "term" in a and it["kind"] == "term" and it["term"] == a["term"]:
                return True
        return False

    def is_gold(self, i, gold):
        it = self.items[i]
        if "section_seq" in gold:
            return it["kind"] == "chunk" and it["doc_id"] == gold["doc_id"] and gold["section_seq"] in it["seqs"]
        if "section_any" in gold:   # 표 재추출 번호 이전(s43): 옛 섹션 하나가 새 섹션 여러 개에 나뉜 경우
            return it["kind"] == "chunk" and it["doc_id"] == gold["doc_id"] and bool(set(gold["section_any"]) & it["seqs"])
        if "clause_id" in gold:
            return it["kind"] == "clause" and it["clause_id"] == gold["clause_id"]
        return it["kind"] == "term" and it["term"] == gold.get("term")   # 표 질문(s42)처럼 문서만 있는 정답은 term 없음


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
            for t in (q["type"][0] if q["type"][0] in "AB" else q["type"], "전체"):   # common·control 같은 유형은 이름 그대로
                per[t].append(r)
                per_any[t].append(ra)
        res[m] = per
        for label, P in (("엄격", per), ("보완", per_any)):
            line = []
            for t in [k for k in P if k != "전체"] + ["전체"]:
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
