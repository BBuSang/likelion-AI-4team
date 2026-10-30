"""v2 검색 (plan_v2 P4). 검색 계정(rag_user)으로 rag 테이블만 읽음.

    from search import Searcher
    s = Searcher(opts)                    # 문서 쪽 정보(역할·지역·상태·개체)를 한 번 읽어 둠
    s.search_text("고객 김민준 건만 보여줘", k=10)   # 질의 가명처리 → 순위
    s.search_masked(qrec, k=10)          # 이미 가명처리된 질의 (pseudo_queries.jsonl 한 줄)

순위 = 아래 값을 앞에서부터 비교 (사전 검증 2차에서 고른 조합, plan_v2 3-1장)
  1. 토큰 일치 수          질의 토큰을 가진 문서가 위로 (v1과 같음)
  2. 역할 (opts.role)      질의 "고객 김민준" → 역할이 다르다고 확인된 같은 이름 문서는 토큰 일치에서 뺌 (필터)
                           opts.entity 면 문서 하나의 역할 대신 같은 개체(rag.entities) 문서들의 역할을 씀
  3. 처리 상태 (opts.status) 질의 토큰과 같은 줄에 원하는 상태(처리 중/완료)가 있는 문서
  4. 문서 종류 (opts.intent) 질의에 문서 종류 이름(위임·한도·감사 지적…)이 있으면 그 종류
  5. 지역 (opts.region)    질의의 시·도/시·군·구와 같은 주소를 가진 문서
  6. 의미 점수             가명 질의 ↔ 가명 본문 임베딩 코사인
  (4와 5 사이) 같은 줄 단어 (opts.line) 질의 토큰이 있는 표·목록 줄에 질의 단어(앞 두 글자)가 몇 개 있는가.
                           VOC 표처럼 한 문서에 여러 사람 줄이 있을 때 "그 사람 줄의 내용"을 봄 (청크 없이). 부서명은 빼고 비교
  + 1위 개체 기준 (opts.anchor): 위 순서로 1위가 된 문서의 그 사람(개체)과 충돌하는 같은 이름 개체
                           (역할·부서·같은 종류 식별 토큰이 다름)의 문서를 토큰 일치에서 빼고 다시 정렬.
                           질의에 역할 표현이 없거나 같은 역할끼리의 동명이인(부서·번호로만 구분)을 위한 것
"""
import collections
import re
import sys
from pathlib import Path

V2 = Path(__file__).resolve().parent
sys.path.insert(0, str(V2))
import db  # noqa: E402
from clients import embed  # noqa: E402
from pseudo.pipeline import SIDO_ALIAS  # noqa: E402

TOKEN_RE = re.compile(r"<[A-Z]+_[0-9a-f]{8}>")
INTENT = {
    "공지": ["공지"], "녹취요약": ["녹취"], "출장보고": ["출장"], "한도심사결과": ["한도"],
    "분쟁조정": ["분쟁"], "감사지적": ["감사 지적", "감사지적"], "VOC요약": ["VOC"],
    "법인카드소명": ["법인카드", "소명"], "경조사공지": ["경조사", "부고"], "카드발급심사": ["카드 발급", "카드발급"],
    "가맹점변경": ["가맹점"], "고객편지": ["편지"], "채용메일": ["채용"], "권한신청": ["권한"],
    "피해접수": ["피해"], "영문혼용메일": ["영문"], "대리인위임확인": ["위임"], "자동이체신청서": ["자동이체"],
    "우편반송처리": ["반송"], "해외송금접수": ["해외송금"],
}
# 처리 상태 어휘 두 벌 (worklog 43번 과적합 점검)
#   query  : holdout 질의 문구("아직", "안 끝난", "마무리")를 보고 넣은 어휘
#   general: 문서 상태 값(처리 중·처리 완료)과 표준 동의어만, 질의를 보기 전 기준으로 정한 어휘
STATUS_WORDS = {
    "query": (["처리 중", "진행 중", "아직", "안 끝난", "미처리"], ["완료", "마무리", "해결", "끝난"]),
    "general": (["처리 중", "처리중", "진행 중", "미처리", "미완료"], ["처리 완료", "완료", "종결", "해결"]),
}
# 기본값 = G6 (worklog 44번, 새 질의 세트 조정용으로 고정): 상태 일반 어휘, 같은 줄 일치 끔.
# 같은 줄 일치·질의 어휘는 holdout 질의를 보고 만든 것이라 새 질의에서 효과 0 → 끔 (holdout에서만 Hit@1 +0.057)
# anchor는 동명이인 혼동 0.045 → 0.041 대신 Recall@10 0.990 → 0.984 라 끔 (worklog 39번)
ALL_OPTS = {"intent": 1, "region": 1, "status": 1, "status_vocab": "general", "line": 0,
            "role": 1, "entity": 1, "anchor": 0}
LINE_STOP = {"고객", "고객님", "님", "건", "문서", "기록", "내역", "관련", "직원", "담당"}
ORG_WORD_RE = re.compile(r"[가-힣A-Za-z]+(?:팀|센터|본부|지점|실)(?![가-힣])")
PARTICLE_RE = re.compile(r"(님이|님의|님|이|가|은|는|을|를|에|의|로|으로|도|만|이랑|랑|하고)$")


def query_stems(masked):
    """질의 단어의 앞 두 글자 (토큰·흔한 말 제외). 형태소 분석 없이 어간 근사."""
    words = re.findall(r"[가-힣A-Za-z]+", TOKEN_RE.sub(" ", masked))
    stems = set()
    for w in words:
        w = PARTICLE_RE.sub("", w) if len(w) > 2 else w
        if len(w) >= 2 and w not in LINE_STOP:
            stems.add(w[:2])
    return stems
ID_PREFIX = ("<PHONE_", "<RRN_", "<EMAIL_", "<ACCOUNT_", "<CARD_", "<ADDRESS_")


def query_bigrams(masked):
    """질의 한글 단어의 글자 2-gram (토큰 제외). 손으로 고른 불용어·어간 없이 쓰는 표준 겹침 단위."""
    grams = set()
    for w in re.findall(r"[가-힣]+", TOKEN_RE.sub(" ", masked)):
        grams |= {w[i:i + 2] for i in range(len(w) - 1)}
    return grams


def parse_query(masked, mentions, opts=None):
    """질의 의도 (ops.query_log.parsed). 가명 질의와 질의 언급 역할만 봄."""
    opts = opts or {}
    intent = sorted(dt for dt, words in INTENT.items() if any(w in masked for w in words))
    open_w, done_w = STATUS_WORDS[opts.get("status_vocab", "query")]
    status = "처리 중" if any(w in masked for w in open_w) else \
        "처리 완료" if any(w in masked for w in done_w) else None
    region = None
    words = re.findall(r"[가-힣]+", masked)
    for i, w in enumerate(words):
        if w in SIDO_ALIAS:
            sgg = [re.sub(r"(에|쪽|으로|로)$", "", x) for x in words[i + 1:i + 3]]
            region = [SIDO_ALIAS[w], [x for x in sgg if re.fullmatch(r"[가-힣]+[시군구]", x)]]
            break
    roles = {m["token"]: m["role"] for m in mentions if m["label"] == "person" and m.get("role")}
    units = query_bigrams(masked) if opts.get("line_mode") == "bigram" else query_stems(masked)
    return {"intent": intent, "status": status, "region": region, "roles": roles, "stems": sorted(units)}


class Searcher:
    def __init__(self, opts=None, conn=None):
        self.opts = dict(ALL_OPTS if opts is None else opts)
        self.conn = conn or db.rag_conn()
        q = self.conn.execute
        self.doc_type = dict(q("SELECT doc_id, doc_type FROM rag.documents"))
        self.doc_lines_raw = {d: t.split("\n") for d, t in q("SELECT doc_id, masked_text FROM rag.documents")}
        self.doc_lines = {d: [TOKEN_RE.sub(" ", l) for l in ls] for d, ls in self.doc_lines_raw.items()}
        self.tokens = {d: set(t) for d, t in q("SELECT doc_id, tokens FROM rag.chunks WHERE kind = 'doc'")}
        self.lines = collections.defaultdict(set)      # (doc, token) → 줄 번호
        self.roles = collections.defaultdict(set)      # (doc, token) → 역할
        self.entity = {}                               # (doc, token) → entity_id
        for d, t, ln, role, eid in q("SELECT doc_id, token, line_no, role, entity_id FROM rag.mentions"):
            self.lines[(d, t)].add(ln)
            if role:
                self.roles[(d, t)].add(role)
            if eid:
                self.entity[(d, t)] = eid
        self.entity_roles = collections.defaultdict(set)
        for (d, t), eid in self.entity.items():
            self.entity_roles[eid] |= self.roles.get((d, t), set())
        self.entity_orgs = collections.defaultdict(set)
        for eid, org in q("SELECT entity_id, org_unit FROM rag.mentions WHERE entity_id IS NOT NULL AND org_unit IS NOT NULL"):
            self.entity_orgs[eid].add(org)
        self.entity_ids = collections.defaultdict(set)
        for eid, t in q("SELECT entity_id, token FROM rag.entity_links"):
            self.entity_ids[eid].add(t)
        self.address_region = {t: f"{s} {g or ''}".strip()
                               for t, s, g in q("SELECT token, sido, sigungu FROM rag.address_regions")}
        self.regions = collections.defaultdict(list)
        for d, sido, sgg in q("SELECT m.doc_id, a.sido, a.sigungu FROM rag.mentions m "
                              "JOIN rag.address_regions a ON a.token = m.token"):
            self.regions[d].append((sido, set((sgg or "").split())))
        self.status = collections.defaultdict(set)     # doc → {(line, 상태)}
        for d, ln, v in q("SELECT doc_id, line_no, value FROM rag.doc_fields WHERE field = 'status'"):
            self.status[d].add((ln, v))

    def _doc_roles(self, d, t):
        if self.opts.get("entity") and (d, t) in self.entity:
            return self.entity_roles[self.entity[(d, t)]]
        return self.roles.get((d, t), set())

    def entity_conflict(self, a, b):
        ra, rb = self.entity_roles[a], self.entity_roles[b]
        oa, ob = self.entity_orgs[a], self.entity_orgs[b]
        if (ra and rb and not ra & rb) or (oa and ob and not oa & ob):
            return True
        for pre in ID_PREFIX:
            ia = {x for x in self.entity_ids[a] if x.startswith(pre)}
            ib = {x for x in self.entity_ids[b] if x.startswith(pre)}
            if ia and ib and not ia & ib:
                return True
        return False

    def rank(self, q_tokens, parsed, sem):
        ranked, keys = self._rank(q_tokens, parsed, sem, set())
        if not (self.opts.get("anchor") and ranked):
            return ranked, keys
        top, drop = ranked[0], set()
        for t in set(q_tokens) & self.tokens.get(top, set()):
            e0 = self.entity.get((top, t))
            if not e0:
                continue
            for d in sem:
                e = self.entity.get((d, t))
                if e and e != e0 and self.entity_conflict(e0, e):
                    drop.add((d, t))
        return self._rank(q_tokens, parsed, sem, drop) if drop else (ranked, keys)

    def _rank(self, q_tokens, parsed, sem, drop):
        o, qt = self.opts, set(q_tokens)
        keys = {}
        for d, s in sem.items():
            hits = {t for t in qt & self.tokens.get(d, set()) if (d, t) not in drop}
            tok, role = len(hits), 0
            if o.get("role"):
                for t, want in parsed["roles"].items():
                    if t in hits:
                        have = self._doc_roles(d, t)
                        if have and want not in have:
                            role = -1
                if role < 0:
                    tok = 0
            status = 0
            if o.get("status") and parsed["status"]:
                want_lines = {ln for t in hits for ln in self.lines[(d, t)]}
                status = int(any(v == parsed["status"] and (ln in want_lines or not qt) for ln, v in self.status[d]))
            line = 0
            if o.get("line") and hits and parsed["stems"]:
                for ln in {ln for t in hits for ln in self.lines[(d, t)]}:
                    text = self.doc_lines[d][ln]
                    if text.count("|") < 2:         # 표·목록 줄만 (일반 문장에 쓰면 의미 점수를 덮어 Q063·Q066 하락)
                        continue
                    if o.get("line_org", 1):            # 부서명 칸(여신심사팀·인사팀)이 "심사"·"인사"와 맞는 것 방지
                        text = ORG_WORD_RE.sub(" ", text)
                    line = max(line, sum(1 for st in parsed["stems"] if st in text))
            intent = int(bool(o.get("intent") and self.doc_type.get(d) in parsed["intent"]))
            region = 0
            if o.get("region") and parsed["region"]:
                sido, sgg = parsed["region"]
                for rs, rw in self.regions.get(d, []):
                    if rs == sido:
                        region = max(region, 2 if sgg and all(x in rw for x in sgg) else 1 if not sgg else 0)
            keys[d] = (tok, role, status, intent, line, region, s)
        return sorted(keys, key=keys.get, reverse=True), keys

    def semantic(self, masked):
        v = embed(masked)
        return {d: float(s) for d, s in self.conn.execute(
            "SELECT doc_id, 1 - (embedding <=> %s) FROM rag.chunks WHERE kind = 'doc'", (v,))}

    def search_masked(self, qrec, k=10, sem=None):
        parsed = parse_query(qrec["masked_query"], qrec.get("mentions", []), self.opts)
        sem = sem if sem is not None else self.semantic(qrec["masked_query"])
        ranked, keys = self.rank(qrec["query_tokens"], parsed, sem)
        return ranked[:k], parsed, {d: keys[d] for d in ranked[:k]}

    def search_text(self, query, k=10):
        from pseudo.pipeline import process
        r = process(query)
        qrec = {"masked_query": r["masked"], "query_tokens": sorted({m["token"] for m in r["mentions"]}),
                "mentions": [{k_: v for k_, v in m.items() if k_ not in ("raw", "normalized")} for m in r["mentions"]]}
        ranked, parsed, keys = self.search_masked(qrec, k)
        return qrec, ranked, parsed, keys
