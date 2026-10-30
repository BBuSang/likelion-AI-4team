"""v2 답변 (plan_v2 P5): 질의 가명처리 → 검색 → 문맥 만들기 → HCX-007 → 자동 점검 → ops 기록.

    from chat import Chat
    bot = Chat()
    r = bot.answer("고객 김민준 건만 보여줘")
    r["answer"], r["checks"], r["docs"]

외부로 나가는 것은 가명 질의와 가명 문맥뿐. 검색·답변은 rag_user 계정만 씀 (vault 접근 없음).

문맥 만들기 (v1에서 본 오류 대응)
- 표·목록 문서는 질의 토큰이 있는 줄 + 표 머리 줄 + 제목만 넘김 → 다른 고객 줄을 붙이는 오류 차단
- 같은 이름 토큰이 검색된 문서들에서 서로 다른 개체(rag.entities)이고 역할·부서가 충돌하면
  "인물 A(직원·고객센터) / 인물 B(고객)"로 표시 → LLM이 동명이인을 섞지 않게
자동 점검
- cited_docs     답변이 인용한 문서
- unknown_tokens 답변에 나왔지만 질의·문맥에 없는 토큰 (지어낸 토큰)
- row_mismatch   한 답변 줄에 함께 나온 토큰·접수번호가 원문의 같은 줄에 함께 있지 않음 (다른 줄 내용을 붙임)
"""
import re
import sys
import time
import uuid
from pathlib import Path

V2 = Path(__file__).resolve().parent
sys.path.insert(0, str(V2))
from clients import chat  # noqa: E402
from search import Searcher  # noqa: E402

K = 3
TOKEN_RE = re.compile(r"<[A-Z]+_[0-9a-f]{8}>")
CASE_RE = re.compile(r"\b[A-Z]{2,5}-\d{3,}\b")
CITE_RE = re.compile(r"\[(H\d{4})\]")
NOT_FOUND = "관련 문서를 찾지 못했습니다."
ROLE_KO = {"customer": "고객", "staff": "직원", "external": "외부", "agent": "직원"}

SYSTEM = f"""너는 금융회사 직원의 질문에 사내 문서로 답하는 도우미다. 문서와 질문의 개인정보는 가명 토큰(<PERSON_…>, <PHONE_…> 등)으로 바뀌어 있다.
규칙
1. 제공된 [문서]에 있는 내용만 쓴다. 질문과 관련된 내용이 전혀 없으면 "{NOT_FOUND}" 한 줄만 쓴다.
2. 내용마다 출처 문서를 [H0001] 형식으로 붙인다.
3. 가명 토큰은 글자 그대로 옮긴다. 실명이나 번호를 추측하지 않는다.
4. 같은 토큰이라도 [인물 A], [인물 B]처럼 나뉘어 있으면 서로 다른 사람이다. 질문이 가리키는 인물(역할·소속)의 문서만 답하고, 다른 인물 문서는 "동명이인으로 보여 제외"라고 한 줄로 밝힌다. 구분 표시가 없으면 같은 사람이라고 단정하지 않는다.
5. 표는 질문한 사람의 줄만 보여 준다. 보이지 않는 줄의 내용을 만들지 않는다.
6. "건·기록·내역"은 업무 문서를 뜻한다. 질문한 사람은 직원이다(귀하·고객님이라고 부르지 않는다).
7. 문서마다 한 줄, 5줄 이내. 굵은 글씨·표·머리말 없이 쓴다. 규칙 번호를 말하지 않는다.
형식 예시
- [H0023] VOC-8293 포인트 소멸 문의, 처리 중
- [H0024] 직원 <PERSON_…>의 출장 보고 → 동명이인으로 보여 제외"""


# HCX-007은 시스템 지시의 형식 규칙을 자주 넘김 → 질문 바로 뒤에 다시 붙임
FORMAT = ("[답변 형식] 관련 문서마다 '- [H0000] 핵심 내용' 한 줄씩, 최대 5줄. "
          "문서 원문을 옮겨 적지 말고 요약한다. 요약·종합·참고·안내 문구를 덧붙이지 않는다.")


def clean(text):
    """형식 규칙을 넘긴 경우 정리: 굵은 글씨·머리말·구분선·※ 덧붙임 줄 제거, 깨진 인용(-H0165]) 복구."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*-{3,}\s*$", "", text, flags=re.M)
    text = re.sub(r"^\s*※.*$", "", text, flags=re.M)
    text = re.sub(r"(?<!\[)\b(H\d{4})\]", r"[\1]", text)
    text = re.sub(r"^-\[", "- [", text, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class Chat:
    def __init__(self, k=K, opts=None, log=True):
        self.k = k
        self.search = Searcher(opts)
        self.log = log

    # ───────── 문맥 ─────────
    def _strong_conflict(self, a, b):
        """LLM에 '다른 사람'이라고 알릴 만큼 확실한 충돌: 역할·부서·주민번호가 다름.
        전화·이메일·주소는 한 사람이 여러 개 쓸 수 있어 제외 (⑥ 사무실 번호 vs 휴대폰으로 같은 직원을 둘로 표시한 오류)"""
        s = self.search
        ra, rb, oa, ob = s.entity_roles[a], s.entity_roles[b], s.entity_orgs[a], s.entity_orgs[b]
        if (ra and rb and not ra & rb) or (oa and ob and not oa & ob):
            return True
        ia = {x for x in s.entity_ids[a] if x.startswith("<RRN_")}
        ib = {x for x in s.entity_ids[b] if x.startswith("<RRN_")}
        return bool(ia and ib and not ia & ib)

    def _persons(self, ranked, q_tokens):
        """질의 이름 토큰별로 검색 문서의 개체를 묶고, 충돌하는 개체가 2개 이상이면 인물 A/B 이름표."""
        labels = {}
        s = self.search
        for t in q_tokens:
            if not t.startswith("<PERSON_"):
                continue
            ents = []
            for d in ranked:
                e = s.entity.get((d, t))
                if e and e not in ents:
                    ents.append(e)
            if len(ents) < 2 or not any(self._strong_conflict(a, b) for a in ents for b in ents if a < b):
                continue
            for i, e in enumerate(ents):
                desc = ",".join(sorted(ROLE_KO.get(r, r) for r in s.entity_roles[e])) or "역할 미상"
                if s.entity_orgs[e]:
                    desc += "·" + ",".join(sorted(s.entity_orgs[e]))
                labels[(t, e)] = f"인물 {chr(65 + i)}({desc})"
        return labels

    def _doc_context(self, d, q_tokens, labels):
        s = self.search
        lines = s.doc_lines_raw[d]
        qt = set(q_tokens)
        hit = {i for i, l in enumerate(lines) if any(t in l for t in qt)}
        table = [i for i, l in enumerate(lines) if l.count("|") >= 2]
        if hit and any(i in hit for i in table):
            keep = {0} | {i for i in hit}
            for i in table:                        # 표 머리 줄 (다음 줄이 --- 구분선)
                if i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3}", lines[i + 1]):
                    keep.add(i)
            body = "\n".join(lines[i] for i in sorted(keep)) + "\n(표는 질문한 사람의 줄만 표시)"
        else:
            body = "\n".join(lines)
        notes = [f"※ 이 문서의 {t} = [{lab.split('(')[0]}] ({lab.split('(', 1)[1]}"
                 for (t, e), lab in labels.items() if s.entity.get((d, t)) == e]
        # 주소 토큰은 일반화한 지역(시·도 시·군·구)을 함께 알려 줌 (rag.address_regions, 비개인정보)
        for t in sorted(set(TOKEN_RE.findall(body))):
            if t in s.address_region:
                notes.append(f"※ {t} = {s.address_region[t]} (시·군·구까지만 표시한 주소)")
        head = f"[{d}] {s.doc_type[d]}"
        return head + ("\n" + "\n".join(notes) if notes else "") + "\n" + body

    # ───────── 답변 ─────────
    def prepare(self, query):
        t0 = time.perf_counter()
        qrec, ranked, parsed, keys = self.search.search_text(query, self.k)
        labels = self._persons(ranked, qrec["query_tokens"])
        doc_contexts = {d: self._doc_context(d, qrec["query_tokens"], labels) for d in ranked}
        return {"query_id": uuid.uuid4().hex[:12], "masked_query": qrec["masked_query"],
                "query_tokens": qrec["query_tokens"], "parsed": parsed, "docs": ranked, "keys": keys,
                "labels": {f"{t}|{e}": v for (t, e), v in labels.items()},
                "doc_contexts": doc_contexts, "context": "\n\n".join(doc_contexts.values()),
                "search_sec": time.perf_counter() - t0}

    def generate(self, prep, reasoning=None):
        t0 = time.perf_counter()
        text = chat([{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": f"[문서]\n{prep['context']}\n\n[질문]\n{prep['masked_query']}\n\n{FORMAT}"}],
                    reasoning)
        return clean(text), time.perf_counter() - t0

    def check(self, prep, answer):
        known = set(TOKEN_RE.findall(prep["context"])) | set(prep["query_tokens"])
        unknown = sorted(set(TOKEN_RE.findall(answer)) - known)
        cited = sorted(set(CITE_RE.findall(answer)))
        mismatch = []
        items_of = lambda l: set(TOKEN_RE.findall(l)) | set(CASE_RE.findall(l))  # noqa: E731
        for line in answer.split("\n"):
            items, docs = items_of(line), CITE_RE.findall(line)
            persons = {x for x in items if x.startswith("<PERSON_")}
            values = items - persons                 # 접수번호·전화·계좌 등 원문에서 줄 하나에 붙어 있는 값
            if not (persons and values and docs):
                continue
            src = [l for d in docs for l in self.search.doc_lines_raw.get(d, [])]
            # 값이 다른 사람 토큰과 함께 있는 줄에만 있고, 그 줄들 어디에도 답변 줄의 사람이 없으면 다른 사람 줄을 붙인 것
            # (녹취 "고객: <PHONE>이요"처럼 사람 토큰이 없는 줄은 판단하지 않음)
            holders = [l for l in src if values & items_of(l) and "<PERSON_" in l]
            if holders and not any(persons & items_of(l) for l in holders):
                mismatch.append(line.strip())
        not_found = answer.strip().startswith(NOT_FOUND)
        return {"cited_docs": cited, "unknown_tokens": unknown, "row_mismatch": mismatch,
                "not_found": not_found, "uncited_docs": [d for d in prep["docs"] if d not in cited]}

    def answer(self, query, reasoning=None):
        prep = self.prepare(query)
        text, llm_sec = self.generate(prep, reasoning)
        checks = self.check(prep, text)
        if self.log:
            self._log(prep, text, checks, llm_sec, reasoning)
        return {**prep, "answer": text, "checks": checks, "llm_sec": llm_sec}

    def _log(self, prep, text, checks, llm_sec, reasoning):
        from psycopg.types.json import Jsonb
        conn = self.search.conn
        with conn.cursor() as cur:
            cur.execute("INSERT INTO ops.query_log (query_id, masked_query, query_tokens, parsed, mode, k) "
                        "VALUES (%s, %s, %s, %s, 'v2', %s)",
                        (prep["query_id"], prep["masked_query"], prep["query_tokens"], Jsonb(prep["parsed"]), self.k))
            for i, d in enumerate(prep["docs"], 1):
                key = prep["keys"][d]
                cur.execute("INSERT INTO ops.retrieval_log (query_id, rank, chunk_id, doc_id, s_token, s_semantic, s_final) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                            (prep["query_id"], i, f"{d}#d", d, key[0], key[-1], 1.0 / i))
            cur.execute("INSERT INTO ops.answer_log (query_id, answer_masked, cited_docs, unknown_tokens, row_mismatch, "
                        "llm_sec, reasoning) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                        (prep["query_id"], text, checks["cited_docs"], checks["unknown_tokens"],
                         len(checks["row_mismatch"]), llm_sec, reasoning or "low(default)"))
        conn.commit()
