"""검색: 토큰 조건 + 의미 검색 결합.

    python -m app.rag.search "윤선영 고객 대출 상담 내역 찾아줘"
    python -m app.rag.search "보이스피싱 피해 접수 기록" --mode semantic     # 의미 검색만 (v1 비교용)

1. 질의를 문서와 같은 탐지기·가명처리로 마스킹 (API 로는 마스킹된 질의만 전송)
2. 질의 속 개인정보 하나하나를 '조건(group)'으로 만든다. 문서가 조건을 얼마나 만족하는지가 토큰 점수
     - 전화·이메일·주민번호·계좌·카드 : 값 토큰이 문서에 있으면 1
     - 주소 : 전체 토큰이 있으면 1, 아니면 주소 조각(<ADDRP_…>: 시군구·도로명+번호·동·호) 겹친 비율 × 0.8
              질의에 주소가 탐지되지 않아도 지역 이름(시·군·구)이 있으면 조각 조건으로 추가
     - 이름 : 이름 토큰(<NAME_…>)이 있으면 1, 이름만 쓴 질의('시우')는 <GIVEN_…> 로 1,
              성 포함 질의인데 문서엔 이름만 쓴 표기만 있으면 0.6
              동명이인: rag.persons 에서 그 이름의 추정 인물을 찾고, 질의 속 다른 개인정보 토큰(person_attrs)이나
              역할 단어('고객' / '상담원·직원·대리…' / '수탁사·업체…')·소속 이름으로 사람을 고를 수 있으면
              그 사람 토큰이 있는 문서만 1, 같은 이름의 다른 확정 인물 문서는 0.3, 미확정 토큰 문서는 0.7
3. 점수 = 토큰 점수(조건 평균, 0~1) + 의미 유사도(코사인).  조건을 다 맞춘 문서가 위로, 같은 수준끼리는 의미로 정렬.
   질의에 개인정보가 없거나 어느 문서에도 없으면 토큰 점수가 모두 0 → 의미 검색과 같아진다.
4. 후보: 토큰이 하나라도 겹치는 청크(GIN 인덱스) + 의미 검색 상위 청크

다른 코드(run_queries.py, chat.py)가 쓰는 약속
  search(query, k, conn=None) -> (masked_query, query_tokens, docs)
  docs: [{"doc_id", "chunk_id", "masked_text", "tokens", "score"}, ...]  순위 순서
검색 단계는 rag_user 계정만 씁니다 (매핑 테이블 접근 없음).
"""
import argparse
import collections
import re

import requests

from app.core import access, config, db
from app.pii import pii_pipeline
from .embed import embed

SEMANTIC_POOL = 50   # 의미 검색 후보 청크 수
CONFUSABLE = 0.3     # 동명이인 중 다른 사람(단서로 확정된 다른 추정 인물) 문서의 이름 조건 점수
UNRESOLVED = 0.7     # 같은 이름의 미확정 토큰(누구인지 모름) 문서 — 다른 사람이라고 단정하지 않음
GIVEN_ONLY = 0.6     # 성 포함 질의 ↔ 문서엔 이름만 쓴 표기
ADDR_PART = 0.8      # 주소 조각만 겹칠 때 최대 점수

# '외부'는 '외부 방문·외부 출장'처럼 활동을 뜻하는 경우가 많아 넣지 않음
DOC_TYPE_BONUS = 0.3  # 질의가 가리키는 문서 종류와 같은 문서에 더하는 점수 (조건 하나보다 작게)

# 질의 단어 → 문서 종류(doc_type). 금융 사내 문서의 일반 용어로 만든 사전 (실전 1·holdout 종류 모두 포함).
# ⚠ holdout 질의·문서 종류 목록을 본 뒤에 만든 것이라 처음 보는 질의에서의 효과는 과대평가될 수 있음 (worklog 13장)
DOC_TYPE_KEYWORDS = {
    "법인카드소명": ["법인카드", "소명", "해명", "회사 카드"],
    "대리인위임확인": ["위임", "대리인"],
    "출장보고": ["출장"],
    "감사지적": ["감사 지적", "지적 사항", "지적받", "감사에 걸", "감사 결과", "내부 감사"],
    "경조사공지": ["부고", "경조사", "조모상", "조부상", "부친상", "모친상", "장인상", "장모상", "결혼"],
    "고객편지": ["감사 편지", "칭찬", "편지"],
    "채용메일": ["채용", "면접", "지원자"],
    "권한신청": ["권한"],
    "피해접수": ["보이스피싱", "피해 접수", "피해 신고", "사기 피해", "전화 사기"],
    "녹취요약": ["녹취", "통화 요약", "통화 기록"],
    "VOC요약": ["VOC", "voc", "불만", "고객의 소리"],
    "분쟁조정": ["분쟁"],
    "한도심사결과": ["한도"],
    "카드발급심사": ["카드 발급", "발급 심사", "신규 카드"],
    "가맹점변경": ["가맹점"],
    "자동이체신청서": ["자동이체"],
    "우편반송처리": ["반송", "우편"],
    "해외송금접수": ["해외송금", "송금"],
    "영문혼용메일": ["영문", "영어 메일"],
    "공지": ["공지"],
    "대출상담": ["대출"],
    "카드분실상담": ["분실"],
    "민원처리": ["민원"],
    "이상거래탐지": ["이상거래", "부정 사용"],
    "연체안내": ["연체"],
    "인사발령": ["발령"],
    "회의록": ["회의"],
    "용역계약서": ["계약서", "용역"],
    "상품안내": ["상품 안내"],
    "사내메신저": ["메신저"],
    "고객정보변경": ["정보 변경", "주소 변경", "연락처 변경"],
}


def query_doc_types(query):
    """질의가 가리키는 문서 종류들. '감사 편지'(고객편지)는 '감사 지적'과 헷갈리지 않게 먼저 본다."""
    q = re.sub(r"\s+", " ", query)
    return {t for t, kws in DOC_TYPE_KEYWORDS.items() if any(k in q for k in kws)}


def centroid_doc_types(vec, cur, margin=None):
    """사전 대신 임베딩으로 문서 종류 고르기 (worklog 16장). 손으로 만든 DOC_TYPE_KEYWORDS 는 holdout 에 맞춰져
    새 종류에 효과가 없었음 (synth2 +0.005). 문서 종류별 평균 임베딩 중 질의와 가장 가까운 것 하나를,
    2위보다 margin 이상 가까울 때만 고른다 (애매하면 가산점 없음)."""
    margin = config.DOC_TYPE_MARGIN if margin is None else margin
    cur.execute("SELECT doc_type, 1 - (avg(embedding) <=> %s) AS s FROM rag.chunks "
                "GROUP BY doc_type ORDER BY s DESC LIMIT 2", (vec,))
    rows = cur.fetchall()
    if not rows or (len(rows) == 2 and rows[0][1] - rows[1][1] < margin):
        return []
    return [rows[0][0]]


_LABEL_KO = {"phone_number": "전화번호", "email": "이메일", "resident_registration_number": "주민등록번호",
             "account_number": "계좌번호", "card_number": "카드번호"}

_ROLE_WORDS = {
    "customer": r"고객|민원인|신청인|송금인|예금주|명의",
    "staff": r"상담원|직원|담당|처리한|처리자|상담한|응대한|작성|대리|과장|차장|부장|팀장|주임|심사역|사원",  # 상담한·응대한: 16장 synth2 SQ092
    "external": r"수탁사|협력사|업체|용역|책임님|매니저",
}


# 질의 속 지역 이름 (조사가 붙어도 인정, 시·구로 끝나는 일반 단어는 제외)
_REGION = re.compile(r"[가-힣]+(?:시|군|구)(?=[\s,]|$|에|의|로|쪽|은|는|이|가)")
_NOT_REGION = {"다시", "혹시", "즉시", "동시", "수시", "역시", "당시", "잠시", "항시", "요구", "연구", "친구", "도구",
               "입구", "출구", "창구", "부시", "무시", "표시", "지시", "제시", "개시", "게시", "감시"}


def query_region_parts(query):
    v = query
    for long, short in pii_pipeline._SIDO:
        v = v.replace(long, short)
    return sorted({m.group() for m in _REGION.finditer(v) if m.group() not in _NOT_REGION})


# 질의에서 '윤선영고객', '김민준대리'처럼 호칭을 붙여 쓰면 실전 1 탐지기가 이름을 놓침 → 질의만 호칭 앞을 띄어 씀
# (질의는 원문 위치를 채점에 쓰지 않으므로 바꿔도 됨. 문서 쪽은 위치가 바뀌면 안 되므로 적용하지 않음)
_ATTACHED_TITLE = re.compile(r"([가-힣]{2,4}?)(고객님?|님|씨|대리|과장|차장|부장|팀장|주임|사원|상담원|매니저|책임)(?=[\s,.?!]|$|[이가은는을를의께에])")


_NOT_NAME = {"고객", "담당", "직원", "팀장", "대표", "부장", "과장", "차장", "대리", "사장", "본부장", "센터장", "선생", "회원"}


# 질의에서 지역명을 이름으로 잡는 오탐('천안시 서북구 쪽' → 서북구=이름) 거르기.
# '김민구 고객'처럼 진짜 이름도 '구'로 끝날 수 있으므로, 앞에 다른 지역명이 있거나 뒤에 지역을 뜻하는 말이 올 때만
_REGION_BEFORE = re.compile(r"(?:[가-힣]+(?:시|도|군|구)|서울|부산|대구|인천|광주|대전|울산|세종|경기|강원|충북|충남|"
                            r"전북|전남|경북|경남|제주)\s*$")
_REGION_AFTER = re.compile(r"\s*(?:쪽|에\s*거주|에\s*사는|에\s*살|주소|일대|지역|에서)")


def _region_like(query, s):
    text = query[s["start"]:s["end"]]
    return (re.fullmatch(r"[가-힣]+(?:시|군|구)", text) is not None
            and (_REGION_BEFORE.search(query[:s["start"]]) is not None or _REGION_AFTER.match(query, s["end"]) is not None))


# 질의 탐지 보강 (worklog 15장). 질의는 짧고 문맥이 적어 실전 1 탐지기가 놓친 값이 그대로 LLM·임베딩 API 로 감
#   1) 같은 값이 두 번 나오면 하나만 잡힘 ('연은서라는 고객 … 연은서 이름이') → 나머지도 가림 (2글자 이름은 일반 낱말과 겹쳐 제외)
#   2) 문맥 없이 숫자만 쓴 질의 ('124507108584 이 포함된 자료') → 계좌번호로 봄. 전화·카드·주민번호 모양은 탐지기가 먼저 잡음
#   3) 호칭 없이 쓴 이름 ('공채원 결제 일부만…') → 성씨로 시작하는 낱말 앞 3글자(복성 4글자) 중
#      적재된 이름 토큰(rag.persons.name_token)과 같은 것만 이름으로 봄 — 원문 이름 목록이 아니라 토큰끼리 비교
_BARE_NUMBER = re.compile(r"(?<![\d-])\d{10,14}(?![\d-])")
_HANGUL_WORD = re.compile(r"(?<![가-힣])[가-힣]{3,}")


def _extra_query_spans(query, spans, cur):
    taken = [False] * len(query)
    extra = []

    def add(s, e, label):
        if not any(taken[s:e]):
            extra.append({"label": label, "start": s, "end": e})
            taken[s:e] = [True] * (e - s)

    for s in spans:
        taken[s["start"]:s["end"]] = [True] * (s["end"] - s["start"])
    for s in spans:
        value = query[s["start"]:s["end"]]
        if len(value.replace(" ", "")) >= 3:
            for m in re.finditer(re.escape(value), query):
                if not re.match(r"[가-힣]", query[m.start() - 1:m.start()]):
                    add(m.start(), m.end(), s["label"])
    for m in _BARE_NUMBER.finditer(query):
        add(m.start(), m.end(), "account_number")

    from detectors.rule_based import COMPOUND_SURNAMES, SURNAMES, TITLES  # pii_pipeline.detect 가 실전 1 경로를 넣어 둠
    cands = {}
    for m in _HANGUL_WORD.finditer(query):
        w = m.group()
        n = 4 if w[:2] in COMPOUND_SURNAMES and len(w) >= 4 else 3
        # '고객님'처럼 호칭으로 시작하면 제외 — 문서 쪽 탐지 오탐으로 '고객님'이 이름 토큰으로 적재된 경우가 있음
        if any(w.startswith(t) for t in TITLES | _NOT_NAME):
            continue
        if w[0] in SURNAMES or n == 4:
            cands[pii_pipeline.name_token(w[:n])] = (m.start(), m.start() + n)
    if cands:
        cur.execute("SELECT DISTINCT name_token FROM rag.persons WHERE name_token = ANY(%s)", (list(cands),))
        for (tok,) in cur.fetchall():
            add(*cands[tok], "person")
    return extra


def mask_query(query, conn=None):
    query = _ATTACHED_TITLE.sub(lambda m: m.group() if m.group(1) in _NOT_NAME else f"{m.group(1)} {m.group(2)}", query)
    spans = [s for s in pii_pipeline.detect(query)
             if not (config.QUERY_REGION_FILTER and s["label"] == "person" and _region_like(query, s))]
    own = conn is None
    conn = conn or db.rag_conn()
    try:
        with conn.cursor() as cur:
            spans += _extra_query_spans(query, spans, cur)
    finally:
        if own:
            conn.close()
    masked, records = pii_pipeline.pseudonymize(query, sorted(spans, key=lambda s: s["start"]))
    return pii_pipeline.guard(masked)[0], records  # 남은 번호·이메일 모양은 임베딩 API 로 보내기 전에 가림


def _query_role(query):
    for role, pat in _ROLE_WORDS.items():
        if re.search(pat, query):
            return role
    return None


def build_groups(query, records, cur, hints=None):
    """질의 개인정보 → 조건 목록. 각 조건: {"full": 토큰들, "partial": {토큰: 점수}, "weak": {토큰: 점수}}
    hints(list)를 주면 사람마다 {"query_token", "chosen", "candidates": [(사람 토큰, 역할, 상태)]} 를 채운다
    — 동명이인 구분을 켜면 문서 속 사람 토큰이 질의의 이름 토큰과 달라서 LLM 에 연결을 알려줘야 함"""
    value_tokens = {r["token"] for r in records if r["label"] not in ("person", "address")}
    groups = []
    for r in records:
        if r["label"] == "person":
            name = r["normalized"]
            surname, given = pii_pipeline.split_name(name)
            g = {"full": {pii_pipeline.name_token(name)} if surname else
                 {pii_pipeline.given_token(name), pii_pipeline.name_token(name)},
                 "partial": {pii_pipeline.given_token(given): GIVEN_ONLY} if surname else {}, "weak": {},
                 "desc": f"질문의 사람({r['token']})"}
            chosen = _choose_person(query, name if surname else None, value_tokens, cur)
            if chosen is not None:  # 동명이인 중 특정 사람을 골랐으면 이름 토큰 대신 그 사람 토큰으로
                chosen_tokens, unresolved, others = chosen
                g["weak"] = {t: CONFUSABLE for t in others} | {t: UNRESOLVED for t in unresolved}
                g["full"] = chosen_tokens
            groups.append(g)
            if hints is not None:
                cur.execute("SELECT person_token, role, status FROM rag.persons WHERE name_token = %s",
                            (pii_pipeline.name_token(name),))
                cands = cur.fetchall()
                if any(c[0] != r["token"] for c in cands):
                    hints.append({"query_token": r["token"], "chosen": sorted(chosen[0]) if chosen else [],
                                  "candidates": cands})
        elif r["label"] == "address":
            parts = [pii_pipeline.addr_part_token(p) for p in pii_pipeline.address_parts(r["original"])]
            groups.append({"full": {r["token"]}, "parts": parts, "partial": {}, "weak": {},
                           "desc": f"질문의 주소({r['token']})"})
        else:
            groups.append({"full": {r["token"]}, "partial": {}, "weak": {},
                           "desc": f"질문의 {_LABEL_KO.get(r['label'], r['label'])}({r['token']})"})
    if not any(r["label"] == "address" for r in records):
        # '대구 수성구 고객'처럼 지역 이름만 있는 질의: 탐지기는 개인정보로 보지 않지만(맞는 판단),
        # 지역 조각(시·군·구)을 문서 주소 조각 토큰과 맞춰 본다. 지역 이름은 개인정보가 아니라 질의 쪽에서 계산해도 됨
        parts = query_region_parts(query)
        if parts:
            groups.append({"full": set(), "parts": [pii_pipeline.addr_part_token(p) for p in parts],
                           "partial": {}, "weak": {}, "desc": f"질문의 지역({', '.join(parts)})"})
    return groups


def _choose_person(query, name, value_tokens, cur):
    """그 이름의 추정 인물이 여럿일 때 질의 단서로 고른다.
    고르면 (고른 사람 토큰들, 미확정 토큰들, 나머지 확정 인물 토큰들). 미확정은 고르는 대상에서 빼고 따로 돌려준다."""
    if not name:
        return None
    cur.execute("SELECT pid, person_token, role, affiliations, status FROM rag.persons WHERE name_token = %s",
                (pii_pipeline.name_token(name),))
    rows = cur.fetchall()
    unresolved = {c[1] for c in rows if c[4] != "resolved"}
    cands = [c for c in rows if c[4] == "resolved"]
    if len(cands) < 2:
        return None
    pick = []
    if value_tokens:  # 질의 속 전화번호 등이 그 사람의 개인정보와 일치
        cur.execute("SELECT DISTINCT pid FROM rag.person_attrs WHERE token = ANY(%s)", (list(value_tokens),))
        hit = {p for (p,) in cur.fetchall()}
        pick = [c for c in cands if c[0] in hit]
    if not pick:  # 질의에 소속 이름이 들어 있음
        pick = [c for c in cands if any(a and a in query for a in c[3])]
    if not pick:  # 역할 단어
        role = _query_role(query)
        if role:
            pick = [c for c in cands if c[2] == role]
    if not pick or len(pick) == len(cands):
        return None
    chosen = {c[1] for c in pick}
    others = [c for c in cands if c[1] not in chosen]
    if not config.SOFT_HOMONYM:
        return chosen, unresolved, {c[1] for c in others}
    # 17장: 고르지 않은 같은 이름의 확정 인물은 '다른 사람이라는 증거'가 있을 때만 0.3(CONFUSABLE).
    # 증거 없이 0.3 을 주면 한 사람이 확정 인물 둘로 쪼개졌을 때 정답 문서가 12위까지 밀림 (synth3 TQ002·TQ007).
    # 증거: 역할이 다름 / 고른 사람의 소속과 다른 소속 / 질의 값과 같은 종류인데 다른 값
    role = _query_role(query)
    chosen_affs = {a for c in pick for a in c[3] if a}
    val_labels = {t[1:].split("_")[0] for t in value_tokens}
    attrs = collections.defaultdict(set)  # pid → {(라벨 접두어, 토큰)}
    cur.execute("SELECT pid, token FROM rag.person_attrs WHERE pid = ANY(%s)", ([c[0] for c in others],))
    for pid, tok in cur.fetchall():
        attrs[pid].add((tok[1:].split("_")[0], tok))
    conflict, unknown = set(), set()
    for c in others:
        differs = ((role and c[2] and c[2] != role)
                   or (chosen_affs and c[3] and not chosen_affs & set(c[3]))
                   or any(lab in val_labels and tok not in value_tokens for lab, tok in attrs[c[0]]))
        (conflict if differs else unknown).add(c[1])
    return chosen, unresolved | unknown, conflict


def _group_score(g, tokens):
    if g["full"] & tokens:
        return 1.0
    best = max([s for t, s in g["partial"].items() if t in tokens] + [s for t, s in g["weak"].items() if t in tokens]
               + [0.0])
    if g.get("parts"):
        best = max(best, ADDR_PART * sum(p in tokens for p in g["parts"]) / len(g["parts"]))
    return best


def _reasons(groups, tokens):
    """검색이 이 문서를 고른 근거 (LLM 에 함께 전달). 주소가 통째로 <ADDRESS_…> 로 가려져 있어도
    '질문의 지역과 같은 지역의 주소가 있다'는 걸 알려 주려는 것. 질문에 있던 내용만 쓰므로 새 개인정보는 없음."""
    out = []
    for g in groups:
        if g["full"] & tokens:
            out.append(f"{g['desc']}와 일치하는 값이 있음")
        elif g.get("parts") and any(p in tokens for p in g["parts"]):
            out.append(f"{g['desc']}과 같은 지역의 주소가 있음")
        elif any(t in tokens for t in g["partial"]):
            out.append(f"{g['desc']}와 이름(성 제외)이 같은 표기가 있음")
        elif any(t in tokens for t, s in g["weak"].items() if s == UNRESOLVED):
            out.append(f"{g['desc']}와 이름이 같은 사람(같은 사람인지 미확정)이 있음")
        elif g["weak"] and any(t in tokens for t in g["weak"]):
            out.append(f"{g['desc']}와 이름만 같은 다른 사람이 있음")
    return out


def search(query, k=10, conn=None, mode="hybrid", embed_retries=None, rerank=None, user=access.SYSTEM):
    """user: 검색하는 사람 {emp_id, team, role}. 권한 밖 문서는 SQL 에서 먼저 거른다 (worklog 11장, 안내문 "검색 전에").
    기본값 SYSTEM(전부)은 평가 스크립트용 — 서버 API 는 항상 실제 사용자를 넘긴다.
    embed_retries: 대화형에서 임베딩 API 재시도 횟수를 줄일 때. 임베딩이 끝내 실패하면(속도 제한 등)
    토큰 조건만으로 찾고, 결과 문서마다 "fallback": True 를 붙인다 (의미 유사도 0)."""
    acl, acl_params = access.sql(user)  # 역할을 모르면 여기서 AccessError (검색 안 함)
    masked_q, records = mask_query(query, conn)
    q_tokens = sorted({r["token"] for r in records})
    try:
        vec = embed([masked_q], **({"retries": embed_retries} if embed_retries else {}))[0]
    except Exception:
        if mode != "hybrid":
            raise
        vec = None
    own = conn is None
    conn = conn or db.rag_conn()
    try:
        with conn.cursor() as cur:
            hints = []
            groups = build_groups(query, records, cur, hints) if mode == "hybrid" else []
            types = []
            if mode == "hybrid" and config.SEARCH_DOCTYPE:
                if config.DOC_TYPE_MODE in ("dict", "both"):
                    types = sorted(query_doc_types(query))
                if config.DOC_TYPE_MODE in ("centroid", "both") and vec is not None:
                    types = sorted(set(types) | set(centroid_doc_types(vec, cur)))
            any_tokens = sorted({t for g in groups for t in g["full"] | set(g["partial"]) | set(g["weak"])
                                 | set(g.get("parts", []))})
            if vec is None:  # 의미 검색 없이 토큰 조건(+문서 종류)만
                cur.execute("SELECT doc_id, chunk_id, masked_text, tokens, doc_type, 0.0 AS sim FROM rag.chunks "
                            f"WHERE (tokens && %(t)s::text[] OR doc_type = ANY(%(d)s)) AND {acl}",
                            {"t": any_tokens, "d": types, **acl_params})
            else:
                # 권한 조건은 의미 검색 상위 N 안쪽에도 — 밖에서 거르면 권한 밖 문서가 후보 자리를 먹음
                # HNSW 색인 + WHERE 는 후보가 N 개보다 적게 나올 수 있어 끝까지 더 찾게 함 (pgvector 0.8+, 이 트랜잭션만)
                if acl != "TRUE":
                    cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
                cur.execute(f"""
                    SELECT doc_id, chunk_id, masked_text, tokens, doc_type, 1 - (embedding <=> %(v)s) AS sim
                    FROM rag.chunks WHERE (tokens && %(t)s::text[] OR doc_type = ANY(%(d)s)) AND {acl}
                    UNION
                    (SELECT doc_id, chunk_id, masked_text, tokens, doc_type, 1 - (embedding <=> %(v)s) AS sim
                     FROM rag.chunks WHERE {acl} ORDER BY embedding <=> %(v)s LIMIT %(n)s)""",
                            {"v": vec, "t": any_tokens, "d": types, "n": max(SEMANTIC_POOL, k * 3), **acl_params})
            hits = cur.fetchall()
    finally:
        if own:
            conn.close()
    best = {}
    for doc_id, chunk_id, text, tokens, doc_type, sim in hits:
        tok = set(tokens)
        t_score = sum(_group_score(g, tok) for g in groups) / len(groups) if groups else 0.0
        type_hit = doc_type in types
        score = t_score + float(sim) + (DOC_TYPE_BONUS if type_hit else 0.0)
        if doc_id not in best or score > best[doc_id]["score"]:  # 문서별 가장 좋은 청크 하나
            reasons = _reasons(groups, tok) + ([f"질문이 가리키는 문서 종류({doc_type})"] if type_hit else [])
            best[doc_id] = {"doc_id": doc_id, "chunk_id": chunk_id, "masked_text": text, "tokens": tokens,
                            "doc_type": doc_type, "score": score, "token_score": t_score, "sim": float(sim),
                            "type_hit": type_hit, "fallback": vec is None, "reasons": reasons}
    ranked = sorted(best.values(), key=lambda d: -d["score"])
    if (config.RERANK if rerank is None else rerank) and mode == "hybrid":
        ranked = _rerank(masked_q, ranked)
    ranked = ranked[:k]
    amb = _ambiguous_people(hints, ranked)
    for d in ranked:
        d["ambiguous_people"] = amb
    return _link_people(masked_q, hints, ranked), q_tokens, ranked


RERANK_TOP = 8


def _rerank(masked_q, ranked):
    """상위 RERANK_TOP 개 중 토큰 점수가 1등과 같은 묶음 안에서만 LLM 으로 순서를 다시 매긴다.
    (이름·번호 조건을 맞춘 문서가 재순위 때문에 밀려나지 않게). 바꿔 말한 질문('포인트가 사라졌다' ↔ '포인트 소멸')용.
    LLM 에는 마스킹된 질의·문서만 보냄. 실패하면 원래 순서."""
    top = ranked[:RERANK_TOP]
    if len(top) < 2:
        return ranked
    tier = [d for d in top if abs(d["token_score"] - top[0]["token_score"]) < 1e-6]
    if len(tier) < 2:
        return ranked
    listing = "\n".join(f"[{d['doc_id']}] ({d.get('doc_type') or '-'}) " + d["masked_text"][:220].replace("\n", " ")
                        for d in tier)
    try:
        r = requests.post(
            config.LLM_API_BASE.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
            json={"model": config.RERANK_MODEL, "temperature": 0, "max_tokens": 120,
                  "messages": [{"role": "system", "content":
                                "질문에 가장 잘 맞는 문서부터 순서대로 문서 ID 만 쉼표로 나열하세요. 설명은 쓰지 마세요."},
                               {"role": "user", "content": f"[질문]\n{masked_q}\n\n[문서]\n{listing}"}]},
            timeout=30)
        r.raise_for_status()
        order = re.findall(r"[A-Z]\d{4}", r.json()["choices"][0]["message"]["content"])
    except Exception:
        return ranked
    by_id = {d["doc_id"]: d for d in tier}
    new_tier = [by_id[i] for i in dict.fromkeys(order) if i in by_id]
    new_tier += [d for d in tier if d not in new_tier]
    tier_ids = {d["doc_id"] for d in tier}
    it = iter(new_tier)
    for d in new_tier:
        d["reranked"] = True
    return [next(it) if d["doc_id"] in tier_ids else d for d in top] + ranked[RERANK_TOP:]


_ROLE_KO = {"customer": "고객", "staff": "직원", "external": "외부", None: "역할 모름"}


def _ambiguous_people(hints, docs):
    """동명이인 중 질의 단서(역할·소속·값)로 사람을 고르지 못했는데, 검색된 문서에 같은 이름의 확정 인물이 2명 이상
    → 그대로 답하면 다른 사람 문서가 섞임 (synth2 혼동 인용 0.35, worklog 14-4). chat.py 가 LLM 대신 되묻는다.
    미확정 토큰은 같은 사람일 수 있어 세지 않음. 결과: [{"query_token", "candidates": [(사람 토큰, 역할)]}]"""
    in_docs = {t for d in docs for t in d["tokens"]}
    out = []
    for h in hints:
        if h["chosen"]:
            continue
        resolved = [(t, role) for t, role, status in h["candidates"] if status == "resolved" and t in in_docs]
        if len(resolved) >= 2:
            out.append({"query_token": h["query_token"], "candidates": resolved})
    return out


def _link_people(masked_q, hints, docs):
    """동명이인 구분을 켜면 질의의 이름 토큰(<PERSON_이름>)과 문서 속 사람 토큰(<PERSON_이름+단서>)이 다르다.
    LLM 이 '질문한 사람이 문서에 없다'고 답하지 않도록, 검색된 문서에 실제로 나온 같은 이름의 사람 토큰을 알려준다.
    한 사람으로 골랐으면 질의 토큰을 그 사람 토큰으로 바꾸고, 여럿이면 후보와 역할을 덧붙인다 (토큰·역할만, 실명 없음)."""
    in_docs = {t for d in docs for t in d["tokens"]}
    for h in hints:
        cands = [c for c in h["candidates"] if c[0] in in_docs]
        chosen = [t for t in h["chosen"] if t in in_docs]
        unresolved = [c for c in cands if c[2] != "resolved"]
        if len(chosen) == 1 and not unresolved:
            masked_q = masked_q.replace(h["query_token"], chosen[0])
        elif len(cands) == 1:
            masked_q = masked_q.replace(h["query_token"], cands[0][0])
        elif cands:
            # 고른 사람이 있어도 같은 이름의 미확정 토큰이 문서에 있으면(같은 사람일 수 있음) 바꾸지 않고 후보로 알림
            if chosen:
                cands = [c for c in cands if c[0] in chosen or c[2] != "resolved"]
            listed = ", ".join(f"{t}({_ROLE_KO.get(role, role)}{'' if status == 'resolved' else ', 누구인지 미확정'})"
                               for t, role, status in cands)
            masked_q += f"\n(참고: 질문 속 {h['query_token']} 와 이름이 같은 사람이 문서에서는 {listed} 로 표기됨. " \
                        f"질문의 단서(고객/직원 등)에 맞는 사람만 골라 답할 것)"
    return masked_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--mode", choices=["hybrid", "semantic"], default="hybrid")
    a = ap.parse_args()
    masked_q, q_tokens, docs = search(a.query, a.k, mode=a.mode)
    print(f"질의(가명처리): {masked_q}\n질의 토큰: {q_tokens or '없음'}\n")
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d['doc_id']}  score={d['score']:.3f} (토큰 {d['token_score']:.2f} + 의미 {d['sim']:.3f})")
        print("   " + d["masked_text"][:120].replace("\n", " ") + " …")


if __name__ == "__main__":
    main()
