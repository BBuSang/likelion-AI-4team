"""동명이인 구분 (추정 인물 묶기).  정답 인물 ID 는 쓰지 않고, 원문 속 단서만으로 '이 이름은 어느 사람인가'를 정한다.

적재할 때 전체 문서의 탐지 결과를 한 번에 받아 이름 언급마다 person_key 를 붙인다.
pseudonymize() 는 person_key 가 있으면 HMAC("person|이름|key") 로 사람 단위 토큰을 만든다.

언급 하나에서 뽑는 단서
  - 닻(anchor)  : 이름 뒤 같은 문단 안, 다음 이름이 나오기 전까지의 전화·이메일·주민번호·계좌·카드  (가장 확실)
                 예) '민원인: 김민준 (연락처 010-0144-3268)', '고객명: 김민준\n연락처: 010-…'
  - 소속(aff)   : 이름 바로 뒤 괄호 '(고객센터)', '(미르보안)' / 앞 단어가 '…팀·센터·본부' / '미르보안 이서연 책임' 꼴
  - 역할(role)  : 고객 쪽 ('민원인:', '고객명:', '… 고객') / 직원 쪽 ('상담원:', '처리자:', '참석:', '… 대리')
묶는 순서
  1. 같은 문서 안 이름만 쓴 표기('시우')는 그 문서에 성 포함 이름('문시우')이 하나뿐이면 그 이름으로 본다
  2. 이름별로 닻·소속을 공유하는 언급끼리 묶는다 (Union-Find)
  3. 역할이 같고 닻·소속이 서로 어긋나지 않는 묶음은 합친다 (같은 고객이 문서마다 전화/이메일을 따로 남긴 경우)
  4. 단서 없는 언급: 같은 문서의 같은 이름 → 역할이 맞는 묶음이 하나 → 묶음이 하나뿐 → 아니면 미확정
     · 35장: 붙일 묶음을 고를 때 직급(이름 바로 뒤 '대리'·'과장')이 다른 묶음, 문서의 고객 번호(카드·계좌·주민번호)가
       묶음의 같은 종류 번호와 모두 다른 묶음은 뺀다. 번호 때문에 빠진 고객은 그 번호를 단서로 새 사람(doc_id_new)
     · 직급이 다르다고 묶음끼리 합치는 걸 막지는 않음 — 같은 사람을 문서마다 다른 직급으로 부르는 경우가 있음(synth2 20명)
     · 미확정은 이름 기준 토큰(지금까지 방식)으로 둔다. 잘못 합치는 것이 쪼개지는 것보다 위험하므로 억지로 붙이지 않는다.
한 문서에 같은 이름의 다른 사람이 함께 나올 수 있으므로(예: 내부 점검자 이서연 + 수탁사 담당자 이서연)
'한 문서 안 같은 이름 = 같은 사람' 가정은 단서 없는 언급에만 쓴다.
"""
import collections
import re

from .pii_pipeline import normalize, split_name

ANCHOR_LABELS = ("phone_number", "email", "resident_registration_number", "account_number", "card_number")
ANCHOR_WINDOW = 100   # 이름 뒤 몇 글자까지 닻을 찾을지 (문단 경계·다음 이름에서 먼저 끊김)

TITLES = ("본부장", "센터장", "팀장", "실장", "부장", "차장", "과장", "대리", "주임", "사원", "매니저", "책임", "선임",
          "수석", "이사", "대표", "상담원", "심사역", "계장", "행원")
_TITLE_RE = "|".join(TITLES)
_NOT_AFF = set(TITLES) | {"연차", "휴가", "외근", "출장", "본인", "대리인", "고객", "부재", "퇴사", "병가", "교육"}
_ORG_SUFFIX = re.compile(r"(팀|센터|본부|부서|지점|사업부|실)$")
_CUSTOMER_LABEL = r"고객명?|민원인|신청인|송금인|수취인|예금주|명의인|가입자|차주|보증인|위임인|피해자|변경\s*요청\s*고객|고객님"
_STAFF_LABEL = (r"상담원|처리자|처리\s*담당|담당자?|작성자|참석자?|불참|보고자|점검자|심사역|결재자?|승인자|접수자|"
                r"발신자?|수신자?|참조|보낸\s*사람|받는\s*사람|갑(?=\s*[:：])")
# 외부(수탁사·협력사 등) 사람: 내부 직원과 이름이 같아도 다른 사람일 수 있어 따로 본다
_EXTERNAL_LABEL = r"수탁사|협력사|위탁사|업체|외부|용역사|을(?=\s*[:：])"
# 항목 이름은 단어 전체가 맞을 때만 ('고객보호팀'의 '고객', '보고서를'의 '을'은 아님)
_W = r"(?<![가-힣])(?:{})(?![가-힣])"
_LABEL_RE = re.compile("(?P<x>" + _W.format(_EXTERNAL_LABEL) + r"(?:\s*담당자?)?)|(?P<c>" + _W.format(_CUSTOMER_LABEL)
                       + ")|(?P<s>" + _W.format(_STAFF_LABEL) + ")")
_JOSA_END = re.compile(r"[은는이가을를에의도로와과께서만]$")


# ---------------------------------------------------------------- 언급 단서 뽑기
def _line_bounds(text, s, e):
    ls = text.rfind("\n", 0, s) + 1
    le = text.find("\n", e)
    return ls, (len(text) if le < 0 else le)


# 이름(또는 앞 식별자)과 다음 식별자 사이에 허용하는 글: 문장부호 + 짧은 단어 하나 + 항목 이름
_GAP_OK = re.compile(r"[\s(（,/:：·]*(?:[가-힣]{1,4}\s*)?(?:연락처|휴대폰|휴대전화|전화번호?|핸드폰|이메일|메일|계좌번호?|"
                     r"카드번호?|주민등록번호|주민번호)?[\s(（,/:：]*")
# '고객명: 김민준' 다음 줄의 '연락처: 010-…' 처럼 항목: 값 줄. 16장: 글머리표 '- 고객명:' '• 연락처:' 도 허용
_KV_LINE = re.compile(r"\s*(?:[-*•·]\s*)?[가-힣A-Za-z ]{1,12}\s*[:：]\s*")
# 16장: '고객명: 석준서 고객님' 처럼 이름 뒤에 호칭만 있는 줄도 '항목: 이름' 줄로 봄
_NAME_LINE_REST = re.compile(r"[\s()（）]*(?:고객님|고객|님|씨)?[\s()（）]*")


def _anchors(text, e, spans, i):
    """이름(spans[i])에 딸린 식별자. 이름 바로 뒤(사이엔 문장부호·짧은 단어만) 또는 '항목: 이름' 다음의 '항목: 값' 줄."""
    if re.match(r"\s*[:：]", text[e:e + 2]):
        return []  # '이윤서: 그 고객 연락처 010-…' — 말한 사람·항목 이름이면 뒤 내용은 그 사람 것이 아님
    ids = sorted((s2 for s2 in spans if s2["label"] in ANCHOR_LABELS and s2["start"] >= e), key=lambda s: s["start"])
    nxt_person = next((s2["start"] for s2 in spans[i + 1:] if s2["label"] == "person" and s2["start"] >= e), len(text))
    found, pos = [], e
    for s2 in ids:  # 같은 줄: 이름 → (식별자 →)* 사이가 짧을 때만 이어 붙임
        if s2["start"] >= nxt_person or "\n" in text[pos:s2["start"]] or not _GAP_OK.fullmatch(text[pos:s2["start"]]):
            break
        found.append(s2)
        pos = s2["end"]
    _, le = _line_bounds(text, e, e)
    if not found and _KV_LINE.match(text[text.rfind("\n", 0, e) + 1:e]) and _NAME_LINE_REST.fullmatch(text[e:le]):
        # '고객명: 김민준' 줄 → 바로 이어지는 '항목: 값' 줄들 (빈 줄·다른 이름에서 멈춤)
        line_start = le + 1
        while line_start < len(text) and line_start < nxt_person:
            line_end = text.find("\n", line_start)
            line_end = len(text) if line_end < 0 else line_end
            kv = _KV_LINE.match(text, line_start)
            if not kv or line_end == line_start:
                break
            hit = [s2 for s2 in ids if s2["start"] == kv.end()]
            found += hit
            line_start = line_end + 1
    return [(s2["label"], normalize(s2["label"], text[s2["start"]:s2["end"]])) for s2 in found]


def _affiliation(before, after):
    m = re.match(rf"\s*(?:{_TITLE_RE})?\s*[(（]([^)）0-9@]{{2,15}})[)）]", after)
    if m and m.group(1).strip() not in _NOT_AFF:
        return re.sub(r"\s+", "", m.group(1))
    words = re.findall(r"[가-힣A-Za-z0-9]+", before)
    if words:
        prev = words[-1]
        if before.rstrip().endswith(prev):  # 이름 바로 앞 단어 (사이에 ':' ',' 등이 없을 때)
            if _ORG_SUFFIX.search(prev) and len(prev) >= 3:
                return prev
            if (re.match(rf"\s*(?:{_TITLE_RE})", after) and len(prev) >= 2 and not _LABEL_RE.fullmatch(prev)
                    and not _JOSA_END.search(prev)):
                return prev  # '미르보안 이서연 책임' 꼴 ('문의는 정서현 과장' 같은 조사 붙은 단어는 제외)
    return None


def _internal(aff):
    return bool(_ORG_SUFFIX.search(aff))  # …팀·센터·본부 = 우리 회사 조직


def _role(before, after, aff):
    """customer(고객) / staff(내부 직원) / external(수탁사·협력사 등 외부 사람) / None."""
    if re.match(r"\s*고객(?!센터)", after):
        return "customer"
    if aff:
        return "staff" if _internal(aff) else "external"
    last = None
    for m in _LABEL_RE.finditer(before):
        last = "external" if m.group("x") else "customer" if m.group("c") else "staff"
    if last:
        return last
    if re.match(rf"\s*(?:{_TITLE_RE})", after) or aff:
        return "staff"
    return None


def _aff_kind(after, aff):
    """소속을 어떻게 찾았나: paren '(고객센터)' / suffix '…팀' / prefix '미르보안 이서연 책임' (앞 단어 + 직함)."""
    if re.match(rf"\s*(?:{_TITLE_RE})?\s*[(（]", after):
        return "paren"
    return "suffix" if _ORG_SUFFIX.search(aff) else "prefix"


# 직급(35장): 이름 바로 뒤 직급 ('김민준 대리', '김민준대리님'). 같은 이름의 직원끼리 가르는 단서.
# 상담원·심사역처럼 직무 이름은 직급이 아니라서 뺌. 본부장·센터장·대표·이사는 드물어 같이 둠
_RANK = ("본부장", "센터장", "팀장", "실장", "부장", "차장", "과장", "대리", "주임", "사원", "매니저", "책임", "선임", "수석", "이사",
         "대표", "계장")
_RANK_RE = re.compile(r"\s?(" + "|".join(_RANK) + r")(?![가-힣]{2})")


def _rank(after):
    m = _RANK_RE.match(after)
    return m.group(1) if m else None


def mention_features(text, spans, i):
    s, e = spans[i]["start"], spans[i]["end"]
    ls, le = _line_bounds(text, s, e)
    before, after = text[ls:s], text[e:le]
    aff = _affiliation(before, after)
    return {"anchors": _anchors(text, e, spans, i), "aff": aff, "role": _role(before, after, aff), "rank": _rank(after),
            "_aff_kind": _aff_kind(after, aff) if aff else None, "_ctx": (before, after)}


def _check_prefix_affs(mentions):
    """'앞 단어 + 직함'으로 찾은 소속은 코퍼스 안에서 괄호·조직 접미사로도 확인된 조직명이고 한글일 때만 인정.
    처음 보는 문서의 'Dear 이민재 과장님', '서명 김성호 부장' 처럼 앞 단어가 조직이 아닌 경우를 걸러낸다
    (실전 1 dev/test 에선 이 꼴의 소속이 모두 다른 문서에 괄호·접미사로도 나오는 회사명이었음). 정답 없이 코퍼스만 봄."""
    vocab = {s["aff"] for _, s in mentions if s.get("_aff_kind") in ("paren", "suffix")}
    for _, s in mentions:
        if s.get("_aff_kind") == "prefix" and (s["aff"] not in vocab or not re.fullmatch(r"[가-힣]+", s["aff"])):
            before, after = s["_ctx"]
            s["aff"], s["_aff_kind"] = None, None
            s["role"] = _role(before, after, None)


# ---------------------------------------------------------------- 묶기
class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def resolve(docs, spans_by_doc):
    """docs: [{"id", "text"}], spans_by_doc: {doc_id: [span]}.

    반환: (spans_by_doc 사본 — person 구간에 "normalized", "person_key", "basis", "role", "aff" 추가, persons)
    persons: {(이름, key): {"name", "key", "role", "affs", "anchors"}}
    """
    texts = {d["id"]: d["text"] for d in docs}
    out = {doc_id: [dict(s) for s in sorted(spans, key=lambda s: s["start"])] for doc_id, spans in spans_by_doc.items()}

    # 0. 언급 목록 + 단서. 같은 문서 안 이름만 쓴 표기는 성 포함 이름으로 펼친다
    mentions = []
    for doc_id, spans in out.items():
        text = texts[doc_id]
        persons = [s for s in spans if s["label"] == "person"]
        fulls = {normalize("person", text[s["start"]:s["end"]]) for s in persons}
        fulls = {n for n in fulls if split_name(n)[0]}
        for i, s in enumerate(spans):
            if s["label"] != "person":
                continue
            name = normalize("person", text[s["start"]:s["end"]])
            if not split_name(name)[0] or (len(name) == 3 and name.endswith("이")):
                # 성 없는 이름('시우', '서연이'랑) → 이 문서의 성 포함 이름 중 하나뿐이면 그 이름
                cands = {f for f in fulls if split_name(f)[1] in (name, name[:-1]) and f != name}
                if len(cands) == 1:
                    name = cands.pop()
            s.update(normalized=name, **mention_features(text, spans, i))
            mentions.append((doc_id, s))
    _check_prefix_affs(mentions)

    # 1~2. 이름별로 닻·소속을 공유하는 언급끼리 묶기
    uf = _UF()
    for doc_id, s in mentions:
        nodes = [f"a:{lab}:{v}" for lab, v in s["anchors"]] + ([f"f:{s['aff']}"] if s["aff"] else [])
        nodes = [f"{s['normalized']}|{n}" for n in nodes]
        s["_nodes"] = nodes
        for n in nodes[1:]:
            uf.union(nodes[0], n)

    def cluster_info():
        info = collections.defaultdict(lambda: {"roles": collections.Counter(), "anchors": collections.defaultdict(set),
                                                "affs": set(), "ranks": set(), "name": None})
        for _, s in mentions:
            if s["_nodes"]:
                c = info[uf.find(s["_nodes"][0])]
                c["name"] = s["normalized"]
                if s["role"]:
                    c["roles"][s["role"]] += 1
                for lab, v in s["anchors"]:
                    c["anchors"][lab].add(v)
                if s["aff"]:
                    c["affs"].add(s["aff"])
                if s.get("rank"):
                    c["ranks"].add(s["rank"])
        for c in info.values():
            c["role"] = c["roles"].most_common(1)[0][0] if c["roles"] else None
        return info

    # 3. 역할이 같고 서로 어긋나지 않는 묶음 합치기
    def compatible(a, b):
        if a["role"] and b["role"] and a["role"] != b["role"]:
            return False  # 고객 ↔ 직원
        for lab in set(a["anchors"]) & set(b["anchors"]):
            if not a["anchors"][lab] & b["anchors"][lab]:
                return False  # 같은 종류 식별자가 서로 다름 → 다른 사람
        if a["affs"] and b["affs"] and not a["affs"] & b["affs"]:
            return False  # 소속이 다름
        if bool(a["affs"]) != bool(b["affs"]) and not (a["role"] and a["role"] == b["role"]):
            return False  # 소속 있는 직원 묶음과 식별자만 있는 묶음은 역할이 같다고 확인될 때만
        return True

    info = cluster_info()
    by_name = collections.defaultdict(list)
    for root, c in info.items():
        by_name[c["name"]].append(root)
    for name, roots in by_name.items():
        for i, r1 in enumerate(roots):
            for r2 in roots[i + 1:]:
                if uf.find(r1) != uf.find(r2) and compatible(info[uf.find(r1)], info[uf.find(r2)]):
                    uf.union(r1, r2)
                    info = cluster_info()
    info = cluster_info()
    clusters = collections.defaultdict(list)  # 이름 -> [root]
    for root, c in info.items():
        clusters[c["name"]].append(root)

    # 4. 언급마다 key 정하기
    for _, s in mentions:
        if s["_nodes"]:
            s["person_key"], s["basis"] = uf.find(s["_nodes"][0]).split("|", 1)[1], "anchor" if s["anchors"] else "aff"
    doc_ids = collections.defaultdict(set)  # 문서에 나온 식별자 (약한 닻: 이름 옆이 아니어도 같은 문서)
    for doc_id, spans in out.items():
        for s2 in spans:
            if s2["label"] in ANCHOR_LABELS:
                doc_ids[doc_id].add((s2["label"], normalize(s2["label"], texts[doc_id][s2["start"]:s2["end"]])))

    # 35장: 고객 본인 것인 경우가 대부분인 번호(카드·계좌·주민번호). 문서에 있는 값이 묶음의 같은 종류 값과 모두 다르면 다른 고객
    OWN_IDS = ("card_number", "account_number", "resident_registration_number")

    def ids_conflict(doc_id, root):
        for lab in OWN_IDS:
            mine = {v for l2, v in doc_ids[doc_id] if l2 == lab}
            theirs = info[root]["anchors"].get(lab, set())
            if mine and theirs and not mine & theirs:
                return True
        return False

    def own_id(doc_id):
        """문서에 한 값만 있는 고객 번호 → 단서 키 (묶음 키와 같은 모양 'a:종류:값')"""
        for lab in OWN_IDS:
            vals = sorted(v for l2, v in doc_ids[doc_id] if l2 == lab)
            if len(vals) == 1:
                return f"a:{lab}:{vals[0]}"
        return None

    def role_ok(role, root, rank=None, doc_id=None):
        """35장: 단서 없는 언급을 붙일 묶음인가 — 역할, 직급(아는 경우), 고객 번호 충돌"""
        if role and info[root]["role"] not in (role, None):
            return False
        if rank and info[root]["ranks"] and rank not in info[root]["ranks"]:
            return False  # 직급을 아는 언급은 직급이 다른 묶음에 안 붙임 ('김민준 과장' → '김민준 대리' 묶음 X)
        if doc_id and role == "customer" and ids_conflict(doc_id, root):
            return False
        return True

    # 16장에서 '닻 없는 이름을 역할로 나누기'를 시험했으나 역할 판단 오류로 쪼개짐이 더 늘어 넣지 않음 (synth2 +6, test +1)
    for doc_id, s in mentions:
        if s["_nodes"]:
            continue
        name, roots = s["normalized"], clusters.get(s["normalized"], [])
        root_of = {r.split("|", 1)[1]: r for r in roots}
        # 같은 문서의 같은 이름이라도 역할이 어긋나면(내부 점검자 ↔ 수탁사 담당자) 같은 사람으로 보지 않음
        rank = s.get("rank")
        same_doc = {x["person_key"] for d2, x in mentions
                    if d2 == doc_id and x["normalized"] == name and x.get("person_key") in root_of
                    and role_ok(s["role"], root_of[x["person_key"]], rank)}
        doc_anchor = [r for r in roots if role_ok(s["role"], r, rank)
                      and any((lab, v) in doc_ids[doc_id] for lab, vs in info[r]["anchors"].items() for v in vs)]
        # 역할을 아는 언급은 역할이 같거나 역할을 모르는 묶음과만 맞춤 (직급을 알면 직급도, 35장)
        role_match = [r for r in roots if s["role"] and role_ok(s["role"], r, rank, doc_id)]
        loose = [r for r in roots if s["role"] and info[r]["role"] in (s["role"], None)]  # 35장 규칙 전 기준
        if len(same_doc) == 1:
            s["person_key"], s["basis"] = same_doc.pop(), "same_doc"
        elif len(doc_anchor) == 1 and len(roots) > 1:
            s["person_key"], s["basis"] = doc_anchor[0].split("|", 1)[1], "doc_anchor"
        elif len(role_match) == 1:
            s["person_key"], s["basis"] = role_match[0].split("|", 1)[1], "role"
        elif loose and not role_match and s["role"] == "customer" and own_id(doc_id):
            # 35장: 번호가 달라 기존 고객이 아님 → 이 문서의 고객 번호를 단서로 (다른 문서의 같은 번호·이름과 이어짐)
            s["person_key"], s["basis"] = own_id(doc_id), "doc_id_new"
        elif loose and not role_match and s["role"] == "staff":
            # 35장: 직급이 다른 같은 이름 직원 → 직급이 있으면 '이름+직급' 한 사람, 없으면 미확정
            s["person_key"], s["basis"] = (f"r:staff:{rank}", "role_new") if rank else (None, "unresolved")
        elif s["role"] and roots and not role_match:
            # 역할(·직급)이 다른 같은 이름 → 새 사람. 직급까지 넣어 '대리 김민준'과 '과장 김민준'을 나눔 (35장)
            s["person_key"], s["basis"] = f"r:{s['role']}" + (f":{rank}" if rank else ""), "role_new"
        elif len(roots) == 1 and not s["role"]:
            s["person_key"], s["basis"] = roots[0].split("|", 1)[1], "unique"
        else:
            s["person_key"], s["basis"] = None, "unresolved" if roots else "name_only"

    # 역할로 만든 새 사람(r:…)도 같은 이름·같은 역할이면 같은 사람
    persons = {}
    for _, s in mentions:
        k = (s["normalized"], s["person_key"])
        p = persons.setdefault(k, {"name": s["normalized"], "key": s["person_key"], "roles": collections.Counter(),
                                   "affs": set(), "anchors": set()})
        if s["role"]:
            p["roles"][s["role"]] += 1
        if s["aff"]:
            p["affs"].add(s["aff"])
        p["anchors"].update(s["anchors"])
    for p in persons.values():
        p["role"] = p.pop("roles").most_common(1)[0][0] if p.get("roles") else None
    for _, s in mentions:
        for k in ("_nodes", "_aff_kind", "_ctx"):
            s.pop(k, None)
    return out, persons
