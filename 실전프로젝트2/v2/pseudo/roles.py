"""이름 언급의 역할·부서 추출 (v2 rag.mentions.role / title / org_unit).

원문 주변 단어를 적재 단계에서 로컬로만 보고, 결과는 비개인정보 값(customer·staff·external, 부서명)만 남긴다.
규칙은 dev/test 정답(roles)으로 만들고 확인한다. holdout 정답은 없음.

    extract(text, spans) -> [{"role", "title", "org"}, ...]   spans: [{"start", "end", "key"}]
      key 가 같은 언급(같은 토큰)은 문서 안에서 역할을 공유한다 (역할 단서가 한 곳에만 있어도 전파).

판단 순서 (앞이 우선)
  1. 외부 단서: 같은 줄의 수탁사·협력사·대표이사, 이름 바로 앞·괄호 안의 회사명 → external
  2. 이름 바로 뒤: 고객·대표 → customer / 직함(대리·팀장…) → staff
  3. 이름 바로 앞 단어: 담당·상담사 등 → staff / 고객 → customer
  4. 줄의 항목명 "상담원: 김은영", "참석자: A(팀장), B, C" → 항목명으로 판단
     항목명이 성명·이름이면 위 3줄 안의 제목(고객 정보 등)으로 판단
  5. 괄호: (고객센터)·(팀장) → staff
  6. 표 칸: 다음 칸이 부서명 → staff, 그 밖의 소속 → external
"""
import collections
import re

STAFF_TITLES = ("직원", "대리", "과장", "팀장", "매니저", "사원", "부장", "차장", "주임", "상담사", "상담원",
                "본부장", "지점장", "선임", "책임", "수석", "실장", "센터장", "심사역", "행원", "계장", "인턴")
STAFF_BEFORE = ("담당", "담당자", "상담사", "상담원", "작성자", "처리자", "직원", "심사역", "점검자", "검토자")
STAFF_LABELS = ("상담원", "상담사", "담당자", "담당", "처리자", "작성자", "점검자", "참석자", "참석", "불참",
                "문의", "발신", "발신자", "보고자", "심사자", "심사역", "승인자", "결재", "접수자", "확인자",
                "검토자", "처리 담당", "담당 직원", "수신",
                "소명자")                          # holdout 서식 (법인카드소명). dev/test에는 없음
CUSTOMER_LABELS = ("고객", "고객명", "대상", "대상자", "요청 고객", "민원인", "피해자", "가입자", "회원",
                   "차주", "예금주", "고객 성명",
                   "신청인", "대표자")             # holdout 서식 (분쟁조정·가맹점변경). dev/test에는 없음
NAME_LABELS = ("성명", "이름")
EXTERNAL_CUES = ("수탁사", "협력사", "협력업체", "대표이사", "외부 업체")
COMPANY_RE = re.compile(r"(시스템즈|시스템|솔루션|보안|테크|코리아|파트너스|컨설팅|주식회사|㈜|\(주\)|콜|소프트|네트웍스)$")
ORG_RE = re.compile(r"([가-힣A-Za-z]+(?:팀|센터|본부|지점|부|실))$")
LABEL_RE = re.compile(r"(?:^|[\s\-·•])([가-힣]+(?: [가-힣]+)?)\s*:\s*([^:]*)$")
LIST_OK = re.compile(r"^[가-힣\s,()·/]*$")


def _strip_title(after):
    for t in sorted(STAFF_TITLES, key=len, reverse=True):
        if after.startswith(t):
            return t
    return None


def _role_one(text, s, e):
    ls = text.rfind("\n", 0, s) + 1
    le = text.find("\n", e)
    le = len(text) if le < 0 else le
    line, before, after = text[ls:le], text[ls:s].rstrip(" "), text[e:le].lstrip(" ")
    words = before.split()
    prev_word = words[-1] if words else ""
    if prev_word == "소속" and len(words) >= 2:     # "IT운영팀 소속 박도윤" → 부서는 소속 앞 단어
        prev_word = words[-2]
    role, title, org = None, None, None

    # 1. 외부 단서
    # 외부 단서는 이름 바로 앞(12자 이내)에 있을 때만: "수탁사 담당자: 문은정" O / "협력사 점검 출장 같이 간 김민준 직원" X
    if (any(c in before[-12:] for c in EXTERNAL_CUES) or COMPANY_RE.search(prev_word)
            or re.search(r"(?:^|\s)을\s*:\s*$", before)):     # 계약서 을 = 상대 업체
        role = "external"
        org = prev_word if COMPANY_RE.search(prev_word) else None
    paren = re.match(r"\(([^)]*)\)", after)
    if paren and COMPANY_RE.search(paren.group(1).strip()):
        role, org = "external", paren.group(1).strip()

    # 2. 이름 바로 뒤
    t = _strip_title(after)
    if after.startswith(("고객", "대표")) and not after.startswith("대표이사"):
        role, title = role or "customer", "고객" if after.startswith("고객") else "대표"
    elif re.match(r"님?\s*귀하", after):          # 안내문 받는 사람 "정하은 님 귀하" (holdout 서식)
        role = role or "customer"
    elif t:
        role, title = role or "staff", t
    elif e - s <= 2 and re.match(r"씨(가|는|를|도|에게|께|[^가-힣]|$)", after):
        # 이름만 부른 "소율 씨가" = 동료를 부르는 말. "김서연 씨"(성 포함)는 고객에게도 써서 제외
        role = role or "staff"

    # 3. 이름 바로 앞 단어
    if role is None:
        if prev_word.endswith(STAFF_BEFORE) or prev_word in STAFF_TITLES:
            role = "staff"
        elif prev_word.endswith("고객"):
            role = "customer"

    # 4. 줄의 항목명
    if role is None:
        m = LABEL_RE.search(before)
        if m and LIST_OK.match(m.group(2)) and len(m.group(2)) <= 40:
            label = m.group(1).strip()
            if label.endswith(CUSTOMER_LABELS):
                role = "customer"
            elif label.endswith(STAFF_LABELS):
                role = "staff"
            elif label in NAME_LABELS:
                head = "\n".join(text[:ls].split("\n")[-4:])
                role = "customer" if "고객" in head else "staff" if ("직원" in head or "담당" in head) else None

    # 5. 괄호
    if paren and role is None:
        inner = paren.group(1).strip()
        if _strip_title(inner) or ORG_RE.search(inner):
            role = "staff"
            org = inner if ORG_RE.search(inner) else org
            title = title or _strip_title(inner)

    # 6. 표 칸
    if "|" in line and after.startswith("|"):
        cell = after[1:].split("|")[0].strip()
        if ORG_RE.search(cell):
            role, org = role or "staff", org or cell
        elif cell and not cell[0].isdigit() and "@" not in cell and role is None:
            role, org = "external", cell

    # 부서: 이름 앞의 부서명
    if org is None:
        m = ORG_RE.search(prev_word)
        if m:
            org = m.group(1)
            role = role or "staff"
    return {"role": role, "title": title, "org": org}


def extract(text, spans):
    """spans: [{"start", "end", "key"}] → 같은 순서의 [{"role", "title", "org"}]. 같은 key 는 역할 공유."""
    out = [_role_one(text, sp["start"], sp["end"]) for sp in spans]
    by_key = collections.defaultdict(list)
    for sp, r in zip(spans, out):
        if r["role"]:
            by_key[sp["key"]].append(r["role"])
    doc_role = {k: collections.Counter(v).most_common(1)[0][0] for k, v in by_key.items()}
    for sp, r in zip(spans, out):
        r["own_role"] = r["role"]
        r["role"] = doc_role.get(sp["key"], None)
    return out
