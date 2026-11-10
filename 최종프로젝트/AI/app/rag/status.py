"""처리 상태 추출 (worklog 19장): 문서·질의에서 접수 / 처리 중 / 처리 완료 / 보류.

검색에서 문서 종류는 맞혔는데 상태가 다른 문서를 1등으로 올리는 실패가 많아서 (18장) 상태를 조건으로 쓴다.
- 문서: 상태를 적는 자리(‘상태: 접수’, ‘상태는 '보류'’, 표의 상태 칸, ‘[보류]’ 제목, ‘현재 처리 중’)에서만 읽는다.
  본문 곳곳의 ‘접수일’, ‘안내 완료’ 같은 말은 상태가 아님.
- 한 문서에 여러 사람·여러 상태가 있는 표(VOC 목록 등)는 줄마다 따로 → 질의의 사람이 있는 줄의 상태를 씀 (line_statuses).
- 질의: ‘아직 안 끝난’, ‘마무리된’, ‘접수만 된’ 같은 말 → 상태 하나 (query_status). 못 정하면 None.
가명처리된 글에 그대로 쓰는 규칙이라 외부 호출 없음.
"""
import re

STATUSES = ("접수", "처리 중", "처리 완료", "보류")
_V = r"(접수|처리\s?중|진행\s?중|처리\s?완료|완료|보류)"
_Q = r"[\"'‘’“”\[\(]?"


def norm(v):
    v = re.sub(r"\s+", "", v)
    return {"접수": "접수", "처리중": "처리 중", "진행중": "처리 중", "처리완료": "처리 완료", "완료": "처리 완료",
            "보류": "보류"}.get(v)


# 상태를 적는 자리 (앞쪽일수록 강함). 문서 상태는 가장 강한 자리에서 처음 나온 값
_DOC_PATTERNS = [
    re.compile(r"상태\s*[:：]\s*" + _Q + _V),                       # 상태: 접수 / 처리 상태: '보류' / 반송 상태: 접수
    re.compile(r"상태(?:는|가|은)\s*(?:지금|현재)?\s*" + _Q + _V),    # 현재 상태는 '접수' / 상태는 지금 처리 중
    re.compile(r"현황\s*[:：]\s*" + _Q + _V),                       # 접수 현황: 처리 중
    re.compile(r"\[" + _V + r"\]"),                                  # Subject: [보류] …
    re.compile(r"현재\s*" + _Q + _V + r"(?![가-힣]*(?:예정|후))"),   # 현재 보류입니다 / 현재 처리 중
    re.compile(r"(?:^|\s)" + _V + r"\s*(?:상태|단계)(?:입니다|이며|로|이고|에|임|\.|$)"),  # 보류 상태입니다
    re.compile(r"(?:^|\s)" + _V + r"\s?(?:으로|로)\s?(?:처리|변경|전환)"),  # 보류로 처리 중입니다 → 보류
    re.compile(r"(?:건|요청|신청)(?:이|은|는)\s*" + _V + r"\s?(?:되었|됐|되어|됨)"),  # 환불 건이 처리 완료되었습니다
    re.compile(r"(?:^|[\s,])" + _V + r"(?:입니다|이에요|예요|임)(?![가-힣])"),  # …, 처리 완료입니다 ('완료 예정'은 안 잡음)
]
_CELL = re.compile(r"\|\s*" + _V + r"\s*(?=\|)|\|\s*" + _V + r"\s*$")  # 표 칸: | 처리 중 |


def _first(text):
    for p in _DOC_PATTERNS:
        m = p.search(text)
        if m:
            return norm(m.group(1))
    return None


def doc_status(text):
    """문서 전체의 상태 하나 (없으면 None). 표에 상태 칸만 있으면 그 칸 값이 모두 같을 때만"""
    s = _first(text)
    if s:
        return s
    cells = {norm(a or b) for line in text.splitlines() for a, b in _CELL.findall(line)}
    return cells.pop() if len(cells) == 1 else None


def line_statuses(text):
    """[(줄, 상태)] — 줄 안의 상태 자리 또는 표의 상태 칸. 사람별 상태(표의 한 줄 = 한 사람)에 씀"""
    out = []
    for line in text.splitlines():
        s = _first(line)
        if not s:
            cells = [norm(a or b) for a, b in _CELL.findall(line)]
            s = cells[-1] if cells else None
        if s:
            out.append((line, s))
    return out


# 질의 → 상태 집합. '아직 결론 안 난'은 접수·처리 중·보류 어느 것이든 될 수 있어서 '안 끝남' 묶음 (synth2 정답에서 셋 다 나옴)
UNFINISHED = frozenset({"접수", "처리 중", "보류"})
_QUERY_PATTERNS = [
    (UNFINISHED, r"(아직|안\s?끝난|안\s?끝났|끝나지\s?않|완료(되지|가)?\s?(않|안\s?된)|결론이\s?(아직\s?)?(안|나지\s?않)|"
                 r"결과가?\s?(아직\s?)?(안\s?나|나오지\s?않)|미처리|처리\s?안\s?(된|끝난))"),
    (frozenset({"보류"}), r"보류|멈춰\s?있|중단된"),
    (frozenset({"처리 중"}), r"처리\s?중|진행\s?중|진행되고|검토\s?중|심사\s?중|어떻게\s?되고\s?있"),
    (frozenset({"처리 완료"}), r"완료|해결된|해결\s?완료|마무리|끝난|끝났|종결"),
    (frozenset({"접수"}), r"접수(된|만|\s?상태|\s?중|\s?건|\s?내역|\s?기록|$|\s?(거|것)|하신|한\s?(거|건))|처리\s?접수"),
]
_NOT_STATUS = re.compile(r"(피해|분실|민원|사고|신고)\s?접수(?!\s?(된|만|상태))")  # '피해 접수'는 문서 종류


def query_status(query):
    """질의가 가리키는 상태 집합 (frozenset) 또는 None"""
    q = _NOT_STATUS.sub(" ", query)
    for statuses, pat in _QUERY_PATTERNS:
        if re.search(pat, q):
            return statuses
    return None
