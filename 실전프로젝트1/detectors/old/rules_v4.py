"""규칙 기반 탐지기 v4 — 정규식 + 이름 문맥 규칙 (+ 선택: Kiwi 형태소 분석).

사용법
    python detectors/rules.py data/pii_synth_finance_dev.jsonl preds/rules_dev.jsonl
    python detectors/rules.py data/pii_synth_finance_dev.jsonl preds/rules_kiwi_dev.jsonl --kiwi

이름 탐지 방식 (2단계)
    1) 씨앗 찾기: '성씨 + 두 글자' 모양이면서, 확실한 문맥(뒤에 호칭·직급, 앞에 '성명:' 같은 항목명,
       메신저 '이름:' 형식)에 있는 것만 이름으로 인정
    2) 퍼뜨리기: 같은 문서 안에서 씨앗과 같은 이름이 다시 나오면(띄어 쓴 형태 포함) 문맥이 약해도 이름으로 인정
       + 씨앗의 이름 부분(성 뺀 두 글자)이 '씨'·'아/야' 앞에 오면 이름으로 인정
    --kiwi: Kiwi가 고유명사(NNP)로 판단한 '성씨 + 두 글자' 단어도 씨앗에 추가
"""
import json
import re
import sys
import time

# ------------------------------------------------------------------ 숫자·이메일
# (?<!\d) / (?!\d): 앞뒤가 숫자가 아닐 때만 매칭 → 12자리 민원번호 같은 긴 숫자 속 일부를 잘못 잡지 않음
# 위에서부터 차례로 자리를 차지하므로 순서가 중요 (주민번호 → 카드 → 전화 → 계좌)
PATTERNS = {
    # 앞 6자리가 실제 날짜(YYMMDD)일 때만 + 뒷자리 첫 숫자 1~8(외국인 5~8 포함), 하이픈·공백·없음
    "resident_registration_number":
        r"(?<!\d)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])[- ]?[1-8]\d{6}(?!\d)",
    # 16자리(4-4-4-4, 구분자는 하이픈·공백·없음 중 하나로 통일) + 아멕스 15자리(4-6-5)
    "card_number": r"(?<!\d)(?:\d{4}([- ]?)\d{4}\1\d{4}\1\d{4}|3[47]\d{2}[- ]\d{6}[- ]\d{5})(?!\d)",
    # 휴대전화(010·011·016~019), 지역번호(02, 031~064), 인터넷전화(070), 국제 표기(+82 10-…)
    # 대표번호(1588-…)는 개인 번호가 아니므로 일부러 뺌
    "phone_number": r"(?<![\d+])(?:\+82[- ]?(?:1[016789]|2|[3-6][1-5]|70)|01[016789]|02|0[3-6][1-5]|070)"
                    r"[-. ]?\d{3,4}[-. ]?\d{4}(?!\d)",
    # 하이픈으로 나뉜 숫자 덩어리 3~4개, 숫자 합계 10~14자리 (은행마다 자릿수가 달라서 넓게)
    # 날짜(2026-09-23)는 8자리라 걸리지 않음
    "account_number": r"(?<![\d-])(?=(?:\d-?){10,14}(?![\d]))\d{2,6}-\d{2,6}-\d{2,7}(?:-\d{1,3})?(?![\d-])",
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
}
# 하이픈 없는 계좌번호는 앞에 '계좌·입금' 같은 단서가 있을 때만 (괄호 안 숫자 부분만 잡음)
CONTEXT_PATTERNS = {
    "account_number": r"(?:계좌|입금|출금|환불|이체)[^\d\n]{0,8}?(\d{10,14})(?!\d)",
}

# ------------------------------------------------------------------ 주소
SIDO = (r"(?:서울특별시|부산광역시|대구광역시|인천광역시|광주광역시|대전광역시|울산광역시|세종특별자치시"
        r"|경기도|강원특별자치도|강원도|충청북도|충청남도|전북특별자치도|전라북도|전라남도|경상북도|경상남도|제주특별자치도)")
ADDRESS = re.compile(
    SIDO
    + r"(?:\s[가-힣]+(?:시|군|구))+"           # 시·군·구 (여러 개 가능: 성남시 분당구)
    + r"\s[가-힣0-9]+(?:로|길)\s\d+(?:-\d+)?"   # 도로명 + 건물번호
    + r"(?:,\s?(?:\d+동\s?)?\d+(?:호|층))?"     # 동·호수·층 (있으면)
)

# ------------------------------------------------------------------ 이름
# 일반 성씨 목록 (데이터셋 인물 목록은 쓰지 않음 — 정답을 미리 보는 것과 같기 때문)
SURNAMES = "김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진나지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용예경봉사부황"
TITLES = ("고객", "씨", "님", "팀장", "부장", "과장", "대리", "차장", "사원", "주임", "선임", "책임", "매니저",
          "이사", "대표", "실장", "본부장", "지점장", "상담원", "드림", "올림",
          # 보강: 다른 업종·직급 체계의 직함
          "원장", "소장", "센터장", "행장", "부행장", "사장", "부사장", "전무", "상무", "계장", "수석",
          "연구원", "위원", "회원", "차주", "보증인", "변호사", "세무사", "회계사", "선생님", "교수", "기자", "담당")
# 이름 바로 뒤에 붙어도 되는 조사 (정답 범위에서는 빠져 있음)
JOSA = ("에게서", "께서", "에게", "이랑", "랑", "께", "은", "는", "이", "가", "을", "를", "의", "과", "와", "도", "만", "님")
# 항목명 — 뒤에 오는 이름(쉼표로 이어진 목록 포함)을 씨앗으로 인정
FIELDS = ("성명", "고객명", "상담원", "담당자", "담당", "민원인", "검토자", "참석자", "참석", "불참", "작성자", "작성",
          "처리자", "점검자", "신청인", "수신", "발신", "보고자", "고객", "갑", "을",
          # 보강: 금융 서류·메일에서 흔한 항목명
          "예금주", "신청자", "소유자", "명의자", "대리인", "계약자", "보증인", "차주", "피해자", "책임자",
          "승인자", "기안자", "결재자", "수신자", "발신자", "보낸 사람", "받는 사람", "보낸사람", "받는사람",
          "이름", "고객 성명")

H = "[가-힣]"
COMPOUND = "남궁|황보|제갈|선우|독고|사공"  # 두 글자 성
FULL = rf"(?<!{H})(?:{COMPOUND}|[{SURNAMES}]) ?{H}{{2}}"  # '김은영', '김 은영', '남궁다은'
TITLE_RE = "|".join(TITLES)
JOSA_RE = "|".join(JOSA)
FIELD_RE = "|".join(FIELDS)

# 씨앗 규칙 — 이름(키)은 진단표에서 쓰임 (tools/diagnose_rules.py)
SEED_RULES = {
    # 뒤에 호칭·직급: '윤선영 고객', '김은영 대리', '정하은님'
    "씨앗_뒤호칭": re.compile(rf"({FULL})(?= ?(?:{TITLE_RE}))"),
    # 뒤에 괄호로 연락처·내선·직급: '박도윤 (010-…)', '김은영(팀장)'
    "씨앗_뒤괄호": re.compile(rf"({FULL})(?= ?\((?:0|내선|연락처|고객|팀장|부장|과장|대리|차장|연차))"),
    # 메신저 형식: 줄 맨 앞 '김서연: …', '[10:21] 김서연: …', '김서연(09:15): …', '김서연 > …'
    # ('- 김서연:'처럼 목록 기호가 붙은 형식은 '- 유형별:' 같은 일반 목록과 구분이 안 돼서 뺌)
    "씨앗_메신저": re.compile(
        rf"(?m)^(?:\[[^\]\n]{{1,20}}\]\s*)?({FULL})(?=(?:\s?\(\d{{1,2}}:\d{{2}}\))?\s?[:>])"),
    # 소속·직함 뒤: 'IT운영팀 박도윤', '고객센터 정성호', '대표이사 김예준', '소장 오민석이', '차주 윤태호'
    # ('영업부 ○○○'의 '부'는 '일부 서비스', '내부 ○○' 같은 일반어와 겹쳐서 뺌)
    "씨앗_소속뒤": re.compile(
        rf"(?:팀|센터|지점|본부|지사|영업점|법인|대표이사|대표|원장|소장|차주|보증인) ({FULL})"
        rf"(?=$|[^가-힣]|입니다|(?:{JOSA_RE}|{TITLE_RE}))"),
    # 자기소개: '정보보호팀 정서현입니다'
    "씨앗_입니다": re.compile(rf"({FULL})(?=입니다)"),
}
FIELD_LINE = re.compile(rf"(?m)(?:^|[\s\-·])(?:{FIELD_RE})\s?:\s?([^\n]+)")
# 이름 부분처럼 보이지만 실제로는 직급·일반어인 두 글자 (예: '허 부장', '남 과장')
NOT_GIVEN = set(TITLES) | {"부장님", "과장님", "고객님", "대리님", "팀장님", "운영", "보호", "관리", "심사", "정보", "센터"}
# 성씨로 시작하는 세 글자 일반어 (항목명 등) — 이름 후보에서 제외
NOT_NAME = {"민원인", "고객명", "연락처", "담당자", "작성자", "검토자", "참석자", "처리자", "점검자", "신청인", "보고자",
            "예금주", "신청자", "소유자", "명의자", "기안자", "이메일", "주소지", "부서명", "소속팀", "신청일", "기안일"}
# 이름 끝 글자로 거의 안 쓰이는 조사형 글자: '문의는', '전화를', '전체에'
BAD_LAST = set("는를에의가을")


def norm_name(s):
    return s.replace(" ", "")


KIWI = None  # --kiwi 옵션을 주면 Kiwi 객체가 들어감


DISABLED = set()  # 진단용: 여기 넣은 규칙 이름은 끔 (예: {"씨앗_입니다"})


def find_seeds(text):
    """{이름: 그 이름을 찾은 규칙들} 을 돌려줌."""
    seeds = {}

    def add(name, rule):
        seeds.setdefault(norm_name(name), set()).add(rule)

    if KIWI is not None and "씨앗_Kiwi" not in DISABLED:
        for t in KIWI.tokenize(text):
            if t.tag == "NNP" and len(t.form) == 3 and t.form[0] in SURNAMES:
                add(t.form, "씨앗_Kiwi")
    for rule_name, rule in SEED_RULES.items():
        if rule_name in DISABLED:
            continue
        for m in rule.finditer(text):
            add(m.group(1), rule_name)
    if "씨앗_항목명" not in DISABLED:
        for m in FIELD_LINE.finditer(text):  # '참석: 이상훈, 김민준, 정성호' 처럼 쉼표 목록까지
            for part in re.split(r"[,/·]", m.group(1)):
                mm = re.match(rf"\s*({FULL})(?!{H})", part)
                if mm:
                    add(mm.group(1), "씨앗_항목명")
    return {s: r for s, r in seeds.items()
            if s not in NOT_NAME and s[-1] not in BAD_LAST and s[-2:] not in NOT_GIVEN}


def split_name(name):
    """'남궁다은' → ('남궁', '다은'), '김은영' → ('김', '은영')"""
    return (name[:2], name[2:]) if len(name) == 4 else (name[0], name[1:])


def find_names(text):
    """(시작, 끝, 찾은 규칙들) 목록을 돌려줌. 퍼뜨려 찾은 이름은 그 씨앗을 찾은 규칙으로 기록."""
    seeds = find_seeds(text)
    ents = []
    end_ok = rf"(?=$|[^가-힣]|(?:{JOSA_RE}|{TITLE_RE}|입니다))"
    for name, src in seeds.items():
        sur, given = split_name(name)
        # 전체 이름 (붙여 쓴 것 / 띄어 쓴 것)
        for m in re.finditer(rf"(?<!{H}){sur} ?{given}{end_ok}", text):
            ents.append((m.start(), m.end(), src))
        # 성 뺀 이름: '은영 씨', '승현아,'
        for m in re.finditer(rf"(?<!{H}){given}(?= 씨|씨|아[, ]|야[, ]|님|이랑|랑)", text):
            ents.append((m.start(), m.end(), src))
    # 씨앗이 없어도 성 뺀 이름을 잡는 규칙: '소율 씨가', 메신저 '지영아,'
    if "단독_이름씨" not in DISABLED:
        for m in re.finditer(rf"(?<!{H})({H}{{2}})(?= 씨)", text):
            if m.group(1) not in NOT_GIVEN:
                ents.append((m.start(), m.end(), {"단독_이름씨"}))
    if "단독_메신저호격" not in DISABLED:
        for m in re.finditer(rf"(?m)(?<=[:>] )({H}{{2}})(?=[아야], )", text):
            ents.append((m.start(), m.end(), {"단독_메신저호격"}))
    return ents


# ------------------------------------------------------------------ 합치기
def detect(text):
    ents, taken = [], set()

    def add(label, s, e):
        span = set(range(s, e))
        if span & taken:  # 먼저 잡힌 것과 겹치면 버림
            return
        taken.update(span)
        ents.append({"label": label, "start": s, "end": e})

    for label, pat in PATTERNS.items():  # 위에 있는 패턴이 먼저 자리를 차지 (주민번호·카드가 전화번호보다 우선)
        for m in re.finditer(pat, text):
            add(label, m.start(), m.end())
    for label, pat in CONTEXT_PATTERNS.items():
        for m in re.finditer(pat, text):
            add(label, m.start(1), m.end(1))
    for m in ADDRESS.finditer(text):
        add("address", m.start(), m.end())
    for s, e, _ in sorted(find_names(text), key=lambda x: (x[0], -x[1])):
        add("person", s, e)
    return sorted(ents, key=lambda e: e["start"])


def run(in_path, out_path):
    with open(in_path, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            t0 = time.perf_counter()
            ents = detect(doc["text"])
            ms = (time.perf_counter() - t0) * 1000
            out.write(json.dumps({"id": doc["id"], "entities": ents, "ms": ms}, ensure_ascii=False) + "\n")
    print(f"저장: {out_path}")


if __name__ == "__main__":
    if "--kiwi" in sys.argv:
        from kiwipiepy import Kiwi
        KIWI = Kiwi()  # 사전 로딩에 1초쯤 걸리므로 문서마다가 아니라 한 번만 만듦
    paths = [a for a in sys.argv[1:] if not a.startswith("--")]
    run(paths[0], paths[1])
