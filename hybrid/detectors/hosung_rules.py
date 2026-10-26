"""hosung 저장소 실전프로젝트1/detectors/rule_based.py 사본 (팀장님 규칙 기반, 수정하지 않음).

이 저장소에서 바로 돌리려고 import 두 줄만 바꿨다:
    postprocess.drop_venue_addresses → 아래에 그대로 옮김, detectors.analyzers → 같은 폴더 사본
개선은 detectors/rules_best.py 에서 한다 (이 파일은 원본과 비교할 수 있게 그대로 둠).

----- 원본 설명 -----
방식 1:규칙 기반 탐지기 (정규식 + Kiwi 형태소 분석).

- 번호·이메일·주소: 정규식
- 계좌 vs 전화: 같은 모양이면 앞 문맥('계좌', '은행' 등)으로 판단 (토스처럼 휴대폰 번호를 계좌로 쓰는 경우)
- 이름: Kiwi 명사 덩어리 + 일반 성씨 목록, 필드명('성명:', '예금주:')·호칭·직함 문맥 규칙
- 데이터셋 인물 목록은 쓰지 않는다 (정답을 미리 보는 것과 같음)

v3 (엣지 케이스 대응): 지역번호·070·050x·+82·괄호 지역번호, 은행별 계좌 자릿수·4묶음·하이픈 없는 계좌,
지번·읍면리·시도 줄임말·세종·번길 주소, 외자·4글자·복성·외국인·영문 이름, 하이픈 없는/가린 카드번호,
아멕스, 외국인등록번호, 가린 주민번호, 다른 종류의 대시(–, —), 우회 표기 이메일([at])
"""
import re


# ------------------------------------------------------------------ 장소 주소 (원본 postprocess.py)
# 팀 라벨 기준: 집 주소가 아닌 장소 주소(웨딩홀·장례식장·회식 장소 등)는 개인을 특정하지 않으므로 가리지 않는다.
# 집 주소를 놓치는 쪽이 훨씬 위험해서 비대칭으로 판단한다:
#   장소를 뜻하는 말이 확실히 있고, 개인을 가리키는 말이 없을 때만 주소에서 뺀다. 문맥이 없으면 가린다.
VENUE_BEFORE = re.compile(r"장소|위치|식장|장례식장|빈소|회식|모임|오시는 ?길|찾아오시는|매장|본사|지점|사무실|행사장|"
                          r"집합|면접|교육장|진료|병원")
VENUE_IN_OR_AFTER = re.compile(r"웨딩홀|예식장|웨딩|장례식장|호텔|카페|식당|레스토랑|병원|파티룸|컨벤션|연회장|공원|"
                               r"역 ?\d+번 ?출구|본사|사옥|매장|지점")
PERSONAL_BEFORE = re.compile(r"주소|거주|자택|집|배송|수령|받는 ?곳|받으실|변경|등기|임차|임대|전입|우편|발송지|"
                             r"보내|송부|재발급|실거주|이사")


def _clause_before(text, start, window=30):
    line_start = text.rfind("\n", 0, start) + 1
    before = text[max(line_start, start - window):start]
    return re.split(r"[.,/;]\s", before)[-1]


def drop_venue_addresses(text: str, ents: list) -> list:
    out = []
    for e in ents:
        if e["label"] == "address":
            before = _clause_before(text, e["start"])
            span_after = text[e["start"]:min(len(text), e["end"] + 15)]
            venue = VENUE_BEFORE.search(before) or VENUE_IN_OR_AFTER.search(span_after)
            # 동·호수가 있으면 사는 집일 수 있다 ('직거래 … 902동 505호에서'). 병원·장례식장의 '호실'은 제외
            residence = re.search(r"\d+동|\d+호(?!실)", text[e["start"]:e["end"]])
            if venue and not residence and not PERSONAL_BEFORE.search(before):
                continue
        out.append(e)
    return out


SURNAMES = set(
    "김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진나지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국어은편용"
)
COMPOUND_SURNAMES = {"남궁", "선우", "제갈", "황보", "독고", "사공", "서문"}
TITLES = {"과장", "대리", "팀장", "부장", "차장", "사원", "주임", "실장", "이사", "상무", "전무",
          "대표", "고객", "상담", "본부", "센터", "지점", "선임", "책임", "수석", "위원", "매니저", "담당"}
_TITLE_ALT = "|".join(sorted(TITLES, key=len, reverse=True))
JOB_TITLES = ["본부장", "센터장", "지점장", "매니저", "과장", "대리", "팀장", "부장", "차장", "사원", "주임",
              "실장", "이사", "상무", "전무", "선임", "책임", "수석"]
# '인사팀', '고객명', '연회비' 처럼 명사+접미사 덩어리는 이름이 아니다
NON_NAME_SUFFIX = {"팀", "명", "비", "부", "실", "과", "점", "사", "처", "국", "소", "청", "원장", "센터"}
# '신사동', '서울시', '제주도', '서울숲', '신림역', '금정로' 처럼 장소 이름 끝 글자 (실제 이름 697개 중 해당 없음.
# '리'는 '장유리'·'김미리' 같은 이름이 많아 제외)
PLACE_SUFFIX = set("동시도로길역숲")
# '하늘빛은행', '새봄카드' 처럼 바로 뒤에 기관 접미사가 붙으면 이름이 아니다
ORG_FOLLOW = re.compile(r"은행|카드|캐피탈|저축|증권|보험|지점|팀")

# ------------------------------------------------------------------ 번호
DASH = r"\-‐‑–—－"                   # 하이픈 외에 워드프로세서가 바꿔 넣는 대시들 (문자 클래스용으로 - 이스케이프)
SEP = rf" ?[{DASH}.]? ?"            # '-', '.', ' ', ' - ', 또는 없음
# 지역번호(02, 031~064), 인터넷전화 070, 안심번호 0502~0508, 휴대폰 010·011·016~019
AREA = r"(?:2|3[1-3]|4[1-4]|5[1-5]|6[1-4]|70|50[2-8]|1[016-9])"
# 괄호는 '(051) 742-0091'처럼 지역번호만 감쌀 때만 번호에 포함한다. '고객(010-…)'의 괄호는 번호가 아님.
# 앞이 숫자·대시면 계좌 등 더 긴 숫자의 일부('3333-02-7654321'의 '02-7654321')라서 제외
PHONE = (rf"(?<![\d+{DASH}.])(?:\+82[ {DASH}]?(?:\(0\))?{AREA}|\(0{AREA}\)|0{AREA})"
         rf"{SEP}\d{{3,4}}{SEP}\d{{4}}(?!\d)")
CSEP = rf"[{DASH} ]?"
CARD = (rf"(?<![\d*])\d{{4}}{CSEP}[\d*]{{4}}{CSEP}[\d*]{{4}}{CSEP}\d{{4}}(?![\d*])"   # 16자리 (가운데 * 가림 허용)
        rf"|(?<!\d)3[47]\d{{2}}{CSEP}\d{{6}}{CSEP}\d{{5}}(?!\d)")                   # 아멕스 15자리 4-6-5
# 주민·외국인등록번호(뒷자리 첫 숫자 1~8), 뒷자리 6개를 * 로 가린 형태 허용
RRN = (rf"(?<!\d)\d{{2}}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]) ?[{DASH}]? ?[1-8](?:\d{{6}}|\*{{6}})(?!\d)")
# 은행마다 자릿수가 다르다: 국민 6-2-6, 신한 3-3-6, 우리 4-3-6, 농협 3-4-4-2, 카카오 4-2-7, 토스 4-4-4 …
# → 2~6자리 숫자 묶음 3~4개. 날짜(2026-06-25)는 제외
ACCOUNT = (rf"(?<![\d{DASH}])(?!(?:19|20)\d{{2}}[{DASH}]\d{{1,2}}[{DASH}]\d{{1,2}}(?!\d))"
           rf"\d{{2,6}}(?:[{DASH}]\d{{2,7}}){{2,3}}(?![\d{DASH}])")
DIGITS_ONLY = re.compile(r"(?<!\d)\d{10,14}(?!\d)")
# 계좌 문맥 단어. 쉼표·마침표·슬래시로 문장이 끊기면 문맥이 이어지지 않는다고 본다
BANK_WORDS = re.compile(r"계좌|예금주|입금|송금|이체|환불|은행|뱅크|농협|신협|새마을금고|우체국|저축")
# 계좌 모양이지만 계좌가 아닌 번호: 사업자등록번호·송장·주문·접수번호 등 (v6, 일상 문서에서 발견)
NOT_ACCOUNT_WORDS = re.compile(r"사업자\s?(?:등록)?\s?번호|송장|운송장|추적\s?번호|주문\s?번호|접수\s?번호|예약\s?번호|"
                               r"신청\s?번호|관리\s?번호|ISBN|등록번호")
# 회사 대표번호 문맥 (데이터셋 기준: 대표번호는 개인정보 아님). 080 은 수신자 부담 회사 번호
# '고객센터'만으로는 직원 직통번호와 구분이 안 돼서 넣지 않는다
COMPANY_PHONE_WORDS = re.compile(r"대표\s?번호|대표\s?전화")
# 번호 뒤에 오는 계좌 문맥: '10920304012로 송금', '503012-0113245에서 송금'
AFTER_BANK = re.compile(r"\)?\s?(?:으로|로|에서|에)?\s?(?:송금|입금|이체|환불|납부|상환|출금)")
# 하이픈·점으로 나뉜 2~4묶음, 숫자 10~16개 (계좌 문맥이 있을 때만 계좌로 봄)
LOOSE_NUMBER = re.compile(r"(?<![\d.\-])\d{2,8}(?:[-.]\d{2,8}){1,3}(?![\d.\-])")
# 주민·외국인등록번호 문맥이 있으면 날짜·성별 자리가 맞지 않아도 6-7자리를 주민번호로 본다
RRN_WORDS = re.compile(r"주민\s?(?:등록)?\s?번호|외국인\s?등록\s?번호|외국인번호")
RRN_LOOSE = re.compile(rf"(?<!\d)\d{{6}} ?[{DASH}] ?\d{{7}}(?!\d)|(?<!\d)\d{{13}}(?!\d)")
# '국민 123456…', '신한 110…' 처럼 '은행'을 생략한 은행 이름. '우리 팀', '하나 더', '기업 고객' 같은 일반 단어와
# 겹치므로 번호 바로 앞(공백·콜론만 사이에 둠)에 있을 때만 인정한다
BANK_PREFIX = re.compile(r"(?:국민|신한|우리|하나|농협|기업|IBK|KB|SC제일|씨티|수협|신협|카카오|토스|케이)"
                         r"\s?(?:은행|뱅크)?\s?[:：]?\s?$")
# ------------------------------------------------------------------ 이메일
EMAIL = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"   # \w 는 한글도 포함 → ASCII 만
EMAIL_OBFUSCATED = r"[A-Za-z0-9._%+-]+ ?[\[(](?:at|AT|골뱅이)[\])] ?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"
# ------------------------------------------------------------------ 주소
SIDO = "|".join(sorted([
    "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시", "울산광역시",
    "세종특별자치시", "경기도", "강원특별자치도", "강원도", "충청북도", "충청남도", "전북특별자치도", "전라북도",
    "전라남도", "경상북도", "경상남도", "제주특별자치도",
    "서울시", "부산시", "대구시", "인천시", "광주시", "대전시", "울산시", "세종시",
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원",
    "충북", "충남", "전북", "전남", "경북", "경남", "제주",
], key=len, reverse=True))
SGG = r"[가-힣]{1,6}(?:시|군|구)"
# 번지 뒤에 '3건', '5개' 같은 단위가 오면 주소가 아니다
_NUM_END = r"(?!\d|건|개|명|원|회|번째|차|일|월|년|분|초|%)"
ROAD = rf"(?:[가-힣]{{1,6}}(?:읍|면) )?[가-힣0-9]{{1,12}}(?:로|길)(?: ?\d+번?길)? ?\d+(?:-\d+)?{_NUM_END}"
JIBUN = rf"(?:[가-힣]{{1,6}}(?:읍|면) )?[가-힣0-9]{{1,8}}(?:동|리|가) (?:산 ?)?\d+(?:-\d+)?(?:번지)?{_NUM_END}"
# 건물 이름: '은파아파트', '현대오피스텔', '혜화빌딩', '새길병원 장례식장' (v6)
BUILDING = (r"[가-힣A-Za-z0-9]{1,10}(?:아파트|빌라|빌딩|타워|오피스텔|맨션|타운|하우스|프라자|스퀘어|센터|호텔|병원|"
            r"장례식장|웨딩홀|파티룸)")
DETAIL = (rf"(?:,? ?{BUILDING}){{0,2}}(?:,? ?[A-Za-z0-9]{{1,4}}동)?(?:,? ?(?:\d+층|\d+호실?|[Bb]\d+층|지하 ?\d+층)){{0,2}}"
          # 도로명 주소 뒤 참고항목: '(역삼동)', '(역삼동, 강남파이낸스센터)'
          r"(?: ?\([가-힣0-9]+(?:동|가|리|읍|면)(?:, ?[가-힣A-Za-z0-9 ]{1,20})?\))?")
# 시도로 시작하거나, 시도 없이 '강남구 테헤란로 152'처럼 2글자 이상 시군구로 시작
ADDRESS = (rf"(?:(?:{SIDO}) (?:{SGG} ){{0,2}}|(?:[가-힣]{{2,5}}(?:시|군|구) ){{1,2}})(?:{ROAD}|{JIBUN}){DETAIL}")

# 순서가 곧 우선순위: 앞 패턴이 잡은 글자는 뒤 패턴이 다시 잡지 않는다
PATTERNS = [
    ("email", EMAIL),
    ("email", EMAIL_OBFUSCATED),
    ("resident_registration_number", RRN),
    ("card_number", CARD),
    ("phone_number", PHONE),
    ("account_number", ACCOUNT),
    ("address", ADDRESS),
]

# ------------------------------------------------------------------ 이름 문맥 규칙
# '성명: 홍길동', '예금주: John Smith', '고객명: 응우옌 반 안' 처럼 이름 칸
NAME_FIELDS = ("성명|이름|고객명|고객 성명|예금주|수신인|받는 분|민원인|신청인|요청자|가입자|명의자|계약자|"
               "피보험자|보호자|대리인|담당자|작성자|상담원|처리자|보고자|수신|발신|참조|From|To|Cc|To\\.|"
               "예약자명|예약자|주문자|상담자|신청자|수령자|수령인|발송자|보내는 분|받는 사람|보내는 사람|임차인|임대인|"
               "구매자|판매자|입금자명|입금자|송금자|문의자|환자명|환자|학생명|학부모|회원명|성함|대표자|고객|처리 담당")
# 필드 뒤 구분 기호(: 또는 .)는 필수: 없으면 '상담원 황보미래 대리가 응대함'처럼 문장을 통째로 잡는다
FIELD_NAME = re.compile(rf"(?<![가-힣A-Za-z])(?:{NAME_FIELDS}) ?[:：.] ?"
                        # 한글 단어는 끝에 경계를 둔다: '리스크관리팀'에서 '리스크관리'만 떼어 잡지 않도록
                        r"([가-힣]{2,5}(?![가-힣])(?: [가-힣]{1,4}(?![가-힣])){0,2}"
                        r"|[A-Za-z][A-Za-z'.-]+(?: [A-Za-z][A-Za-z'.-]+){1,3})")
# '윤 선영 고객', '박 예린 팀장', '제갈 성렬 대리', '선우 진 과장' 처럼 띄어 쓴 이름. 뒤 문맥을 호칭·직함·줄끝으로
# 제한해야 '고객(010…)이 기존'의 '이 기존' 같은 오탐을 막을 수 있다
SPACED_NAME = re.compile(rf"(?<![가-힣)])([가-힣]{{1,2}}) ([가-힣]{{1,3}})(?=\s?(?:님|씨|{_TITLE_ALT})|\s*$|\s*[/,(])", re.M)
# '소율 씨가', '지영 씨, 안녕하세요' 처럼 성 없이 이름 + 씨
GIVEN_SSI = re.compile(r"(?<![가-힣])([가-힣]{2}) 씨")
# 메신저 '김서연: 승현아, ...' 처럼 부르는 말
GIVEN_VOCATIVE = re.compile(r"(?<=: )([가-힣]{2})(?=아,|야,)")
# 2~4글자 + 님/씨/직급/고객: 외자 '김솔님', '신승현 과장', 4글자 '김하늘별님', '남궁민수 고객'.
# Kiwi 가 '신(접두사)+승+현(관형사)'처럼 엉뚱하게 쪼개 명사 덩어리 규칙에서 빠지는 이름을 문맥으로 잡는다.
# '고객센터'·'지점'은 뒤에 와도 직급이 아니다 ('문의는 고객센터', '여의도지점' 오탐 방지)
HONOR_WORDS = ["선생님", "교수님", "원장님", "사장님", "기사님", "회원님", "고객님", "담당자", "총무", "회장", "간사"]
NAME_HONORIFIC = re.compile(
    rf"(?<![가-힣])([가-힣]{{2,4}})(?=님|씨|\s?(?:{'|'.join(JOB_TITLES)})|\s(?:{'|'.join(HONOR_WORDS)}|고객(?!센터)"
    rf"|명의|이름으로))")
# 서명·인사말: '채민호 드림', '김 올림', '안녕하세요, OO팀 표지원입니다'
SIGNATURE = re.compile(r"(?<![가-힣])([가-힣]{2,4})(?:\s?(?:드림|올림|배상)(?![가-힣])|입니다)")
# 참석자 목록: '참석: 김솔, 남궁 하늘, Daniel Park, 정다운' → 쉼표로 나눠 하나씩 이름 판단
LIST_FIELD = re.compile(r"(?m)(?<![가-힣])(?:참석자?|불참자?|참여자|배석자?|명단)\s?[:：]\s?(.+)$")
# 메신저 말하는 사람: 줄 맨 앞 '오세린: …'. Kiwi 가 '오세린'을 '오(숫자)+세(나이)+린'으로 읽는 것처럼
# 분석기가 이름으로 못 보는 경우를 줄 형식으로 잡는다. '주소:', '민원인:' 같은 필드명은 제외
SPEAKER = re.compile(r"(?m)^([가-힣]{2,4}) ?: ")
FIELD_WORDS = {"주소", "일시", "장소", "제목", "수신", "발신", "참조", "대상", "내용", "처리", "결과", "비고", "연락처",
               "직업", "계좌", "성명", "이름", "고객", "고객명", "민원인", "신청인", "요청자", "작성자", "보고자", "처리자",
               "담당자", "담당", "상담원", "참석", "불참", "회의명", "안건", "문의", "일자", "기간", "목적", "요청", "답변"}
# 이름 뒤에 붙을 수 있는 조사·호칭 (문서 안 이름 퍼뜨리기에서 경계 확인용)
NAME_FOLLOW = re.compile(r"(?:이랑|에게서|에게|한테|께서|께|님|씨|은|는|이|가|을|를|의|도|만|과|와|랑|아|야|이가|이는|이를|이도)?"
                         r"(?![가-힣])")


class RuleBasedDetector:
    name = "rules"

    def __init__(self, context_rules: bool = True, analyzer: str = "kiwi", compound_filter: bool = True):
        """context_rules=False 면 문서 문구(필드명·호칭·직함·메신저 말투)에 기대는 이름 규칙을 끄고
        정규식 + 명사 덩어리 규칙만 쓴다 → 처음 보는 템플릿에서의 하락 폭 추정용.
        analyzer: 형태소 분석기 (kiwi 기본, komoran·kkma·hannanum·okt 는 비교 실험용, detectors/analyzers.py)"""
        from detectors.analyzers import Analyzer
        self.context_rules = context_rules
        self.compound_filter = compound_filter
        self.an = Analyzer(analyzer)
        if analyzer != "kiwi":
            self.name = f"rules_{analyzer}"

    def detect(self, text: str) -> list:
        ents, taken = [], [False] * len(text)

        def add(s, e, label):
            if not any(taken[s:e]):
                ents.append({"label": label, "start": s, "end": e})
                for i in range(s, e):
                    taken[i] = True

        def remove(x):
            ents.remove(x)
            for i in range(x["start"], x["end"]):
                taken[i] = False

        for label, pat in PATTERNS:
            for m in re.finditer(pat, text):
                add(m.start(), m.end(), label)

        # 문맥으로 번호 종류를 바로잡는다 (v6)
        for e in list(ents):
            value, before = text[e["start"]:e["end"]], self._clause_before(text, e["start"])
            if e["label"] == "account_number" and (NOT_ACCOUNT_WORDS.search(before) or value.startswith("080")):
                remove(e)  # 사업자등록번호·송장번호·주문번호, 080 회사 번호
            elif e["label"] == "phone_number" and (COMPANY_PHONE_WORDS.search(before)
                                                   or re.match(r"\s?\(?대표", text[e["end"]:e["end"] + 4])):
                remove(e)  # 대표번호
            elif e["label"] == "phone_number" and self._phone_is_account(before):
                e["label"] = "account_number"  # 토스뱅크 등 휴대폰 번호 계좌
            elif e["label"] == "account_number" and self._last_is_card(before):
                e["label"] = "card_number"  # '카드번호 1234-00-1234-5678' 처럼 계좌 모양 카드번호
        # 주민번호 문맥이 있으면 날짜가 맞지 않는 6-7자리도 주민번호
        for m in RRN_LOOSE.finditer(text):
            if RRN_WORDS.search(self._clause_before(text, m.start(), 30)):
                for e in [x for x in ents if x["start"] < m.end() and m.start() < x["end"]]:
                    remove(e)
                add(m.start(), m.end(), "resident_registration_number")
        # 하이픈 없는 숫자, 2묶음·점 구분 번호는 주문번호 등과 구분이 안 되므로 계좌 문맥이 있을 때만 계좌로 본다
        for pat in (DIGITS_ONLY, LOOSE_NUMBER):
            for m in pat.finditer(text):
                digits = len(re.sub(r"\D", "", m.group()))
                before = self._clause_before(text, m.start())
                if 10 <= digits <= 16 and not NOT_ACCOUNT_WORDS.search(before) and (
                        self._bank_context(text, m.start()) or AFTER_BANK.match(text, m.end())
                        or BANK_PREFIX.search(text[max(0, m.start() - 12):m.start()])):
                    add(m.start(), m.end(), "account_number")

        if self.context_rules:
            self._context_names(text, add)

        # 분석기(기본 Kiwi)는 사전에 없는 이름을 '모르는 단어(NNG)'로 두거나 '신승+현', '남성+호' 처럼 엉뚱하게 쪼갠다.
        # 그래서 품사(NNP)보다 '빈틈없이 붙은 명사 조각을 합친 덩어리'를 보고 판단한다:
        #   성씨로 시작하는 3글자(복성이면 4글자) 덩어리 → 이름 후보. '여의도지점'처럼 더 길면 이름 아님
        #   단, '서비스'·'임직원'처럼 사전에 있는 일반명사 한 단어면 이름이 아님
        for s, e, toks in self._noun_chunks(text):
            word = text[s:e]
            known_common = len(toks) == 1 and toks[0].tag == "NNG" and toks[0].known
            # 접미사는 Kiwi 조각이 아니라 마지막 글자로 본다: '연회비'를 문맥마다 다르게 쪼개기 때문
            if known_common or (len(toks) > 1 and word[-1] in NON_NAME_SUFFIX) \
                    or ORG_FOLLOW.match(text, e):
                continue
            # '신사동', '서울숲', '제주도' 처럼 사전에 있는 장소 이름, '금정로 215' 처럼 뒤에 번지가 오는 도로명
            if word[-1] in PLACE_SUFFIX and (len(toks) == 1 and toks[0].known or re.match(r"\s?\d", text[e:e + 2])):
                continue
            # '신청+일', '배송+건' 처럼 사전 일반명사 + 사전 조각으로 된 합성어 (실제 이름도 일부 걸려 옵션)
            if self.compound_filter and self._is_compound(toks):
                continue
            if word[:2] in COMPOUND_SURNAMES and len(word) == 4:
                add(s, e, "person")
            elif word[0] in SURNAMES and len(word) == 3 and word[1:] not in TITLES:
                add(s, e, "person")

        if self.context_rules:
            for s, e in self._propagate_names(text, ents):
                inside = [x for x in ents if x["start"] < e and s < x["end"]]
                if not inside:
                    add(s, e, "person")
                elif all(x["label"] == "person" and s <= x["start"] and x["end"] <= e for x in inside)                         and e - s > max(x["end"] - x["start"] for x in inside):
                    # '응우옌 티 마이' 안에서 '마이'만 따로 잡혀 있었으면 긴 이름으로 바꾼다
                    for x in inside:
                        ents.remove(x)
                        for i in range(x["start"], x["end"]):
                            taken[i] = False
                    add(s, e, "person")
        # 팀 라벨 기준: 집 주소가 아닌 장소 주소는 가리지 않는다 (postprocess.drop_venue_addresses)
        return drop_venue_addresses(text, ents)

    def _propagate_names(self, text, ents):
        """문서 안에서 이미 찾은 이름의 다른 표기를 같은 문서에서 더 찾아 (start, end) 목록으로 돌려준다.
        '박지훈'을 찾았으면 '박 지훈'(띄어쓰기)과 '지훈'(이름만)도, 'Daniel Park'·'응우옌 티 마이'는 그대로 찾는다.
        템플릿 문구에 기대지 않아서 처음 보는 문서에서도 통하는 규칙."""
        variants = set()
        for e in ents:
            if e["label"] != "person":
                continue
            raw = text[e["start"]:e["end"]]
            if re.search(r"[A-Za-z]", raw) or len(raw.replace(" ", "")) >= 5 or raw.count(" ") >= 2:
                variants.add(re.escape(raw))  # 영문·외국인 이름은 표기 그대로
            name = raw.replace(" ", "")
            if not re.fullmatch(r"[가-힣]{3,4}", name):
                continue
            sur_len = 2 if name[:2] in COMPOUND_SURNAMES and len(name) == 4 else 1
            sur, given = name[:sur_len], name[sur_len:]
            variants.add(re.escape(sur) + r" ?" + re.escape(given))
            # 이름만 쓴 형태는 '하늘', '한결'처럼 일반 단어일 수 있어 사전에 없는 경우만
            if len(given) >= 2 and not self._is_common_word(given):
                variants.add(re.escape(given))
        # 긴 표기부터: '박 지훈'을 먼저 잡아야 그 안의 '지훈'만 따로 잡히지 않는다
        found = []
        for v in sorted(variants, key=len, reverse=True):
            for m in re.finditer(rf"(?<![가-힣A-Za-z])({v})(?![A-Za-z])", text):
                if NAME_FOLLOW.match(text, m.end()):
                    found.append((m.start(1), m.end(1)))
        return found

    def _clause_before(self, text, start, window=25):
        line_start = text.rfind("\n", 0, start) + 1
        before = text[max(line_start, start - window):start]
        return re.split(r"[.,/;]\s", before)[-1]

    def _phone_is_account(self, before):
        """전화 모양 번호를 계좌로 볼지: 계좌 단어 뒤에 '(휴대폰 번호):' 같은 설명만 있고 다른 말이 없을 때만.
        '계좌 변경 요청은 관리사무소 010…'처럼 계좌 단어가 앞쪽에만 있으면 전화로 둔다 (v6)"""
        last = None
        for last in BANK_WORDS.finditer(before):
            pass
        if last is None:
            return False
        between = before[last.end():]
        return bool(re.fullmatch(r"(?:번호)?[\s:：()]*(?:\(?(?:휴대폰|휴대전화|핸드폰|전화)\s?번호\)?)?[\s:：()]*(?:은|는|로|으로)?\s*",
                                 between))

    def _last_is_card(self, before):
        card = [m.end() for m in re.finditer(r"카드", before)]
        bank = [m.end() for m in BANK_WORDS.finditer(before)]
        return bool(card) and (not bank or card[-1] > bank[-1])

    def _is_compound(self, toks):
        return (len(toks) >= 2 and toks[0].tag == "NNG" and toks[0].known and len(toks[0].form) >= 2
                and all(t.known and t.tag in ("NNG", "NNB", "XSN", "NNP") for t in toks[1:]))

    def _bank_context(self, text, start, window=25):
        line_start = text.rfind("\n", 0, start) + 1
        before = text[max(line_start, start - window):start]
        before = re.split(r"[.,/;]\s", before)[-1]  # 문장이 끊긴 앞부분은 보지 않는다
        return bool(BANK_WORDS.search(before))

    def _context_names(self, text, add):
        """필드명·호칭·직함·말투 같은 주변 문구로 찾는 이름 규칙."""
        for m in LIST_FIELD.finditer(text):
            pos = m.start(1)
            for item in re.split(r"([,、/])", m.group(1)):
                if item not in ",、/":
                    name = re.sub(r"\s*\(.*?\)\s*$", "", item).strip()  # '이우리 (연차)' → '이우리'
                    if name and self._is_list_name(name):
                        s = pos + item.index(name)
                        add(s, s + len(name), "person")
                pos += len(item)
        for m in SPEAKER.finditer(text):
            w = m.group(1)
            if w not in FIELD_WORDS and (w[:2] in COMPOUND_SURNAMES or w[0] in SURNAMES)                     and w[-1] not in NON_NAME_SUFFIX and self._looks_like_name(w):
                add(m.start(1), m.end(1), "person")
        for m in FIELD_NAME.finditer(text):
            span = self._trim_field_name(m.group(1))
            if span:
                add(m.start(1), m.start(1) + len(span), "person")
        for m in SPACED_NAME.finditer(text):
            sur, given = m.group(1), m.group(2)
            if (len(sur) == 1 and sur in SURNAMES and len(given) == 2 and given not in TITLES) \
                    or (sur in COMPOUND_SURNAMES and given not in TITLES):
                add(m.start(), m.end(), "person")
        for pat in (GIVEN_SSI, GIVEN_VOCATIVE):
            for m in pat.finditer(text):
                if m.group(1) not in TITLES:
                    add(m.start(1), m.end(1), "person")
        for pat in (NAME_HONORIFIC, SIGNATURE):
            for m in pat.finditer(text):
                w = m.group(1)
                sur_ok = w[:2] in COMPOUND_SURNAMES or w[0] in SURNAMES
                if sur_ok and w[-1:] not in NON_NAME_SUFFIX and self._looks_like_name(w):
                    add(m.start(1), m.end(1), "person")

    def _is_list_name(self, name):
        if re.fullmatch(r"[A-Za-z][A-Za-z'.-]+(?: [A-Za-z][A-Za-z'.-]+){1,3}", name):
            return True
        parts = name.split(" ")
        if len(parts) == 2 and (parts[0] in SURNAMES or parts[0] in COMPOUND_SURNAMES)                 and 1 <= len(parts[1]) <= 3 and parts[1] not in TITLES:
            return True  # '남궁 하늘', '이 도윤'
        return len(parts) == 1 and 2 <= len(name) <= 4 and (name[:2] in COMPOUND_SURNAMES or name[0] in SURNAMES)             and name[-1] not in NON_NAME_SUFFIX and self._looks_like_name(name)

    def _trim_field_name(self, value):
        """필드 값에서 이름만 남긴다: '윤선영 연락처' → '윤선영', '김은영 대리' → '김은영'."""
        if re.match(r"[A-Za-z]", value):
            return value
        words = value.split(" ")
        first = words[0]
        if first[-1] in NON_NAME_SUFFIX or not self._looks_like_name(first):
            return None  # '채권관리팀', '고객센터' 같은 부서·일반 단어
        keep = [first]
        for w in words[1:]:
            if any(w.startswith(t) for t in TITLES | set(JOB_TITLES)) or (len(w) >= 2 and self._is_common_word(w)) \
                    or self._ends_with_josa(w):
                break  # 직급, 일반 단어, 조사로 끝나는 단어('대리가')에서 이름이 끝난다
            keep.append(w)
        # 성씨로 시작하는 한국 이름은 한 단어(띄어 쓴 이름 제외)면 충분. 여러 단어는 외국인 이름
        return " ".join(keep)

    def _is_common_word(self, word):
        """Kiwi 사전에 있는 한 단어인지 (고유명사 제외). 이름은 대부분 사전에 없다."""
        toks = self.an.tokenize(word)
        return len(toks) == 1 and toks[0].known and toks[0].tag != "NNP"

    def _ends_with_josa(self, word):
        toks = self.an.tokenize(word)
        return bool(toks) and toks[-1].tag.startswith("J")

    def _looks_like_name(self, word):
        toks = self.an.tokenize(word)
        if not toks:
            return False
        if len(toks) == 1 and toks[0].known and toks[0].tag != "NNP":
            return False  # '모든', '이번', '기존' 처럼 사전에 있는 일반 단어
        if toks[-1].tag.startswith("J") and toks[-1].form in ("은", "는", "이", "가", "을", "를", "의", "에", "도", "로"):
            stem = word[:len(word) - len(toks[-1].form)]
            # '문의+는'처럼 앞부분이 일반 단어면 이름이 아님. '박경+은'처럼 앞부분이 일반 단어가 아니면
            # Kiwi 가 이름 끝 글자를 조사로 잘못 본 것이므로 통과 (v6)
            if len(stem) < 2 or self._is_common_word(stem):
                return False
            if len(stem) >= 3:
                return False  # '김서준+은'은 '이름+조사'. 이름(김서준)은 다른 규칙이 따로 잡는다
        # 어미·용언 조각은 2·4글자에서만 거른다. 3글자 이름은 Kiwi 가 '손하은'을 '손+하+은(어미)'처럼
        # 잘못 분석하는 일이 많아 이 기준을 쓰면 실제 이름을 놓친다
        if len(word) != 3 and any(t.tag[0] in "EV" for t in toks):
            return False
        if len(word) == 4 and word[:2] not in COMPOUND_SURNAMES:
            # 성 1 + 이름 3: '김하늘+별'(NNP 포함)은 이름, '신용+카드'(일반명사뿐)는 아님
            return any(t.tag == "NNP" or not t.known for t in toks)
        return True

    def _noun_chunks(self, text):
        chunk = []
        for tok in list(self.an.tokenize(text)) + [None]:
            # '님'·'씨'는 이름 덩어리에 붙이지 않는다 ('노성호님'이 4글자가 되어 빠지는 것 방지)
            if tok is not None and tok.tag in ("NNG", "NNP", "NNB") and tok.form not in ("님", "씨") \
                    and (not chunk or tok.start == chunk[-1].start + chunk[-1].len):
                chunk.append(tok)
                continue
            if chunk:
                yield chunk[0].start, chunk[-1].start + chunk[-1].len, chunk
            chunk = [tok] if tok is not None and tok.tag in ("NNG", "NNP") else []
