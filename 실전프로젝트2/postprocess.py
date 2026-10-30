"""토큰 분류 모델 출력 후처리: 경계 보정 + 미끼 오탐 제거 (+ 장소 주소 제외는 규칙에도 공통 적용).

v1 test 오류에서 확인한 모델의 실수를 일반 규칙으로 고친다.
- 경계: 번호 앞뒤 숫자를 잘라먹음('1-853629-72'), 이름 뒤 호칭·부르는 말이 붙음('승현아', '노성호님')
- 오탐: 대표번호('159[9-0000]'), 12자리 민원번호를 주민번호로, '허 부장'의 '허' 같은 성+직함
"""
import re

NUMBER_LABELS = {"phone_number", "resident_registration_number", "account_number", "card_number"}
TITLES = {"과장", "대리", "팀장", "부장", "차장", "사원", "주임", "실장", "이사", "상무", "전무", "대표",
          "매니저", "책임", "선임", "수석", "본부장", "센터장", "지점장"}
COMPANY_PHONE = re.compile(r"1[5-9]\d{2}-?\d{4}")  # 대표번호: 1588-xxxx, 1599-xxxx 등
# v8 (holdout 오류 분석 23장에서 추가)
# 부고 단어: '김미영 사원 조모상' 의 '조모상'
CONDOLENCE = re.compile(r"(?:장인|장모|빙부|빙모|부친|모친|조부|조모|외조부|외조모|시부|시모|부군|배우자|본인)상$")
# 회사 이름 끝말: '미르보안', '데이터온시스템즈'
ORG_END = re.compile(r"(?:보안|시스템즈?|소프트|솔루션|테크|컴퍼니|코리아|은행|캐피탈|증권|보험|카드|전자|건설|산업|물산|홀딩스)$")
# 이름 바로 뒤에 올 수 있는 글자 (조사·호칭·서술). 이 밖의 한글이 붙어 있으면 단어 중간을 자른 것('【미수】신')
NAME_NEXT = set("은는이가을를의에께님씨도만과와랑아야로인입이")
ROLE_BEFORE = re.compile(r"(?:회원|고객|담당|상담원|선생|학생|환자|기사|사원|대리|과장|차장|부장|팀장|주임|실장|이사|대표|"
                         r"팀|센터|본부|지점|부서)$")  # v9: 부서 뒤에 붙여 쓴 이름 '영업팀하늘에게'
# v9: 이름 뒤에 붙어 잡히는 조사·호칭·직함 (synth_hard 개선용 오류: '최민재님', '정다영의', '이 준형이랑', '최세진고객').
# 긴 것부터. 은·는·가 처럼 이름 끝 글자로 흔한 것('하은', '지은')은 넣지 않는다
PERSON_TAIL = re.compile(r"(?:고객님|고객|대리님|대리|과장님|과장|팀장님|팀장|에게서|에게|께서|이랑|이한테|한테|이에요|인데요|"
                         r"입니다|이가|이는|이의|님|씨|께|의|과|와)$")
# 번호 뒤 '(대표)', ' 대표번호' 처럼 회사 번호라고 밝힌 경우
COMPANY_AFTER = re.compile(r"\s?(?:\(\s?대표|대표\s?(?:번호|전화))")  # '대표자 홍길동'은 해당 없음
# 번호 앞 문맥: 사건·접수·주문번호 등은 개인정보 아님, '내선' 뒤 번호도 회사 번호
# '○○ 번호' 꼴만 본다: '민원인: … (연락처 010…)'의 '민원', '주민등록번호'의 '등록번호'에 걸리지 않도록
NOT_PII_NUMBER_BEFORE = re.compile(r"(?:사건|접수|주문|송장|운송장|사업자\s?(?:등록)?|민원|관리|예약|신청)\s?번호|내선")
YEAR_SERIAL = re.compile(r"(?:19|20)\d{2}-\d{5,7}")  # 연도-일련번호 '2026-822765'
SIGN_OFF = re.compile(r"\s?(?:드림|올림|배상)$")


def _is_num(c):
    return c.isdigit()


def strip_person_tail(word: str) -> str:
    """'최민재님' → '최민재'. 떼고 남는 이름이 2글자 미만이면 그대로 둔다 ('김씨'의 '김'만 남기지 않도록)."""
    m = PERSON_TAIL.search(word)
    if m and len(word[:m.start()].replace(" ", "")) >= 2 and re.search(r"[가-힣]$", word[:m.start()]):
        return word[:m.start()].rstrip()
    return word


def trim_person_tail(text: str, ents: list) -> list:
    out = []
    for e in ents:
        if e["label"] == "person":
            v = strip_person_tail(text[e["start"]:e["end"]])
            e = {**e, "end": e["start"] + len(v)}
        out.append(e)
    return out


def fix_boundaries(text: str, ents: list) -> list:
    out = []
    for e in ents:
        s, t, lab = e["start"], e["end"], e["label"]
        if lab in NUMBER_LABELS:
            # 숫자 덩어리 중간에서 끊겼으면 양옆으로 넓힌다 (구분자 - . 를 사이에 둔 숫자까지)
            while s > 0 and (_is_num(text[s - 1]) or (text[s - 1] in "-." and s > 1 and _is_num(text[s - 2]))):
                s -= 1
            while t < len(text) and (_is_num(text[t]) or (text[t] in "-." and t + 1 < len(text) and _is_num(text[t + 1]))):
                t += 1
        elif lab == "person":
            # 이름 뒤에 붙은 호칭 제거 (사람 이름이 '님'·'씨'로 끝나는 일은 거의 없다)
            while t - s > 2 and text[t - 1] in "님씨":
                t -= 1
            # 메신저 '승현아,' 처럼 부르는 말은 뒤에 쉼표·느낌표가 올 때만 제거 ('허수아 부장'의 '아'는 이름)
            if t - s > 2 and text[t - 1] in "아야" and t < len(text) and text[t] in ",!":
                t -= 1
            # v8: 서명 '시우 드림'의 '드림', '심사역 오가은이 조건'의 주격 조사 '이' (뒤에 공백·문장부호가 올 때만)
            m = SIGN_OFF.search(text[s:t])
            if m and m.start() >= 2:
                t = s + m.start()
            if t - s >= 3 and text[t - 1] == "이" and (t == len(text) or text[t] in " \n,.)"):
                t -= 1
            # v9: 영문 이름을 단어 중간에서 자름 'oria Kim' → 'Victoria Kim' (서브워드 조각 경계)
            while s > 0 and text[s - 1].isascii() and text[s - 1].isalpha() and text[s].isascii() and text[s].isalpha():
                s -= 1
            while t < len(text) and text[t].isascii() and text[t].isalpha() and text[t - 1].isascii() \
                    and text[t - 1].isalpha():
                t += 1
            t = s + len(strip_person_tail(text[s:t]))  # v9: '최민재님', '정다영의'
        elif lab == "address":
            # v8: 주소 앞 우편번호 '(05299) '
            m = re.search(r"\(\d{5}\) ?$", text[max(0, s - 8):s])
            if m:
                s -= len(m.group())
            # v9: 주소 끝에 붙은 조사·쉼표 '…1301호로', '…29-7로', '…로 95,' (번지·호·층·괄호 뒤일 때만)
            m = re.search(r"(?<=[\d호층)])\s?(?:으로|로|에서|에|가|이|는|은|를|을|의|와|과|,)$", text[s:t])
            if m:
                t = s + m.start()
        out.append({**e, "start": s, "end": t})
    return resolve_name_i(text, out)


# v10: '이'로 끝나는 이름('백송이')과 '이름 + 주격 조사 이'('안윤이 조건 안내')는 글자로는 구분이 안 된다.
# 주격 조사 '이' 뒤에는 직함·호칭이 바로 올 수 없으므로 '백송이 대리'·'백송이 님'·'백송이 드림'이면 '이'까지 이름.
# '고객'은 '김미영이 고객 상담을'처럼 일반 명사로도 와서 '고객님'만 본다
NAME_TITLE_AFTER = (r"(?:님|씨|대리|과장|차장|부장|팀장|주임|사원|실장|이사|매니저|책임|선임|수석|고객님|선생|교수|담당|"
                    r"드림|올림|배상)(?![가-힣])|(?:님|씨|대리|과장|차장|부장|팀장|고객님)(?:께|이|은|은요|에게|의|가|과)")
# '이'까지 붙인 이름은 3글자(성 + 2글자 이름)까지만, 4글자는 복성일 때만 ('김미영이'는 4글자 이름이 아님)
_COMPOUND = ("남궁", "황보", "제갈", "선우", "독고", "사공", "서문", "동방", "어금", "망절")


def resolve_name_i(text: str, ents: list) -> list:
    """이름 끝 '이'를 이름에 넣을지 정한다 (규칙·토큰 공통).
    1) 같은 문서 단서: '백송이 대리'·'백송이님'처럼 '이' 뒤에 직함·호칭이 오면 긴 이름, '안윤님'·'안윤의'·'안윤 대리'처럼
       '이' 없이 쓰인 곳이 있으면 짧은 이름. 한쪽만 있으면 그쪽으로 맞춘다
       ('백송이가'·'백송이는'은 받침 있는 이름 + 접사 '이' + 조사로도 읽혀 단서로 쓰지 않음)
    2) 문서 단서가 없으면 이 자리 바로 뒤가 직함·호칭인지
    3) 둘 다 없으면 탐지기가 정한 경계 그대로 ('심사역 백송이 조건 안내'는 구분할 수 없음)"""
    out = []
    for e in ents:
        if e["label"] != "person":
            out.append(e)
            continue
        s, t = e["start"], e["end"]
        v = text[s:t]
        if not re.fullmatch(r"[가-힣]+(?: [가-힣]+)?", v):
            out.append(e)
            continue
        if v.endswith("이") and len(v.replace(" ", "")) >= 3:
            long_end, short_end = t, t - 1
        elif t < len(text) and text[t] == "이" and len(v.replace(" ", "")) >= 2:
            long_end, short_end = t + 1, t
        else:
            out.append(e)
            continue
        long_, short = text[s:long_end], text[s:short_end]
        if long_.endswith(" 이"):
            out.append(e)  # '브라이언 이'처럼 띄어 쓴 성 '이'
            continue
        n = len(long_.replace(" ", ""))
        if n > 4 or (n == 4 and long_.replace(" ", "")[:2] not in _COMPOUND):
            out.append({**e, "end": short_end} if t == long_end else e)  # 긴 쪽이 될 수 없으면 짧은 이름
            continue
        long_ev = re.search(rf"(?<![가-힣]){re.escape(long_)}\s?(?:{NAME_TITLE_AFTER})", text)
        short_ev = re.search(rf"(?<![가-힣]){re.escape(short)}(?:\s?(?:{NAME_TITLE_AFTER})|(?:은|을|의|과|에게|께)(?![가-힣]))",
                             text)
        if long_ev and not short_ev:
            t = long_end
        elif short_ev and not long_ev:
            t = short_end
        elif not long_ev and not short_ev and re.match(rf"\s?(?:{NAME_TITLE_AFTER})", text[long_end:]):
            t = long_end
        out.append({**e, "end": t})
    return out


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


def drop_decoys(text: str, ents: list) -> list:
    out = []
    for e in ents:
        value = text[e["start"]:e["end"]]
        lab = e["label"]
        if lab in NUMBER_LABELS:
            digits = re.sub(r"\D", "", value)
            if COMPANY_PHONE.fullmatch(value):
                continue  # 대표번호
            if value.isdigit() and len(digits) == 12:
                continue  # 민원번호·접수번호 (전화 10~11자리, 주민 13자리와 다름)
            if YEAR_SERIAL.fullmatch(value) or NOT_PII_NUMBER_BEFORE.search(_clause_before(text, e["start"], 15)):
                continue  # v8: 사건번호 '경찰청 2026-822765', 내선 '1067'
            if lab == "phone_number" and len(digits) <= 5:
                continue  # v8: 4~5자리는 내선번호
            if lab == "phone_number" and COMPANY_AFTER.match(text, e["end"]):
                continue  # v9: '0212345678(대표)'
        if lab == "person":
            word = value.replace(" ", "")
            if len(word) <= 1:
                continue  # '허 부장'의 '허'처럼 성 한 글자
            parts = value.split()
            if len(parts) == 2 and len(parts[0]) == 1 and parts[1] in TITLES:
                continue  # '남 부장' 같은 성+직함은 정답 기준상 개인정보 아님
            if word in TITLES:
                continue
            # v8: 부고 단어, 회사 이름, 단어 중간을 자른 것('미르【보안】', '【미수】신')
            if CONDOLENCE.search(word) or ORG_END.search(word):
                continue
            prev = text[e["start"] - 1] if e["start"] > 0 else " "
            nxt = text[e["end"]] if e["end"] < len(text) else " "
            # 앞에 붙은 말이 호칭·직급이면('회원박희정님') 단어 중간이 아니라 띄어쓰기를 안 한 것
            glued_title = ROLE_BEFORE.search(text[max(0, e["start"] - 4):e["start"]])
            if (re.match(r"[가-힣]", prev) and not glued_title) or (re.match(r"[가-힣]", nxt) and nxt not in NAME_NEXT):
                continue
        out.append(e)
    return out
