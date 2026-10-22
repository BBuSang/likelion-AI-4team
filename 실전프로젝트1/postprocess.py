"""토큰 분류 모델 출력 후처리: 경계 보정 + 미끼 오탐 제거.

v1 test 오류에서 확인한 모델의 실수를 일반 규칙으로 고친다.
- 경계: 번호 앞뒤 숫자를 잘라먹음('1-853629-72'), 이름 뒤 호칭·부르는 말이 붙음('승현아', '노성호님')
- 오탐: 대표번호('159[9-0000]'), 12자리 민원번호를 주민번호로, '허 부장'의 '허' 같은 성+직함
"""
import re

NUMBER_LABELS = {"phone_number", "resident_registration_number", "account_number", "card_number"}
TITLES = {"과장", "대리", "팀장", "부장", "차장", "사원", "주임", "실장", "이사", "상무", "전무", "대표",
          "매니저", "책임", "선임", "수석", "본부장", "센터장", "지점장"}
COMPANY_PHONE = re.compile(r"1[5-9]\d{2}-?\d{4}")  # 대표번호: 1588-xxxx, 1599-xxxx 등


def _is_num(c):
    return c.isdigit()


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
        out.append({**e, "start": s, "end": t})
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
        if lab == "person":
            word = value.replace(" ", "")
            if len(word) <= 1:
                continue  # '허 부장'의 '허'처럼 성 한 글자
            parts = value.split()
            if len(parts) == 2 and len(parts[0]) == 1 and parts[1] in TITLES:
                continue  # '남 부장' 같은 성+직함은 정답 기준상 개인정보 아님
            if word in TITLES:
                continue
        out.append(e)
    return out
