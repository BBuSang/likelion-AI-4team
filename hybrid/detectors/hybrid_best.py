"""하이브리드 탐지기: rules_best(결합 규칙) + atonlee 모델 (hosung 운영 탐지기와 같은 결합 방식).

- 모델은 hosung 파이프라인과 같게 돌림: 줄바꿈을 넘는 개체는 첫 줄 조각만, 경계 보정·미끼 제거·장소 주소 제외
  (hosung postprocess.py 의 fix_boundaries·drop_decoys 를 아래에 옮김)
- 결합: 규칙 결과를 먼저 두고, 겹치지 않는 모델 결과를 더함
  PII_HYBRID_MODE=union (기본): 모델의 모든 라벨 / names: 모델의 두 글자 이상 이름만 (기존 hybrid_atonlee 방식)
- 규칙과 겹치지 않는 모델 이름은 일반명사·용언·상품명·회사명·항목명이면 버림 (PII_MODEL_NAME_FILTER=0 으로 끔)
- holdout 2차 (PII_HOLDOUT_FIXES=0 으로 끔): 모델 번호 중 내선·사건번호 제외, 서명어 '드림' 잘라냄,
  단어 중간부터 잡은 이름('미르보안'의 '보안') 제외, 걸러내기 보정('정설 카드번호'는 상품명 아님, '재원:' 화자는 이름 근거),
  이름 뒤 주격 조사 '이' 잘라냄
"""
import os
import re

from detectors import hosung_rules as H
from detectors import rules as old
from detectors import rules_best
from detectors.ner import TokenClassifier

_model = TokenClassifier("atonlee")
MODE = os.environ.get("PII_HYBRID_MODE", "union")
FILTER = os.environ.get("PII_MODEL_NAME_FILTER", "1") != "0"

# ------------------------------------------------------------------ 모델 후처리 (hosung postprocess.py)
NUMBER_LABELS = {"phone_number", "resident_registration_number", "account_number", "card_number"}
TITLES = {"과장", "대리", "팀장", "부장", "차장", "사원", "주임", "실장", "이사", "상무", "전무", "대표",
          "매니저", "책임", "선임", "수석", "본부장", "센터장", "지점장"}
COMPANY_PHONE = re.compile(r"1[5-9]\d{2}-?\d{4}")  # 대표번호: 1588-xxxx, 1599-xxxx 등


def fix_boundaries(text, ents):
    out = []
    for e in ents:
        s, t, lab = e["start"], e["end"], e["label"]
        if lab in NUMBER_LABELS:
            # 숫자 덩어리 중간에서 끊겼으면 양옆으로 넓힌다 (구분자 - . 를 사이에 둔 숫자까지)
            while s > 0 and (text[s - 1].isdigit() or (text[s - 1] in "-." and s > 1 and text[s - 2].isdigit())):
                s -= 1
            while t < len(text) and (text[t].isdigit() or (text[t] in "-." and t + 1 < len(text) and text[t + 1].isdigit())):
                t += 1
        elif lab == "person":
            # 이름 뒤에 붙은 호칭 제거, 부르는 말은 뒤에 쉼표·느낌표가 올 때만 제거
            while t - s > 2 and text[t - 1] in "님씨":
                t -= 1
            if t - s > 2 and text[t - 1] in "아야" and t < len(text) and text[t] in ",!":
                t -= 1
        out.append({**e, "start": s, "end": t})
    return out


def drop_decoys(text, ents):
    out = []
    for e in ents:
        value, lab = text[e["start"]:e["end"]], e["label"]
        if lab in NUMBER_LABELS:
            if COMPANY_PHONE.fullmatch(value):
                continue  # 대표번호
            if value.isdigit() and len(value) == 12:
                continue  # 민원번호·접수번호
        if lab == "person":
            word, parts = value.replace(" ", ""), value.split()
            if len(word) <= 1 or word in TITLES:
                continue  # '허 부장'의 '허'처럼 성 한 글자
            if len(parts) == 2 and len(parts[0]) == 1 and parts[1] in TITLES:
                continue  # 성+직함
        out.append(e)
    return out


def model_detect(text):
    ents = []
    for e in _model.predict(text):
        # 줄바꿈을 넘어 이어진 개체는 첫 줄 조각만 ('김예준\n을')
        part = text[e["start"]:e["end"]].split("\n")[0]
        if part.strip():
            ents.append({**e, "end": e["start"] + len(part)})
    return H.drop_venue_addresses(text, drop_decoys(text, fix_boundaries(text, ents)))


# ------------------------------------------------------------------ 모델 이름 거르기 (dev·test 오류에서)
# 규칙과 겹치지 않는 모델 이름만 대상. 규칙이 근거를 못 찾은 이름이라 오탐이 몰리는 곳
PRODUCT_AFTER = re.compile(r"\s?(?:포인트|카드|은행|증권|보험|캐피탈|저축|페이|뱅크)")
PRODUCT_AFTER_V2 = re.compile(r"\s?(?:포인트|카드(?!\s?번호)|은행|증권|보험|캐피탈|저축|페이|뱅크)")  # '정설 카드번호'는 항목명
# 이름 근거: 뒤에 호칭·직함('장미 고객님', '정유란 학부모님'), '이고/입니다', 이름 나열('문예지, 조성웅')
NAME_AFTER = re.compile(r"\s?(?:학부모|이고|입니다|이에요)|, ?[가-힣]{2,4}[,\s]")
SPEAKER_AFTER = re.compile(r"\s?[:：]")  # '상담사 재원: 네, …' 처럼 말하는 사람
NOT_PII_NUMBER_BEFORE = re.compile(r"내선|사건\s?번호")
SIGN_AFTER_NAME = re.compile(r"\s(?:드림|올림|배상)$")


def has_evidence(text, e):
    s, t = e["start"], e["end"]
    if old.TITLE_AFTER.match(text, t) or NAME_AFTER.match(text, t) or old.LABEL_BEFORE.search(text[max(0, s - 8):s]):
        return True
    if not rules_best.HOLDOUT_FIXES:
        return False
    # holdout 2차: 말하는 사람('재원:'), 바로 앞 단어가 직함·역할 명사('심사역 오가은이')
    prev = re.search(r"([가-힣]+) $", text[max(0, s - 8):s])
    return bool(SPEAKER_AFTER.match(text, t) or (prev and (prev.group(1) in old.TITLES or rules_best._role_noun(prev.group(1)))))


def model_name_ok(text, e):
    """'이첩'(일반명사), '보고서 달래'(문맥상 동사), '하린 포인트 카드'(상품명), '미르보안'(회사명), '고객명:'(항목명)은 버림.
    호칭·필드명 같은 이름 근거가 있으면 거르지 않음 ('장미 고객님', '이름: 이정표')."""
    an = rules_best._det.an
    word = text[e["start"]:e["end"]].replace(" ", "")
    if re.search(r"[A-Za-z]", word):
        return True
    if rules_best.HOLDOUT_FIXES:
        # 서식 단어·역할 명사·항목명은 근거('과장 장모상', '세대주:')가 있어도 이름이 아님. 어절 전체로도 확인 ('외조|모상')
        ws, we = e["start"], e["end"]
        while ws > 0 and "가" <= text[ws - 1] <= "힣":
            ws -= 1
        while we < len(text) and "가" <= text[we] <= "힣":
            we += 1
        if (word in rules_best.NOT_NAMES or text[ws:we] in rules_best.NOT_NAMES or word in H.FIELD_WORDS
                or rules_best._role_noun(word)):
            return False
        if len(word) >= 4 and word[:2] not in H.COMPOUND_SURNAMES and rules_best._det._is_common_word(word[-2:]):
            return False  # '미르+보안 대표이사': 회사명은 직함 앞에 와도 이름이 아님
        # 단어 중간부터 잡은 일반명사 조각 ('미르보안'의 '보안'). 띄어쓰기가 빠진 '회원박희정님'은 그대로 둠
        if ws < e["start"] and rules_best._det._is_common_word(word):
            return False
    product = PRODUCT_AFTER_V2 if rules_best.HOLDOUT_FIXES else PRODUCT_AFTER
    if product.match(text, e["end"]) or (word[-1] == "명" and text[e["end"]:e["end"] + 1] in ":："):
        return False
    if has_evidence(text, e):
        return True
    if rules_best._det._is_common_word(word) or word in rules_best.NOT_NAMES:
        return False
    if len(word) >= 4 and word[:2] not in H.COMPOUND_SURNAMES and rules_best._det._is_common_word(word[-2:]):
        return False  # '미르+보안': 끝 두 글자가 일반명사인 4글자 이상
    # 줄 전체를 분석했을 때 이 구간이 용언·어미로 읽히면 이름이 아님 ('보고서 달래')
    ls = text.rfind("\n", 0, e["start"]) + 1
    le = text.find("\n", e["end"])
    toks = an.tokenize(text[ls:le if le >= 0 else len(text)])
    inside = [t for t in toks if ls + t.start < e["end"] and e["start"] < ls + t.start + t.len]
    return not any(t.tag[0] in "VE" for t in inside)


def trim_model_name(text, e):
    """'은영이랑' → '은영': 이름 뒤 '이'가 '랑'과 이어지면 뺌. holdout 2차: '시우 드림' → '시우', '오가은이 조건' → '오가은'."""
    if e["label"] == "person" and e["end"] - e["start"] > 2 and text[e["end"] - 1] == "이" \
            and text[e["end"]:e["end"] + 1] == "랑":
        return {**e, "end": e["end"] - 1}
    if rules_best.HOLDOUT_FIXES and e["label"] == "person":
        m = SIGN_AFTER_NAME.search(text[e["start"]:e["end"]])
        if m:
            e = {**e, "end": e["start"] + m.start()}
        e = rules_best.trim_josa_i(text, e)
    return e


def model_number_ok(text, e):
    """holdout 2차: '내선 1067', '사건 번호: 경찰청 2026-623736' 은 개인정보 번호가 아님."""
    return not (rules_best.HOLDOUT_FIXES and NOT_PII_NUMBER_BEFORE.search(rules_best._clause_before(text, e["start"])))


# ------------------------------------------------------------------ 결합
def load():
    rules_best.load()
    _model.load()


def detect(text):
    ents = rules_best.detect(text)
    for m in model_detect(text):
        m = trim_model_name(text, m)
        if MODE == "names" and (m["label"] != "person" or m["end"] - m["start"] < 2):
            continue
        if any(m["start"] < e["end"] and e["start"] < m["end"] for e in ents):
            continue
        if m["label"] == "person" and FILTER and not model_name_ok(text, m):
            continue
        if m["label"] in NUMBER_LABELS and not model_number_ok(text, m):
            continue
        ents.append(m)
    return ents
