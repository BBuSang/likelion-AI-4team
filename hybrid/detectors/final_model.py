"""최종 하이브리드: hybrid_best + hosung v8 규칙 중 hybrid_best에 없는 것.

hosung v8 코드는 저장소에 올라와 있지 않아서(remote hosung = v7 4c4aa0b), worklog.md 24-1 설명대로 다시 구현했다.
v8의 나머지(우편번호·읍면 번지·끝자리 1자리 계좌·주격 '이'·'드림'·내선 번호·단어 중간 조각)는 hybrid_best에 이미 있음.
v8은 holdout 오류를 보고 만든 수정이므로 모두 PII_HOLDOUT_FIXES=0 으로 꺼짐 (그러면 hybrid_best와 같음).

규칙 쪽 (v8 #1·#2·#4)
- 사람만 올 수 있는 칸(수취인·송금인·위임인·소명자·피해자·검토자·신고인·청구인·확인자·승인자·심사역)은
  일반 단어와 같은 이름도 받음 ('수취인: 신설')
- 이름 칸의 띄어 쓴 외자 ('대상 고객: 전 찬'). 둘째 글자 뒤에 한글이 이어지면 받지 않음 ('고객: 김 과장')
- 직함 심사역·조사역·검사역·감사역·상담역 뒤의 이름 ('김찬 심사역'), 앞의 이름 ('심사역 안윤이 …' → '안윤')
- 줄 맨 앞 '이름:' 꼴에서 항목 접미사(지·자·인·역·처·명·일·호)로 끝나고 문서에 한 번만 나오면 이름 아님
토큰 쪽 (v8 #6)
- 회사 이름 끝말(보안·시스템즈·테크·솔루션 …)로 끝나는 이름 제거
- '○○ 번호'(사건·접수·주문·민원·관리·예약·신청·송장) 뒤 번호, 연도-일련번호('2026-623736') 제거
"""
import re

from detectors import hosung_rules as H
from detectors import hybrid_best as HB
from detectors import rules_best as R

PERSON_ONLY = "수취인|송금인|위임인|소명자|피해자|검토자|신고인|청구인|확인자|승인자|심사역"
V8_TITLES = "심사역|조사역|검사역|감사역|상담역"
PERSON_FIELD = re.compile(rf"(?<![가-힣A-Za-z])({PERSON_ONLY}) ?[:：] ?([가-힣]{{2,4}})(?![가-힣])")
SPACED_SINGLE = re.compile(rf"(?<![가-힣A-Za-z])(?:{H.NAME_FIELDS}|{PERSON_ONLY}|대상 고객) ?[:：] ?([가-힣] [가-힣])(?![가-힣])")
NAME_BEFORE_TITLE = re.compile(rf"(?<![가-힣])([가-힣]{{2,4}}) ?(?={V8_TITLES})")
NAME_AFTER_TITLE = re.compile(rf"(?:{V8_TITLES}) ([가-힣]{{2,4}})(?![가-힣])")
SPEAKER_LINE = re.compile(r"(?m)^([가-힣]{2,4}) ?[:：]")
ITEM_SUFFIX = set("지자인역처명일호")
COMPANY_END = re.compile(r"(?:보안|시스템즈|시스템|테크|솔루션|컴퍼니|코리아|네트웍스|소프트|데이터|파트너스|상사|상회|산업)$")
NUMBER_CTX = re.compile(r"(?:사건|접수|주문|민원|관리|예약|신청|송장|운송장)\s?번호")
YEAR_SERIAL = re.compile(r"(?:19|20)\d{2}-\d{4,7}")


def load():
    HB.load()


def _name_ok(name, allow_common=False):
    # 부서·기관 이름 제외: '소명자: 고객센터 / 대리 / 김민준', '인사팀' (hosung NON_NAME_SUFFIX: 팀·센터·부·실·점 …)
    org = any(name.endswith(suf) for suf in H.NON_NAME_SUFFIX)
    return (R._surname_ok(name) and not org and name not in R.NOT_NAMES and name not in H.FIELD_WORDS
            and not R._role_noun(name) and name[1:] not in H.TITLES
            and (allow_common or not R._det._is_common_word(name)))


def v8_rule_names(text, ents):
    """v8 #2·#4: 사람 칸 이름(일반 단어 이름 포함), 띄어 쓴 외자, 새 직함 앞뒤 이름."""
    out = []

    def add(s, e):
        if not R._overlaps(ents + out, s, e):
            out.append({"label": "person", "start": s, "end": e})

    for m in PERSON_FIELD.finditer(text):
        if _name_ok(m.group(2), allow_common=True):
            add(m.start(2), m.end(2))
    for m in SPACED_SINGLE.finditer(text):
        if m.group(1)[0] in H.SURNAMES and m.group(1)[2] not in "님씨":
            add(m.start(1), m.end(1))
    for m in NAME_BEFORE_TITLE.finditer(text):
        if _name_ok(m.group(1)) and R._det._looks_like_name(m.group(1)):
            add(m.start(1), m.end(1))
    for m in NAME_AFTER_TITLE.finditer(text):
        e = R.trim_josa_i(text, {"label": "person", "start": m.start(1), "end": m.end(1)})
        if _name_ok(text[e["start"]:e["end"]]) and R._det._looks_like_name(text[e["start"]:e["end"]]):
            add(e["start"], e["end"])
    return out


def v8_drop_speaker_items(text, ents):
    """v8 #1: 줄 맨 앞 '항목명:' — 항목 접미사로 끝나고 문서에 한 번만 나오면 이름이 아님."""
    items = set()
    for m in SPEAKER_LINE.finditer(text):
        w = m.group(1)
        if w[-1] in ITEM_SUFFIX and len(re.findall(rf"(?<![가-힣]){re.escape(w)}", text)) < 2:
            items.add((m.start(1), m.end(1)))
    return [x for x in ents if x["label"] != "person" or (x["start"], x["end"]) not in items]


def v8_model_ok(text, m):
    """v8 #6: 회사 이름 끝말, '○○ 번호' 뒤 번호, 연도-일련번호."""
    value = text[m["start"]:m["end"]]
    if m["label"] == "person":
        return not COMPANY_END.search(value.replace(" ", ""))
    if m["label"] in HB.NUMBER_LABELS:
        return not (NUMBER_CTX.search(R._clause_before(text, m["start"])) or YEAR_SERIAL.fullmatch(value))
    return True


def detect(text):
    load()
    ents = R.detect(text)
    if R.HOLDOUT_FIXES:
        ents = v8_drop_speaker_items(text, ents)
        ents += v8_rule_names(text, ents)
    # 이하 hybrid_best.detect 와 같은 결합 + v8 토큰 후처리
    for m in HB.model_detect(text):
        m = HB.trim_model_name(text, m)
        if HB.MODE == "names" and (m["label"] != "person" or m["end"] - m["start"] < 2):
            continue
        if any(m["start"] < e["end"] and e["start"] < m["end"] for e in ents):
            continue
        if m["label"] == "person" and HB.FILTER and not HB.model_name_ok(text, m):
            continue
        if m["label"] in HB.NUMBER_LABELS and not HB.model_number_ok(text, m):
            continue
        if R.HOLDOUT_FIXES and not v8_model_ok(text, m):
            continue
        if R.HARD_FIXES:
            m = R.trim_name_tail(text, m)
        ents.append(m)
    return sorted(ents, key=lambda x: x["start"])
