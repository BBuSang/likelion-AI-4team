"""본문 정제 규칙: 표식·반복 문구 제거, 기호 통일, 답변유형, 검색 제외 판단에 쓰는 패턴."""
import re

BOILER = [
    re.compile(r"자세한[ \t]*내용은[ \t]*첨부한?[ \t]*회신문[ \t]*파일을[ \t]*참고하여[ \t]*주시기[ \t]*바랍니다\.?"),
    re.compile(r"^[ \t]*(첨부|붙임)[ \t]*(파일)?[ \t]*(참조|참고)[ \t.]*$", re.M),
    re.compile(r"\n[ \t]*※[ \t]*(비조치의견서|법령해석)의[ \t]*효력[\s\S]*$"),
    re.compile(r"\n[^\n]*\d{4}[ \t]*년[ \t]*\d{1,2}[ \t]*월[ \t]*\d*[ \t]*일?[ \t]*\n?[ \t]*금융(감독원|위원회)[^\n]*\(인\)[ \t]*$"),
    re.compile(r"\[(if [^\]]*|endif)\]"),
    # 워드 처리지시문 찌꺼기: '?xml:namespace prefix = o ns = "…" /', '?xml:namespace prefix = p style="margin'
    # (아래 세 개는 파싱 단계에서 주석·처리지시문을 버리므로 원칙적으로 안 걸린다. 안전장치로 남김, 대소문자 무시)
    re.compile(r"\??xml:namespace\s+prefix\s*=\s*\"?\w+\"?(?:\s*ns\s*=\s*\"[^\"\n]*\")?(?:\s*style=\"[^\"\n]*\"?)?\s*/?", re.I),
    re.compile(r"^[ \t]*\[data-hwpjson\][^\n]*$", re.M),                         # 한컴 웹편집기 내부 데이터
    re.compile(r"rhwp-studio-clipboard:[0-9a-f-]{36}", re.I),                    # 편집기 클립보드 표식
]
# 전화번호: 지역번호가 있거나, T./Tel/☎ 표시가 있거나, 괄호 안에 단독으로 있는 번호만 (연도 범위 "2016-2018" 오인 방지)
PHONE = re.compile(r"(?:0\d{1,2}[-)]\s?\d{3,4}-\d{4}|(?:T\.|Tel\.?|☎)\s*\d{3,4}-\d{4}|(?<=\()\s*(?!19|20)\d{4}-\d{4}\s*(?=\)))")
# 마무리 인사 줄 ("궁금하신 사항은 …(02-…)로 문의하여 주시기 바랍니다") → 줄 전체 삭제 (담당자 이름도 여기 들어 있음)
CLOSING = re.compile(r"^[^\n]*궁금[^\n]*$", re.M)


def strip_phone(t):
    """마무리 인사 줄은 지우고, 판단 내용이 있는 문장에서는 전화번호만 지운다."""
    t = CLOSING.sub(lambda m: "" if PHONE.search(m.group(0)) else m.group(0), t)
    t = re.sub(r"[,，]?\s*" + PHONE.pattern, "", t)
    return re.sub(r"\(\s*\)", "", t)
ANSWER_TYPE = re.compile(r"답변유형[ \t]*[:：][ \t]*([^\n]+)")
# 정해진 답변유형 값 (긴 것부터 비교)
TYPES = ["비조치(무효)", "일부조치", "비조치", "조치", "기타", "행정지도", "감독행정", "협회자율규제"]


def answer_type_of(raw_value):
    v = raw_value.strip()
    return next((t for t in TYPES if v.startswith(t)), None)


# 특수 글꼴(사용자 정의 영역) 문자 → 표준 문자. 문맥으로 확인한 것만.
#   U+F02B1~: 한컴 원문자 ①②③…   U+F081~F089 / U+F08C~F095: 워드 기호 글꼴 원문자   U+F09E: 가운뎃점
PUA_MAP = {**{0xF02B1 + i: chr(0x2460 + i) for i in range(20)},
           **{0xF081 + i: chr(0x2460 + i) for i in range(9)},
           **{0xF08C + i: chr(0x2460 + i) for i in range(10)},
           0xF09E: "·"}


def normalize(t):
    t = (t or "").translate(PUA_MAP)
    t = t.replace("｢", "「").replace("｣", "」")
    # 원문에서 낫표가 물음표로 깨진 법령명 "?보험업법 시행령?" → "「보험업법 시행령」" (12곳, 법령명 꼴만)
    t = re.sub(r"\?([가-힣][^?\n]{1,60}?(?:법률|법|시행령|시행규칙|규정|세칙|규칙))\?", r"「\1」", t)
    t = re.sub(r"[​‌‍﻿\ud800-\udfff�]", "", t)   # 폭 없는 공백, 짝 없는 서로게이트, 깨진 글자
    t = re.sub(r"[    　\t]", " ", t)
    t = t.replace("ㆍ", "·").replace("･", "·").replace("‧", "·").replace("◦", "○")
    t = re.sub(r"[ ]{2,}", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _strip_type_line(m):
    """'답변유형 : 비조치(무효)'처럼 값이 정해진 단어 하나뿐이면 줄을 지우고,
    '답변유형 : 협회자율규제이나 위반시 …'처럼 문장이면 라벨만 지우고 문장은 남긴다."""
    v = m.group(1).strip()
    return "" if v in TYPES else v


def clean(t):
    t = normalize(t)
    for p in BOILER:
        t = p.sub("", t)
    t = re.sub(r"^[ \t]*답변유형[ \t]*[:：][ \t]*([^\n]*)", _strip_type_line, t, flags=re.M)
    t = strip_phone(t)
    return normalize(t)


WITHDRAW = re.compile(r"철회")
REDIRECT = re.compile(r"유권해석으로\s*처리\s*완료|유권해석으로\s*처리완료")
