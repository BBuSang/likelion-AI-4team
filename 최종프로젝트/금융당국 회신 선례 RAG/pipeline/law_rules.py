"""조문 추출 규칙: 법령명 사전·정규화, 조문 해석(resolve_method), 법령 → 분야."""
import re


# ---------- 약칭·오기 → 정식 명칭 (공식 약칭 API로 나중에 검증) ----------
ALIASES = {
    "자본시장법": "자본시장과 금융투자업에 관한 법률",
    "자본시장 및 금융투자업에 관한 법률": "자본시장과 금융투자업에 관한 법률",
    "자본시장과 금융투자에 관한 법률": "자본시장과 금융투자업에 관한 법률",
    "신용정보법": "신용정보의 이용 및 보호에 관한 법률",
    "신정법": "신용정보의 이용 및 보호에 관한 법률",
    "지배구조법": "금융회사의 지배구조에 관한 법률",
    "금융회사지배구조법": "금융회사의 지배구조에 관한 법률",
    "금융회사 지배구조법": "금융회사의 지배구조에 관한 법률",
    "금융사지배구조법": "금융회사의 지배구조에 관한 법률",
    "금융회사 지배구조에 관한 법률": "금융회사의 지배구조에 관한 법률",
    "저축은행법": "상호저축은행법",
    "은행업감독규정시행세칙": "은행업감독업무시행세칙",
    "주금공법": "한국주택금융공사법",
    "금용기관의 업무위탁 등에 관한 규정": "금융기관의 업무위탁 등에 관한 규정",
    "금용회사의 업무 위탁 등에 관한 규정": "금융기관의 업무위탁 등에 관한 규정",
    "금융기관의 업무위탁에 관한 규정": "금융기관의 업무위탁 등에 관한 규정",
    "특정금융정보법 보고 및 감독규정": "특정 금융거래정보 보고 및 감독규정",
    "특금법 보고 및 감독규정": "특정 금융거래정보 보고 및 감독규정",
    "여전업감독규정": "여신전문금융업감독규정",
    "여신전문금융업법감독규정": "여신전문금융업감독규정",
    "상호저축은행감독규정": "상호저축은행업감독규정",
    "저축은행업감독규정": "상호저축은행업감독규정",
    "금융투자업 감독규정": "금융투자업규정",
    "정보처리업무위탁규정": "금융회사의 정보처리 업무 위탁에 관한 규정",
    "신용정보의 보호 및 이용에 관한 법률": "신용정보의 이용 및 보호에 관한 법률",
    "신용정보법의 이용 및 보호에 관한 법률": "신용정보의 이용 및 보호에 관한 법률",
    "자본시장법과 금융투자업에 관한 법률": "자본시장과 금융투자업에 관한 법률",
    "자본시장과 금융투자업에 관한 법": "자본시장과 금융투자업에 관한 법률",
    "자본시장과 금융투장업에 관한 법률": "자본시장과 금융투자업에 관한 법률",
    "여신전문금융법": "여신전문금융업법",
    "특정금융거래의 보고 및 이용 등에 관한 법률": "특정 금융거래정보의 보고 및 이용 등에 관한 법률",
    "독점규제 및 공정경쟁에 관한 법률": "독점규제 및 공정거래에 관한 법률",
    "금융기관 검사 및 제재에 대한 규정": "금융기관 검사 및 제재에 관한 규정",
    "인터넷전문은행법 설립 및 운영에 관한 특례법": "인터넷전문은행 설립 및 운영에 관한 특례법",
    "금융회사지배구조법에 관한 법률": "금융회사의 지배구조에 관한 법률",
    "정보통신망 이용촉진 및 정보보보 등에 관한 법률": "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
    "근로자퇴직급여법": "근로자퇴직급여 보장법",
    "금융회사의 해외진출에 관한 규정": "금융기관의 해외진출에 관한 규정",
    "수협법": "수산업협동조합법",
    "온투업법 감독규정": "온라인투자연계금융업 감독규정",
    "농어업경영체법": "농어업경영체 육성 및 지원에 관한 법률",
    "공공기록물법": "공공기록물 관리에 관한 법률",
    "전자금융법": "전자금융거래법",
    "금융소비자법": "금융소비자 보호에 관한 법률",
    "인터넷전문은행법": "인터넷전문은행 설립 및 운영에 관한 특례법",
    "특정 금융거래정보의 보고 및 이용에 관한 법률": "특정 금융거래정보의 보고 및 이용 등에 관한 법률",
    "금융소비자의 보호에 관한 법률": "금융소비자 보호에 관한 법률",
    "주식회사의 외부감사에 관한 법률": "주식회사 등의 외부감사에 관한 법률",
    "주식회사 외부 감사에 관한 법률": "주식회사 등의 외부감사에 관한 법률",
    "주식회사 외부감사에 관한 법률": "주식회사 등의 외부감사에 관한 법률",
    "대부업법": "대부업 등의 등록 및 금융이용자 보호에 관한 법률",
    "개정 대부업법": "대부업 등의 등록 및 금융이용자 보호에 관한 법률",
    "특정금융정보법": "특정 금융거래정보의 보고 및 이용 등에 관한 법률",
    "특금법": "특정 금융거래정보의 보고 및 이용 등에 관한 법률",
    "금융실명법": "금융실명거래 및 비밀보장에 관한 법률",
    "실명법": "금융실명거래 및 비밀보장에 관한 법률",
    "여전법": "여신전문금융업법",
    "금산법": "금융산업의 구조개선에 관한 법률",
    "금융산업구조개선법": "금융산업의 구조개선에 관한 법률",
    "금융소비자보호법": "금융소비자 보호에 관한 법률",
    "금소법": "금융소비자 보호에 관한 법률",
    "외부감사법": "주식회사 등의 외부감사에 관한 법률",
    "외감법": "주식회사 등의 외부감사에 관한 법률",
    "신협법": "신용협동조합법",
    "개인정보보호법": "개인정보 보호법",
    "개인채무자보호법": "개인금융채권의 관리 및 개인금융채무자의 보호에 관한 법률",
    "채무자회생법": "채무자 회생 및 파산에 관한 법률",
    "기촉법": "기업구조조정 촉진법",
    "공정거래법": "독점규제 및 공정거래에 관한 법률",
    "온투법": "온라인투자연계금융업 및 이용자 보호에 관한 법률",
    "온투업법": "온라인투자연계금융업 및 이용자 보호에 관한 법률",
    "온라인투자연계금융업법": "온라인투자연계금융업 및 이용자 보호에 관한 법률",
    "가상자산이용자보호법": "가상자산 이용자 보호 등에 관한 법률",
    "통신사기피해환급법": "전기통신금융사기 피해 방지 및 피해금 환급에 관한 특별법",
    "정보통신망법": "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
    "전자문서법": "전자문서 및 전자거래 기본법",
    "금융위원회법": "금융위원회의 설치 등에 관한 법률",
    "금융위원회 설치법": "금융위원회의 설치 등에 관한 법률",
    "자본시장과 금융투자업과 관한 법률": "자본시장과 금융투자업에 관한 법률",
    "자본시장과 금융투자업자에 관한 법률": "자본시장과 금융투자업에 관한 법률",
    "자본시장과 금전투자에 관한 법률": "자본시장과 금융투자업에 관한 법률",
    "금융회사의 지배구조에 관한 법률 지배구조법": "금융회사의 지배구조에 관한 법률",
    "전자규정감독규정": "전자금융감독규정",
    "은행업갑독규정": "은행업감독규정",
    "산립조합법": "산림조합법",
    "굼융회사의 정보처리 업무 위탁에 관한 규정": "금융회사의 정보처리 업무 위탁에 관한 규정",
    "기업구조조정촉전법": "기업구조조정 촉진법",
    "주택금융공사법": "한국주택금융공사법",
    "농협법": "농업협동조합법",
    "주식회사의 외부감사 등에 대한 법률": "주식회사 등의 외부감사에 관한 법률",
    "금융기관의 업무위수탁 등에 관한 규정": "금융기관의 업무위탁 등에 관한 규정",
    "금융회사의 업무위수탁 등에 관한 규정": "금융기관의 업무위탁 등에 관한 규정",
    "금융기관 업무위탁 규정": "금융기관의 업무위탁 등에 관한 규정",
    "정보통신망법 이용촉진 및 정보보호 등에 관한 법률": "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
    "정보통신망 이용촉진 및 비밀보장에 관한 법률": "정보통신망 이용촉진 및 정보보호 등에 관한 법률",
    "통신피해사기환급법": "전기통신금융사기 피해 방지 및 피해금 환급에 관한 특별법",
    "상호저축은행법 감독규정": "상호저축은행업감독규정",
    "자금세탁방지 및 공중협박자금조달금지에 관한규정": "자금세탁방지 및 공중협박자금조달금지에 관한 업무규정",
    "금융산업의 구조개선에 관한 법": "금융산업의 구조개선에 관한 법률",
    "금융위원회의 설치 등에 관한 법률": "금융위원회의 설치 등에 관한 법률",
    "금융혁신법": "금융혁신지원 특별법",
    "퇴직급여법": "근로자퇴직급여 보장법",
    "전금법": "전자금융거래법",
    "자산유동화법": "자산유동화에 관한 법률",
    "서민금융법": "서민의 금융생활 지원에 관한 법률",
}
# 전체 이름 기준 보정: 감독'규정' 시행세칙이라고 쓰지만 실제 이름은 감독'업무' 시행세칙
FULL_ALIAS = {
    "은행업감독규정시행세칙": "은행업감독업무 시행세칙",
    "여신전문금융업감독규정시행세칙": "여신전문금융업감독업무 시행세칙",
    "상호금융업감독규정시행세칙": "상호금융업감독업무 시행세칙",
    "상호저축은행감독업무시행세칙": "상호저축은행업감독업무 시행세칙",
    "상호저축은행업감독업무시행세칙": "상호저축은행업감독업무 시행세칙",
    "상호저축은행업감독규정시행세칙": "상호저축은행업감독업무 시행세칙",
    "보험업감독규정시행세칙": "보험업감독업무 시행세칙",
}
# 절차 규정: 회신 요청 자체의 처리 근거라서 '어떤 조문을 해석했나'에서 뺀다
STOP = {"법령해석및비조치의견서업무처리에관한운영규칙", "법령해석및비조치의견서업무처리에대한운영규칙"}

SUFFIX = r"(?:법률|특별법|법|시행령|시행규칙|시행세칙|감독규정|업무규정|규정|세칙|규칙)"
TAILS = ("시행령", "시행규칙", "시행세칙")


def key(s):
    return re.sub(r"\s+", "", s or "")


def fkey(s):
    """같은 법령을 묶는 느슨한 키: 띄어쓰기, 조사 '의', 가운뎃점 종류를 무시 ('금융위원회 설치 등에 관한 법률' = '금융위원회의 설치 …')"""
    return re.sub(r"[·ㆍ・․‧]", "", key(s)).replace("의", "")


# 이름만으로는 어느 법인지 모르는 일반어, 앞 법령을 가리키는 말 → 법령명 사전에 넣지 않는다
GENERIC = {"법", "법률", "특별법", "시행령", "시행규칙", "시행세칙", "감독규정", "업무규정", "규정", "세칙", "규칙",
           "본법", "이법", "동법", "같은법", "관련법", "관계법", "개정법", "현행법", "구법", "신법", "모법",
           "동규정", "같은규정", "본규정", "위규정", "동시행령", "같은시행령", "동시행규칙", "동시행세칙", "동세칙", "동감독규정",
           "위탁규정", "수탁규정", "이행규정", "시행규정", "현행규정", "관련규정", "해당규정"}
# 수식어는 뒤에 공백이 있을 때만 뗀다 ("이자제한법", "본인서명사실…", "위탁규정"의 첫 글자를 떼지 않기 위해)
RELATIVE_PREFIX = re.compile(r"^(?:동|같은|위|이|해당|관련|관계|상기|당해|본|구|현행|개정|각)\s+")


DEF_INSIDE = re.compile(r"\s*\(\s*이하[^)]*\)\s*")


def strip_prefix(name):
    """'구 증권거래법' → '증권거래법', '현행 은행법' → '은행법' (수식어 제거)
    '자본시장과 금융투자업에 관한 법률(이하 ‘자본시장법’)' → 괄호 안 약칭 정의 제거"""
    name = DEF_INSIDE.sub("", name)
    return RELATIVE_PREFIX.sub("", re.sub(r"\s+", " ", name).strip())


def is_generic(name):
    """수식어를 떼고 남은 말이 일반어면 True ('동법 시행령', '같은 법', '본 규정' …)"""
    rest = strip_prefix(name)
    base, _ = split_tail(rest)
    return key(name) in GENERIC or key(rest) in GENERIC or key(base) in GENERIC or not key(base)


def kind_of(name):
    if name.endswith("시행령"):
        return "시행령"
    if name.endswith("시행규칙"):
        return "시행규칙"
    if re.search(r"(법률|특별법|법)$", name):
        return "법률"
    return "행정규칙"   # 감독규정·업무규정·시행세칙·규칙 등


def split_tail(name):
    """'보험업법시행령' → ('보험업법', '시행령'), '전자금융감독규정시행세칙' → ('전자금융감독규정', '시행세칙')"""
    n = re.sub(r"\s+", " ", name).strip()
    for t in TAILS:
        if key(n).endswith(t) and key(n) != t:
            base = re.sub(r"\s*" + r"\s*".join(t) + r"$", "", n).strip()
            return base, t
    return n, None


class Canon:
    """표기 → 정식 명칭. 띄어쓰기만 다른 표기는 가장 많이 쓴 표기로 통일."""

    def __init__(self, spellings):
        # spellings: {표기: 횟수} (「」 안 표기 전체). 같은 법령(느슨한 키 기준)은 가장 많이 쓴 표기로 통일
        self.alias = {fkey(k): v for k, v in ALIASES.items()}
        best = {}
        for s, c in spellings.items():
            if is_generic(s):
                continue
            b, _ = split_tail(strip_prefix(s))
            k = fkey(b)
            if k not in best or c > best[k][1]:
                best[k] = (b, c)
        self.display = {k: v[0] for k, v in best.items()}
        for v in ALIASES.values():           # 사전의 정식 명칭이 최우선
            self.display[fkey(v)] = v
        self.known = set(self.display) | set(self.alias)

    def __call__(self, name, local=None):
        base, tail = split_tail(strip_prefix(name))
        k = fkey(base)
        if local and key(base) in local:
            defined = fkey(local[key(base)])
            if defined in self.alias or defined in self.display:   # 오타가 든 정의문이면 약칭 사전으로 해석한다
                k = defined
        if k in self.alias:
            k = fkey(self.alias[k])
        disp = self.display.get(k, base)
        out = f"{disp} {tail}" if tail else disp
        return FULL_ALIAS.get(key(out).replace("의", ""), out)


# 「정식명」(이하 "약칭") / 「정식명(이하 '약칭')」 / 정식명(이하 “약칭“) — 따옴표 종류가 제각각이다
_Q = "“”\"‘’'`"
DEF_ALIAS = re.compile(
    r"(?:[「｢]([^」｣\n]{2,80}?)[」｣]|([가-힣][가-힣·\s]{1,60}?(?:법률|법|시행령|규정|세칙)))\s*\(\s*이하\s*,?\s*[" + _Q + r"]?\s*[「｢]?"
    r"([^」｣" + _Q + r")\n]{1,25}?)[」｣]?\s*[" + _Q + r"]?\s*(?:이라|라)?\s*(?:한다|함|합니다)?\s*\)"
    r"|[「｢]([^」｣\n(]{2,80}?)\s*\(\s*이하\s*,?\s*[" + _Q + r"]?\s*([^" + _Q + r")\n]{1,25}?)\s*[" + _Q + r"]?\s*(?:이라\s*한다)?\s*\)\s*[」｣]"
)
QUOTED = re.compile(r"[「｢]\s*([^「｢」｣\n]{2,80}?)\s*[」｣]")
# 호: '제5호', '제5호의2', '제5의2호'
_ITEM = (r"(?:\s*(?:제\s*)?(?:(?P<item>\d+)\s*호(?:\s*의\s*(?P<isub>\d+)(?![\d.]*\s*(?:개|명|년|월|일|배|%|조|항)))?"
         r"|(?P<item2>\d+)\s*의\s*(?P<isub2>\d+)\s*호))?")
_PARA = r"(?:\s*(?:제\s*)?(?P<para>\d+)\s*항|(?P<pc>[①-⑳])(?:\s*항)?)?"
ARTICLE = re.compile(
    r"(?P<sup>부칙\s*)?제\s*(?P<art>\d+(?:-\d+)*)\s*조(?:\s*의\s*(?P<sub>\d+)|(?P<sub2>\d{1,2})(?!\d)(?=의|에|,|\s*제|\s*\(|\s+\d+\s*항))?"   # '제11조2의'는 제11조의2 오타
    + _PARA + _ITEM
)
# '§2①7', '§11조의2제2항': 법령 앞뒤 문맥은 일반 조문과 같게 해석한다
SECTION = re.compile(
    r"§\s*(?P<art>\d+(?:-\d+)*)(?:\s*조)?(?:\s*의\s*(?P<sub>\d+))?"
    r"(?:\s*(?:제\s*)?(?P<para>\d+)\s*항|\s*(?P<pc>[①-⑳])(?:\s*항)?(?:[·ㆍ]?\s*(?P<item3>\d+)(?![\d조항호]))?)?" + _ITEM
)
# '신협법 28조11항', '보험업감독규정 7-45조의2': '제'가 없으면 금액 등과 섞이므로 법령명 바로 뒤일 때만 인정한다
BARE_ARTICLE = re.compile(
    r"(?<![\d\-.,제가-힣~∼§])(?<!제\s)(?<!제\s\s)(?<!§\s)(?P<art>\d{1,3}(?:-\d+)*)\s*조(?!\s*(?:원|달러|억|만))(?:\s*의\s*(?P<sub>\d+)|(?P<sub2>\d{1,2})(?!\d)(?=\s+\d+\s*항))?"
    + _PARA + _ITEM
)
SAME_ARTICLE = re.compile(r"(?:같은|동)\s*조\s*(?:제\s*)?(?P<para>\d+)\s*항" + _ITEM)
# 앞 법령을 가리키는 말
RELATIVE = re.compile(
    r"(?<![가-힣])(?:동|같은|위|이|해당)\s*(?P<what>법률|법|규정|감독규정)?\s*(?P<tail>시행령|시행규칙|시행세칙)?(?=\s*(?:[」｣(]|제\s*\d|상|에|의|은|는|및|,|\s))"
)
BASE_REF = re.compile(r"(?<![가-힣])(?P<what>특별법|법|영|시행령|규칙|시행규칙|감독규정|업무규정|규정|세칙|시행세칙)\s*\*{0,3}\s*(?:\([^)\n]{0,20}\)\s*)?(?=제\s*\d|§\s*\d)")


def local_aliases(text, known=None):
    """본문의 약칭 정의를 읽는다. 괄호 없는 정의('…법률(이하 "금융실명법")')는 앞 글이 함께 잡히므로
    사전(known: fuzzy key 집합)에 있는 법령명과 끝부분이 일치할 때만 인정하고 그 이름으로 자른다."""
    out = {}
    for m in DEF_ALIAS.finditer(text):
        short = (m.group(3) or m.group(5) or "").strip()
        if m.group(2):  # 괄호 없는 정의
            words = m.group(2).split()
            full = next((" ".join(words[i:]) for i in range(len(words))
                         if known is not None and fkey(" ".join(words[i:])) in known), None)
            if full is None:
                continue
        else:
            full = (m.group(1) or m.group(4) or "").strip()
        full = RELATIVE_PREFIX.sub("", full)
        if re.search(SUFFIX + r"$", full) and 2 <= len(key(short)) and not is_generic(full) and key(short) not in GENERIC:
            out[key(short)] = full
    return out


def build_bare_pattern(names):
    """사전 이름(정식·약칭·본문 표기)을 띄어쓰기 무시로 찾는 정규식. 긴 것부터."""
    # 2글자 이름은 '상법'·'민법'처럼 법으로 끝나는 것만 (다른 말과 겹칠 위험)
    names = {strip_prefix(n) for n in names if not is_generic(n)}
    pats = sorted({key(n) for n in names if len(key(n)) >= 3 or (len(key(n)) == 2 and key(n).endswith("법"))},
                  key=len, reverse=True)
    alt = "|".join(r"\s*".join(map(re.escape, p)) for p in pats)
    return re.compile(r"(?<![가-힣A-Za-z])(?:" + alt + r")(?:\s*(?:시행령|시행규칙|시행세칙))?")

GENERIC_SHORT = {"법", "영", "시행령", "규칙", "시행규칙", "규정", "세칙", "시행세칙", "감독규정", "업무규정"}
# 약칭 정의 안의 이름이 법령으로 잡히면 '’이라 함)'이 조문 앞에 남는다
ADJACENT = re.compile(r"[\s」｣”\"'’)\]]*(?:이라\s*(?:함|한다|칭함|합니다)\s*\)?)?(?:\(\s*이하[^)\n]*\))?\s*(?:상|의|에|에서|중|은|는)?\s*\(?\s*")
_CONT_TOKEN = (r"[,，·ㆍ、]|및|또는|과|와|내지|~|∼|-|부터|까지|이나|각\s*호|본문|단서|전단|후단"
               r"|제\s*\d+\s*호\s*의\s*\d+|제\s*\d+\s*의\s*\d+\s*호|제\s*\d+\s*(?:항|호)"
               r"|[가-하]목|\([^)\n]{0,30}\)|\s+")
CONTINUE = re.compile(r"(?:" + _CONT_TOKEN + r")*")
NONLAW_DOC = re.compile(r"모범규준|방법서|약관|가이드라인|지침|협약|정관|내규|약정서|계약서|요령|규약|매뉴얼|안내서")
NAMED_BEFORE = re.compile(r"(?:관한|대한|위한|시장)\s*$")     # '파생상품시장 업무규정'도 사전에 없는 다른 규정
UNKNOWN_LAW_BEFORE = re.compile(r"(?:법률|법|령)['’”\"」\s]*$")   # 사전에 없는 법령명 바로 뒤의 조문
METHOD_RANK = {"explicit": 0, "context": 1, "default": 2, "unresolved": 3}


def local_generic(text, known=None):
    """사례 본문의 '(이하 "법")', '(이하 "규정")' 정의를 읽는다. 이런 줄임말은 그 사례 안에서 정의된 법령을 가리킨다."""
    out = {}
    for m in DEF_ALIAS.finditer(text):
        short = key(m.group(3) or m.group(5) or "")
        if short not in GENERIC_SHORT:
            continue
        if m.group(2):
            words = m.group(2).split()
            full = next((" ".join(words[i:]) for i in range(len(words))
                         if known is not None and fkey(" ".join(words[i:])) in known), None)
            if full is None:
                continue
        else:
            full = (m.group(1) or m.group(4) or "").strip()
        full = RELATIVE_PREFIX.sub("", full)
        if re.search(SUFFIX + r"$", full) and not is_generic(full):
            out.setdefault(short, full)
    return out


def related(root, reg):
    """모법과 행정규칙이 같은 업권인가. 은행업감독규정 안의 '영 제N조'를 보험업법 시행령으로 잇지 않기 위해 쓴다."""
    a, b = category_of(root), category_of(reg)
    return a is not None and a == b


def item_of(m):
    """정규식 결과에서 호와 호의 가지번호를 꺼낸다."""
    d = m.groupdict()
    item, isub = d.get("item") or d.get("item2") or d.get("item3"), d.get("isub") or d.get("isub2")
    return {"item": int(item) if item else None, "item_sub": int(isub) if isub else None}


def para_of(m):
    """항 번호. '제2항'과 '②' 표기를 모두 숫자로."""
    if m.group("para"):
        return int(m.group("para"))
    return ord(m.group("pc")) - 0x245F if m.group("pc") else None


class ArtRef:
    """조문 표기 정규식 결과를 같은 모양으로 다룬다. bare: '제' 없는 표기."""

    def __init__(self, m, bare=False):
        self.d, self.text, self.bare = m.groupdict(), m.group(0), bare

    def group(self, k=0):
        return self.text if k == 0 else self.d.get(k)

    def groupdict(self):
        return self.d


def extract(text, section, canon, bare_pat, local, local_gen=None, default_law=None, default_reg=None, single_law=False):
    """한 섹션에서 법령 언급과 조문을 뽑는다. 조문마다 resolve_method를 남긴다.
    explicit: 법령명 바로 뒤, 또는 그 조문에 이어지는 나열
    context:  동법·같은 법·시행령·정의된 줄임말·조문 인용 블록, 법령이 하나뿐인 사례의 조문
    default:  칸 안에 앞선 법령이 없어 사례의 주 법령으로 해석
    unresolved: 어느 법령인지 판단 불가 (law_name 없음, law_guess에 후보만 남김)"""
    local_gen = local_gen or {}
    events, taken = [], []
    for m in QUOTED.finditer(text):
        name = re.sub(r"\s+", " ", m.group(1)).strip()
        if key(name) in GENERIC_SHORT:
            events.append((m.start(), m.end(), "base", key(name)))
            taken.append((m.start(), m.end()))
        elif re.search(SUFFIX + r"$", name) and not is_generic(name):
            events.append((m.start(), m.end(), "law", name))
            taken.append((m.start(), m.end()))
    for m in bare_pat.finditer(text):
        if any(s <= m.start() < e for s, e in taken):
            continue
        events.append((m.start(), m.end(), "law", m.group(0)))
        taken.append((m.start(), m.end()))
    if local:
        lp = re.compile(r"(?<![가-힣])(?:" + "|".join(r"\s*".join(map(re.escape, k)) for k in sorted(local, key=len, reverse=True)) + r")")
        for m in lp.finditer(text):
            if not any(s <= m.start() < e for s, e in taken):
                events.append((m.start(), m.end(), "law", m.group(0)))
                taken.append((m.start(), m.end()))
    for m in RELATIVE.finditer(text):
        if (m.group("what") or m.group("tail")) and not any(s <= m.start() < e for s, e in taken):
            events.append((m.start(), m.end(), "rel", (m.group("what"), m.group("tail"))))
            taken.append((m.start(), m.end()))     # '동 규정'의 '규정'을 다시 줄임말로 읽지 않는다
    for m in BASE_REF.finditer(text):
        if not any(s <= m.start() < e for s, e in taken):
            events.append((m.start(), m.end(), "base", m.group("what")))
    for m in SAME_ARTICLE.finditer(text):
        events.append((m.start(), m.end(), "same", m))
    for m in ARTICLE.finditer(text):
        events.append((m.start(), m.end(), "art", ArtRef(m)))
    for m in SECTION.finditer(text):
        if "별표" not in text[max(0, m.start() - 14):m.start()]:   # '<별표5의2> §9'는 별표 안의 항목
            events.append((m.start(), m.end(), "art", ArtRef(m)))
    for m in BARE_ARTICLE.finditer(text):
        events.append((m.start(), m.end(), "art", ArtRef(m, bare=True)))
    events.sort(key=lambda x: x[0])

    mentions, arts = [], []
    cur, root, src, cur_end, law_end = None, None, None, 0, 0
    last, last_end, last_heading = None, 0, False

    def set_cur(name, how, end):
        nonlocal cur, src, cur_end
        cur, src, cur_end = name, how, end

    def is_far(pos):
        """모법이 사례의 주 법률과 다르고 법령명이 멀리 앞에 있으면, '법·동법·영'이 그 법을 뜻한다고 단정하지 않는다."""
        near = text.count("\n", law_end, pos) == 0 and pos - law_end <= 60
        return bool(root and default_law and root != default_law and not near)

    paren_close, paren_root = None, None
    for s, e, kind, val in events:
        if paren_close is not None and s > paren_close:   # 괄호 안에서 인용한 법령은 괄호를 벗어나면 모법이 아니다
            if paren_root is not None or root != default_law:
                root = paren_root
            paren_close = None
        if kind == "law":
            name = canon(val, local)
            if key(name) in STOP:
                set_cur("__STOP__", "law", e)
                continue
            set_cur(name, "law", e)
            law_end = e
            ls = text.rfind("\n", 0, s) + 1
            if paren_close is None and text.rfind("(", ls, s) > text.rfind(")", ls, s):
                close = text.find(")", e)
                if close != -1 and "\n" not in text[e:close] and not text[e:close].lstrip("」｣ ").startswith("("):
                    paren_close, paren_root = close, root
            base, tail = split_tail(name)
            if kind_of(name) == "법률":
                root = name
            elif tail in ("시행령", "시행규칙"):
                root = base
            mentions.append(name)
        elif kind == "rel":
            what, tail = val
            if cur in (None, "__STOP__"):
                set_cur(None, None, e)
                continue
            base = split_tail(cur)[0]
            # 행정규칙을 말하다가 나온 '동법'이 업권이 다른 법률을 가리킬 수는 없다
            if kind_of(cur) == "행정규칙" and what in ("법", "법률") and root and not related(root, cur):
                set_cur(None, None, e)
                continue
            if (what in ("법", "법률") or tail in ("시행령", "시행규칙")) and is_far(s):
                set_cur(None, None, e)     # 앞선 법령이 있는데 멀다: 어느 법인지 단정하지 않는다
                continue
            if tail:
                name = f"{root or base} {tail}" if tail in ("시행령", "시행규칙") else f"{base} {tail}"
                name = canon(name, local)
                set_cur(name, "rel", e)
                mentions.append(name)
            elif what in ("법", "법률"):
                set_cur(root or base, "rel", e)
            elif kind_of(cur) == "행정규칙":
                set_cur(cur, "rel", e)     # 동 규정 → 앞 규정
            else:
                set_cur(None, None, e)     # 법률을 말하다가 '동 규정'은 어느 규정인지 알 수 없다
        elif kind == "base":
            if val in local_gen:           # 사례 안에서 정의된 줄임말이 최우선
                name = canon(local_gen[val], local)
                set_cur(name, "rel", e)
                if kind_of(name) == "법률":
                    root = name
                elif split_tail(name)[1] in ("시행령", "시행규칙"):
                    root = split_tail(name)[0]
                mentions.append(name)
                continue
            if val in ("규정", "업무규정", "감독규정", "세칙", "규칙") and NAMED_BEFORE.search(text[max(0, s - 6):s]):
                set_cur(None, None, e)     # '…에 관한 규정 제N조': 사전에 없는 다른 규정
                continue
            # '금융위원회 설치법 시행령'처럼 사전에 없는 법령의 시행령이면 앞 법령으로 잇지 않는다
            if val in ("영", "시행령", "규칙", "시행규칙") and re.search(r"[가-힣]법\s*$", text[max(0, s - 12):s]):
                set_cur(None, None, e)
                continue
            val = "법" if val == "특별법" else val
            is_reg = cur not in (None, "__STOP__") and kind_of(cur) == "행정규칙"
            # '감독규정'·'업무규정'은 그 이름을 가진 규정만 가리킨다 (행정지도 운영규칙 등 제외)
            if is_reg and val in ("감독규정", "업무규정") and val not in key(cur):
                is_reg = False
            far_root = is_far(s)
            if val == "법":
                if is_reg and root and not related(root, cur):
                    set_cur(None, None, e)
                elif far_root:
                    set_cur(None, None, e)
                elif root:
                    set_cur(root, "rel", e)
                elif default_law:
                    root = default_law
                    set_cur(default_law, "default", e)
                else:
                    set_cur(None, None, e)
            elif val in ("영", "시행령", "규칙", "시행규칙"):
                tail = "시행령" if val in ("영", "시행령") else "시행규칙"
                if is_reg and root and not related(root, cur):
                    set_cur(None, None, e)
                elif far_root:
                    set_cur(None, None, e)
                elif root:
                    set_cur(canon(f"{root} {tail}", local), "rel", e)
                elif default_law:
                    root = default_law
                    set_cur(canon(f"{default_law} {tail}", local), "default", e)
                else:
                    set_cur(None, None, e)
            elif val in ("세칙", "시행세칙"):
                if is_reg:
                    base = split_tail(cur)[0]
                    set_cur(canon(f"{base} 시행세칙", local), "rel", e)
                elif default_reg:
                    set_cur(canon(f"{split_tail(default_reg)[0]} 시행세칙", local), "default", e)
                else:
                    set_cur(None, None, e)
            else:                          # 감독규정·업무규정·규정
                if is_reg:
                    set_cur(split_tail(cur)[0] if split_tail(cur)[1] == "시행세칙" else cur, "rel", e)
                elif default_reg and (val == "규정" or val in key(default_reg)):
                    set_cur(default_reg, "default", e)
                else:
                    set_cur(None, None, e)
        elif kind == "same":
            if last is not None and cur != "__STOP__":
                a = dict(last, paragraph=int(val.group("para")), **item_of(val),
                         raw=re.sub(r"\s+", "", val.group(0)), section=section, pos=s)
                # 줄을 넘긴 '동조'인데 앞 줄에 법령이 둘 이상이면, 끼어든 인용과 본 조문을 구분할 수 없다
                if "\n" in text[last_end:s]:
                    start = text.rfind("\n", 0, last["pos"]) + 1
                    if len({x["law_name"] for x in arts if x["pos"] >= start}) > 1:
                        a.update(law_guess=last["law_name"], law_name=None, law_kind=None, resolve_method="unresolved")
                arts.append(a)
                last_end = e
        elif kind == "art":
            if val.bare and not (cur not in (None, "__STOP__") and ADJACENT.fullmatch(text[cur_end:s])):
                continue
            if cur == "__STOP__":
                last, last_end = None, e
                continue
            a = {"law_name": None, "law_kind": None, "section": section, "article": val.group("art"),
                 "article_sub": int(val.group("sub") or val.group("sub2")) if (val.group("sub") or val.group("sub2")) else None,
                 "paragraph": para_of(val), **item_of(val),
                 "supplementary": bool(val.group("sup")), "raw": re.sub(r"\s+", "", val.group(0)),
                 "resolve_method": "unresolved", "law_guess": cur, "pos": s}
            method, heading = None, False
            if last is not None and CONTINUE.fullmatch(text[last_end:s]):
                method = last["resolve_method"]            # 앞 조문에 이어지는 나열은 같은 법령
                name = last["law_name"]
                heading = last_heading
                a["supplementary"] = a["supplementary"] or last["supplementary"]   # '부칙 제3조 및 제10조'
            elif cur is not None and ADJACENT.fullmatch(text[cur_end:s]):
                method = {"law": "explicit", "rel": "context", "default": "default"}[src]
                name = cur
            elif cur is not None:
                line_start = text.rfind("\n", 0, s) + 1
                is_heading = not re.sub(r"[\s□■○●ㅇ◎※\-\*·]", "", text[line_start:s]) and text[e:e + 1] == "("
                sentence = re.split(r"[.\n]", text[max(0, s - 200):s])[-1]
                if is_heading and (text.count("\n", cur_end, s) <= 2 or last_heading):
                    method, name, heading = "context", cur, True   # 법령명 바로 아래의 조문 인용 블록
                elif (single_law and src == "law" and not NONLAW_DOC.search(sentence)
                      and not UNKNOWN_LAW_BEFORE.search(text[max(0, s - 14):s])):
                    method, name = "context", cur          # 법령이 하나뿐인 사례 (모범규준·약관 등을 말하는 문장은 제외)
            if heading and "부칙" in text[cur_end:s]:
                a["supplementary"] = True
            # '제4-90조'형 번호는 규정에만 있다
            if method and method != "unresolved" and "-" in val.group("art") and kind_of(name) != "행정규칙":
                method = None
            if method and method != "unresolved":
                a.update(law_name=name, law_kind=kind_of(name), resolve_method=method, law_guess=None)
            arts.append(a)
            last, last_end, last_heading = a, e, heading
    return mentions, arts


# ---------- B2 법령 → 분야 ----------
# 포털 분야: 공통 / 은행 / 보험 / 전자금융 / 정책금융 / 자본시장 / 여신전문금융업 / 상호·저축은행업
LAW_CATEGORY = [
    (r"^은행법|^은행업감독|^외국환거래|^외국환거래규정|^금융지주회사", "은행"),
    (r"^한국산업은행법|^중소기업은행법|^한국수출입은행법|^신용보증기금법|^기술보증기금법|^서민의금융생활|^한국주택금융공사법|^주택도시기금법", "정책금융"),
    (r"^보험업|^보험업감독|^보험사기", "보험"),
    (r"^자본시장과금융투자업|^금융투자업규정|^증권의발행및공시|^금융투자회사의영업및업무|^자산유동화|^부동산투자회사법|^간접투자|^증권거래법|^자본시장조사", "자본시장"),
    (r"^전자금융|^전자서명법|^전자문서및전자거래|^온라인투자연계|^가상자산", "전자금융"),
    (r"^여신전문금융업|^대부업|^대부업등감독", "여신전문금융업"),
    (r"^상호저축은행|^신용협동조합|^상호금융업감독|^농업협동조합|^수산업협동조합|^산림조합|^새마을금고", "상호·저축은행업"),
]


def category_of(law_name):
    k = key(law_name)
    for pat, cat in LAW_CATEGORY:
        if re.search(pat, k):
            return cat
    return None
