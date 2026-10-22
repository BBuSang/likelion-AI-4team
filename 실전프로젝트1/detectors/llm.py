"""LLM 탐지기: OpenAI 호환 Chat Completions API (기본 gpt-4.1-mini, temperature 0).

- LLM에는 원문을 보내고, 개인정보 문자열과 라벨만 JSON으로 받음 (위치 계산은 LLM에 맡기지 않음)
- 받은 문자열을 원문에서 찾아 모든 등장 위치를 구간으로 만듦 (긴 문자열 먼저, 겹침은 base.postprocess가 정리)
- 원문에 없는 문자열(LLM이 고치거나 지어낸 값)은 버림
- 설정: OPENAI_API_KEY (없으면 '../API key/env' 파일의 openai-api-key=... 사용),
        LLM_MODEL (기본 gpt-4.1-mini), LLM_BASE_URL (기본 https://api.openai.com/v1, 로컬 서버도 가능)
- 가상 데이터 탐지 실험에 한해 원문 전송 허용 (운영 환경에서는 사용 불가 — 보고서 참고)
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

from detectors.rules import TITLE_RE

SURNAME_TITLE = re.compile(rf"[가-힣]\s?(?:{TITLE_RE})")
MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
KEY_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "API key", "env")

LABEL_DESC = {
    "person": "사람 이름 (전체 이름, 이름만, 띄어 쓴 이름 '윤 선영' 포함)",
    "phone_number": "개인·직원 전화번호 (하이픈 없는 형식 포함)",
    "email": "이메일 주소",
    "address": "주소 (시·도부터 동·호수까지)",
    "resident_registration_number": "주민등록번호",
    "account_number": "계좌번호",
    "card_number": "카드번호",
}

# 팀 기준: 기본은 성+직함 제외. PII_SURNAME_TITLE=1이면 포함 (run_detect.py --surname-title)
SURNAME_TITLE_ON = os.environ.get("PII_SURNAME_TITLE") == "1"
SURNAME_TITLE_RULE = (
    "- 성 한 글자 + 직함 표현도 person이다. 조사·님은 빼고 성과 직함까지 적는다:\n"
    "  '허 부장님이' → '허 부장', '김 과장이' → '김 과장', '정대리가' → '정대리'." if SURNAME_TITLE_ON else
    "- 잡지 않는 것: 성 한 글자 + 직함뿐인 표현('허 부장', '김 과장'처럼 이름 부분이 없는 것).")

SYSTEM = f"""너는 금융사 사내 문서에서 개인정보를 찾는 탐지기다.
라벨(이 7종만 사용):
{chr(10).join(f'- {k}: {v}' for k, v in LABEL_DESC.items())}

규칙
- 원문에 나온 문자열을 한 글자도 바꾸지 말고 그대로 적는다 (공백·하이픈 유지).
- 고객뿐 아니라 직원·상담원·담당자·메신저 화자·회의 참석자 이름도 모두 person이다.
- 이름 뒤 조사·호칭·직함은 빼고 이름만 적는다:
  '홍길동은' → '홍길동', '홍길동 상담원' → '홍길동', '홍길동 부장에게' → '홍길동', '길동 씨' → '길동'.
- 띄어 쓴 이름도 이름이다: '정 다온 팀장이' → '정 다온', '정 다온 고객' → '정 다온'.
- 성 없이 이름만 부른 것도 이름이다: '가람아' → '가람', '나래이랑' → '나래', '보람 씨가' → '보람'.
{SURNAME_TITLE_RULE}
- 잡지 않는 것: 회사 대표번호(1599-0000 등)·팩스·내선번호, 날짜, 민원·접수번호, 금액, 회사·부서·상품명.
- 같은 값이 여러 번 나오면 한 번만 적어도 된다. 표기가 다르면(예: '윤 선영', '윤선영') 각각 적는다.
- 개인정보가 없으면 빈 목록.

출력: JSON 객체 하나 {{"entities": [{{"label": "...", "text": "..."}}, ...]}}"""

_key = None


def load():
    global _key
    if _key:
        return
    _key = os.environ.get("OPENAI_API_KEY")
    if not _key and os.path.exists(KEY_FILE):
        with open(KEY_FILE, encoding="utf-8") as f:
            for line in f:
                k, sep, v = line.partition("=")
                if sep and "key" in k.lower():
                    _key = v.strip().strip('"').strip("'")
                    break
    if not _key:
        raise RuntimeError("API 키 없음: OPENAI_API_KEY 환경변수를 설정하세요")


def call(text, retries=5):
    body = json.dumps({
        "model": MODEL, "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
    }).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL}/chat/completions", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {_key}", "Content-Type": "application/json"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"]
        except (urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if i == retries - 1 or (code is not None and code < 500 and code != 429):
                raise
            time.sleep(2 ** i)


def parse(content):
    try:
        items = json.loads(content).get("entities", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    return [(e["label"], e["text"]) for e in items
            if isinstance(e, dict) and e.get("label") in LABEL_DESC and isinstance(e.get("text"), str)
            and e["text"].strip()]


def name_variants(name):
    """띄어쓰기 변형: '노 성호' ↔ '노성호' (LLM이 한 표기만 적고 다른 표기를 빠뜨리는 경우 보완)."""
    joined = name.replace(" ", "")
    out = {name, joined}
    if len(joined) == 3:
        out.add(joined[0] + " " + joined[1:])
    return out


def bad_person(s):
    """한 글자('정 부장'의 '정')나 성+직함('김 대리')은 버림: 팀 기준상 개인정보가 아니고,
    한 글자는 원문 전체에서 찾으면 '정지'·'일정' 같은 단어 안까지 걸림."""
    return len(s.replace(" ", "")) < 2 or (not SURNAME_TITLE_ON and bool(SURNAME_TITLE.fullmatch(s)))


def locate(text, items):
    """(라벨, 문자열)을 원문의 모든 등장 위치로 바꿈. 긴 문자열 먼저 두어 겹칠 때 우선.
    person은 띄어쓰기 변형까지 문서 안에서 찾음."""
    items = {(label, v) for label, s in items
             if not (label == "person" and bad_person(s.strip()))
             for v in (name_variants(s.strip()) if label == "person" else {s})}
    ents = []
    for label, s in sorted(items, key=lambda x: -len(x[1])):
        s = s.strip()
        i = text.find(s)
        while i != -1:
            ents.append({"label": label, "start": i, "end": i + len(s)})
            i = text.find(s, i + 1)
    return sorted(ents, key=lambda e: (e["start"], e["start"] - e["end"]))


def detect(text):
    load()
    return locate(text, parse(call(text)))
