"""2주차 추가 평가 데이터 생성 (synth2): 문서 ~1,000건 + 검색 질의 ~200개. 과적합 점검용 — 결과를 보고 고치지 않는다.

멘토 데이터(holdout)와 같은 방식
  1. 코드가 가상 인물(고객·직원·외부)과 값(전화·이메일·계좌·카드·주민번호·주소)을 만든다. 동명이인 쌍을 일부러 넣는다
  2. LLM(OpenAI gpt-4.1)은 빈칸 템플릿만 쓴다 — '{고객명} 고객님({고객전화})이 …'. 실제 이름·번호는 쓰지 않는다
  3. 코드가 빈칸을 채운다 → 정답 위치·인물 ID 가 정확. 이름 표기(전체/띄어 씀/이름만)와 번호 서식은 무작위
  4. 질의: 코드가 대상(사람·종류·값·상태·지역)을 고르고 정답 문서를 계산, LLM 은 자연스러운 문장만 쓴다
문서 종류 25종 중 15종은 search.DOC_TYPE_KEYWORDS 에 없는 새 종류 (사전이 holdout 에 맞춰졌는지 확인)

    python tools/gen_synth2.py personas     # 인물 만들기 (LLM 없음)
    python tools/gen_synth2.py templates    # 템플릿 생성 (OpenAI, 끊겨도 이어서)
    python tools/gen_synth2.py fill         # 문서 1,000건 채우기 (LLM 없음)
    python tools/gen_synth2.py queries      # 질의 200개 (OpenAI)
    python tools/gen_synth2.py all

출력 (data/synth2/)
  synth2_labeled.jsonl                {"id","split","doc_type","text","entities":[{label,text,start,end,person_id,form}]}
  synth2_persons_gold.jsonl           {"id","persons":[{start,end,text,person_id,form}]}       ← 채점 전용
  synth2_retrieval_queries.jsonl      {"qid","query"}
  synth2_retrieval_queries_gold.jsonl {"qid","query","type","relevant_doc_ids","confusable_doc_ids","spec"}  ← 채점 전용
  _work/  personas.json, templates_raw/, templates.json, usage.jsonl
OpenAI 키: 프로젝트 맨 위 .env 의 openai-api-key (가상 데이터 생성에만 사용)
"""
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "synth2"
WORK = OUT / "_work"
SEED = 20261030
MODEL = "gpt-4.1"
TEMPLATES_PER_TYPE = 8
FILLS_PER_TEMPLATE = 5
# gen_synth3.py 가 바꿔 쓰는 값
NAME, DOC_PREFIX, Q_PREFIX = "synth2", "S", "SQ"
N_PEOPLE = {"customer": 250, "staff": 80, "external": 20}
HOMONYMS = {"cust_staff": 5, "cust_cust": 5, "staff_staff": 0}
NAME_FORM = (0.15, 0.27)  # 두 번째 언급부터: 띄어 씀 15%, 이름만 12%

# ---------------------------------------------------------------- 문서 종류 (new=사전에 없는 새 종류)
DOC_TYPES = {
    # 새 종류 15
    "보험금청구": ("new", "카드 부가 보험(여행자·상해) 보험금 청구 접수 기록", ["고객"]),
    "카드해지상담": ("new", "고객의 카드 해지 요청 상담 기록과 해지 처리 결과", ["고객"]),
    "포인트이관신청": ("new", "포인트를 다른 카드·가족에게 옮기는 신청서", ["고객"]),
    "결제취소요청": ("new", "가맹점 결제 취소·환불 요청 처리 기록", ["고객"]),
    "명세서재발송": ("new", "이용대금 명세서 재발송 요청 처리 기록", ["고객"]),
    "연회비환불": ("new", "연회비 환불 요청과 처리 결과", ["고객"]),
    "리볼빙상담": ("new", "결제금액 일부만 내고 나머지를 이월하는 리볼빙 서비스 상담", ["고객"]),
    "카드론상담": ("new", "카드론(장기카드대출) 상담 기록", ["고객"]),
    "할부전환신청": ("new", "일시불 결제를 할부로 바꾸는 신청 처리", ["고객"]),
    "가족카드발급": ("new", "본인 회원의 가족카드 발급 신청 심사", ["고객"]),
    "재발급신청": ("new", "분실·훼손 카드 재발급 신청 처리", ["고객"]),
    "개인정보열람청구": ("new", "고객이 자기 개인정보 처리 내역 열람을 청구한 기록", ["고객"]),
    "계정회수보고": ("new", "퇴사·이동 직원의 시스템 계정 회수 결과 보고", ["직원"]),
    "교육이수보고": ("new", "직원 개인정보보호 교육 이수 현황 보고", ["직원"]),
    "장애보고": ("new", "카드 승인 시스템 장애 발생·조치 보고", ["직원", "외부"]),
    # 기존 종류 10 (holdout 과 같은 이름)
    "VOC요약": ("old", "고객 불만·문의(VOC) 주간 요약", ["고객"]),
    "녹취요약": ("old", "상담 통화 녹취 AI 요약", ["고객"]),
    "분쟁조정": ("old", "고객과 가맹점 사이 분쟁 조정 검토 의견", ["고객"]),
    "한도심사결과": ("old", "카드 이용한도 상향 심사 결과 안내", ["고객"]),
    "피해접수": ("old", "보이스피싱·사기 피해 접수 기록", ["고객"]),
    "법인카드소명": ("old", "직원 법인카드 사용 소명서", ["직원"]),
    "출장보고": ("old", "협력사 점검 출장 결과 보고", ["직원", "외부"]),
    "감사지적": ("old", "내부 감사 지적 사항과 조치", ["직원", "고객"]),
    "권한신청": ("old", "시스템 접근 권한 신청과 승인", ["직원"]),
    "우편반송처리": ("old", "카드·명세서 우편물 반송 처리 기록", ["고객"]),
}
STATUSES = ["접수", "처리 중", "처리 완료", "보류"]
TEAMS = ["고객센터", "카드운영팀", "여신심사팀", "리스크관리팀", "정보보호팀", "고객보호팀", "IT운영팀", "인사팀", "채권관리팀"]
TITLES = ["사원", "주임", "대리", "과장", "차장", "팀장"]
COMPANIES = ["한결솔루션", "누리시큐", "다온콜센터", "가람데이터", "새온보안"]

SURNAMES = list("김이박최정강조윤장임한오서신권황안송류전홍고문양손배백허유남심노하곽성차주우구민진나지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국") + ["남궁", "황보", "제갈", "선우"]
GIVEN = ["민준", "서연", "지훈", "하은", "도윤", "서윤", "예준", "지우", "시우", "수아", "주원", "지아", "하준", "채원", "은우",
         "윤서", "건우", "다은", "현우", "소율", "우진", "예린", "선우", "가은", "지호", "유나", "준서", "하린", "민재", "서아",
         "태윤", "나연", "승현", "다인", "재윤", "아린", "정우", "수빈", "도현", "연우", "지안", "예서", "시현", "채은", "민서",
         "로운", "하율", "이안", "온유", "은서", "태오", "하람", "규리", "세아", "지율", "라온", "해솔", "보람", "새봄", "한결"]
SIDO = [("서울특별시", "서울", ["강남구", "송파구", "마포구", "종로구", "관악구", "노원구"]),
        ("부산광역시", "부산", ["해운대구", "수영구", "동래구"]), ("대구광역시", "대구", ["수성구", "달서구"]),
        ("인천광역시", "인천", ["연수구", "남동구"]), ("광주광역시", "광주", ["서구", "북구"]),
        ("대전광역시", "대전", ["유성구", "서구"]), ("경기도", "경기", ["성남시 분당구", "수원시 영통구", "고양시 일산동구", "용인시 수지구"]),
        ("충청남도", "충남", ["천안시 서북구", "아산시"]), ("경상남도", "경남", ["창원시 성산구", "김해시"]),
        ("제주특별자치도", "제주", ["제주시"])]
ROADS = ["새솔로", "한빛로", "가온길", "누리로", "마루길", "도담로", "은하수길", "푸른로", "다솜길", "하늬로"]
BANKS = ["하늘빛은행", "온누리은행", "가람저축은행", "새봄은행", "누리은행"]


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 1. 인물
def make_personas():
    rnd = random.Random(SEED)
    used = set()

    def name():
        while True:
            n = rnd.choice(SURNAMES) + rnd.choice(GIVEN)
            if n not in used:
                used.add(n)
                return n

    def digits(n):
        return "".join(rnd.choice("0123456789") for _ in range(n))

    def address():
        long, short, sgg = rnd.choice(SIDO)
        return {"sido_long": long, "sido_short": short, "sigungu": rnd.choice(sgg), "road": rnd.choice(ROADS),
                "num": str(rnd.randint(1, 250)), "detail": rnd.choice(["", f"{rnd.randint(2, 15)}층",
                                                                        f"{rnd.randint(101, 120)}동 {rnd.randint(101, 1505)}호"]),
                "postal": digits(5)}

    people = []
    for i in range(N_PEOPLE["customer"]):
        n = name()
        people.append({"pid": f"C{i + 1:03d}", "role": "customer", "name": n,
                       "phone": "010" + digits(8), "email": f"{rnd.choice(['kim', 'lee', 'park', 'choi', 'user', 'mail'])}{digits(3)}@{rnd.choice(['example.com', 'example.net', 'mail.example.org'])}",
                       "account": {"bank": rnd.choice(BANKS), "num": digits(12)}, "card": "94" + digits(14),
                       "rrn": f"{rnd.randint(60, 99)}{rnd.randint(1, 12):02d}{rnd.randint(1, 28):02d}{rnd.choice('12')}{digits(6)}",
                       "address": address()})
    for i in range(N_PEOPLE["staff"]):
        n = name()
        people.append({"pid": f"S{i + 1:03d}", "role": "staff", "name": n, "team": rnd.choice(TEAMS),
                       "title": rnd.choice(TITLES), "phone": "010" + digits(8),
                       "email": f"{rnd.choice(['s', 'staff', 'emp'])}{digits(4)}@saebomcard.example.com"})
    for i in range(N_PEOPLE["external"]):
        n = name()
        people.append({"pid": f"X{i + 1:03d}", "role": "external", "name": n, "company": rnd.choice(COMPANIES),
                       "title": rnd.choice(["매니저", "책임", "선임"]), "phone": "010" + digits(8)})
    # 동명이인 10쌍: 고객↔직원 5, 고객↔고객 5 (뒤 사람의 이름을 앞 사람 이름으로 바꿈)
    cust = [p for p in people if p["role"] == "customer"]
    staff = [p for p in people if p["role"] == "staff"]
    homonyms = []
    h1, h2, h3 = HOMONYMS["cust_staff"], HOMONYMS["cust_cust"], HOMONYMS["staff_staff"]
    pairs = (list(zip(cust[:h1], staff[:h1])) + list(zip(cust[h1:h1 + h2], cust[h1 + h2:h1 + 2 * h2]))
             + list(zip(staff[h1:h1 + h3], staff[h1 + h3:h1 + 2 * h3])))
    for a, b in pairs:
        b["name"] = a["name"]
        homonyms.append([a["pid"], b["pid"]])
    WORK.mkdir(parents=True, exist_ok=True)
    json.dump({"people": people, "homonyms": homonyms}, open(WORK / "personas.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"인물 {len(people)}명 {N_PEOPLE}, 동명이인 {len(homonyms)}쌍 → {WORK / 'personas.json'}")


# ---------------------------------------------------------------- 2. 템플릿 (LLM)
SLOTS_DOC = """사용할 수 있는 빈칸 (중괄호 그대로):
- 고객: {고객명} {고객전화} {고객이메일} {고객계좌} {고객카드} {고객주민번호} {고객주소}
- 담당 직원: {직원명} {직원팀} {직원직급} {직원전화} {직원이메일}
- 두 번째 직원(필요할 때만): {직원2명} {직원2팀} {직원2직급}
- 외부 업체 사람(필요할 때만): {외부명} {외부회사} {외부직급} {외부전화}
- 기타: {날짜} {시각} {금액} {상태} {접수번호}"""

TEMPLATE_PROMPT = """가상의 카드사 '새봄카드' 사내 문서의 빈칸 템플릿을 {n}개 만들어 주세요.
문서 종류: {dtype} — {desc}
문서에 나오는 사람: {who}

{slots}

규칙
1. 사람 이름·전화번호·이메일·계좌·카드번호·주민번호·주소는 반드시 위 빈칸으로만 쓰고, 실제처럼 보이는 값을 직접 쓰지 마세요.
2. 이름 빈칸 뒤에는 '고객님', '님', '고객', '씨', '{직원직급}', '대리' 같은 호칭을 붙이고, 조사(은/는/이/가)를 이름 바로 뒤에 붙이지 마세요. 예: '{고객명} 고객님은', '{직원명} {직원직급}이'
3. 같은 사람을 두 번 이상 부르는 문장을 1~2곳 넣으세요 (예: 앞에서 '{고객명} 고객님', 뒤에서 다시 '{고객명}님').
4. {상태} 는 '접수/처리 중/처리 완료/보류' 중 하나가 들어갑니다. 처리 상태 문장에 한 번 쓰세요.
5. 템플릿마다 구성(항목: 값 서식, 이메일, 메신저 대화, 표 같은 목록, 서술형 보고)과 내용(사유·경과·조치)을 서로 다르게 하세요. 150~450자.
6. 고객이 나오는 종류면 {고객명} 을 꼭 넣고, 고객 개인정보 빈칸은 템플릿마다 2~4개를 골라 쓰세요.

JSON 으로만 답하세요: {{"templates": ["템플릿1", "템플릿2", ...]}}"""


def _fill_prompt(tpl, **kw):
    """프롬프트 안에 '{고객명}' 같은 빈칸 예시가 많아 str.format 대신 지정한 자리만 바꾼다."""
    for k, v in kw.items():
        tpl = tpl.replace("{" + k + "}", str(v))
    return tpl.replace("{{", "{").replace("}}", "}")


def _client():
    from openai import OpenAI
    s = (ROOT.parent / ".env").read_text(encoding="utf-8")
    key = re.search(r"^openai-api-key=(.*)$", s, re.M).group(1).strip().strip("\"'")
    return OpenAI(api_key=key)


def _chat(client, prompt, temperature):
    for attempt in range(4):
        try:
            r = client.chat.completions.create(model=MODEL, temperature=temperature, max_tokens=12000,
                                               response_format={"type": "json_object"},
                                               messages=[{"role": "user", "content": prompt}])
            with open(WORK / "usage.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), "model": MODEL, "in": r.usage.prompt_tokens,
                                    "out": r.usage.completion_tokens}) + "\n")
            return json.loads(r.choices[0].message.content)
        except Exception as e:  # 일시 오류·JSON 깨짐
            print("  재시도:", type(e).__name__, str(e)[:120])
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("LLM 호출 실패")


_ALLOWED = set(re.findall(r"\{([^{}]+)\}", SLOTS_DOC))
_LEAK = re.compile(r"01[016789][-. ]?\d{3,4}[-. ]?\d{4}|\d{6}-?\d{7}|\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}|[\w.]+@[\w.]+")


def _template_ok(t, who):
    slots = re.findall(r"\{([^{}]+)\}", t)
    if not slots or any(s not in _ALLOWED for s in slots):
        return False
    if _LEAK.search(re.sub(r"\{[^{}]+\}", "", t)):
        return False  # 빈칸 밖에 실제처럼 보이는 번호·이메일
    if "고객" in who and "고객명" not in slots:
        return False
    if who == ["직원"] and "직원명" not in slots:
        return False
    return 100 <= len(t) <= 700


def make_templates():
    client = _client()
    raw = WORK / "templates_raw"
    raw.mkdir(parents=True, exist_ok=True)
    for dtype, (_, desc, who) in DOC_TYPES.items():
        p = raw / f"{dtype}.json"
        if p.exists():
            continue
        prompt = _fill_prompt(TEMPLATE_PROMPT, n=TEMPLATES_PER_TYPE + 2, dtype=dtype, desc=desc, who=", ".join(who),
                              slots=SLOTS_DOC)
        res = _chat(client, prompt, 0.9)
        json.dump(res, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  {dtype}: {len(res.get('templates', []))}개")
    out, dropped = {}, 0
    for dtype, (_, _, who) in DOC_TYPES.items():
        ts = json.load(open(raw / f"{dtype}.json", encoding="utf-8")).get("templates", [])
        ok = [t for t in ts if _template_ok(t, who)]
        dropped += len(ts) - len(ok)
        out[dtype] = ok[:TEMPLATES_PER_TYPE]
    json.dump(out, open(WORK / "templates.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"템플릿 {sum(len(v) for v in out.values())}개 (검사에서 뺀 것 {dropped}) → {WORK / 'templates.json'}")


# ---------------------------------------------------------------- 3. 채우기
def _fmt_phone(d, rnd):
    return rnd.choice([f"{d[:3]}-{d[3:7]}-{d[7:]}", d, f"{d[:3]}.{d[3:7]}.{d[7:]}", f"{d[:3]} {d[3:7]} {d[7:]}",
                       f"{d[:3]}-{d[3:7]}-{d[7:]}"])


def _fmt_card(d, rnd):
    g = [d[i:i + 4] for i in range(0, 16, 4)]
    return rnd.choice(["-".join(g), " ".join(g), d, "-".join(g)])


def _fmt_account(a, rnd):
    n = a["num"]
    return rnd.choice([f"{n[:3]}-{n[3:9]}-{n[9:]}", f"{n[:4]}-{n[4:8]}-{n[8:]}", n])


def _fmt_rrn(r, rnd):
    return rnd.choice([f"{r[:6]}-{r[6:]}", r, f"{r[:6]}-{r[6:]}"])


def _fmt_address(a, rnd):
    sido = rnd.choice([a["sido_long"], a["sido_short"], a["sido_long"]])
    base = f"{sido} {a['sigungu']} {a['road']} {a['num']}"
    if a["detail"]:
        base += rnd.choice([", ", " "]) + a["detail"]
    return (f"({a['postal']}) " if rnd.random() < 0.3 else "") + base


def _name_form(name, rnd, first):
    """첫 언급은 전체 이름, 이후엔 가끔 띄어 쓴 이름·이름만."""
    sur = name[:2] if len(name) >= 4 and name[:2] in ("남궁", "황보", "제갈", "선우") else name[:1]
    given = name[len(sur):]
    if first:
        return name, "full"
    x = rnd.random()
    if x < NAME_FORM[0]:
        return f"{sur} {given}", "spaced"
    if x < NAME_FORM[1]:
        return given, "given"
    return name, "full"


def fill():
    rnd = random.Random(SEED + 1)
    P = json.load(open(WORK / "personas.json", encoding="utf-8"))
    people = P["people"]
    templates = json.load(open(WORK / "templates.json", encoding="utf-8"))
    by_role = {r: [p for p in people if p["role"] == r] for r in ("customer", "staff", "external")}
    # 고객은 여러 문서에 나오도록 순서대로 돌려 씀 (동명이인은 앞쪽에 있어 자주 나옴)
    cust_cycle = by_role["customer"][:]
    docs, gold_persons = [], []
    n = 0
    for dtype, ts in templates.items():
        for t in ts:
            for _ in range(FILLS_PER_TEMPLATE):
                n += 1
                if not cust_cycle:
                    cust_cycle = by_role["customer"][:]
                    rnd.shuffle(cust_cycle)
                c = cust_cycle.pop(rnd.randrange(len(cust_cycle))) if rnd.random() < 0.7 else rnd.choice(by_role["customer"])
                s1, s2 = rnd.sample(by_role["staff"], 2)
                x = rnd.choice(by_role["external"])
                status = rnd.choice(STATUSES)
                text, ents, seen = "", [], set()
                pos = 0
                for m in re.finditer(r"\{([^{}]+)\}", t):
                    text += t[pos:m.start()]
                    slot = m.group(1)
                    val, label, pid, form = _slot_value(slot, c, s1, s2, x, status, rnd, seen)
                    if label:
                        ents.append({"label": label, "text": val, "start": len(text), "end": len(text) + len(val),
                                     "person_id": pid, "form": form})
                    text += val
                    pos = m.end()
                text += t[pos:]
                doc_id = f"{DOC_PREFIX}{n:04d}"
                meta = {"customer": c["pid"] if "{고객" in t else None,
                        "staff": [s1["pid"]] + ([s2["pid"]] if "{직원2" in t else []) if "{직원명}" in t else [],
                        "external": x["pid"] if "{외부명}" in t else None,
                        "status": status if "{상태}" in t else None}
                docs.append({"id": doc_id, "split": NAME, "doc_type": dtype, "text": text, "entities": ents,
                             "meta": meta})
                gold_persons.append({"id": doc_id, "persons": [
                    {k: e[k] for k in ("start", "end", "text", "person_id", "form")}
                    for e in ents if e["label"] == "person"]})
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT / f"{NAME}_labeled.jsonl", docs)
    write_jsonl(OUT / f"{NAME}_persons_gold.jsonl", gold_persons)
    print(f"문서 {len(docs)}건 → {OUT / f'{NAME}_labeled.jsonl'}")


def _slot_value(slot, c, s1, s2, x, status, rnd, seen):
    """빈칸 → (값, 라벨, 인물 ID, 이름 표기)."""
    def person(p):
        first = p["pid"] not in seen
        seen.add(p["pid"])
        v, form = _name_form(p["name"], rnd, first)
        return v, "person", p["pid"], form
    table = {
        "고객명": lambda: person(c), "직원명": lambda: person(s1), "직원2명": lambda: person(s2),
        "외부명": lambda: person(x),
        "고객전화": lambda: (_fmt_phone(c["phone"], rnd), "phone_number", c["pid"], None),
        "고객이메일": lambda: (c["email"], "email", c["pid"], None),
        "고객계좌": lambda: (_fmt_account(c["account"], rnd), "account_number", c["pid"], None),
        "고객카드": lambda: (_fmt_card(c["card"], rnd), "card_number", c["pid"], None),
        "고객주민번호": lambda: (_fmt_rrn(c["rrn"], rnd), "resident_registration_number", c["pid"], None),
        "고객주소": lambda: (_fmt_address(c["address"], rnd), "address", c["pid"], None),
        "직원전화": lambda: (_fmt_phone(s1["phone"], rnd), "phone_number", s1["pid"], None),
        "직원이메일": lambda: (s1["email"], "email", s1["pid"], None),
        "외부전화": lambda: (_fmt_phone(x["phone"], rnd), "phone_number", x["pid"], None),
        "직원팀": lambda: (s1["team"], None, None, None), "직원직급": lambda: (s1["title"], None, None, None),
        "직원2팀": lambda: (s2["team"], None, None, None), "직원2직급": lambda: (s2["title"], None, None, None),
        "외부회사": lambda: (x["company"], None, None, None), "외부직급": lambda: (x["title"], None, None, None),
        "날짜": lambda: (f"2026-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}", None, None, None),
        "시각": lambda: (f"{rnd.randint(9, 18):02d}:{rnd.randint(0, 59):02d}", None, None, None),
        "금액": lambda: (f"{rnd.randint(1, 300) * 10000:,}원", None, None, None),
        "상태": lambda: (status, None, None, None),
        "접수번호": lambda: (f"REQ-{rnd.randint(10000, 99999)}", None, None, None),
    }
    return table[slot]()


# ---------------------------------------------------------------- 4. 질의
QUERY_PROMPT = """새봄카드 직원이 사내 문서 검색 챗봇에 입력할 질문을 만들어 주세요. 아래 각 항목(spec)마다 질문 1개씩.
규칙
- spec 의 must_include 에 있는 문자열은 질문에 글자 그대로 넣으세요 (이름·번호를 바꾸지 마세요).
- style: 'keyword'=짧은 키워드 나열, 'casual'=반말 구어체, 'polite'=존댓말 문장, 'context'=상황 설명을 덧붙인 긴 문장
- paraphrase=true 이면 문서 종류 이름과 그 핵심 단어를 쓰지 말고 다른 말로 풀어서 표현하세요 (예: '리볼빙' → '결제금액 일부만 내고 나머지는 다음 달로 넘기는 거').
- 질문에 spec 에 없는 사람 이름·번호를 새로 만들어 넣지 마세요.
JSON 으로만: {{"queries": [{{"i": 번호, "query": "질문"}}, ...]}}

spec 목록:
{specs}"""

Q_PLAN = [("name_topic", 30), ("person_all", 15), ("value_exact", 20), ("value_format_variant", 20), ("role_agent", 20),
          ("homonym_role", 15), ("hybrid_status", 20), ("address_partial", 15), ("semantic_no_pii", 20),
          ("hybrid_paraphrase", 25)]


def make_queries():
    rnd = random.Random(SEED + 2)
    P = json.load(open(WORK / "personas.json", encoding="utf-8"))
    pp = {p["pid"]: p for p in P["people"]}
    docs = load_jsonl(OUT / f"{NAME}_labeled.jsonl")
    by_person, by_type = {}, {}
    for d in docs:
        m = d["meta"]
        for pid in [m["customer"], m["external"]] + m["staff"]:
            if pid:
                by_person.setdefault(pid, []).append(d)
        by_type.setdefault(d["doc_type"], []).append(d)
    homonym_of = {}
    for a, b in P["homonyms"]:
        homonym_of[a], homonym_of[b] = b, a
    cust_docs = [d for d in docs if d["meta"]["customer"]]
    staff_docs = [d for d in docs if d["meta"]["staff"]]
    styles = ["keyword", "casual", "polite", "context"]
    specs = []

    def ids(ds):
        return sorted({d["id"] for d in ds})

    def conf_for(pid, rel):
        h = homonym_of.get(pid)
        return sorted({d["id"] for d in by_person.get(h, [])} - set(rel)) if h else []

    def add(qtype, spec, rel, conf=()):
        if rel:
            specs.append({"type": qtype, "spec": spec, "relevant": rel, "confusable": list(conf)})

    used = set()
    for qtype, n in Q_PLAN:
        tries = 0
        while sum(s["type"] == qtype for s in specs) < n and tries < 2000:
            tries += 1
            st = rnd.choice(styles)
            if qtype in ("name_topic", "hybrid_paraphrase", "hybrid_status"):
                d = rnd.choice(cust_docs)
                c = pp[d["meta"]["customer"]]
                if qtype == "hybrid_status" and not d["meta"]["status"]:
                    continue
                key = (qtype, c["pid"], d["doc_type"])
                if key in used:
                    continue
                used.add(key)
                rel = [x for x in by_person[c["pid"]] if x["doc_type"] == d["doc_type"]
                       and (qtype != "hybrid_status" or x["meta"]["status"] == d["meta"]["status"])]
                spec = {"must_include": [c["name"]], "who": f"고객 {c['name']}", "doc_type": d["doc_type"],
                        "doc_desc": DOC_TYPES[d["doc_type"]][1], "style": st, "paraphrase": qtype == "hybrid_paraphrase"}
                if qtype == "hybrid_status":
                    spec["status"] = d["meta"]["status"]
                add(qtype, spec, ids(rel), conf_for(c["pid"], ids(rel)))
            elif qtype == "person_all":
                c = pp[rnd.choice(cust_docs)["meta"]["customer"]]
                if ("pa", c["pid"]) in used or len(by_person[c["pid"]]) < 2:
                    continue
                used.add(("pa", c["pid"]))
                add(qtype, {"must_include": [c["name"]], "who": f"고객 {c['name']}", "ask": "이 고객이 나온 문서 전부",
                            "style": st}, ids(by_person[c["pid"]]), conf_for(c["pid"], ids(by_person[c["pid"]])))
            elif qtype in ("value_exact", "value_format_variant"):
                d = rnd.choice(cust_docs)
                vals = [e for e in d["entities"] if e["label"] in ("phone_number", "account_number", "card_number",
                                                                    "resident_registration_number", "email")
                        and e["person_id"] == d["meta"]["customer"]]
                if not vals:
                    continue
                e = rnd.choice(vals)
                c = pp[e["person_id"]]
                if qtype == "value_format_variant":
                    if e["label"] == "email":
                        continue
                    raw = {"phone_number": c["phone"], "account_number": c["account"]["num"], "card_number": c["card"],
                           "resident_registration_number": c["rrn"]}[e["label"]]
                    alt = [f for f in _formats(e["label"], raw) if f != e["text"]]
                    value = rnd.choice(alt)
                else:
                    value = e["text"]
                if (qtype, value) in used:
                    continue
                used.add((qtype, value))
                # 같은 사람의 같은 종류 값이 나온 문서 전부 (서식 무관)
                rel = [x for x in by_person[c["pid"]] if any(y["label"] == e["label"] and y["person_id"] == c["pid"]
                                                          for y in x["entities"])]
                add(qtype, {"must_include": [value], "value_label": e["label"], "ask": "이 값이 나온 문서",
                            "style": st}, ids(rel))
            elif qtype == "role_agent":
                d = rnd.choice(staff_docs)
                s = pp[d["meta"]["staff"][0]]
                key = (qtype, s["pid"], d["doc_type"])
                if key in used:
                    continue
                used.add(key)
                rel = [x for x in by_person[s["pid"]] if x["doc_type"] == d["doc_type"]]
                add(qtype, {"must_include": [s["name"]], "who": f"{s['team']} {s['name']} {s['title']} (직원)",
                            "doc_type": d["doc_type"], "doc_desc": DOC_TYPES[d["doc_type"]][1],
                            "ask": "이 직원이 담당·처리한 이 종류 문서", "style": st}, ids(rel),
                    conf_for(s["pid"], ids(rel)))
            elif qtype == "homonym_role":
                a, b = rnd.choice(P["homonyms"])
                pid = rnd.choice([a, b])
                p = pp[pid]
                if ("hr", pid) in used or pid not in by_person:
                    continue
                used.add(("hr", pid))
                role_ko = {"customer": "고객", "staff": "직원", "external": "외부"}[p["role"]]
                hint = role_ko if p["role"] == "customer" else f"{p['team']} 직원"
                rel = ids(by_person[pid])
                add(qtype, {"must_include": [p["name"]], "who": f"{hint} {p['name']}",
                            "ask": f"{role_ko} 쪽 {p['name']} 의 문서만 (같은 이름의 다른 사람 말고)", "style": st},
                    rel, conf_for(pid, rel))
            elif qtype == "address_partial":
                d = rnd.choice(cust_docs)
                c = pp[d["meta"]["customer"]]
                a = c["address"]
                region = f"{a['sido_short']} {a['sigungu']}"
                if (qtype, region) in used:
                    continue
                used.add((qtype, region))
                rel = [x for x in cust_docs if any(e["label"] == "address" and e["person_id"] == x["meta"]["customer"]
                                                   and pp[x["meta"]["customer"]]["address"]["sigungu"] == a["sigungu"]
                                                   and pp[x["meta"]["customer"]]["address"]["sido_short"] == a["sido_short"]
                                                   for e in x["entities"])]
                add(qtype, {"must_include": [region], "ask": "이 지역에 사는 고객 문서", "style": st}, ids(rel))
            elif qtype == "semantic_no_pii":
                dtype = rnd.choice(list(DOC_TYPES))
                status = rnd.choice(STATUSES)
                if (qtype, dtype, status) in used:
                    continue
                used.add((qtype, dtype, status))
                rel = [x for x in by_type[dtype] if x["meta"]["status"] == status]
                if not rel or len(rel) > 25:
                    continue
                add(qtype, {"must_include": [], "doc_type": dtype, "doc_desc": DOC_TYPES[dtype][1], "status": status,
                            "ask": "사람 이름·번호 없이 이 종류·상태 문서", "style": st}, ids(rel))
    # LLM 으로 문장 만들기 (20개씩)
    client = _client()
    out_q, out_g = [], []
    for start in range(0, len(specs), 20):
        batch = specs[start:start + 20]
        listing = "\n".join(json.dumps({"i": i, **s["spec"]}, ensure_ascii=False) for i, s in enumerate(batch))
        for attempt in range(3):
            res = _chat(client, _fill_prompt(QUERY_PROMPT, specs=listing), 0.8)
            got = {q["i"]: q["query"] for q in res.get("queries", []) if isinstance(q, dict) and "i" in q}
            bad = [i for i, s in enumerate(batch) if i not in got or not all(m in got[i] for m in s["spec"]["must_include"])]
            if not bad:
                break
            print(f"  must_include 누락 {len(bad)}개 → 다시 생성")
        for i, s in enumerate(batch):
            q = got.get(i)
            if not q or not all(m in q for m in s["spec"]["must_include"]):
                continue  # 끝내 못 맞춘 질의는 버림
            qid = f"{Q_PREFIX}{len(out_q) + 1:03d}"
            out_q.append({"qid": qid, "query": q})
            out_g.append({"qid": qid, "query": q, "type": s["type"], "relevant_doc_ids": s["relevant"],
                          "confusable_doc_ids": s["confusable"], "spec": s["spec"]})
        print(f"  질의 {len(out_q)}개")
    write_jsonl(OUT / f"{NAME}_retrieval_queries.jsonl", out_q)
    write_jsonl(OUT / f"{NAME}_retrieval_queries_gold.jsonl", out_g)
    print(f"질의 {len(out_q)}개 → {OUT / f'{NAME}_retrieval_queries.jsonl'}")


def _formats(label, d):
    if label == "phone_number":
        return [f"{d[:3]}-{d[3:7]}-{d[7:]}", d, f"{d[:3]}.{d[3:7]}.{d[7:]}", f"{d[:3]} {d[3:7]} {d[7:]}"]
    if label == "card_number":
        g = [d[i:i + 4] for i in range(0, 16, 4)]
        return ["-".join(g), " ".join(g), d]
    if label == "account_number":
        return [f"{d[:3]}-{d[3:9]}-{d[9:]}", f"{d[:4]}-{d[4:8]}-{d[8:]}", d]
    return [f"{d[:6]}-{d[6:]}", d]


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    for name, fn in [("personas", make_personas), ("templates", make_templates), ("fill", fill), ("queries", make_queries)]:
        if step in (name, "all"):
            print(f"== {name}")
            fn()


if __name__ == "__main__":
    main()
