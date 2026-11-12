"""탐지 모델 추가 학습용 합성 데이터 (worklog 31장). 개선·확인 세트(synth_hard_*, synth_dev, synth2·3, holdout)는 안 씀.

    python scripts/ft_make_data.py frames      # ① gpt-6-luna 로 문장 틀(빈칸만, 개인정보 없음) 만들기 → outputs/ft/frames.jsonl
    python scripts/ft_make_data.py fill        # ② 틀에 가짜 값 채워 학습 데이터 → outputs/ft/train.jsonl, valid.jsonl

틀의 빈칸: ⟦이름⟧ ⟦전화⟧ ⟦이메일⟧ ⟦주소⟧ ⟦주민⟧ ⟦계좌⟧ ⟦카드⟧
이름 뒤 조사는 받침에 따라 바뀌므로 {가}{는}{를}{과}{랑}{이}{으로} 로 받아서 채울 때 고름.
이름 형태 (30장 미탐 유형): 성+이름, 띄어 쓴 이름, 이름만(순우리말 낱말 이름 포함), 영문, 복성, 외자, 앞말에 붙여 쓰기.
반대 예문: '하늘·다운·보람'을 낱말로 쓴 문장, 성+직함('박 과장', 정책상 안 가림) → 전부 O
"""
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.agents import llm  # noqa: E402
from app.core import config  # noqa: E402

OUT = config.OUT_DIR / "ft"
SLOT = re.compile(r"⟦(이름|전화|이메일|주소|주민|계좌|카드)⟧")
LABEL = {"이름": "person", "전화": "phone_number", "이메일": "email", "주소": "address", "주민": "resident_registration_number",
         "계좌": "account_number", "카드": "card_number"}

FRAME_SYS = """한국 카드·은행 회사의 사내 업무 글을 만드는 도우미입니다. 개인정보 자리는 빈칸으로만 쓰고 실제 값은 절대 쓰지 마세요.
빈칸: ⟦이름⟧ ⟦전화⟧ ⟦이메일⟧ ⟦주소⟧ ⟦주민⟧ ⟦계좌⟧ ⟦카드⟧
⟦이름⟧ 바로 뒤 조사는 받침에 따라 바뀌므로 다음 표시를 쓰세요: {가}(이/가) {는}(은/는) {를}(을/를) {과}(와/과) {랑}(이랑/랑) {으로}(으로/로)
구어체로 이름 뒤에 '이'를 붙이는 경우(하늘이한테, 민준이가)는 ⟦이름⟧{이}한테 처럼 쓰세요. 님·씨·께서·에게·한테·의·도·만·고객·대리님 등은 그대로 쓰세요.
JSON 으로만 답하세요: {"frames": ["...", ...]}"""

FRAME_USER = """다음 조건으로 서로 다른 업무 글 {n}개를 만드세요.
- 종류: {kind}
- 각 글은 1~4문장, 줄바꿈 가능. ⟦이름⟧을 1~4번, 다른 빈칸을 0~3번 쓰세요
- 이름이 나오는 모양을 다양하게: 문장 맨 앞, 목록('⟦이름⟧, ⟦이름⟧, ⟦이름⟧'), 따옴표 안, 괄호 안, 서명('- ⟦이름⟧ 드림'), 호칭 뒤('회원 ⟦이름⟧ 님', '담당 ⟦이름⟧ 대리님'), 부서 뒤('영업팀 ⟦이름⟧'), 자기소개('저 ⟦이름⟧인데요')
- 조사도 다양하게: {{랑}}, {{이}}한테, 께서, 에게, {{과}}, {{는}}, {{가}}, 하고, 의
- 같은 문장 틀을 반복하지 마세요. 날짜·금액·상품명·접수번호 같은 개인정보가 아닌 값은 자유롭게 써도 됩니다"""

SLOT_USER = """다음 조건으로 서로 다른 업무 글 {n}개를 만드세요.
- 종류: {kind}
- 각 글은 1~4문장. ⟦이름⟧ 0~2번, 그리고 ⟦전화⟧ ⟦이메일⟧ ⟦주소⟧ ⟦주민⟧ ⟦계좌⟧ ⟦카드⟧ 중 2~4개를 꼭 쓰세요 (특히 {focus})
- 값 앞에 항목 이름이 있는 경우('연락처: ⟦전화⟧')와 없는 경우('⟦계좌⟧로 입금') 모두, 괄호 안, 문장 중간 등 다양하게"""

NEG_USER = """개인정보(사람 이름·연락처 등)가 전혀 없는 사내 업무 글 {n}개를 만드세요. 빈칸(⟦ ⟧)도 쓰지 마세요.
- 모든 글에 다음 낱말 중 하나 이상이 이름이 아닌 원래 뜻으로 꼭 들어가게: {words}
  (예: '하늘'=날씨·하늘색, '다운'=다운로드·시스템 다운, '보람'=일의 보람, '새롬'·'나래'·'가람'=상품·지점·행사 이름, '이상'=이상 거래·10만 원 이상)
- 사람을 가리킬 때는 '박 과장님', '최 대리', '김 팀장'처럼 성+직함만 쓰세요 (이름 없이)
- 1~3문장. JSON 으로만: {{"frames": ["...", ...]}}"""

KINDS = ["상담원 통화 메모", "사내 메신저 대화", "고객 민원 처리 기록", "카드 분실·재발급 접수 기록", "대출 상담 일지",
         "부서 간 업무 요청 메일", "고객 안내 문자·안내문", "보이스피싱 피해 접수 메모", "교육·행사 공지", "회의록 요약",
         "인사·경조사 공지", "지점 방문 상담 기록"]
NEG_WORDS = ["하늘", "다운", "보람", "한결", "새롬", "나래", "슬기", "가람", "바다", "별", "이상", "최초", "한 번", "이 건",
             "소망", "기쁨", "누리", "아름", "다솜", "미소", "은하", "초롱", "솔바람", "해솔", "봄", "여름"]


def frames():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("pos", FRAME_USER.format(n=25, kind=KINDS[i % len(KINDS)])) for i in range(24)]
    focus = ["⟦주민⟧ ⟦계좌⟧", "⟦카드⟧ ⟦전화⟧", "⟦주소⟧ ⟦이메일⟧", "⟦계좌⟧ ⟦주소⟧", "⟦전화⟧ ⟦주민⟧", "⟦이메일⟧ ⟦카드⟧"]
    jobs += [("pos", SLOT_USER.format(n=25, kind=KINDS[i % len(KINDS)], focus=focus[i % len(focus)])) for i in range(12)]
    jobs += [("neg", NEG_USER.format(n=25, words=", ".join(random.Random(i).sample(NEG_WORDS, 5)))) for i in range(16)]

    def one(job):
        kind, user = job
        for _ in range(3):
            try:
                out, meta = llm.chat_json(FRAME_SYS, user, provider="openai", timeout=180)
                return kind, [f for f in (out or {}).get("frames", []) if isinstance(f, str)], meta["cost_usd"]
            except Exception as e:  # 일시 오류
                err = e
        print("실패", type(err).__name__)
        return kind, [], 0.0
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, jobs))
    n, cost = 0, 0.0
    with open(OUT / "frames.jsonl", "w", encoding="utf-8") as f:
        for kind, fs, c in res:
            cost += c
            for t in fs:
                if kind == "neg" and ("⟦" in t or "{" in t):
                    continue
                f.write(json.dumps({"kind": kind, "frame": t}, ensure_ascii=False) + "\n")
                n += 1
    print(f"틀 {n}개 · 비용 ${cost:.3f}")


# v2: v1 학습 뒤 늘어난 오탐 3가지 (31장) — 영문 머리말('Subject', 'To'), 순우리말 낱말이 든 회사·가게 이름('다운푸드'),
#     짧은 업무 낱말('요망', '이용한도', '콜백') → 반대 예문 + 같은 모양에 이름이 오는 비교 예문
MORE = [
    ("neg", """개인정보(사람 이름·연락처)가 전혀 없는 사내 이메일·메신저 글 {n}개. 빈칸(⟦ ⟧)도 쓰지 마세요.
- 영문 머리말을 꼭 쓰세요: Subject:, To:, From:, CC:, Re:, Fwd:, Sub: 와 받는 쪽은 '팀 전체', '영업팀', 'All', '관계 부서'처럼 사람 이름 없이
- 본문에 ATM, ARS, SMS, VIP, OTP, OK, FAQ, PDF, Excel 같은 영문 낱말을 자연스럽게
- 2~5줄. JSON 으로만: {{"frames": ["...", ...]}}"""),
    ("pos", """사내 이메일·메신저 글 {n}개. 영문 머리말(Subject:, To:, From:, CC:, Re:)을 쓰고, 받는 사람·보낸 사람·본문에 ⟦이름⟧을 1~3번,
⟦이메일⟧이나 ⟦전화⟧를 0~2번 쓰세요. 머리말 낱말 자체(Subject 등)는 이름이 아닙니다. 2~5줄."""),
    ("neg", """개인정보가 전혀 없는 업무 글 {n}개. 빈칸(⟦ ⟧)도 쓰지 마세요.
- 순우리말 낱말이 들어간 회사·가게·상품·지점 이름을 꼭 쓰세요 (예: 다운푸드, 한결금융, 슬기상회, 가람물산, 누리시큐리티, 새롬카드, 하늘정원, 마루상회, 봄날베이커리, 해솔지점)
- 가맹점 결제·제휴·납품·지점 공지 같은 내용. 1~3문장. JSON 으로만: {{"frames": ["...", ...]}}"""),
    ("neg", """개인정보가 전혀 없는 짧은 업무 메모·처리 기록 {n}개. 빈칸(⟦ ⟧)도 쓰지 마세요.
- 이런 짧은 업무 낱말을 많이, 표나 '항목: 값' 모양으로도: 요망, 콜백, 이용한도, 진행상태, 처리요망, 청구, 회신, 접수완료, 보류, 반려, 확인요, 재발송, 한도상향, 결제일, 연체
- 1~4줄. JSON 으로만: {{"frames": ["...", ...]}}"""),
]


# v3: v2 뒤 남은 오탐 — 이름 바로 옆 항목 낱말('【기준】 명민재님', '상주: 박찬기', '(내선 …)', '요망')
FIELD_WORDS = "기준, 내선, 비고, 상기, 공유, 요망, 상주, 발인, 조화, 성함, 처리결과, 담당, 문의, 참고, 수신, 발신, 결재, 검토, 확인자, 작성자"
MORE3 = [
    ("pos", """표나 '항목: 값' 모양의 업무 기록 {n}개. 사람 이름 ⟦이름⟧이 항목 낱말 바로 옆에 붙어 나오게 쓰세요
(예: '기준 ⟦이름⟧님', '작성자: ⟦이름⟧', '⟦이름⟧ 팀장님(내선 1234)', '검토 요망 - ⟦이름⟧', '상주: ⟦이름⟧', '- 비고: ⟦이름⟧ 확인').
항목 낱말 예: """ + FIELD_WORDS + """. 항목 낱말 자체는 이름이 아닙니다. ⟦전화⟧·⟦이메일⟧ 0~2번. 2~5줄."""),
    ("neg", """표나 '항목: 값' 모양의 업무 기록 {n}개. 사람 이름·개인정보 없이(빈칸 ⟦ ⟧도 쓰지 말고), 다음 항목 낱말을 많이 쓰세요: """
     + FIELD_WORDS + """. 값은 날짜·금액·상태·부서명. 2~5줄. JSON 으로만: {{"frames": ["...", ...]}}"""),
]


# v4: 데이터 크게 늘리기 (사용자 제안) + v3 뒤 남은 것 — 오탐: 방·층 번호('2층 5호', '1호실'), 순우리말이 든 회사·상품 이름
#     ('누리시큐', '봄날'), 짧은 업무 낱말('미준수', '이용한도', '하자') / 미탐: 띄어 쓴 이름('홍 영'), 앞말에 붙은 이름('담당이재훈')
MORE4 = [
    ("pos", """서로 다른 업무 글 {n}개. 사람 이름 ⟦이름⟧을 2~4번 쓰고, 이름끼리 '⟦이름⟧이랑 ⟦이름⟧하고', '⟦이름⟧, ⟦이름⟧, ⟦이름⟧'처럼
나란히 나오는 문장과, 직함·부서 바로 뒤('담당 ⟦이름⟧ 대리님', '영업2팀 ⟦이름⟧', '회원 ⟦이름⟧ 고객님'), 자기소개('저 ⟦이름⟧입니다'),
메신저 대화('상담사: …', '고객: …')를 섞으세요. 종류: 상담 기록·메신저·메일·처리 메모 중 아무거나. 1~4문장."""),
    ("neg", """개인정보가 전혀 없는 행사·조문·회의 안내 {n}개 (빈칸 ⟦ ⟧ 쓰지 말 것). 장소는 건물·홀·장례식장·회의실 이름과
층·호실만 (예: '라비앙홀 3층', '청아장례식장 2호실', 'K타워 405호', '본사 12층 대회의실'). 도로명 주소는 쓰지 마세요. 1~3문장."""),
    ("neg", """개인정보가 전혀 없는 업무 글 {n}개 (빈칸 ⟦ ⟧ 쓰지 말 것). 따옴표 안 상품·계정·서비스·협력사 이름과 줄인 회사 이름을 꼭 쓰세요
(예: '봄날' 적금, "가람" 카드, 협력사 누리시큐, 블루씨드, 하늘마루, 새온보안, 반딧불 앱, 한별마트). 1~3문장."""),
    ("neg", """개인정보가 전혀 없는 짧은 감사·점검·민원 처리 기록 {n}개 (빈칸 ⟦ ⟧ 쓰지 말 것). 이런 낱말을 많이, 끝맺음은 '~임.', '~함.', '~요망.'처럼:
미준수, 이용한도, 하자, 주거래, 미처리, 재확인, 누락, 지연, 보완, 시정, 부적정, 미흡, 적정, 이상없음. 1~4줄."""),
]


# v5: v4 를 holdout 형식에 돌렸을 때 나온 유형을 일반화 (holdout 은 이제 개선용, 새 확인 세트 = 1주차 synth_fresh, 31장)
#     오탐: 영문 브랜드·약어('Saebom Card', AI, USD), '이상입니다'·'미수신' / 미탐: 메신저 말하는 사람 줄('상담사 재원:')
MORE5 = [
    ("neg", """개인정보가 전혀 없는 한국어 업무 글 {n}개 (빈칸 ⟦ ⟧ 쓰지 말 것). 영문 회사·브랜드·서비스 이름과 약어를 한국어 문장에 섞으세요
(예: Saebom Card, Hana Pay, KB, AI 녹취 요약, USD 9,000, KRW, ATM, CS팀, VIP, QR, IT, HR, Fintech Lab, [Notice], Dear Customer).
메일 제목·서명('Marketing Strategy Team, Saebom Card')도 섞어서. 1~4줄."""),
    ("neg", """개인정보가 전혀 없는 상담·안내 글 {n}개 (빈칸 ⟦ ⟧ 쓰지 말 것). 이런 표현을 많이: '이상입니다', '이상입니다. 감사합니다',
'미수신', '미납', '미등록', '정상 처리', '확인 완료', '누리은행·한빛은행 같은 은행 이름 + 계좌', '결제 알림 미수신'. 표('항목 | 값')도 섞어서. 1~4줄."""),
    ("pos", """메신저·통화 녹취 정리 글 {n}개. 줄마다 '역할 ⟦이름⟧: 말' 또는 '⟦이름⟧: 말' 모양으로 말하는 사람을 쓰세요
(예: '상담사 ⟦이름⟧: 네, 확인해 드릴게요', '고객 ⟦이름⟧: 카드가 안 돼요', '⟦이름⟧ 대리: 처리했습니다'). 역할 낱말(상담사·고객·팀장)은 이름이 아닙니다.
⟦전화⟧·⟦계좌⟧·⟦카드⟧ 0~2번. 3~6줄."""),
]


# v6: v5 의 '말하는 사람 줄' 예문 부작용 — 콜론 앞 낱말을 이름으로 배움 ('성명:', '사유:', '연락처:', '본문:' 오탐)
#     → 한 글 안에 '항목명: 값' 줄과 '말하는 사람: 말' 줄을 섞어 구분하게. 은행 이름('누리은행')·'이 건도' 반대 예문
MORE6 = [
    ("pos", """접수서·처리 기록 {n}개. 앞부분은 '항목명: 값' 줄 3~6개, 뒷부분은 통화·메신저 대화 2~4줄로 쓰세요.
항목명 예: 성명, 이름, 연락처, 주소, 사유, 본문, 소속, 접수일, 처리결과, 비고, 금주 일정, 요청 내용, 담당자.
값 자리에는 ⟦이름⟧ ⟦전화⟧ ⟦주소⟧ ⟦계좌⟧ ⟦이메일⟧ 또는 일반 내용. 대화 줄은 '상담사 ⟦이름⟧: …', '고객: …', '⟦이름⟧: …'.
항목명(성명·사유 등)과 역할 낱말(상담사·고객)은 이름이 아닙니다."""),
    ("neg", """개인정보가 전혀 없는 '항목명: 값' 모양 서식 {n}개 (빈칸 ⟦ ⟧ 쓰지 말 것). 항목명: 성명, 연락처, 주소, 사유, 본문, 소속, 금주, 요청, 결과, 비고.
값은 '본인 확인 후 기재', '(미기재)', '해당 없음', '변경 요청', 날짜·금액·부서명. 3~6줄."""),
    ("neg", """개인정보가 전혀 없는 업무 글 {n}개 (빈칸 ⟦ ⟧ 쓰지 말 것). 순우리말이 든 은행·지점·상품 이름(누리은행, 가람저축은행, 한빛은행 본점,
새롬지점, 하늘정원점)과 '이 건도', '이 건은', '그 건도' 같은 표현을 꼭 쓰세요. 1~3문장."""),
]


def frames_more(more=None, reps=6):
    jobs = [(kind, user.format(n=25)) for kind, user in (more or MORE) for _ in range(reps)]

    def one(job):
        kind, user = job
        for _ in range(3):
            try:
                out, meta = llm.chat_json(FRAME_SYS, user, provider="openai", timeout=180)
                return kind, [f for f in (out or {}).get("frames", []) if isinstance(f, str)], meta["cost_usd"]
            except Exception:
                pass
        return kind, [], 0.0
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(one, jobs))
    n, cost = 0, 0.0
    with open(OUT / "frames.jsonl", "a", encoding="utf-8") as f:
        for kind, fs, c in res:
            cost += c
            for t in fs:
                if kind == "neg" and ("⟦" in t or "{" in t):
                    continue
                f.write(json.dumps({"kind": kind, "frame": t, "v": 2}, ensure_ascii=False) + "\n")
                n += 1
    print(f"틀 {n}개 추가 · 비용 ${cost:.3f}")


# ------------------------------------------------------------------ 가짜 값
SURNAMES = list("김이박최정강조윤장임한오서신권황안송류홍전고문양손배백허유남심노하곽성차주우구민진지엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹")
SURNAMES_W = SURNAMES[:12] * 4 + SURNAMES
COMPOUND = ["남궁", "제갈", "선우", "황보", "독고", "사공"]
SYL1 = list("민서지하도윤수예주시현준우채유은다재건태소연승혜성진영동경가나라마보슬아")
SYL2 = list("준서윤우호현민진아은연원빈희수영율린름결솔람운람별온찬혁석훈규")
NATIVE = ["하늘", "다운", "보람", "한결", "새롬", "나래", "슬기", "가람", "바다", "누리", "아름", "다솜", "미소", "은하", "초롱",
          "해솔", "솔바람", "하람", "가온", "나린", "다올", "라온", "마루", "새벽", "여름", "단비", "이슬", "한별", "샘", "봄"]
FOREIGN = ["John", "Anna", "Robert", "Sophia", "Michael", "Emily", "David", "Grace", "James", "Olivia", "Daniel", "Chloe",
           "Kim Min", "Lee Ji", "Park Sol", "Tanaka", "Nguyen", "Maria", "Ivan", "Leo", "Mia", "Ethan", "Hana Lee", "KIM O"]
GLUE_BEFORE = re.compile(r"(팀|부|회원|고객|담당|창구|상담원|마케팅|센터|담당자|본부|지점) $")


def given(r):
    return r.choice(SYL1) + r.choice(SYL2)


def person(r):
    """(값, 형태)"""
    x = r.random()
    if x < 0.27:  # v4: 띄어쓰기 15→20%, 외자 7→10% (30장 미탐 유형)
        return r.choice(SURNAMES_W) + given(r), "full"
    if x < 0.47:
        return r.choice(SURNAMES_W) + " " + given(r), "spaced"
    if x < 0.63:
        return given(r), "given"
    if x < 0.83:  # v5: 영문 이름 뺌 (사용자: 실제로 드묾 + 'Saebom'·'AI'·'USD' 오탐의 원인) → 그 몫은 순우리말 이름으로
        return r.choice(NATIVE), "native"
    if x < 0.90:
        c = r.choice(COMPOUND)
        return c + r.choice(["", " "]) + r.choice([given(r), r.choice(SYL2)]), "compound"
    return r.choice(SURNAMES_W) + r.choice(["", " "]) + r.choice(SYL2), "short"  # 외자 (김린, 이 윤)


def digits(r, n):
    return "".join(r.choice("0123456789") for _ in range(n))


def phone(r):
    a, b = r.choice(["010", "010", "010", "011", "02", "031", "051"]), digits(r, 4)
    mid = digits(r, 3 if a in ("02", "031", "051") and r.random() < 0.5 else 4)
    return r.choice(["-", "", ".", " "]).join([a, mid, b])


def email(r):
    u = r.choice(["kim", "lee", "park", "sol", "hana", "j.min", "dev", "sky", "minho", "ys"]) + r.choice(["", "_", "."]) \
        + r.choice(["", digits(r, 2), r.choice(["k", "j", "lee"])])
    return u + "@" + r.choice(["example.com", "mail.example.org", "test.co.kr", "saebom.test", "corp.example.net"])


SIDO = ["서울특별시", "서울", "부산광역시", "경기도", "인천", "대구광역시", "광주광역시", "대전", "경상남도", "전라북도", "강원도", "제주특별자치도"]
GU = ["강남구", "종로구", "마포구", "수원시 영통구", "해운대구", "중구", "서구", "유성구", "성남시 분당구", "청주시 상당구", "창원시 의창구"]
RO = ["테헤란로", "세종대로", "중앙로", "대학로", "한빛로", "새봄길", "산업로", "도곡로", "역삼로", "해안로"]
APT = ["래미안", "자이", "푸르지오", "힐스테이트", "새봄빌", "한빛아파트", "더샵", "e편한세상"]


def address(r):
    x = r.random()
    if x < 0.6:
        s = f"{r.choice(SIDO)} {r.choice(GU)} {r.choice(RO)} {r.randint(1, 300)}"
        if r.random() < 0.5:
            s += f", {r.choice(APT)} {r.randint(101, 120)}동 {r.randint(101, 1505)}호"
        return s
    if x < 0.95:  # v3: 동·호수만(아래) 20% → 5% — '2층 205호' 같은 부분 주소 오탐 (31장)
        return f"{r.choice(GU)} {r.choice(['역삼동', '청운동', '방화동', '서초동', '연산동', '삼성동'])} {r.randint(1, 900)}-{r.randint(1, 30)}"
    return f"{r.choice(APT)}{r.randint(101, 120)}동{r.randint(101, 1505)}호"


def rrn(r):
    return f"{r.randint(60, 99)}{r.randint(1, 12):02d}{r.randint(1, 28):02d}{r.choice(['-', '', ' '])}{r.choice('1234')}{digits(r, 6)}"


def account(r):
    pat = r.choice([(3, 2, 6), (6, 2, 6), (4, 3, 6), (3, 6, 5), (3, 4, 4, 2)])
    return r.choice(["-", "-", ""]).join(digits(r, n) for n in pat)


def card(r):
    return r.choice(["-", " ", ""]).join(digits(r, 4) for _ in range(4))


GEN = {"전화": phone, "이메일": email, "주소": address, "주민": rrn, "계좌": account, "카드": card}


def batchim(name):
    c = name.rstrip()[-1]
    if "가" <= c <= "힣":
        return (ord(c) - 0xAC00) % 28 != 0
    return c.lower() not in "aeiouy"  # 영문: 모음으로 끝나면 받침 없음으로


JOSA = {"가": ("이", "가"), "는": ("은", "는"), "를": ("을", "를"), "과": ("과", "와"), "랑": ("이랑", "랑"), "이": ("이", ""),
        "으로": ("으로", "로")}


def fill(frame, r):
    """(글, 구간 목록)"""
    out, spans, pos = "", [], 0
    for m in SLOT.finditer(frame):
        out += frame[pos:m.start()]
        kind = m.group(1)
        if kind == "이름":
            v, _ = person(r)
            if GLUE_BEFORE.search(out) and r.random() < 0.5:  # '영업팀 슬기' → '영업팀슬기' (30장 앞말에 붙음)
                out = out[:-1]
        else:
            v = GEN[kind](r)
        spans.append({"label": LABEL[kind], "start": len(out), "end": len(out) + len(v)})
        out += v
        pos = m.end()
        jm = re.match(r"\{(가|는|를|과|랑|이|으로)\}", frame[pos:])
        if jm:
            yes, no = JOSA[jm.group(1)]
            out += yes if batchim(v) else no
            pos += jm.end()
    out += frame[pos:]
    out = re.sub(r"\{(가|는|를|과|랑|이|으로)\}", "", out)  # 빈칸 뒤가 아닌 곳에 잘못 쓴 표시
    return out, spans


def build():
    rows = [json.loads(l) for l in open(OUT / "frames.jsonl", encoding="utf-8")]
    r = random.Random(42)
    data = []
    for i, row in enumerate(rows):
        reps = 7 if row["kind"] == "pos" else 3
        for k in range(reps):
            text, spans = fill(row["frame"], r) if row["kind"] == "pos" else (row["frame"], [])
            data.append({"id": f"F{i:04d}-{k}", "frame": i, "text": text, "entities": spans})
    frames_ids = sorted({d["frame"] for d in data})
    r.shuffle(frames_ids)
    valid_frames = set(frames_ids[: len(frames_ids) // 10])  # 검증은 틀 단위로 떼어 둠 (같은 틀이 양쪽에 안 가게)
    for name, part in (("train", [d for d in data if d["frame"] not in valid_frames]),
                       ("valid", [d for d in data if d["frame"] in valid_frames])):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for d in part:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(name, len(part), "문장 ·", sum(len(d["entities"]) for d in part), "구간")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"frames": frames, "more": frames_more, "more3": lambda: frames_more(MORE3), "more4": lambda: frames_more(MORE4, reps=12), "more5": lambda: frames_more(MORE5, reps=10), "more6": lambda: frames_more(MORE6, reps=10), "fill": build}[sys.argv[1]]()
