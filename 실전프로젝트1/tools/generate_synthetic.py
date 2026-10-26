"""일상 카테고리의 가상 문서를 OpenAI 모델로 대량 생성한다 (규칙과 무관한 시험 데이터).

편향 줄이기
- 프롬프트에는 탐지 규칙을 전혀 넣지 않는다. 카테고리·문체와 라벨 표시 기준(데이터셋 정답 기준)만 준다.
- 값의 형식(전화번호 모양, 주소 형식, 이름 등)도 모델이 정한다.
- 생성 결과는 사람이(그리고 규칙 작성자가) 읽지 않고 parse_synthetic.py 로 형식만 검사한다.

카테고리 20 × 문체 5 = 100묶음, 묶음당 20건 → 2,000건. 이미 만든 묶음은 건너뛴다(이어하기).

사용법
  python -X utf8 tools/generate_synthetic.py --batches 1      # 시범 1묶음
  python -X utf8 tools/generate_synthetic.py                  # 전체 100묶음
결과: data/generated/_work/raw/<모델>_<묶음>.txt, data/generated/_work/usage.jsonl

2차 평가 세트 (문서 유형 25 × 문체 4 = 100묶음 → 2,000건, 고치는 데 쓰지 않고 한 번만 채점)
  python -X utf8 tools/generate_synthetic.py --set fresh --batches 2 --raw-dir raw_fresh_pilot   # 시범 (평가에서 제외)
  python -X utf8 tools/generate_synthetic.py --set fresh                                          # 전체 → raw_fresh/
어려운 표기 세트 (문서 유형 25 × 문체 2 = 50묶음 → 1,000건, 문서마다 어려운 표기 2~3개)
  python -X utf8 tools/generate_synthetic.py --set hard                                           # → raw_hard/
"""
import argparse
import json
import os
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 맨 위 폴더의 common·run_compare·detectors 사용

from concurrent.futures import ThreadPoolExecutor

from common import DATA, load_env

CATEGORIES = ["상담 기록", "환불 요청", "배송·택배 문의", "병원·식당·숙소 예약", "민원 접수", "중고거래",
              "이사·부동산 계약", "보험금 청구", "채용·입사 지원", "학교·학원 공지", "카드 분실·정지", "대출·금융 상담",
              "회사 업무 메일", "단체 대화방", "고객 문의 게시판", "택배·퀵 접수", "경조사 안내", "관리비·공과금 안내",
              "동호회·모임 공지", "항공·여행 예약"]
STYLES = ["공식 문서·기록", "이메일", "메신저 대화", "짧은 메모", "게시판 글"]
DOCS_PER_BATCH = 20

PROMPT = """당신은 한국어 문서 작성자입니다. 개인정보 탐지기를 시험하기 위한 "가상의" 한국어 문서를 만들어 주세요.

[이번 요청]
- 카테고리: {category}
- 문체: {style}
- 개수: {n}건
- 묶음 번호: {batch}

[작성 원칙]
1. 실제 사람이 일상이나 업무에서 쓰는 글처럼 자연스럽게 써 주세요. 사람마다 쓰는 습관이 다르듯 표기·형식·길이·말투를 문서마다 다양하게 해 주세요.
2. 문서 길이는 2줄짜리 짧은 메모부터 15줄 정도의 긴 기록까지 섞어 주세요.
3. {n}건 중 2~3건은 개인정보가 전혀 없는 문서로 써 주세요.
4. 개인정보가 아닌 숫자·명칭(날짜, 시간, 금액, 주문번호, 접수번호, 회사 대표번호, 회사·기관·상품 이름 등)도 자연스럽게 섞어 주세요.
5. 모든 개인정보는 실존하지 않는 가상 값으로 써 주세요.
   - 이메일 도메인은 example.com, example.net, example.co.kr 같은 예시 도메인만 사용
   - 전화번호는 실제 번호와 겹치지 않도록 국번(가운데 자리)을 0으로 시작
   - 주민등록번호·카드번호·계좌번호는 실제로 유효하지 않은 임의의 숫자

[개인정보 표시 방법]
아래 7종류의 개인정보는 반드시 [[값|라벨]] 형식으로 표시해 주세요. 표시를 빠뜨리면 안 됩니다.
- person: 사람 이름 (고객, 직원, 지인 등 모든 사람)
- phone_number: 전화번호
- email: 이메일 주소
- address: 주소 (상세 주소까지 포함)
- resident_registration_number: 주민등록번호 또는 외국인등록번호 (둘 다 13자리 숫자)
- account_number: 은행 계좌번호
- card_number: 카드번호

표시 기준
- 문서에 적힌 글자 그대로 표시합니다. 예: [[윤 선영|person]], [[010.0349.2063|phone_number]]
- 이름 뒤의 조사·호칭·직함은 표시 밖에 둡니다. 예: [[김민수|person]]님은, [[이소율|person]] 과장
- 같은 사람이 여러 번 나오면 나올 때마다 표시합니다.
- 다음은 개인정보가 아니므로 표시하지 않습니다: 성과 직함만 있는 표현(예: 김 과장), 회사 대표번호, 회사·기관·은행·상품 이름, 날짜, 금액, 주문·접수번호

[출력 형식]
설명이나 인사말 없이 아래 형식만 출력하세요. 각 문서는 "=== " 줄로 시작합니다.

=== {batch}-01 | {category} | {style}
(문서 본문)

=== {batch}-02 | {category} | {style}
(문서 본문)

... {n}건까지"""


# --set fresh: 2차 평가 전용 세트 (2026-10-26). 탐지기를 고치는 데 쓰지 않고 한 번만 채점한다.
# dev·test·일상 문서·멘토 holdout 어디에도 없던 금융 업무 유형 + 표기 다양화. 탐지 규칙은 여전히 넣지 않고,
# 사람들이 실제로 쓰는 표기 습관만 묶음마다 무작위로 몇 개씩 준다 (모든 문서에 같은 변형이 몰리지 않도록).
FRESH_CATEGORIES = ["여신 심사 메모", "콜센터 통화 녹취 전사", "보이스피싱 피해 상담", "상속인 금융거래 조회",
                    "법인 계좌 개설 서류 안내", "외화 환전 예약", "퇴직연금 상담", "투자상품 판매 기록",
                    "전세자금대출 서류 안내", "자동차 할부 계약", "개인정보 열람·정정 청구", "채무조정 신청",
                    "급여이체·주거래 변경", "영업점 일일 업무 보고", "사내 교육 수료 안내", "전산 장애·접속 기록 보고",
                    "고객 방문 예약 확인", "보험 계약 변경 요청", "압류·법원 송달 처리", "사망자 계좌 해지",
                    "사내 행사·회식 안내", "외국인 고객 계좌 개설", "명의 도용 신고", "휴면예금 찾기 안내",
                    "협력업체 정산 메일"]
FRESH_STYLES = ["'항목: 값' 형식의 서식·기록", "이메일", "메신저 또는 통화 대화", "표나 목록이 섞인 보고서"]
# 문서마다 표기 계획을 무작위로 정해 준다. 묶음 단위 힌트만 주면 모델이 기본 형식(010-0000-0000, 3글자 이름)으로
# 돌아가거나 한 형식으로 몰려서(시범 1·2차), 형식별 비율을 코드로 정한다. (표기 모양, 가중치)
FRESH_PLAN = {
    "개인정보 양": [("개인정보 없음", 8), ("개인정보 1~2개만", 30), ("개인정보 여러 개", 45), ("여러 사람이 나오고 개인정보가 많음", 17)],
    "전화번호": [("010-0123-4567처럼 하이픈", 30), ("점으로 구분(010.0123.4567)", 10), ("공백으로 구분(010 0123 4567)", 10),
              ("붙여쓰기(01001234567)", 10), ("국제번호(+82 10-0123-4567 또는 +82-10-…)", 10), ("지역번호(02, 031, 051, 064 등)", 15),
              ("괄호 지역번호((02)0123-4567)", 5), ("050 안심번호", 5), ("070 인터넷전화", 5)],
    "이름": [("보통 3글자 이름", 45), ("외자 이름(성 + 한 글자)", 15), ("복성 이름(남궁·황보·제갈·선우·독고 등)", 8),
           ("띄어 쓴 이름(성과 이름 사이 한 칸)", 10), ("성 없이 이름만 부르기", 12), ("외국인 이름(한글 표기 또는 영문)", 10)],
    "이름 뒤 말": [("님·씨", 30), ("에게·께서·께", 15), ("이랑·하고·와", 10), ("의·은·는·이·가", 30), ("인데요·이에요·입니다", 15)],
    "주민등록번호": [("하이픈(000000-0000000)", 50), ("붙여쓰기 13자리", 20), ("공백 구분(000000 0000000)", 15),
                ("외국인등록번호(13자리, 뒷자리 첫 숫자 5~8)", 15)],
    "계좌번호": [("3-2-6 묶음", 12), ("4-3-6 묶음", 12), ("3-6-5 묶음", 12), ("6-2-6 묶음", 12), ("3-3-6 묶음", 10),
              ("4묶음(3-2-6-1 등)", 10), ("붙여쓰기", 12), ("휴대폰 번호 모양의 계좌(010…)", 10), ("가상계좌(14자리 안팎)", 10)],
    "카드번호": [("4-4-4-4 하이픈", 50), ("4-4-4-4 공백", 20), ("16자리 붙여쓰기", 20), ("15자리 4-6-5", 10)],
    "주소": [("도로명 주소", 30), ("지번 주소(동·리 + 번지)", 20), ("읍·면이 들어간 주소", 10), ("시도 줄임말(서울, 경기, 부산)", 15),
           ("우편번호를 붙인 주소", 10), ("아파트·빌라 이름 + 동·호수", 15)],
    "헷갈리는 값": [("없음", 40), ("회사 대표번호(1588-…)", 10), ("사업자등록번호", 8), ("사건·접수·주문·송장번호", 12), ("내선번호", 8),
              ("사람 이름처럼 보이는 회사·상품·가게 이름", 10), ("성 + 직함만 쓴 표현(박 대리)", 7), ("회사·지점·행사장 주소", 5)],
}


# 문서에 넣을 개인정보 종류 (이름 말고는 가중치로 뽑음: 제공 데이터·holdout 비율보다 번호류를 조금 더)
FRESH_KINDS = [("전화번호", 25), ("이메일", 12), ("주소", 14), ("주민등록번호", 10), ("계좌번호", 11), ("카드번호", 8)]


def _pick(rng, key):
    opts = FRESH_PLAN[key]
    return rng.choices([v for v, _ in opts], [w for _, w in opts])[0]


def doc_plan(rng):
    amount, decoy = _pick(rng, "개인정보 양"), _pick(rng, "헷갈리는 값")
    if amount == "개인정보 없음":
        return f"개인정보 없음 (헷갈리는 값: {decoy})"
    n_other = {"개인정보 1~2개만": rng.choice([0, 1]), "개인정보 여러 개": rng.randint(1, 3),
               "여러 사람이 나오고 개인정보가 많음": rng.randint(2, 4)}[amount]
    with_name = amount != "개인정보 1~2개만" or rng.random() < 0.6
    names = [k for k, _ in FRESH_KINDS]
    picked = set(rng.choices(names, [w for _, w in FRESH_KINDS], k=n_other if with_name else max(n_other, 1)))
    kinds = (["이름"] if with_name else []) + [k for k in names if k in picked]
    parts = [f"넣을 개인정보: {', '.join(kinds)} ({amount})"]
    for k in kinds:
        if k == "이메일":
            continue
        parts.append(f"{k}: {_pick(rng, k)}")
        if k == "이름":
            parts.append(f"이름 뒤 말: {_pick(rng, '이름 뒤 말')}")
    parts.append(f"헷갈리는 값: {decoy}")
    return " / ".join(parts)


FRESH_PROMPT = """당신은 금융회사 직원입니다. 개인정보 탐지기를 시험하기 위한 "가상의" 한국어 업무 문서를 만들어 주세요.

[이번 요청]
- 문서 유형: {category}
- 문체: {style}
- 개수: {n}건
- 묶음 번호: {batch}

[작성 원칙]
1. 실제 금융회사 직원·고객이 쓰는 글처럼 자연스럽게 써 주세요. 사람마다 쓰는 습관이 다르듯 표기·형식·길이·말투를 문서마다 다르게 해 주세요.
   **같은 묶음 안에서 한 가지 표기(예: 모든 전화번호를 같은 모양으로, 모든 이름을 띄어 쓰기)를 반복하지 마세요.**
2. 문서 길이는 5줄부터 20줄 정도까지 섞어 주세요. 문서마다 들어가는 개인정보의 종류와 개수도 다르게 해 주세요
   (이름 하나만 있는 문서, 번호 두세 개만 있는 문서, 여러 사람이 나오는 문서 등). 모든 항목을 다 채운 같은 틀을 반복하지 마세요.
3. 문서마다 아래 "문서별 계획"을 따라 주세요. "넣을 개인정보"에 적힌 종류만 넣고(적히지 않은 종류는 넣지 마세요),
   각 종류는 적힌 모양으로 써 주세요. "헷갈리는 값"은 개인정보처럼 보이지만 표시하지 않는 값입니다.
{hints}
4. 같은 사람이 문서 안에서 여러 번 나오면 표기를 섞어도 됩니다(전체 이름, 이름만 등).
5. 모든 개인정보는 실존하지 않는 가상 값으로 써 주세요.
   - 이메일 도메인은 example.com, example.net, example.co.kr 같은 예시 도메인만 사용
   - 전화번호는 실제 번호와 겹치지 않도록 국번(가운데 자리)을 0으로 시작
   - 주민등록번호·카드번호·계좌번호는 실제로 유효하지 않은 임의의 숫자
   - 값의 일부를 별표(*)로 가린 형태는 쓰지 마세요
   - 1234-5678, 0000, 1111처럼 티 나게 반복·연속되는 숫자 대신 무작위로 보이는 숫자를 쓰세요

[개인정보 표시 방법]
아래 7종류의 개인정보는 반드시 [[값|라벨]] 형식으로 표시해 주세요. 표시를 빠뜨리면 안 됩니다.
- person: 사람 이름 (고객, 직원, 지인 등 모든 사람. 외국인 이름 포함)
- phone_number: 개인 전화번호 (직원 직통번호·휴대폰 포함)
- email: 이메일 주소
- address: 개인의 주소 (집·거주지·우편물 받는 곳, 상세 주소까지)
- resident_registration_number: 주민등록번호 또는 외국인등록번호 (둘 다 13자리 숫자)
- account_number: 은행 계좌번호 (은행 이름은 표시 밖에 둠. 예: 신한은행 [[계좌번호|account_number]])
- card_number: 카드번호

표시 기준
- 값 자체를 [[ ]] 안에 넣고 | 뒤에 라벨을 씁니다. 표시 뒤의 조사는 원래 글처럼 띄우지 않고 붙입니다.
  예: 담당자 [[박서준|person]] 대리에게 전달, [[박서준|person]]의 신청 건
- 문장 속에 나온 개인의 집·거주지·우편물 받는 주소도 빠짐없이 표시합니다. 예: 주소는 [[경기도 광명시 광명동 482-12|address]]로 확인
- 틀린 형식: 박서준 [[person]] (값을 밖에 두고 라벨만 쓰기), [[박서준]] (라벨 없음) — 이렇게 쓰지 마세요
- 문서에 적힌 글자 그대로(띄어쓰기·구분 기호 포함) 표시합니다.
- 이름 뒤의 조사·호칭·직함은 표시 밖에 둡니다. 예: [[김민수|person]]님은, [[이소율|person]] 과장
- 같은 사람이 여러 번 나오면 나올 때마다 표시합니다.
- 다음은 개인정보가 아니므로 표시하지 않습니다: 성과 직함만 있는 표현(예: 김 과장), 회사 대표번호(1588-…처럼 회사 전체가 쓰는 번호), 내선번호, 회사·기관·은행·상품·가게 이름, 회사·지점·매장·행사장·병원처럼 개인을 가리키지 않는 장소의 주소, 날짜, 금액, 사업자등록번호, 주문·접수·사건번호

[출력 형식]
설명이나 인사말 없이 아래 형식만 출력하세요. 각 문서는 "=== " 줄로 시작합니다.

=== {batch}-01 | {category} | {style}
(문서 본문)

=== {batch}-02 | {category} | {style}
(문서 본문)

... {n}건까지"""


# --set hard: 어려운 표기 세트 (2026-10-26). fresh 와 같은 프롬프트에 문서마다 "꼭 넣을 어려운 표기" 2~3개를 더한다.
# 지금까지 탐지기가 약했던 표기 중 일상·업무 글에서 실제로 쓸 만한 것만 골랐다. 이름 쪽 비중을 높게. (설명, 가중치)
HARD_STYLES = ["메신저 또는 통화 대화", "이메일이나 서식 문서"]
HARD_CASES = [
    # 이름
    ("성 + 외자 이름을 붙여 쓴 2글자 이름 (성 1글자 + 이름 1글자)", 10),
    ("성 + 외자 이름을 띄어 쓴 이름 (성과 한 글자 이름 사이 공백)", 7),
    ("일반 단어와 같은 순우리말 이름 (하늘, 보람, 한결, 다운, 나래, 슬기, 가람, 새롬 같은 이름을 성과 함께 또는 이름만)", 10),
    ("메신저에서 이름을 부르는 말 (이름 뒤에 아/야, 이랑, 이가, 이는, 이한테 같은 말이 붙음)", 10),
    ("띄어쓰기 없이 앞말·뒷말에 붙은 이름 (호칭·부서 바로 뒤에 붙거나 이름 뒤에 대리님·고객님이 붙어 씀)", 7),
    ("서명 (이름 + 드림/올림/배상, 또는 줄 끝에 '- 이름')", 5),
    ("영문 이름과 한글 이름을 함께 씀 (영문 이름(한글 이름) 등)", 4),
    ("처음엔 전체 이름, 뒤에서는 이름만 또는 '성 + 직함'으로 같은 사람을 부르기", 6),
    ("사람 이름처럼 보이는 일반 단어·회사·가게·상품 이름을 개인정보가 아닌 값으로 섞기", 6),
    # 전화번호
    ("지역번호 전화번호를 붙여 쓰기 (02·031 등으로 시작하는 9~10자리)", 4),
    ("0507·0504 안심번호나 070 인터넷전화", 3),
    ("국제번호 여러 모양 (+82 10…, +82-10-…, 82-10-…, +82 (0)10-…)", 3),
    ("번호 두 개를 슬래시·쉼표로 이어 쓰기, 또는 직통번호 뒤 괄호 내선 (내선 숫자는 개인정보 아님)", 4),
    ("콜론·괄호 바로 뒤에 공백 없이 붙은 번호 (연락처:010…, (010…))", 3),
    # 주소
    ("시도를 생략하거나 구 이름을 줄인 주소 (구부터 시작, '서울 강남 ○○동 …')", 4),
    ("도로명 주소 뒤 괄호 동 이름 '(○○동)', 건물명·층·호를 쉼표 없이 이어 쓰기", 4),
    # 계좌·카드·주민·이메일
    ("은행 이름과 계좌번호를 붙여 쓰거나(국민123…) 계좌번호 바로 뒤에 예금주 이름", 4),
    ("계좌번호를 공백이나 점으로 구분", 3),
    ("카드사 이름을 붙여 쓴 카드번호 또는 점으로 구분한 카드번호", 3),
    ("주민등록번호 하이픈 양옆에 공백 (000000 - 0000000) 또는 외국인등록번호", 3),
    ("꺾쇠·괄호 안 이메일, 대문자가 섞이거나 서브도메인이 있는 이메일, @ 대신 ' at '·'[at]'으로 쓴 이메일", 4),
]
HARD_RULES = """
[어려운 표기의 표시 기준] (표시 기준은 위와 같음)
- 이름 뒤 호격·조사·직함·'드림'은 표시 밖: [[이름|person]]아, [[이름|person]]이랑, [[이름|person]]대리님, [[이름|person]] 드림
- 앞말에 붙은 이름도 이름 부분만 표시: 회원[[이름|person]]님, 영업팀[[이름|person]]
- 일반 단어와 같은 이름도 사람 이름으로 쓰였으면 표시, 사람이 아닌 단어로 쓰였으면 표시하지 않음
- 은행·카드사 이름은 표시 밖: 국민[[계좌번호|account_number]], 직통번호 [[번호|phone_number]](내선 211)
- '@' 대신 'at'으로 쓴 이메일도 적힌 그대로 email로 표시
"""


def batches(kind="v1"):
    cats, styles = {"fresh": (FRESH_CATEGORIES, FRESH_STYLES), "hard": (FRESH_CATEGORIES, HARD_STYLES)}.get(
        kind, (CATEGORIES, STYLES))
    for i in range(len(cats) * len(styles)):
        yield f"{i + 1:03d}", cats[i // len(styles)], styles[i % len(styles)]


def hard_plan(rng):
    plan = doc_plan(rng)
    if plan.startswith("개인정보 없음"):
        return plan + " / 꼭 넣을 어려운 표기: 사람 이름처럼 보이는 일반 단어·회사·가게·상품 이름 (표시하지 않음)"
    picks = []
    while len(picks) < rng.choice([2, 3]):
        c = rng.choices([c for c, _ in HARD_CASES], [w for _, w in HARD_CASES])[0]
        if c not in picks:
            picks.append(c)
    return plan + " / 꼭 넣을 어려운 표기: " + "; ".join(picks)


def make_prompt(kind, batch, category, style):
    if kind == "v1":
        return PROMPT.format(category=category, style=style, n=DOCS_PER_BATCH, batch=batch)
    import random
    rng = random.Random(f"{kind}-{batch}")  # 묶음 번호로 고정 → 다시 돌려도 같은 프롬프트
    plan = hard_plan if kind == "hard" else doc_plan
    plans = "\n".join(f"   - {batch}-{i + 1:02d}: {plan(rng)}" for i in range(DOCS_PER_BATCH))
    prompt = FRESH_PROMPT.format(category=category, style=style, n=DOCS_PER_BATCH, batch=batch, hints=plans)
    if kind == "hard":  # 어려운 표기의 표시 기준을 [출력 형식] 앞에 넣는다
        prompt = prompt.replace("\n[출력 형식]", HARD_RULES + "\n[출력 형식]")
        prompt = prompt.replace("각 종류는 적힌 모양으로 써 주세요.",
                                "각 종류는 적힌 모양으로 써 주세요. \"꼭 넣을 어려운 표기\"는 반드시 자연스럽게 넣어 주세요"
                                "(그 표기에 필요한 종류는 넣을 개인정보에 없어도 넣음).")
    return prompt


def generate(client, model, temperature, batch, category, style, out_dir, kind="v1", retries=3):
    path = out_dir / f"{model}_{batch}.txt"
    if path.exists() and path.stat().st_size > 0:
        return None  # 이미 만든 묶음
    prompt = make_prompt(kind, batch, category, style)
    for attempt in range(retries):
        try:
            t0 = time.time()
            res = client.chat.completions.create(model=model, temperature=temperature, max_tokens=16000,
                                                 messages=[{"role": "user", "content": prompt}])
            path.write_text(res.choices[0].message.content, encoding="utf-8")
            return {"batch": batch, "model": model, "category": category, "style": style,
                    "prompt_tokens": res.usage.prompt_tokens, "completion_tokens": res.usage.completion_tokens,
                    "finish_reason": res.choices[0].finish_reason, "sec": round(time.time() - t0, 1)}
        except Exception as e:  # 일시 오류(속도 제한 등)는 잠시 쉬고 재시도
            err = str(e)[:200]
            time.sleep(5 * (attempt + 1))
    return {"batch": batch, "model": model, "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--batches", type=int, default=100, help="앞에서부터 몇 묶음 만들지")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--set", default="v1", choices=["v1", "fresh", "hard"],
                    help="fresh: 2차 평가 전용 세트 → raw_fresh/, hard: 어려운 표기 세트 → raw_hard/")
    ap.add_argument("--raw-dir", default=None, help="저장 폴더 (기본: v1 → raw, fresh → raw_fresh)")
    ap.add_argument("--start", type=int, default=0, help="몇 번째 묶음부터 (시범 묶음을 따로 뽑을 때)")
    a = ap.parse_args()
    load_env()
    from openai import OpenAI
    client = OpenAI()

    out_dir = DATA / "generated" / "_work" / (a.raw_dir or {"fresh": "raw_fresh", "hard": "raw_hard"}.get(a.set, "raw"))
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = list(batches(a.set))[a.start:a.start + a.batches]
    usage_path = out_dir.parent / f"usage{'' if a.set == 'v1' else '_' + a.set}.jsonl"
    done = 0
    with ThreadPoolExecutor(a.workers) as ex, open(usage_path, "a", encoding="utf-8") as log:
        for r in ex.map(lambda b: generate(client, a.model, a.temperature, *b, out_dir, a.set), todo):
            if r is None:
                continue
            log.write(json.dumps(r, ensure_ascii=False) + "\n")
            log.flush()
            done += 1
            msg = r.get("error") or f"{r['completion_tokens']} 출력 토큰, {r['finish_reason']}, {r['sec']}초"
            print(f"[{done}] 묶음 {r['batch']}: {msg}", flush=True)
    print("완료")


if __name__ == "__main__":
    main()
