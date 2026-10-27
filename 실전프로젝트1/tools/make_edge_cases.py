"""제공 데이터에 없는 형식(엣지 케이스)을 시험하는 문서를 만든다 → data/edge_cases.jsonl

제공 데이터는 템플릿 23종에 값을 채워 넣는 방식이라 형식이 단조롭다
(전화는 010·02만, 계좌는 3-6-2 한 가지, 주소는 도로명만, 이름은 2~3글자 위주).
실제 문서에서 나올 형식을 팀이 직접 적고 정답을 붙여, 규칙 수정 전후를 run_compare.py 로 비교한다.

작성법: 정답은 [[값|라벨]] 로 표시. 모든 값은 가짜(가상) 값이다.
주의: 규칙을 만드는 사람이 시험 문서도 쓰므로 점수가 낙관적일 수 있다. 다른 팀원이 케이스를 추가해 주면 좋다.

사용법
  python -X utf8 tools/make_edge_cases.py
  python -X utf8 run_compare.py --gold data/edge_cases.jsonl --pred rules=preds/rules_edge.jsonl
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 맨 위 폴더의 common·run_compare·detectors 사용


from common import DATA, LABELS

# (태그, 문서) — 태그는 run_compare 의 hard_cases 자리에 들어가 어떤 종류의 케이스인지 표시한다
CASES = [
    # ---------------- 전화번호: 지역번호·인터넷전화·안심번호·국제번호·구형 번호·표기 변형
    ("phone_area", "경기 본사 경영지원팀 [[031-8017-2245|phone_number]]로 연락 바랍니다."),
    ("phone_area", "부산 지점 연락처는 [[(051) 742-0091|phone_number]]입니다."),
    ("phone_area", "제주 고객센터 담당 직통 [[064-721-3302|phone_number]], 광주 [[062.0381.4421|phone_number]]"),
    ("phone_area", "서울 지점 [[02-731-4567|phone_number]] (3자리 국번)"),
    ("phone_internet", "재택 근무자 인터넷 전화 [[070-4012-3398|phone_number]]로 회신 요청."),
    ("phone_safe", "배달 주문에 안심번호 [[0504-1234-0987|phone_number]]가 기록되어 있었음."),
    ("phone_intl", "해외 체류 중이라 [[+82-10-0482-7713|phone_number]] 또는 [[+82 10 0482 7713|phone_number]]로 연락 부탁."),
    ("phone_old", "구형 번호 [[011-0234-5567|phone_number]]에서 010으로 변경 요청."),
    ("phone_format", "연락처 [[010 - 0482 - 7713|phone_number]] (띄어쓴 하이픈), 보조 [[010–0482–7714|phone_number]] (다른 대시)"),
    ("decoy_company_phone", "카드 분실은 1544-0000, 대출 문의는 1588-0000, 해외 결제 문의는 1600-0000으로 안내."),
    # ---------------- 계좌번호: 은행별 자릿수, 4묶음, 하이픈 없음, 휴대폰 번호 계좌(토스 등)
    ("account_bank", "환불 계좌: 국민은행 [[123456-01-234567|account_number]] (예금주 [[김민수|person]])"),
    ("account_bank", "신한은행 [[110-234-567890|account_number]]으로 입금 확인됨."),
    ("account_bank", "우리은행 [[1002-345-678901|account_number]] 출금 정지 요청."),
    ("account_bank", "농협 [[301-0123-4567-81|account_number]] 계좌로 이체 처리함."),
    ("account_bank", "카카오뱅크 [[3333-02-7654321|account_number]], 토스뱅크 [[1000-0123-4567|account_number]]"),
    ("account_phone", "토스뱅크 계좌(휴대폰 번호): [[010-0482-7713|account_number]]로 송금 받기로 함."),
    ("account_nohyphen", "환불 받을 계좌 [[11002345678901|account_number]] (하나은행)로 처리."),
    ("decoy_numbers", "주문번호 2026102212345, 민원 접수번호 202610220001 확인."),
    # ---------------- 주소: 지번(동·리), 시도 줄임말, 읍·면, 세종, 번길, 시도 생략
    ("address_jibun", "주소: [[서울특별시 강남구 역삼동 123-45|address]]"),
    ("address_jibun", "변경 주소 [[경기 성남시 분당구 정자동 178-1|address]]로 반영."),
    ("address_jibun", "배송지 [[충남 천안시 동남구 신부동 454-1|address]], 회사 [[서울 마포구 월드컵북로 21|address]]"),
    ("address_rural", "실거주지는 [[경기도 양평군 양평읍 양근리 123|address]]입니다."),
    ("address_sejong", "[[세종특별자치시 한누리대로 2130|address]] 정부청사 인근."),
    ("address_road", "[[경기도 성남시 분당구 판교역로 166번길 10, 3층|address]]으로 이전."),
    ("address_road", "[[제주특별자치도 제주시 첨단로 242|address]], [[강남구 테헤란로 152, 12층|address]]"),
    ("address_road", "[[부산광역시 해운대구 센텀중앙로 79, 101동 1202호|address]]로 카드 발송."),
    # ---------------- 이름: 외자, 성 1 + 이름 3, 복성, 외국인·영문, 일반명사 같은 이름
    ("name_single", "고객 [[김솔|person]]님이 지점을 방문함. [[이현|person]] 고객도 동행."),
    ("name_long", "상담원 [[황보미래|person]] 대리가 응대함. 고객 [[김하늘별|person]]님 문의."),
    ("name_compound", "[[남궁민수|person]] 고객, [[제갈 성렬|person]] 대리, [[선우 진|person]] 과장 참석."),
    ("name_foreign", "고객명: [[응우옌 반 안|person]] / 연락처 [[010-0911-2245|phone_number]]"),
    ("name_foreign", "예금주: [[John Smith|person]] / 성명: [[KIM MINJUN|person]]"),
    ("name_common_word", "[[정다운|person]] 고객과 [[한결|person]] 씨가 함께 방문. 한결같이 친절한 응대 부탁."),
    ("decoy_words", "김 과장님, 이번 주 정보 보호 교육 일정 공유드립니다. 하늘빛은행 제휴 건도 확인."),
    # ---------------- 이메일: 점이 여러 개, 대문자·밑줄·플러스, 조사 붙음, 국가 도메인, 우회 표기
    ("email_format", "회신 메일 [[min.su.kim@example.co.kr|email]]으로 부탁드립니다."),
    ("email_format", "보조 메일 [[Kim_Minsu+bank@Example.COM|email]], 개인 [[m.s.k.2026@mail.example.net|email]]"),
    ("email_obfuscated", "스팸 때문에 [[minsu.kim[at]example.com|email]] 형식으로 적어 둠."),
    # ---------------- 카드번호: 하이픈 없음, 공백, 아멕스 15자리, 일부 가림
    ("card_format", "카드번호 [[5412753100982210|card_number]] 분실 신고."),
    ("card_format", "카드 [[5412 7531 0098 2211|card_number]], 아멕스 [[3782-822463-10005|card_number]]"),
    ("card_masked", "고객 화면 표시값 [[5412-75**-****-2210|card_number]] 확인."),
    # ---------------- 주민등록번호: 외국인등록번호, 일부 가림, 하이픈 없음, 공백
    ("rrn_format", "외국인등록번호 [[900512-5345678|resident_registration_number]] 제출."),
    ("rrn_format", "주민번호 [[9005122345678|resident_registration_number]] / 재입력 [[900512 - 2345678|resident_registration_number]]"),
    ("rrn_masked", "화면 표시: [[900512-2******|resident_registration_number]]"),
    # ---------------- v4 추가: 문서 안 이름 퍼뜨리기, 서명·인사말, 은행명, 건물명 괄호
    ("name_propagation", "[[박지훈|person]] 고객이 지점을 방문함. 서류는 [[박 지훈|person]]에게 전달했고, "
                         "[[지훈|person]]이 직접 서명함."),
    ("name_propagation", "담당: [[문채린|person]] 대리\n고객 요청 사항은 [[채린|person]]에게 공유 완료."),
    ("name_signature", "확인 부탁드립니다.\n감사합니다.\n[[채민호|person]] 드림"),
    ("name_signature", "안녕하세요, 리스크관리팀 [[표지원|person]]입니다. 요청하신 자료 보내드립니다."),
    ("name_signature", "From: [[Lee Jiyoung|person]]\nTo: [[윤 하람|person]] 님"),
    ("decoy_words", "검토 예정입니다. 확인 요청입니다. 하반기 일정은 추후 공지 예정입니다."),
    ("account_bankname", "환불은 국민 [[12345601234567|account_number]], 또는 신한 [[110234567890|account_number]]로 받기로 함."),
    ("decoy_bank_word", "하나 더 말씀드리면 [[010-0482-7717|phone_number]]로 연락 주시면 됩니다. 우리 팀 번호는 "
                        "[[02-0731-4568|phone_number]]입니다."),
    ("address_building", "[[서울특별시 강남구 테헤란로 152 (역삼동, 강남파이낸스센터)|address]]로 방문 예정."),
    ("address_building", "발송지: [[부산광역시 해운대구 센텀중앙로 79 (우동)|address]]"),
    # ---------------- v5 추가: 이름과 일반 단어·은행 이름이 겹치는 경우
    ("name_bank_collision", "고객 [[김하나|person]] [[010-0482-7720|phone_number]]로 연락 바람. 하나 더 확인할 사항은 없음."),
    ("name_bank_collision", "예금주 [[김우리|person]], 신한 [[110-234-567891|account_number]]"),
    ("name_bank_collision", "[[김하나|person]]님은 하나은행 계좌 [[1002-345-678902|account_number]]로 환불을 요청함."),
    ("name_bank_collision", "[[이국민|person]] 고객, 국민 [[123456-01-234569|account_number]] 계좌 확인."),
    ("name_bank_collision", "[[신한솔|person]] 대리가 신한은행 제휴 업무를 담당함."),
    ("name_word_collision", "[[이상한|person]] 고객이 방문함. [[고민수|person]] 과장과 [[문제원|person]] 대리가 응대함."),
    ("name_word_collision", "[[장미|person]] 고객님과 [[하늘|person]] 씨가 함께 방문함."),
    ("decoy_words", "이상한 거래가 감지되어 고객에게 문자를 보냄. 고민 끝에 한도를 줄이기로 함. "
                    "문제 원인은 하나 더 있음. 우리 팀 장미 화분 교체 예정."),
    # ---------------- v5 추가: 여러 개인정보가 섞인 실제 문서 형태
    ("doc_email",
     "제목: [민원 회신] 카드 이용 내역 확인 요청\n"
     "수신: [[박서진|person]] 고객님\n"
     "참조: 고객보호팀 [[이 도윤|person]] 과장\n\n"
     "[[박서진|person]] 고객님, 안녕하세요. 새봄카드 고객보호팀 [[한예슬|person]]입니다.\n"
     "문의하신 카드([[5412 7531 0098 2299|card_number]])의 해외 결제 건은 확인 결과 정상 승인된 거래였습니다.\n"
     "추가 문의는 제 직통번호 [[031-8017-2290|phone_number]] 또는 [[ys.han@saebomcard.example.com|email]]으로 연락 주세요.\n\n"
     "[[한예슬|person]] 드림"),
    ("doc_complaint",
     "[민원 처리 기록]\n"
     "민원인: [[응우옌 티 마이|person]] (외국인등록번호 [[880315-6123456|resident_registration_number]])\n"
     "연락처: [[+82 10 5521 0399|phone_number]]\n"
     "주소: [[경기 안산시 단원구 원곡동 789-12|address]]\n\n"
     "민원 내용: 토스뱅크 계좌([[010-5521-0399|account_number]])로 받은 환불금이 두 번 출금되었다고 주장함.\n"
     "처리: 카드([[5412753100982277|card_number]]) 결제 내역과 대조 후 [[응우옌 티 마이|person]] 고객에게 결과 안내 예정."),
    ("doc_meeting",
     "[주간 회의록]\n"
     "일시: 2026-10-20 10:00\n"
     "참석: [[김솔|person]], [[남궁 하늘|person]], [[Daniel Park|person]], [[정다운|person]]\n"
     "불참: [[이우리|person]] (연차)\n\n"
     "1. [[남궁 하늘|person]] 차장이 지난 분기 민원 처리 현황을 공유함.\n"
     "2. [[Daniel Park|person]] 매니저는 시스템 이관 일정을 [[010.4412.0087|phone_number]]로 따로 안내하기로 함.\n"
     "3. [[다운|person]] 씨가 회의록을 정리하기로 함."),
    ("doc_messenger",
     "[사내 메신저] 2026-10-21\n"
     "[[오세린|person]]: [[태오|person]]야, 고객 [[도하윤|person]]님 서류 받았어?\n"
     "[[강태오|person]]: 네, [[하윤|person]] 고객님 서류는 오늘 오전에 받았습니다.\n"
     "[[오세린|person]]: 고마워. 주소는 [[대구 수성구 범어동 123|address]]로 등록해 줘."),
]

_MARK = re.compile(r"\[\[(.+?)\|(\w+)\]\]")


def parse(marked):
    text, ents, last, pos = [], [], 0, 0
    for m in _MARK.finditer(marked):
        text.append(marked[last:m.start()])
        pos += m.start() - last
        value, label = m.group(1), m.group(2)
        assert label in LABELS, label
        ents.append({"label": label, "text": value, "start": pos, "end": pos + len(value)})
        text.append(value)
        pos += len(value)
        last = m.end()
    text.append(marked[last:])
    return "".join(text), ents


def main():
    out = DATA / "edge_cases.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i, (tag, marked) in enumerate(CASES, 1):
            text, ents = parse(marked)
            doc = {"id": f"E{i:03d}", "split": "edge", "doc_type": tag.split("_")[0], "template": tag,
                   "text": text, "entities": ents, "hard_cases": [tag]}
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"엣지 케이스 {len(CASES)}건, 정답 {sum(len(parse(c)[1]) for _, c in CASES)}개 → {out.name}")


if __name__ == "__main__":
    main()
