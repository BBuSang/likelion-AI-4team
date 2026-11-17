"""structured output용 추출 schema (문서 기준) — 2026-11-09 전처리 2-5~2-7 반영판
- 추출 단위는 '문서 1건'. 상품 ↔ 문서는 N:M이라 상품명·분류는 추출하지 않는다 (products.csv / product_docs.csv).
- LLM이 하지 않는 것 (코드로 이미 만듦):
    섹션 나누기·정형 문구 표시 (2-5·2-7 → data/sections), 심의필 번호·유효기간·기준일 (2-4 → data/meta/doc_meta.csv)
- LLM 입력: 정형 문구가 아닌 섹션의 정리 본문(2-6 text)을 `[S{seq}] 섹션 경로 (p.쪽)` 머리와 함께 + 문서 제목·종류·연결 상품 분류
- 원문에 없는 값은 반드시 null. 추론·계산 금지. 빈칸 서식 '(   )%'는 값이 아님 → null.
- 모든 금리·수수료·혜택에는 근거 섹션 번호(section_seq)와 원문 그대로의 source_text → 코드로 '그 섹션에 정말 있는 문장인지' 검증
"""
from typing import Literal, Optional
from pydantic import BaseModel, Field

SourceText = Field(description="근거 원문을 입력 그대로 복사 (요약·수정 금지, 1~2문장)")
SectionSeq = Field(description="근거가 있는 섹션 번호 [S숫자]의 숫자")


class Rate(BaseModel):
    rate_kind: Literal[
        "base",               # 기본이자율
        "preferential",       # 우대이자율 (조건별로 따로)
        "max",                # 최고이자율
        "early_termination",  # 중도해지이자율
        "after_maturity",     # 만기후이자율
        "recruit_period",     # ELD 모집기간이율
        "index_linked",       # ELD 지수연동이율 (구간·최고 등)
        "loan_base",          # 대출 기준금리
        "loan_spread",        # 대출 가산금리
        "loan_preferential",  # 대출 우대(감면)금리
        "loan_final",         # 대출 최종(적용)금리
        "overdue",            # 연체이자율·연체가산이자율
    ]
    rate_min: Optional[float] = None         # 단일값이면 min = max
    rate_max: Optional[float] = None
    term_min_m: Optional[int] = None         # 적용 기간 구간(개월)
    term_max_m: Optional[int] = None
    amount_min: Optional[float] = None       # 적용 금액 구간 — 원화는 원, 외화는 그 통화 단위 (currency)
    amount_max: Optional[float] = None
    currency: Optional[str] = None           # 외화예금이면 USD 등, 원화는 null
    rate_type: Optional[Literal["fixed", "variable"]] = None
    pre_tax: Optional[bool] = None           # 원문에 '세전'이면 true, '세후'면 false, 없으면 null
    condition: Optional[str] = None          # 적용 조건 (우대 요건, '1가지 충족' / '모두 충족' 등)
    section_seq: int = SectionSeq
    source_text: str = SourceText


class Fee(BaseModel):
    fee_kind: Literal[
        "early_repayment",      # 대출 중도상환해약금
        "early_termination",    # 중도해지수수료 (ELD 등)
        "spread",               # 금·은 매매 스프레드
        "physical_withdrawal",  # 금·은 실물(골드바·실버바) 인출 수수료에만
        "cash_handling",        # 외화 지폐·주화 현찰 수수료
        "stamp_tax",            # 인지세 (고객 부담분)
        "transfer",             # 이체·송금 수수료
        "limit",                # 한도약정·한도미사용 수수료
        "etc",
    ]
    amount: Optional[float] = None           # 정액(원)
    rate_pct: Optional[float] = None         # 정률(%)
    condition: Optional[str] = None
    section_seq: int = SectionSeq
    source_text: str = SourceText


class Benefit(BaseModel):
    benefit_kind: Literal["rate_bonus", "fee_waiver", "discount", "point", "cashback", "service"]
    requirement: str                         # 충족 조건 원문 요약 (예: 급여이체 월 50만원 이상)
    benefit: str                             # 받는 혜택 (예: 연 0.3%p)
    value_num: Optional[float] = None
    value_unit: Optional[str] = None         # %p / % / 원 / 회
    section_seq: int = SectionSeq
    source_text: str = SourceText


# ---- 유형별 detail (document.detail JSONB) — 입력의 detail_kind에 맞는 것 하나만 ----
class DepositDetail(BaseModel):          # 저축상품: 입출금자유·거치식·적립식·주택청약·외화예금
    account_type: Optional[str] = None   # 예금과목 (저축예금, 정기예금, 정기적금 …)
    term_min_m: Optional[int] = None
    term_max_m: Optional[int] = None
    amount_min: Optional[float] = None   # 최소 가입·납입 금액
    amount_max: Optional[float] = None   # 최대 가입·월 납입 한도
    payment_method: Optional[Literal["lump_sum", "fixed_installment", "free_installment", "demand"]] = None
    interest_payment: Optional[str] = None   # 만기일시지급식 / 월이자지급식 …
    currencies: list[str] = []           # 외화예금 가입 통화
    one_per_person: Optional[bool] = None
    tax_free_eligible: Optional[bool] = None # 비과세종합저축 가능 여부
    partial_withdrawal: Optional[str] = None # 일부해지 (예: 만기 전 2회 가능 / 불가)
    transferable: Optional[bool] = None      # 양도 가능
    collateral_allowed: Optional[bool] = None
    auto_renewal: Optional[str] = None       # 재예치·자동갱신


class LoanDetail(BaseModel):             # 대출상품 (공통 설명서는 빈칸 서식이 많음 → 대부분 null이 정상)
    loan_type: Optional[str] = None      # 신용 / 주택담보 / 전세 / 기업 …
    limit_max: Optional[float] = None
    limit_rule: Optional[str] = None     # "연소득의 최대 OO%" 같은 원문
    term_min_m: Optional[int] = None
    term_max_m: Optional[int] = None
    repayment_methods: list[str] = []    # 만기일시 / 원리금균등 / 원금균등 / 한도거래 …
    collateral: Optional[str] = None
    guarantee: Optional[str] = None      # 보증기관
    base_rate_index: Optional[str] = None    # 기준금리 종류 (COFIX, 금융채 …)
    rate_cut_request: Optional[bool] = None  # 금리인하요구권 대상
    required_docs: list[str] = []


class InvestDetail(BaseModel):           # 기타상품(금·은 계좌, 파생결합증권)·복합금융상품(ELD)
    product_kind: Optional[Literal["gold", "silver", "eld", "etc"]] = None
    underlying: Optional[str] = None     # 기초자산 (국제금가격, KOSPI200 …)
    principal_guaranteed: Optional[bool] = None   # 원금보장 여부 (원문 그대로)
    risk_grade: Optional[int] = None     # 위험등급 숫자
    trade_unit: Optional[str] = None     # 거래단위 (0.01g …)
    maturity: Optional[str] = None       # 만기 (없음 / 1년 …)
    participation_rate: Optional[float] = None    # ELD 참여율(%)
    recruit_period: Optional[str] = None # ELD 모집기간


class DocumentExtraction(BaseModel):
    """문서 1건 → schema.sql의 document(요약 컬럼·detail) + doc_rate / doc_fee / doc_benefit.
    doc_id·제목·종류·심의필·기준일·섹션은 코드가 채우므로 여기 없음."""
    doc_title_in_text: Optional[str] = None  # 본문에 적힌 상품/문서 이름 (documents.csv 제목과 대조용)
    summary: Optional[str] = None            # 상품 특징 1~2문장 (원문 표현 위주)
    target_customer: Optional[str] = None    # 가입·신청 대상 원문
    join_channels: list[str] = []            # 영업점 / 신한 SOL뱅크 / 인터넷뱅킹 …
    deposit_protected: Optional[bool] = None # 예금자보호 대상이면 true, '보호되지 않음'이면 false
    risk_grade: Optional[int] = None         # 투자설명서·ELD의 위험등급
    detail: Optional[DepositDetail | LoanDetail | InvestDetail] = None
    rates: list[Rate] = []
    fees: list[Fee] = []
    benefits: list[Benefit] = []
    notes: list[str] = []                    # 표가 깨져 값을 확정할 수 없던 곳 등 사람이 볼 것 (짧게)
