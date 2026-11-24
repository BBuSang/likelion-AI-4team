"""2-5a 표형 설명서의 '행 이름' 사전 — 섹션 분할 기준 (LLM 없음)

근거: data/stage/s3 의 표형 설명서 109개에서 짧은 단독 줄·연속 두 줄을 문서 수로 집계 (2026-11-09).
값(자유적립식, 저축예금, 불가 …)과 표 안의 열 이름(경과기간, 차감률 …)은 행 이름이 아니므로 넣지 않는다.
변형 표기: 공백 무시. '+'는 PDF 줄바꿈으로 쪼개진 연속 줄 (예: '휴면예금및+출연').

출력: data/meta/row_labels.csv (label, group, variants) + 표형 문서별 인식 결과 요약
"""
import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S3, META = ROOT / "data" / "stage" / "s3", ROOT / "data" / "meta"

# label: 대표 이름 / group: 거래조건·이자·해지·권리·ELD / variants: 공백 제거 표기
LABELS = [
    # 거래조건 — 가입
    ("가입대상", "거래조건", ["가입대상", "가입자격"]),
    ("예금과목", "거래조건", ["예금과목"]),
    ("상품유형", "거래조건", ["상품유형"]),
    ("계약기간", "거래조건", ["계약기간", "가입기간", "예금기간"]),
    ("가입금액", "거래조건", ["가입금액", "최저가입금액", "최고가입금액", "최소가입금액", "저축한도"]),
    ("모집", "거래조건", ["모집한도", "모집금액", "모집기간"]),
    ("가입방법", "거래조건", ["가입방법", "거래방법", "신규방법"]),
    ("가입통화", "거래조건", ["가입통화"]),
    # 이자
    ("기본이자율", "이자", ["기본이자율", "이자율", "적용이율"]),
    ("우대이자율", "이자", ["우대이자율", "우대+이자율"]),
    ("우대서비스", "이자", ["우대서비스"]),
    ("최고이자율", "이자", ["최고이자율", "최종이자율"]),
    ("만기후이자율", "이자", ["만기후+이자율", "만기후이자율", "만기후이율"]),
    ("이자지급방법", "이자", ["이자지급방법", "이자의지급방법", "이자의+지급방법", "원금또는이자+지급방법", "원금또는+이자지급방법"]),
    ("이자산출근거", "이자", ["이자산출근거", "만기이자+산출근거", "만기이자산출근거", "이자계산방법"]),
    # 해지
    ("중도해지이자율", "해지", ["중도해지+이자율", "중도해지이자율"]),
    ("중도해지이자 산출근거", "해지", ["중도해지이자+산출근거"]),
    ("중도해지수수료", "해지", ["중도해지수수료"]),
    ("특별중도해지", "해지", ["특별중도해지"]),
    ("계약해지방법", "해지", ["계약해지방법"]),
    ("일부해지", "해지", ["일부해지"]),
    ("재예치", "해지", ["재예치"]),
    ("계약갱신및연장", "해지", ["계약갱신및연장"]),
    ("예금해지시 불이익", "해지", ["예금해지시+불이익", "예금해지시불이익"]),
    # 거래·제한
    ("양도및담보제공", "거래제한", ["양도및+담보제공", "담보제공"]),
    ("자동이체", "거래제한", ["자동이체"]),
    ("거래제한", "거래제한", ["거래제한", "거래중지", "입금제한", "이용제한사항"]),
    ("원금및이자지급제한", "거래제한", ["원금및+이자지급제한", "원금또는이자+지급관련제한"]),
    ("휴면예금및출연", "거래제한", ["휴면예금및+출연", "휴면예금+및출연", "휴면예금및출연"]),
    ("수수료", "거래제한", ["수수료"]),
    # 보호·세제·권리
    ("예금자보호여부", "권리", ["예금자보호여부", "예금자보호+여부"]),
    ("세제혜택", "권리", ["세제혜택"]),
    ("위법계약해지권", "권리", ["위법계약해지권", "위법계약+해지권"]),
    ("자료열람요구권", "권리", ["자료열람+요구권", "자료열람요구권"]),
    ("인용출처", "권리", ["인용출처"]),
    ("기타사항", "권리", ["기타사항", "유의사항"]),
    # 금·은 (골드바 설명서)
    ("상품특징", "금은", ["상품특징"]),
    ("거래방식", "금은", ["거래방식"]),
    ("판매가격", "금은", ["판매가격"]),
    ("스프레드", "금은", ["스프레드"]),
    ("재매입", "금은", ["재매입"]),
    ("부가가치세", "금은", ["부가가치세"]),
    ("실물수수료", "금은", ["실물수수료"]),
    # 별지 연계·제휴서비스 표 (개선 3)
    ("연계제휴 서비스내용", "연계제휴", ["연계ㆍ제휴+서비스내용", "연계·제휴+서비스내용", "연계제휴+서비스내용"]),
    ("연계제휴 이행책임", "연계제휴", ["연계ㆍ제휴서비스+이행책임", "연계·제휴서비스+이행책임", "연계제휴서비스+이행책임"]),
    ("연계제휴 제공기간", "연계제휴", ["연계ㆍ제휴서비스+제공기간", "연계·제휴서비스+제공기간", "연계제휴서비스+제공기간"]),
    ("연계제휴 제공요건", "연계제휴", ["연계ㆍ제휴서비스+제공요건", "연계·제휴서비스+제공요건", "연계제휴서비스+제공요건"]),
    ("연계제휴 변경통지", "연계제휴", ["연계ㆍ제휴서비스+변경내용", "연계·제휴서비스+변경내용", "연계제휴서비스+변경내용"]),
    # ELD (세이프지수연동예금)
    ("수익구조", "ELD", ["수익구조"]),
    ("기초자산", "ELD", ["기초자산"]),
    ("기준지수", "ELD", ["기준지수"]),
    ("만기지수", "ELD", ["만기지수"]),
    ("참여율", "ELD", ["참여율"]),
    ("지수연동이율", "ELD", ["지수연동이율"]),
    ("모집기간이율", "ELD", ["모집기간이율"]),
]
VARIANTS = {v: lab for lab, _, vs in LABELS for v in vs}
TWO = {v for v in VARIANTS if "+" in v}


def nospace(s: str) -> str:
    return re.sub(r"\s", "", s)


# 개선 3: 사전에 없는 행 이름을 잡는 일반 규칙 — 짧은 명사 줄이 행 이름에 흔한 끝말로 끝나면 행 이름으로 본다.
# '기간'·'요건'은 표 안 열 이름(경과기간, 우대요건)에도 쓰여 제외. 값 줄(저축예금, 제한없음 …)은 이 끝말로 끝나지 않음.
GENERIC_SUFFIX = re.compile(r"^[가-힣]{1,8}(제한|금액|방법|한도|사항|발급|전환|대상|자격|과목|유형|혜택|여부|수수료|근거|조건)(등)?$")
# 개선 3: 행 이름과 내용이 한 줄에 붙은 경우 '이자의 지급방법 • 예금의 이자는 …'
INLINE = re.compile(r"^(.{2,14}?)\s*[•:：]\s*\S")


def find_labels(lines: list[str], with_raw: bool = False, generic: bool = False):
    """줄 목록 → [(줄 번호, 대표 이름, 차지한 줄 수[, 원문 표기])]. 두 줄 변형을 먼저 본다.
    generic=True: 사전에 없는 짧은 행 이름(GENERIC_SUFFIX)과 한 줄 붙은 행 이름(INLINE)도 잡는다 (개선 3)"""
    out, i = [], 0

    def add(k, lab, n, raw):
        out.append((k, lab, n, raw) if with_raw else (k, lab, n))

    while i < len(lines):
        a = nospace(lines[i])
        b = nospace(lines[i + 1]) if i + 1 < len(lines) else ""
        if f"{a}+{b}" in TWO:
            add(i, VARIANTS[f"{a}+{b}"], 2, re.sub(r"\s+", " ", f"{lines[i].strip()} {lines[i + 1].strip()}"))
            i += 2
            continue
        if a in VARIANTS:
            add(i, VARIANTS[a], 1, re.sub(r"\s+", " ", lines[i].strip()))
        elif generic and GENERIC_SUFFIX.match(a) and not (out and out[-1][0] + out[-1][2] == i):
            # 다른 행 이름 바로 뒤의 일반 규칙 후보는 하위 열 이름 (예: '변경 내용' / '사전 통지 방법') → 받지 않음
            add(i, a, 1, re.sub(r"\s+", " ", lines[i].strip()))
        elif generic and (m := INLINE.match(lines[i].strip())) and nospace(m.group(1)) in VARIANTS:
            add(i, VARIANTS[nospace(m.group(1))], 1, m.group(1).strip())
        i += 1
    return out


def drop_header_runs(found):
    """행 이름이 '사이에 내용 없이' 바로 이어질 때 (실제 행은 행 이름 다음에 내용 줄이 있다)
    - 3개 이상 이어짐 → 표 안의 열 이름 (예: Npay biz 통장 '기본이자율 / 우대이자율 / 최고 이자율') → 모두 뺌
    - 2개 이어짐     → 묶음 행 + 하위 행 (예: ELD '적용이율 → 모집기간이율', '모집한도 → 모집기간') → 앞의 것만 뺌"""
    keep, i = [], 0
    while i < len(found):
        j = i
        while j + 1 < len(found) and found[j + 1][0] == found[j][0] + found[j][2]:
            j += 1
        run = j - i + 1
        if run == 1:
            keep.append(found[i])
        elif run == 2:
            keep.append(found[j])
        i = j + 1
    return keep


def main() -> None:
    with open(META / "row_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "group", "variants"])
        for lab, grp, vs in LABELS:
            w.writerow([lab, grp, "|".join(vs)])

    survey = {r["doc_id"]: r for r in csv.DictReader(open(META / "doc_survey.csv"))}
    table = [d for d, r in survey.items() if r["format"] == "table"]
    per_doc, freq = {}, Counter()
    for d in table:
        lines = [l.strip() for l in (S3 / f"{d}.txt").read_text().split("\n")
                 if l.strip() and not l.startswith("=== p.")]
        found = find_labels(lines)
        per_doc[d] = found
        freq.update({lab for _, lab, _ in found})

    counts = sorted(len({lab for _, lab, _ in v}) for v in per_doc.values())
    print(f"행 이름 {len(LABELS)}개 (변형 {len(VARIANTS)}개) → data/meta/row_labels.csv")
    print(f"표형 문서 {len(table)}개: 문서당 서로 다른 행 이름 수 최소 {counts[0]} / 중앙 {counts[len(counts)//2]} / 최대 {counts[-1]}")
    low = [(d, survey[d]["title"], len({lab for _, lab, _ in v})) for d, v in per_doc.items()
           if len({lab for _, lab, _ in v}) < 6]
    print("행 이름이 6개 미만인 문서:", low)
    print("행 이름별 등장 문서 수:", ", ".join(f"{k}({v})" for k, v in freq.most_common()))


if __name__ == "__main__":
    main()
