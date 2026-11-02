"""eval_rag_e2e.py 결과의 LLM 원 답변에 지금의 출력 검사(chat.check_output)를 다시 돌려 오탐·잡음을 센다.
LLM 을 다시 부르지 않으므로 출력 검사 방식만 바꿔 비교할 때 쓴다.

    python tools/recheck_outputs.py outputs/rag_e2e_hcx007.jsonl
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from starter import chat, pii_pipeline  # noqa: E402

# 일부러 원문 개인정보를 섞은 답변 (잡아야 하는 것)
PROBES = [
    ("김은영", "<PERSON_03f0d420> 고객은 추가 한도를 문의했고, 상담은 **김은영** 상담원이 맡았습니다."),
    ("윤 선영", "- **고객**: 윤 선영 (연락 요청)"),
    ("박하린", "담당자 박하린 대리에게 전달되었습니다 [H0012]."),
    ("정태오", "정태오님은 연회비 면제 조건을 문의함."),
    ("시우", "시우 님께 회신 예정"),
    ("010-1234-5678", "고객 연락처는 **010-1234-5678**입니다."),
    ("01098765432", "연락처(01098765432)로 안내"),
    ("kim.min@example.com", "메일 kim.min@example.com 으로 발송"),
    ("110-123-456789", "환불 계좌: 110-123-456789"),
    ("9410 2058 5277 2217", "카드번호 9410 2058 5277 2217 정지"),
    ("900101-1234567", "주민등록번호 900101-1234567 확인"),
    ("서울 송파구 도담길 125", "배송지: 서울 송파구 도담길 125"),
]


def main():
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r.get("raw_answer")]
    hits, bad = [], []
    for r in rows:
        allowed = set(pii_pipeline.TOKEN_PATTERN.findall(r["llm_input"]))
        _, chk = chat.check_output(r["raw_answer"], allowed)
        hits += [(r["qid"], p["label"], p["text"]) for p in chk["raw_pii"]]
        bad += [(r["qid"], t) for t in chk["unverified_tokens"]]
    print(f"답변 {len(rows)}개 → 원문 개인정보로 잡힌 것 {len(hits)}건 (이 답변들엔 실제 유출 0 → 전부 오탐): {hits[:10]}")
    print(f"확인 안 된 토큰 {len(bad)}건: {bad}")
    caught = 0
    for val, txt in PROBES:
        fixed, chk = chat.check_output(txt, {"<PERSON_03f0d420>"})
        ok = val not in fixed
        caught += ok
        if not ok:
            print("  놓침:", val, "→", fixed)
    print(f"일부러 섞은 원문 개인정보: {caught}/{len(PROBES)} 가림")


if __name__ == "__main__":
    main()
