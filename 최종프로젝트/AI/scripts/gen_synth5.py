"""동명이인 개선용 synth5 (worklog 35장) — synth4(확인용)와 같은 설정, 시드·문서 종류만 다름. 이 세트로 사람 잇기 규칙을 고침.
(아래는 synth4 설명) 동명이인 확인용 synth4 (worklog 34장): 2주차 생성기(실전프로젝트2/tools/gen_synth2.py)를 설정만 바꿔 씀 (synth3 와 같은 방식).
확인 전용 — 이 세트를 보고 규칙·모델을 고치지 않음.

    python scripts/gen_synth4.py all        (personas / templates / fill / queries)
출력: AI/data/synth4/ (synth4_labeled.jsonl, synth4_persons_gold.jsonl, synth4_retrieval_queries(_gold).jsonl, _work/) — git 제외

synth3 와 다른 점
  - 동명이인 40쌍 (고객↔직원 15, 고객↔고객 15, 직원↔직원 10) — 인물 170명 중 80명
  - 문서 종류 14가지 모두 holdout·synth2·synth3 에 없는 것, 직원 두 명이 나오는 종류를 늘림 (직원↔직원 동명이인이 드러나게)
  - 생성 모델 gpt-6-luna (temperature 를 못 바꾸는 모델이라 호출 부분만 바꿈), 질의는 동명이인 질의 비중을 높임
"""
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1] / "실전프로젝트2" / "tools"))
import gen_synth2 as g  # noqa: E402

from app.core import config  # noqa: E402

g.OUT = ROOT / "data" / "synth5"
g.WORK = g.OUT / "_work"
g.SEED = 20261113
g.MODEL = "gpt-6-luna"
g.NAME, g.DOC_PREFIX, g.Q_PREFIX = "synth5", "V", "VQ"
g.TEMPLATES_PER_TYPE = 4
g.FILLS_PER_TEMPLATE = 8
g.N_PEOPLE = {"customer": 120, "staff": 40, "external": 10}
g.HOMONYMS = {"cust_staff": 15, "cust_cust": 15, "staff_staff": 10}
g.NAME_FORM = (0.2, 0.4)  # 두 번째 언급부터: 띄어 씀 20%, 이름만 20%

g.DOC_TYPES = {
    "카드발급심사": ("new", "신규 카드 발급 신청 심사 메모", ["고객", "직원"]),
    "결제일변경": ("new", "카드 결제일 변경 요청 처리", ["고객"]),
    "가맹점문의": ("new", "고객의 가맹점 결제 내역 문의 응대", ["고객"]),
    "상담예약": ("new", "지점 방문 상담 예약과 담당 배정", ["고객", "직원"]),
    "채권추심위임": ("new", "연체 채권 추심을 외부 업체에 위임한 기록", ["고객", "외부"]),
    "직원평가면담": ("new", "팀장과 팀원의 반기 평가 면담 기록", ["직원"]),
    "당직일지": ("new", "야간·주말 당직 근무 일지와 처리 건", ["직원", "고객"]),
    "고객감사편지": ("new", "고객이 직원 응대에 감사 인사를 보낸 건 정리", ["고객", "직원"]),
    "프로젝트주간보고": ("new", "사내 프로젝트 주간 진행 보고 (담당자별)", ["직원"]),
    "개인회생신고": ("new", "개인회생 신청 고객의 채무 신고 처리", ["고객"]),
    "카드사용정지": ("new", "고객 요청에 따른 카드 일시 정지·해제", ["고객"]),
    "보안교육출석": ("new", "부서별 보안 교육 출석 명단과 미이수자 안내", ["직원"]),
    "외부감사대응": ("new", "외부 감사인 요청 자료 제출 기록", ["직원", "외부"]),
    "우수고객이벤트": ("new", "우수 고객 초청 행사 참석자 관리", ["고객", "직원"]),
}
g.Q_PLAN = [("name_topic", 20), ("person_all", 10), ("value_exact", 10), ("role_agent", 15), ("homonym_role", 25),
            ("hybrid_status", 10)]


def _chat(client, prompt, temperature):
    """gpt-6-luna: temperature 대신 reasoning_effort (config.sampling_params). 원 함수와 같은 재시도·사용량 기록"""
    for attempt in range(4):
        try:
            r = requests.post(config.OPENAI_API_BASE.rstrip("/") + "/chat/completions",
                              headers={"Authorization": f"Bearer {config._openai_key()}"},
                              json={"model": g.MODEL, **config.sampling_params(g.MODEL), "max_completion_tokens": 12000,
                                    "response_format": {"type": "json_object"},
                                    "messages": [{"role": "user", "content": prompt}]}, timeout=300)
            r.raise_for_status()
            body = r.json()
            u = body.get("usage") or {}
            with open(g.WORK / "usage.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), "model": g.MODEL, "in": u.get("prompt_tokens"),
                                    "out": u.get("completion_tokens")}) + "\n")
            return json.loads(body["choices"][0]["message"]["content"])
        except Exception as e:  # 일시 오류·JSON 깨짐
            print("  재시도:", type(e).__name__, str(e)[:120])
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("LLM 호출 실패")


g._chat = _chat
g._client = lambda: None

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    g.main()
