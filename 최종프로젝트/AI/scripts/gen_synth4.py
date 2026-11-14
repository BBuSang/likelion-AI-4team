"""동명이인 확인용 synth4 (worklog 34장): 2주차 생성기(실전프로젝트2/tools/gen_synth2.py)를 설정만 바꿔 씀 (synth3 와 같은 방식).
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

g.OUT = ROOT / "data" / "synth4"
g.WORK = g.OUT / "_work"
g.SEED = 20261112
g.MODEL = "gpt-6-luna"
g.NAME, g.DOC_PREFIX, g.Q_PREFIX = "synth4", "U", "UQ"
g.TEMPLATES_PER_TYPE = 4
g.FILLS_PER_TEMPLATE = 8
g.N_PEOPLE = {"customer": 120, "staff": 40, "external": 10}
g.HOMONYMS = {"cust_staff": 15, "cust_cust": 15, "staff_staff": 10}
g.NAME_FORM = (0.2, 0.4)  # 두 번째 언급부터: 띄어 씀 20%, 이름만 20%

g.DOC_TYPES = {
    "자동이체등록": ("new", "카드대금 자동이체 계좌 등록·변경 신청 처리", ["고객"]),
    "해외출국알림": ("new", "고객 해외 출국 전 카드 해외사용 설정 요청", ["고객"]),
    "포인트소멸안내": ("new", "곧 소멸될 포인트 안내와 고객 문의 응대", ["고객"]),
    "연체안내통화": ("new", "카드대금 연체 고객에게 건 안내 통화 기록", ["고객", "직원"]),
    "부정사용모니터링": ("new", "이상 거래 탐지로 카드 사용을 막고 고객 본인 확인한 기록", ["고객", "직원"]),
    "고객정보변경": ("new", "고객 주소·연락처 변경 요청 처리", ["고객"]),
    "대출금리인하요구": ("new", "고객의 카드론 금리 인하 요구 심사", ["고객", "직원"]),
    "업무인수인계": ("new", "부서 이동 직원이 후임에게 넘긴 담당 고객·업무 목록", ["직원", "고객"]),
    "인사발령공지": ("new", "직원 부서 이동·승진 발령 공지", ["직원"]),
    "회의록": ("new", "팀 주간 회의록 (참석자·안건·담당자)", ["직원"]),
    "민원이관": ("new", "다른 팀으로 넘긴 고객 민원과 이관 사유", ["고객", "직원"]),
    "협력사점검": ("new", "콜센터·보안 협력사 정기 점검 결과", ["직원", "외부"]),
    "경조사지원": ("new", "직원 경조사 지원금 신청과 지급", ["직원"]),
    "VIP상담": ("new", "우수 고객 전담 상담 기록", ["고객", "직원"]),
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
