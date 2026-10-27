"""방식 3: LLM 탐지기 (OpenAI API).

LLM 은 글자 위치를 정확히 세지 못하므로 '값 + 라벨'만 받고, 위치는 원문에서 직접 찾는다.
가상 데이터이므로 원문 전송이 허용된 실험이다. 실제 고객 문서라면 외부 API 로 보내면 안 된다.
"""
import json
import os

from common import LABELS, find_all

PROMPT = """당신은 금융사 사내 문서에서 개인정보를 찾는 탐지기입니다.

[라벨] 아래 7개만 사용
- person: 사람 이름. 고객뿐 아니라 상담원·담당자·작성자·참석자·팀장 등 직원과 협력사 직원 이름도 모두 포함
  (성+이름, 띄어 쓴 이름 '윤 선영', 이름만 쓴 '소율'도 포함)
- phone_number: 휴대전화·유선전화 번호 (하이픈·점·공백·붙여쓰기 모든 형식).
  직원 메일 서명·연락처의 직통번호(02-xxxx-xxxx 등)와, 정보 변경 기록의 '변경 전'·'변경 후' 번호도 모두 포함
- email: 이메일 주소
- address: 도로명 주소 (동·호수·층까지 포함)
- resident_registration_number: 주민등록번호
- account_number: 은행 계좌번호
- card_number: 카드번호

[제외] 개인정보가 아님
- 성+직함만 있는 표현 (예: '김 과장', '허 부장님')
- 회사 대표번호(1588-xxxx, 1599-xxxx 처럼 15xx·16xx·18xx 로 시작), '내선 5448' 처럼 '내선' 뒤 4자리, 민원번호·접수번호
  (0으로 시작하는 전화번호는 회사 소속 직원 번호여도 개인정보로 봄)
- 회사명·은행명·부서명·상품명

[규칙]
- 문서에 적힌 글자 그대로 추출 (띄어쓰기·하이픈 유지)
- 이름 뒤 조사·호칭(은/는/이/가/님/씨/아 등)과 직함은 빼고 이름만
- 같은 값이 여러 번 나와도 한 번만 적어도 됨. 단 표기가 다르면('윤 선영'과 '윤선영') 각각 적기
- 빠뜨리는 것이 더 위험하므로, 애매하면 포함
- 없으면 빈 목록

JSON 으로만 답하세요: {"entities": [{"text": "...", "label": "..."}]}"""


class LLMDetector:
    def __init__(self, model: str | None = None, temperature: float = 0.0):
        from openai import OpenAI
        self.client = OpenAI()  # OPENAI_API_KEY 환경변수 사용 (common.load_env 가 .env 에서 등록)
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4.1-mini")
        self.temperature = temperature
        self.name = "llm"
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def detect(self, text: str) -> list:
        res = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": PROMPT},
                      {"role": "user", "content": f"문서:\n{text}"}],
        )
        self.usage["prompt_tokens"] += res.usage.prompt_tokens
        self.usage["completion_tokens"] += res.usage.completion_tokens
        try:
            items = json.loads(res.choices[0].message.content).get("entities", [])
        except (json.JSONDecodeError, AttributeError):
            return []
        ents = []
        for it in items:
            value, label = str(it.get("text", "")).strip(), it.get("label")
            if value and label in LABELS:
                # LLM 이 한 번만 적어도 원문의 모든 등장 위치를 가린다.
                # 이름은 '윤 선영'만 답해도 '윤선영'까지 찾도록 띄어쓰기를 무시하고 찾는다
                hits = find_all(text, value, ignore_space=(label == "person"))
                ents += [{"label": label, "start": s, "end": e} for s, e in hits]
        return ents
