"""모든 에이전트의 베이스 클래스 (참고 레포 python/agents/base.py 에서 가져옴).

에이전트 설계(worklog 3장)가 정해지면 여기를 상속해 만든다.
  보호(가명처리) → 계획(JSON) → 실행(개인정보 RAG / 일반 RAG 3개) → 종합 → 검증(출력 재검사·인용 확인·복원)
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """에이전트 식별 이름"""

    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """에이전트 메인 실행 메서드"""

    @abstractmethod
    def is_loaded(self) -> bool:
        """내부 컴포넌트(모델 등) 로드 완료 여부"""
