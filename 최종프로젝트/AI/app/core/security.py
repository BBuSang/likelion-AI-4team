"""스프링 서버만 AI 서버를 부를 수 있게 하는 내부 키 확인.

AI 서버는 내부망에만 열고(포트 공개 안 함), 그래도 한 번 더 막기 위해 X-Service-Key 를 비교한다.
사용자 로그인(JWT)은 스프링이 확인하고, 사용자 정보·권한은 스프링이 요청 본문에 담아 넘긴다 (worklog 4-1-1).
"""
import hmac

from fastapi import Header, HTTPException, status

from app.core import config


def require_service_key(x_service_key: str = Header(default="")) -> None:
    expected = config.AI_SERVICE_KEY
    # 시간 차로 키를 맞히지 못하게 compare_digest 로 비교
    if not expected or not hmac.compare_digest(x_service_key.encode(), expected.encode()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid service key")
