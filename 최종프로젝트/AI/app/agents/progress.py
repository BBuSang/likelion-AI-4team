"""답변 스트리밍의 진행 단계 알림.

화면에 "지금 무엇을 하는 중인지"를 보여 주려고, 흐름 곳곳(분류·계획·검색·사례 판단·답변·재검사)에서 step()/done() 을 부름.
알림 받을 곳은 요청마다 contextvar 로 정함 (use) — 스트리밍이 아닌 /ask 나 평가 스크립트에서는 아무 일도 안 함.
같은 스레드 안에서만 전달됨: 병렬 검색(ThreadPoolExecutor) 안에서는 부르지 말고 흐름 쪽에서 부를 것.

단계 이름 (화면 문구는 프론트 ChatPage STEP_KO):
  protect · classify · plan · search · replan · search2 · judge · relax · answer · check
"""
import contextvars

_cb = contextvars.ContextVar("progress_cb", default=None)


def use(cb):
    """cb(kind, **data) 를 이 흐름의 알림 받을 곳으로. 반환 토큰은 reset 에"""
    return _cb.set(cb)


def reset(token):
    _cb.reset(token)


def _send(kind, **data):
    cb = _cb.get()
    if cb is None:
        return
    try:
        cb(kind, **data)
    except Exception:  # 알림 실패가 답변을 막지 않게
        pass


def step(stage, **data):
    """새 단계 시작 (앞 단계는 끝난 것으로 봄). data 예: sources=[...]"""
    _send("stage", stage=stage, **data)


def done(stage, **data):
    """단계 결과 덧붙이기 (예: search 의 건수 counts={"internal": 3, "law": 5}, classify 의 sources)"""
    _send("step_done", stage=stage, **data)
