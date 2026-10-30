"""챗봇 데모 (Streamlit 기본 디자인): 질문 → 가명처리 → 검색 → HCX-007 답변을 한 화면에 보여 줍니다.

    python -X utf8 -m streamlit run app.py        (실전프로젝트2 폴더에서, DB를 띄운 상태)

- 사용자가 쓴 질문 아래에 가명처리된 질문을 보여 줍니다 (외부로 나가는 것은 이 줄뿐).
- 답변은 HCX-007이 쓰는 대로 흘려 보여 주고, 끝나면 답변 아래 접힌 상자에 근거 문서를 문서별 탭으로 붙입니다.
- 답변은 토큰 그대로 표시합니다. 실명 복원·권한은 종합 단계 과제라 vault 에 접근하지 않습니다.
"""
import re
import time

import streamlit as st

from starter import config, pii_pipeline
from starter.chat import finish, prepare, stream

SCENARIOS = [  # 시연 시나리오 5개 (worklog 13·23번)
    ("윤선영 고객 상담 내역 알려줘", "실명 없이 이름으로 찾기"),
    ("윤 선영 님 한도 상향 심사 결과 알려줘", "띄어 쓴 이름도 같은 사람"),
    ("01009335186 번호 쓰는 고객 관련 기록 찾아줘", "번호 형식이 달라도 찾기"),
    ("김민준 고객 관련 문서 보여줘", "동명이인 한계"),
    ("보이스피싱 피해 접수 기록 알려줘", "개인정보 없는 질문"),
]
# 문서 수·검색 방식은 평가로 정한 값으로 고정 (k=3: Hit 상승이 3에서 꺾임, 결합 검색: E3 최고 점수)
K, MODE = 3, "hybrid"
NOT_FOUND = "관련 문서를 찾지 못했습니다"
CITE_RE = re.compile(r"\[(H\d{4})\]")
REDRAW_SEC = 0.15  # 스트리밍 중 화면을 다시 그리는 간격

st.set_page_config(page_title="개인정보 AI 챗봇", layout="wide")


@st.cache_resource(show_spinner="탐지기를 불러오는 중입니다 (처음 한 번, 약 25초)")
def load_detector():
    pii_pipeline._get_detector()
    return True


# ---------------------------------------------------------------- 표시 도우미
def md_tokens(text):
    """토큰 <PERSON_…>이 마크다운에서 HTML 태그로 사라지지 않게 코드 표기로 감싸고, [H0001]은 굵게."""
    out = pii_pipeline.TOKEN_PATTERN.sub(lambda m: f"`{m.group()}`", text)
    return CITE_RE.sub(r"**[\1]**", out)


def meta_text(turn):
    parts = [f'근거 문서 {len((turn.get("res") or {}).get("docs", []))}건']
    res = turn.get("res") or {}
    if res.get("search_sec") is not None:
        parts.append(f'검색 {res["search_sec"]:.1f}초')
    if res.get("llm_sec") is not None:
        parts.append(f'답변 {res["llm_sec"]:.1f}초')
    return " · ".join(parts)


def check_badges(res):
    out = [f":green-badge[인용 문서 {len(res['cited_docs'])}건]" if res["cited_docs"] else ":orange-badge[인용 없음]"]
    n = len(res["unknown_tokens"])
    out.append(":green-badge[지어낸 토큰 0]" if n == 0 else f":red-badge[문서에 없는 토큰 {n}]")
    if any(t.startswith("<PERSON_") for t in res["query_tokens"]):
        out.append(":orange-badge[이름 기준 토큰 · 동명이인 구분 안 함]")
    if not res["query_tokens"]:
        out.append(":gray-badge[질문에 개인정보 없음 · 의미 검색]")
    return " ".join(out)


def show_question(turn):
    with st.chat_message("user"):
        st.write(turn["query"])
        if turn.get("res"):
            st.caption("가명처리된 질문: " + md_tokens(turn["res"]["masked_query"]))


def show_answer(turn):
    """기록된 답변을 그립니다 (스트리밍이 끝난 뒤의 모습)."""
    res = turn.get("res")
    st.caption(f"AI 생성 · {config.LLM_MODEL} · {meta_text(turn)}")
    if turn.get("error"):
        st.error(turn["error"])
    elif res["answer"] is None:
        st.info("LLM 키가 없어 검색 결과만 보여 줍니다. 아래 근거 문서를 확인하세요.")
    elif NOT_FOUND in res["answer"] and not res["cited_docs"]:
        st.warning(f"{NOT_FOUND}. 고객 이름이나 번호, 업무 이름(예: 한도 상향, 결제 알림)을 함께 넣어 다시 질문해 보세요.")
    else:
        st.markdown(md_tokens(res["answer"]))
        st.markdown(check_badges(res))
    if res and res["docs"]:
        show_sources(res)


def show_sources(res):
    """답변 아래 접힌 상자: 문서마다 탭 하나 (LLM에 넘긴 가명처리 본문)."""
    with st.expander(f"근거 문서 {len(res['docs'])}건 열어보기"):
        tabs = st.tabs([f"{d['doc_id']} · {d.get('doc_type') or ''}" for d in res["docs"]])
        for tab, d in zip(tabs, res["docs"]):
            with tab:
                tags = [f":green-badge[토큰 일치 {d['token_hits']}]" if d["token_hits"] else ":gray-badge[의미 검색]"]
                if d["doc_id"] in res["cited_docs"]:
                    tags.append(":blue-badge[답변에서 인용]")
                st.markdown(" ".join(tags))
                st.text(d["masked_text"])


def show_turn(no, turn):
    """대화 한 묶음(질문 + 답변 + 근거 문서)을 테두리 상자 하나에 번호를 붙여 그립니다."""
    with st.container(border=True):
        st.markdown(f"##### 질문 {no}")
        show_question(turn)
        with st.chat_message("assistant"):
            show_answer(turn)


def ask(query, k, mode):
    """질문을 바로 보여 주고, 답변을 스트리밍으로 채운 뒤 근거 문서 탭을 붙여 대화 기록에 남깁니다."""
    turn = {"query": query, "k": k, "mode": mode}
    if st.session_state.turns:
        st.divider()
    with st.container(border=True):
        st.markdown(f"##### 질문 {len(st.session_state.turns) + 1}")
        _ask_body(turn, k, mode)
    st.session_state.turns.append(turn)


def _ask_body(turn, k, mode):
    query = turn["query"]
    q_slot = st.empty()
    with q_slot.container():
        show_question(turn)
    a_box = st.chat_message("assistant")
    with a_box:
        slot = st.empty()
    try:
        with a_box, st.spinner("질문을 가명처리하고 문서를 찾는 중입니다"):
            turn["res"] = prepare(query, k, mode)
        with q_slot.container():
            show_question(turn)
        if config.LLM_API_KEY:
            text, thought, last, t0 = "", 0, 0.0, time.perf_counter()
            for kind, v in stream(turn["res"]):
                if kind == "thinking":
                    thought = v
                else:
                    text += v
                if time.perf_counter() - last > REDRAW_SEC:
                    slot.markdown(md_tokens(text) + " ▌" if text else f"_근거 문서를 읽고 답변을 정리하는 중입니다 · 생각 {thought:,}자_")
                    last = time.perf_counter()
            turn["res"] = finish(turn["res"], text, time.perf_counter() - t0)
    except RuntimeError as e:  # API 호출 한도·응답 오류: 원인과 할 일을 담은 메시지 그대로
        turn["error"] = str(e)[:300]
    except Exception as e:  # DB가 꺼져 있는 경우 등
        turn["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    with slot.container():
        show_answer(turn)


# ---------------------------------------------------------------- 화면
st.session_state.setdefault("turns", [])

with st.sidebar:
    st.subheader("시연 시나리오")
    picked = None
    for i, (q, why) in enumerate(SCENARIOS, 1):
        if st.button(f"{i}. {why}", key=f"scn{i}", help=q, use_container_width=True):
            picked = q
    st.info("실명 복원과 권한 확인은 하지 않습니다. 이름이 같은 다른 사람(동명이인)은 같은 토큰이 되어 "
            "구분하지 못합니다. 답변은 사람이 한 번 더 확인해야 합니다.")
    st.caption(f"문서 holdout 187건 · 검색 {config.EMBED_MODEL} · 답변 {config.LLM_MODEL}")
    if st.session_state.turns and st.button("대화 지우기", key="clear"):
        st.session_state.turns = []
        st.rerun()

st.title("개인정보 AI 챗봇")
st.caption("개인정보를 가명처리하여 답변합니다.")
load_detector()  # 제목을 먼저 그린 뒤 탐지기를 불러옵니다

query = st.chat_input("고객 이름, 번호, 업무 내용으로 질문하세요") or picked
if not st.session_state.turns and not query:
    st.info("이렇게 물어보세요\n\n- 윤선영 고객 상담 내역 알려줘\n- 01009335186 번호 쓰는 고객 관련 기록 찾아줘\n"
            "- 보이스피싱 피해 접수 기록 알려줘\n\n실명은 답변에 나오지 않고 `<PERSON_…>` 같은 가명 토큰으로 보입니다.")
for no, t in enumerate(st.session_state.turns, 1):
    if no > 1:
        st.divider()  # 질문 묶음 사이 구분선
    show_turn(no, t)

if query:
    ask(query, K, MODE)
