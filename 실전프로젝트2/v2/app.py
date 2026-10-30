"""v2 챗봇 데모 (Streamlit 기본 스타일, plan_v2 P7).

    python -X utf8 -m streamlit run v2/app.py --server.port 8502      (실전프로젝트2 폴더, v2 DB 실행 상태)

화면
- 질문 하나 = 테두리 상자 하나 ("질문 1", "질문 2" …), 상자 사이 구분선
- 상자 안: 질문 → 가명처리된 질문 → 답변(문서마다 한 줄) → 점검 배지 → 근거 문서(접힘, 문서별 탭)
- 사이드바: 예시 질문, 화면 설명, 한계
가명 토큰은 코드 표기로 감싸 HTML 태그로 사라지지 않게 하고, 인용 [H0001]은 굵게 표시
"""
import re
import sys
from pathlib import Path

import streamlit as st

V2 = Path(__file__).resolve().parent
sys.path.insert(0, str(V2))
from chat import CITE_RE, TOKEN_RE, Chat  # noqa: E402

EXAMPLES = [
    "윤선영 고객 상담 내역 알려줘",
    "윤 선영 님 한도 상향 심사 결과 알려줘",
    "01009335186 번호 쓰는 고객 관련 기록 찾아줘",
    "김민준 고객 관련 문서 보여줘",
    "직원 김민준 관련 문서 알려줘",
    "대구 수성구 고객 문서 알려줘",
    "남궁나연 고객 아직 처리 안 끝난 불만 있어?",
    "보이스피싱 피해 접수 기록 알려줘",
    "주택담보대출 금리인하요구권 안내 문서 알려줘",
]

st.set_page_config(page_title="개인정보 보호 AI 챗봇", page_icon="🔒", layout="centered")


@st.cache_resource(show_spinner="탐지기와 검색 정보를 불러오는 중입니다 (처음 한 번, 20초 정도)")
def get_bot():
    bot = Chat()
    bot.search.search_text("준비", 1)      # 탐지기 모델을 미리 불러 둠
    return bot


def md(text):
    """문서마다 목록 한 항목: [H0000]으로 시작하는 줄은 항목으로, 이어지는 줄은 같은 항목 안에서 줄바꿈.
    가명 토큰은 코드 표기(HTML 태그로 사라지지 않게), 인용은 굵게."""
    items, gap = [], False
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            gap = True
            continue
        if s.startswith("- "):
            items.append(s)
        elif CITE_RE.match(s):
            items.append("- " + s)
        elif items and not gap:                 # 바로 이어진 줄 = 같은 문서 항목의 둘째 줄
            items[-1] += "  \n  " + s
        else:                                   # 빈 줄 뒤 인용 없는 문장 = 목록 밖 별도 문단
            items.append("\n" + s)
        gap = False
    out = "\n".join(items)
    out = TOKEN_RE.sub(lambda m: f"`{m.group(0)}`", out)
    return CITE_RE.sub(lambda m: f"**[{m.group(1)}]**", out)


def badges(turn):
    c = turn["checks"]
    out = [f":blue-badge[근거 문서 {len(turn['docs'])}건]", f":blue-badge[인용 {len(c['cited_docs'])}건]"]
    out.append(":green-badge[지어낸 토큰 0]" if not c["unknown_tokens"]
               else f":red-badge[지어낸 토큰 {len(c['unknown_tokens'])}]")
    out.append(":green-badge[줄 대응 오류 0]" if not c["row_mismatch"]
               else f":red-badge[줄 대응 오류 {len(c['row_mismatch'])}]")
    if turn["labels"]:
        out.append(":orange-badge[동명이인 구분 표시]")
    if not turn["query_tokens"]:
        out.append(":gray-badge[개인정보 없는 질문 · 의미 검색]")
    return " ".join(out)


def render_docs(turn):
    cited = set(turn["checks"]["cited_docs"])
    bot = get_bot()
    with st.expander(f"근거 문서 {len(turn['docs'])}건 열어보기"):
        tabs = st.tabs([f"{d} · {bot.search.doc_type[d]}" for d in turn["docs"]])
        for tab, d in zip(tabs, turn["docs"]):
            with tab:
                key = turn["keys"][d]
                tags = []
                if d in cited:
                    tags.append(":green-badge[답변에서 인용]")
                if key[0] > 0:
                    tags.append(f":violet-badge[토큰 일치 {key[0]}]")
                tags.append(f":gray-badge[의미 점수 {key[-1]:.2f}]")
                st.markdown(" ".join(tags))
                st.caption("LLM에 전달한 내용 (가명처리됨)")
                st.text(turn["doc_contexts"][d])
                if turn["doc_contexts"][d].endswith("(표는 질문한 사람의 줄만 표시)"):
                    st.caption("문서 전체 (가명처리됨)")
                    st.text("\n".join(bot.search.doc_lines_raw[d]))


def render_turn(i, turn):
    with st.container(border=True):
        st.markdown(f"#### 질문 {i}")
        with st.chat_message("user"):
            st.markdown(turn["query"])
            st.caption("가명처리된 질문: " + TOKEN_RE.sub(lambda m: f"`{m.group(0)}`", turn["masked_query"]))
        with st.chat_message("assistant"):
            st.caption(f"근거 문서 {len(turn['docs'])}건 · 검색 {turn['search_sec']:.1f}초 · 답변 {turn['llm_sec']:.1f}초 · AI 생성")
            if turn["checks"]["not_found"]:
                st.info("관련 문서를 찾지 못했습니다.")
            else:
                st.markdown(md(turn["answer"]))
            st.markdown(badges(turn))
            if turn["checks"]["row_mismatch"]:
                st.warning("다른 사람의 줄 내용을 붙였을 수 있는 문장: " + " / ".join(turn["checks"]["row_mismatch"]))
            render_docs(turn)


def ask(query):
    bot = get_bot()
    with st.status("질문을 가명처리하고 문서를 찾는 중입니다", expanded=False) as status:
        prep = bot.prepare(query)
        status.update(label=f"근거 문서 {len(prep['docs'])}건을 읽고 답변을 작성하는 중입니다")
        try:
            text, llm_sec = bot.generate(prep)
        except RuntimeError as e:
            status.update(label="답변을 만들지 못했습니다", state="error")
            st.error(str(e))
            return None
        checks = bot.check(prep, text)
        bot._log(prep, text, checks, llm_sec, None)
        status.update(label="완료", state="complete")
    return {**prep, "query": query, "answer": text, "checks": checks, "llm_sec": llm_sec}


# ───────────────────────── 화면 ─────────────────────────
st.title("개인정보 보호 AI 챗봇")
st.caption("질문과 문서의 개인정보를 가명 토큰으로 바꾼 뒤 검색하고 답변합니다. 실명·번호는 외부 AI로 보내지 않습니다.")

if "turns" not in st.session_state:
    st.session_state.turns = []

with st.sidebar:
    st.header("예시 질문")
    for q in EXAMPLES:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q
    st.divider()
    if st.button("대화 지우기", use_container_width=True):
        st.session_state.turns = []
        st.rerun()
    st.header("화면 보는 법")
    st.markdown("- `<PERSON_…>` 같은 표기는 가명 토큰입니다. 같은 값은 항상 같은 토큰입니다.\n"
                "- **[H0023]** 은 근거 문서 번호입니다. 답변 아래에서 열어볼 수 있습니다.\n"
                "- 표 문서는 질문한 사람의 줄만 AI에 전달합니다.")
    st.header("한계")
    st.markdown("- 이름만 부른 표기(예: \"시우 님\")는 성까지 쓴 이름과 연결하지 않습니다.\n"
                "- 같은 역할의 동명이인은 번호·부서 단서가 없으면 구분하지 못합니다.\n"
                "- AI 답변은 실행마다 표현이 조금씩 다를 수 있습니다.")

get_bot()
for i, turn in enumerate(st.session_state.turns, 1):
    if i > 1:
        st.divider()
    render_turn(i, turn)

query = st.chat_input("질문을 입력하세요 (예: 윤선영 고객 상담 내역 알려줘)")
query = query or st.session_state.pop("pending", None)
if query:
    if st.session_state.turns:
        st.divider()
    turn = ask(query)
    if turn:
        st.session_state.turns.append(turn)
        st.rerun()
