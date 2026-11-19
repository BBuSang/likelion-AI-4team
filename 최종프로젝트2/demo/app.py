"""금융권 법령 RAG 데모 (기획안 5-7 v0.39) — 화면은 answer() 하나만 부른다 (평가와 같은 함수)

    python -m streamlit run 최종프로젝트2/demo/app.py --server.port 8504
"""
import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from retriever.answer import MIN_SCORE, answer, warmup  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from answer_format import render_answer  # noqa: E402  (화면 표시만: 조문별 문단 · 줄바꿈·목록 · 핵심 구절 빨간색)

LEVELS = ["법률", "시행령", "시행규칙", "행정규칙"]
PARA = chr(10) * 2  # 마크다운 문단 나눔
MANIFEST = ROOT / "data" / "work" / "05_embed" / "manifest.json"
EXAMPLES = [  # 기획안 2-1 예시 질문 (pipeline/colab_bundle_04.py 의 QUESTIONS 와 같음)
    "은행이 고객에게 금융상품을 판매할 때 설명해야 하는 의무는 어떤 법에 있어?",
    "금융회사가 고객 신용정보를 제3자에게 제공하려면 어떤 조건이 필요해?",
    "전자금융사고가 나면 손해배상 책임은 누가 져?",
    "예금자보호 한도는 1인당 얼마야?",
    "대부업자가 받을 수 있는 최고 이자율은?",
    "고객 금융거래 정보를 수사기관에 줄 수 있는 경우는?",
    "자금세탁이 의심되는 거래는 어디에 언제까지 보고해야 해?",
    "보험 청약을 철회할 수 있는 기간은?",
    "개인정보가 유출되면 정보주체에게 언제 알려야 해?",
    "가상자산사업자는 이용자 예치금을 어떻게 보관해야 해?",
    "은행이 대주주에게 해 줄 수 있는 신용공여 한도는?",
    "신용카드 회원을 모집할 때 금지되는 행위는?",
]

st.set_page_config(page_title="금융권 법령 RAG", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner="검색 모델을 올리는 중입니다 (처음 한 번, 10~20초)…")
def ready():
    return warmup()


@st.cache_data
def examples():
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))["questions"]
    except (OSError, KeyError, ValueError):
        return EXAMPLES  # data/ 가 없는 컴퓨터 (저장소에서 받은 경우)


load_sec = ready()

st.title("⚖️ 금융권 법령 RAG")
st.caption("금융 관련 법률·시행령·시행규칙·감독규정(현행) 380개에서 근거 조문을 찾아 답합니다. "
           "답변은 찾은 조문만 근거로 쓰며, 법률 자문이 아닙니다.")

if "q" not in st.session_state:
    st.session_state.q = ""

with st.sidebar:
    st.subheader("예시 질문")
    st.caption("누르면 바로 물어봅니다")
    for i, e in enumerate(examples()):
        if st.button(e, key=f"ex{i}", use_container_width=True):
            st.session_state.q = e
            st.session_state.run = True

with st.form("ask"):
    q = st.text_input("질문", key="q", placeholder="예: 예금자보호 한도는 1인당 얼마야?")
    levels = st.multiselect("법령 구분 필터 (선택, 비우면 전체)", LEVELS, default=[], placeholder="전체")
    go = st.form_submit_button("질문하기", type="primary")

if go or st.session_state.pop("run", False):
    if not st.session_state.q.strip():
        st.warning("질문을 입력해 주세요.")
    else:
        with st.spinner("근거 조문을 찾고 답변을 쓰는 중…"):
            r = answer(st.session_state.q, levels=levels or None)
        st.session_state.last = r

r = st.session_state.get("last")
if r:
    st.subheader("답변")
    if r["error"]:
        st.error(r["answer"])
    elif not r["found"]:
        st.info(r["answer"])
    else:
        st.success(render_answer(r["answer"], r.get("key_phrases", [])))
    t = r["timing"]
    st.caption(f"전체 {t['total']:.2f}초 · 검색 {t['search']:.2f}초 · LLM {t['llm']:.2f}초"
               + (" · 검색 점수가 기준보다 낮아 LLM을 부르지 않음" if r.get("below_min_score") else ""))

    if r["sources"]:
        by_article = any(s.get("explain") for s in r["sources"])  # 5-8-4: 조문별 설명이 있으면 조문별로
        st.subheader(f"{'조문별 답변' if by_article else '근거 조문'} ({len(r['sources'])})")
        for i, s in enumerate(r["sources"], 1):
            where = f"검색 {s['rank']}위" if s["rank"] else ""
            if s.get("linked_from"):
                where += f" · {s['link_kind']} 연결로 추가"
            with st.container(border=True):
                st.markdown(f"**{i}. {s['cite'] or s['title']}**")
                if s.get("explain"):
                    st.markdown(s["explain"])
                st.caption(f"{s['law_level']} · 시행 {s['effective_date'] or '-'} · 점수 {s['score']:.3f} · {where}")
                st.markdown(f"[국가법령정보센터 원문 보기]({s['url']})")
                with st.expander("조문 본문"):
                    st.text(s["text"])

    with st.expander("디버그: LLM에 넘긴 후보 · 시간 · 토큰"):
        st.write(f"후보 {len(r['candidates'])}개 · 답변불가 기준점수 {MIN_SCORE} · 모델 준비 {load_sec}초 (서버 시작 때 한 번)")
        picked = {s["no"] for s in r["sources"]}
        st.dataframe([{"번호": c["no"], "근거": "✔" if c["no"] in picked else "", "조문": c["title"], "점수": c["score"],
                       "검색 순위": c["rank"], "연결": "연결" if c.get("linked_from") else "", "넘긴 글": c["text_mode"]}
                      for c in r["candidates"]], hide_index=True, use_container_width=True)
        st.json({"timing": r["timing"], "usage": r["usage"], "error": r["error"]}, expanded=False)
