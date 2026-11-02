"""실전 2 챗봇 데모 (Streamlit).

    streamlit run app.py          (실전프로젝트2 폴더에서, DB 가 떠 있고 적재가 끝난 상태)

화면
  - 대화: 질문 → 가명처리 → 검색 → LLM 답변(토큰 그대로) → 출력 검사
  - 답변마다 '처리 과정': 가명처리된 질의, 질의 토큰, 검색 문서와 점수(토큰 점수 + 의미 유사도), 출력 검사 결과, 단계별 시간
  - 사이드바: LLM 모델, 참고 문서 수, 검색 방식(토큰+의미 / 의미만), 예시 질문, DB 현황
  - 답변마다 '🔓 실명 보기' 토글: 켜면 답변·참고 문서의 토큰을 원래 값으로 바꿔 보여줌 (starter/restore.py)
검색·답변은 검색 계정(rag_user)만 씁니다. 실명 복원만 매핑 계정(vault_user)으로 하고 vault.restore_log 에 남깁니다.
권한 확인은 아직 없음 (종합에서 역할별로 추가).
"""
import re
import time

import streamlit as st

from starter import chat, config, db, pii_pipeline, restore

st.set_page_config(page_title="가명처리 RAG 챗봇", page_icon="🔒", layout="wide")

# LLM_PROVIDER=openai 로 실행하면 OpenAI 모델 (worklog 17장)
MODELS = (["gpt-4.1", "gpt-4.1-mini"] if config.LLM_PROVIDER == "openai" else ["HCX-007", "HCX-005", "HCX-DASH-002"])
MODEL_HELP = ("gpt-4.1: 기본 (약 1.5초). gpt-4.1-mini: 비슷하게 빠르고 저렴, 인용 정밀도는 holdout 에서 조금 낮음" if config.LLM_PROVIDER == "openai"
              else "HCX-007: 정확하지만 8~15초. HCX-005: 1~3초로 빠르지만 내용을 놓치기도 함")
EXAMPLES = [
    "김서연 고객 한도 상향 심사 어떻게 됐어?",
    "연락처가 010-0791-2924인 건 찾아줘",
    "보이스피싱 피해 접수 기록 알려줘",
    "대구 수성구 고객 문서 보여줘",
    "고객센터 김민준 직원이 처리한 건 있어?",
]
_TOKEN = re.compile(r"`?(<[A-Za-z_]+_[0-9A-Za-z]+>)`?")


@st.cache_resource(show_spinner="탐지 모델 불러오는 중… (처음 한 번, 1~2분)")
def warm_up():
    pii_pipeline.detect("준비 010-0000-0000")
    return True


@st.cache_data(ttl=60)
def db_stats():
    with db.rag_conn() as conn:
        rows = conn.execute("""SELECT
            (SELECT count(*) FROM rag.chunks), (SELECT count(DISTINCT doc_id) FROM rag.chunks),
            (SELECT count(*) FROM rag.persons), (SELECT count(*) FROM rag.persons WHERE status = 'resolved')""").fetchone()
    return dict(zip(["chunks", "docs", "persons", "resolved"], rows))


def show_tokens(text):
    """토큰을 코드 글씨로 (마크다운에서 <…> 가 HTML 태그로 사라지지 않게)."""
    text = _TOKEN.sub(lambda m: f"`{m.group(1)}`", text)
    return text.replace(f"`{chat.UNVERIFIED}`", chat.UNVERIFIED)


def restored(res, qid):
    """답변·참고 문서를 한 번에 복원해 res 에 저장 (토글을 다시 켜도 vault 조회·기록을 반복하지 않게)."""
    if "restored" not in res:
        texts = [res["answer"] or ""] + [d["masked_text"] for d in res["docs"]]
        _, info = restore.restore("\n".join(texts), actor="demo", purpose=qid)
        res["restored"] = info
    return res["restored"]


def reveal(text, info):
    """토큰 → 원래 값 (굵게 표시해서 무엇이 복원됐는지 보이게)."""
    for t, v in info["restored"].items():
        text = text.replace(f"`{t}`", t).replace(t, f"**{v}**")
    return text


def render_answer(res, idx):
    on = st.toggle("🔓 실명 보기", key=f"reveal_{idx}",
                   help="가려진 값(토큰)을 원래 값으로 보여줍니다. 복원 기록이 vault.restore_log 에 남습니다.")
    info = restored(res, f"turn{idx}") if on else None
    answer = res["answer"] or "(LLM 키 없음 — 검색 결과만)"
    st.markdown(reveal(answer, info) if on else show_tokens(answer))
    if on:
        n = len(info["restored"])
        st.caption(f"복원한 값 {n}개" + (f" · 매핑에 없어 복원 안 한 토큰 {len(info['unknown'])}개"
                                          if info["unknown"] else "")
                   + (" · '확인 안 된 토큰'은 복원하지 않음" if chat.UNVERIFIED in answer else ""))
    render_details(res, info)


def render_details(res, info=None):
    if res["docs"] and res["docs"][0].get("fallback"):
        st.info("임베딩 API 사용 한도(분당 60회)에 걸려 이번 답변은 의미 검색 없이 토큰 조건(이름·번호·주소)만으로 찾았어요.")
    if not res["docs"]:
        st.info("찾은 문서가 없어요. (임베딩 API 한도에 걸렸고 질문에 이름·번호·주소도 없으면 찾을 수 없음 — 잠시 뒤 다시 시도)")
    with st.expander("처리 과정 보기", expanded=False):
        st.markdown(f"**가명처리된 질의** (API·검색에는 이것만 사용)\n\n{show_tokens(res['masked_query'])}")
        st.caption("질의 토큰: " + (", ".join(res["query_tokens"]) if res["query_tokens"] else "없음 → 의미 검색만"))
        st.markdown("**검색 결과** (점수 = 토큰 점수 + 의미 유사도)")
        st.dataframe(
            [{"순위": i + 1, "문서": d["doc_id"], "점수": round(d["score"], 3),
              "토큰 점수": round(d.get("token_score", 0), 2), "의미 유사도": round(d.get("sim", 0), 3),
              "내용 (가명처리)": d["masked_text"][:80].replace("\n", " ") + "…"} for i, d in enumerate(res["docs"])],
            hide_index=True, use_container_width=True)
        for d in res["docs"]:
            with st.popover(f"{d['doc_id']} 전체 보기" + (" (실명)" if info else "")):
                if info:
                    st.markdown(reveal(d["masked_text"], info).replace("\n", "  \n"))
                else:
                    st.text(d["masked_text"])
        chk = res.get("check")
        if chk:
            if chk["unverified_tokens"]:
                st.warning("LLM 이 문서에 없는 토큰을 썼어요 → '확인 안 된 토큰'으로 바꿈: "
                           + ", ".join(chk["unverified_tokens"]))
            if chk["raw_pii"]:
                st.warning("답변에 원문 개인정보가 있어서 가렸어요: "
                           + ", ".join(f"{p['label']}" for p in chk["raw_pii"]))
            if not chk["unverified_tokens"] and not chk["raw_pii"]:
                st.success("출력 검사 통과: 답변의 토큰이 모두 문서에 있던 것이고, 원문 개인정보 없음")
        t = res["timing"]
        st.caption("시간: " + " · ".join(f"{k} {v:.1f}s" for k, v in t.items()) + f" · 모델 {res['model']}")


# ---------------------------------------------------------------- 사이드바
with st.sidebar:
    st.header("설정")
    model = st.selectbox("LLM 모델", MODELS, index=0, help=MODEL_HELP)
    k = st.slider("참고 문서 수", 1, 5, 3)
    mode_label = st.radio("검색 방식", ["토큰 조건 + 의미 (v2b)", "의미 검색만 (v1)"])
    mode = "hybrid" if mode_label.startswith("토큰") else "semantic"
    st.divider()
    st.subheader("예시 질문")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    st.divider()
    try:
        s = db_stats()
        st.caption(f"DB: 문서 {s['docs']}건 · 청크 {s['chunks']}개 · 추정 인물 {s['persons']}명 "
                   f"(확정 {s['resolved']}) · 임베딩 {config.EMBED_MODEL}")
    except Exception as e:  # DB 가 꺼져 있을 때
        st.error(f"DB 연결 실패: {e}")
    st.caption("검색·답변은 rag_user · 실명 보기만 vault_user (기록 남김, 권한 확인은 종합에서) · "
               "LLM·임베딩 API 에는 가명처리된 텍스트만 전송")
    if st.button("대화 지우기"):
        st.session_state.history = []

# ---------------------------------------------------------------- 대화
st.title("🔒 가명처리 RAG 챗봇")
st.caption("새봄카드 사내 문서(가상) 187건 · 개인정보는 토큰으로만 저장·검색·답변됩니다")

warm_up()
st.session_state.setdefault("history", [])

for i, turn in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        render_answer(turn["res"], i)

query = st.chat_input("질문을 입력하세요 (예: 윤선영 고객 상담 내역)")
query = query or st.session_state.pop("pending", None)
if query:
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        with st.spinner("가명처리 → 검색 → 답변 생성 중…"):
            t = time.perf_counter()
            try:
                # 임베딩 API 는 분당 60회 한도(키 공유 시 금방 참) → 대화에선 3번만 재시도, 안 되면 토큰 조건만으로 검색
                res = chat.answer(query, k=k, model=model, mode=mode, embed_retries=3)
            except Exception as e:
                st.error(f"오류: {type(e).__name__}: {e}")
                st.stop()
            res["timing"]["total"] = time.perf_counter() - t
        render_answer(res, len(st.session_state.history))
    st.session_state.history.append({"query": query, "res": res})
