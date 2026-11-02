"""실전 2 챗봇 데모 (Streamlit).

    streamlit run app.py          (실전프로젝트2 폴더에서, DB 가 떠 있고 적재가 끝난 상태)

화면 (시안: 왼쪽 아이콘 줄 + 대화 목록 패널 + 가운데 대화)
  - 아이콘 줄: 대화 / 성능 요약 / 사용자(역할) / 복원 기록 / 검사 알림, 아래쪽 설정·사용자
  - 대화 목록: 새 채팅, 대화 검색, 오늘·이전 묶음 (이 브라우저 세션에만 저장 — 질문 원문을 파일로 남기지 않음)
  - 대화: 질문 → 가명처리 → 검색 → LLM 답변(스트리밍, 토큰 그대로) → 출력 검사
  - 답변마다 '🔓 실명 보기': 역할이 팀장일 때만. 복원은 매핑 계정(vault_user)으로 하고 vault.restore_log 에 남김
검색·답변은 검색 계정(rag_user)만 씁니다.
"""
import datetime as dt
import re
import time
import uuid

import streamlit as st

from starter import chat, config, db, pii_pipeline, restore

st.set_page_config(page_title="가명처리 RAG 챗봇", page_icon="🔒", layout="wide",
                   initial_sidebar_state="expanded")

# LLM_PROVIDER=openai 로 실행하면 OpenAI 모델 (worklog 17장)
MODELS = (["gpt-4.1", "gpt-4.1-mini"] if config.LLM_PROVIDER == "openai" else ["HCX-007", "HCX-005", "HCX-DASH-002"])
EXAMPLES = [
    "김서연 고객 한도 상향 심사 어떻게 됐어?",
    "연락처가 010-0791-2924인 건 찾아줘",
    "고객센터 김민준 직원이 처리한 건 있어?",
    "대구 수성구 고객 문서 보여줘",
    "보이스피싱 피해 접수 기록 알려줘",
]
RAIL = [("chat", ":material/home:", "대화"), ("stats", ":material/bar_chart:", "성능 요약"),
        ("user", ":material/person:", "사용자·권한"), ("log", ":material/calendar_today:", "실명 복원 기록"),
        ("alerts", ":material/notifications:", "검사 알림")]
ROLES = {"팀장": "실명 보기 가능 (복원 기록이 남음)", "사원": "토큰만 볼 수 있음"}
_TOKEN = re.compile(r"`?(<[A-Za-z_]+_[0-9A-Za-z]+>)`?")

# 팀 비교용 최종 수치 (worklog 20장, 결과 outputs/final/)
FINAL = [
    ("검색 Hit@1 / MRR", "0.943 / 0.966", "0.930 / 0.950", "0.892 / 0.936"),
    ("처음 실행 Hit@1 / MRR", "—", "0.885 / 0.910", "0.853 / 0.901"),
    ("의미 검색만 Hit@1", "0.261", "0.120", "0.157"),
    ("쪼개짐", "6.3%", "21.1%", "33.3%"),
    ("정답 문서 인용", "0.92", "0.89", "0.97"),
    ("인용 정밀도 / 혼동 인용", "0.90 / 0.27", "0.85 / 0.29", "0.82 / 0.36"),
    ("원문 유출 (LLM 입력 / 답변)", "0 / 0", "0 / 0", "0 / 0"),
    ("시간 (검색 + LLM)", "1.2 + 2.7초", "1.1 + 2.6초", "1.2 + 2.8초"),
]


# ---------------------------------------------------------------- 상태
def _init():
    ss = st.session_state
    ss.setdefault("chats", {})          # id → {"title", "created", "turns": [{"query", "res"}]}
    ss.setdefault("current", None)
    ss.setdefault("view", "chat")
    ss.setdefault("role", "팀장")
    ss.setdefault("searching", False)
    ss.setdefault("cfg", {"model": MODELS[0], "k": 3, "mode": "hybrid"})


def new_chat():
    st.session_state.current = None
    st.session_state.view = "chat"


def open_chat(cid):
    st.session_state.current = cid
    st.session_state.view = "chat"


def set_view(view):
    st.session_state.view = view


def toggle_search():
    st.session_state.searching = not st.session_state.searching


def ask_example(q):
    st.session_state.pending = q


def current_turns():
    c = st.session_state.chats.get(st.session_state.current)
    return c["turns"] if c else []


def add_turn(query, res):
    ss = st.session_state
    if ss.current not in ss.chats:
        ss.current = uuid.uuid4().hex[:8]
        ss.chats[ss.current] = {"title": query, "created": dt.datetime.now(), "turns": []}
    ss.chats[ss.current]["turns"].append({"query": query, "res": res})


# ---------------------------------------------------------------- 공통
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
        _, info = restore.restore("\n".join(texts), actor=f"demo:{st.session_state.role}", purpose=qid)
        res["restored"] = info
    return res["restored"]


def reveal(text, info):
    """토큰 → 원래 값 (굵게 표시해서 무엇이 복원됐는지 보이게)."""
    for t, v in info["restored"].items():
        text = text.replace(f"`{t}`", t).replace(t, f"**{v}**")
    return text


# ---------------------------------------------------------------- 답변
def render_answer(res, key):
    can = st.session_state.role == "팀장"
    on = st.toggle("🔓 실명 보기", key=f"reveal_{key}", disabled=not can,
                   help="가려진 값(토큰)을 원래 값으로 보여줍니다. 복원 기록이 vault.restore_log 에 남습니다."
                   if can else "팀장만 볼 수 있어요 (왼쪽 사용자 아이콘에서 역할 변경)")
    on = on and can
    info = restored(res, key) if on else None
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
            hide_index=True, width="stretch")
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


# ---------------------------------------------------------------- 화면 모양
def css():
    view = st.session_state.view
    st.markdown(f"""<style>
    header[data-testid="stHeader"], [data-testid="stSidebarHeader"], [data-testid="stSidebarCollapseButton"],
    [data-testid="stToolbar"], #MainMenu {{ display: none !important; }}
    .stApp, [data-testid="stBottom"] > div {{ background: #f0f1f3; }}
    [data-testid="stMainBlockContainer"] {{ max-width: 920px; padding-top: 40px; padding-bottom: 140px; }}
    [data-testid="stBottomBlockContainer"] {{ max-width: 1240px; padding-bottom: 56px; }}

    /* 사이드바 = 흰 아이콘 줄(80px) + 회색 대화 목록 패널 */
    section[data-testid="stSidebar"] {{ width: 360px !important; min-width: 360px !important; max-width: 360px !important;
        background: #ffffff; border: none; }}
    section[data-testid="stSidebar"] > div {{ background: #ffffff; }}
    [data-testid="stSidebarContent"], [data-testid="stSidebarUserContent"] {{ padding: 0 !important; margin: 0 !important; }}
    .st-key-rail, .st-key-rail_bottom {{ position: fixed; left: 16px; width: 48px; z-index: 5; gap: 10px; }}
    .st-key-rail {{ top: 24px; }}
    .st-key-rail_bottom {{ bottom: 28px; }}
    .st-key-rail button, .st-key-rail_bottom button {{ width: 48px; height: 48px; min-height: 48px; padding: 0;
        border-radius: 50%; color: #3b4048; justify-content: center; }}
    .st-key-rail button:hover, .st-key-rail_bottom button:hover {{ background: #f0f1f3; color: #111; }}
    .st-key-rail_{view} button {{ background: #f0f1f3; }}
    .st-key-avatar button {{ background: #0e5e5a !important; color: #fff !important; font-weight: 700; }}
    .st-key-panel {{ margin-left: 80px; width: calc(100% - 80px) !important; box-sizing: border-box; background: #f7f8fa;
        min-height: 100vh; border-radius: 0 24px 24px 0; padding: 100px 16px 24px 16px; }}
    /* 새 채팅(늘어남) + 검색(52px 고정) — 비율 칸이면 좁을 때 검색 버튼이 패널 밖으로 밀림 */
    .st-key-toprow [data-testid="stHorizontalBlock"] {{ gap: 12px; flex-wrap: nowrap; }}
    .st-key-toprow [data-testid="stColumn"]:first-child {{ flex: 1 1 auto !important; width: auto !important; min-width: 0; }}
    .st-key-toprow [data-testid="stColumn"]:last-child {{ flex: 0 0 52px !important; width: 52px !important; min-width: 52px; }}
    .st-key-new_chat button {{ background: #9e9e9e; color: #fff; border: none; border-radius: 999px; height: 50px;
        font-size: 16px; }}
    .st-key-new_chat button:hover {{ background: #8a8a8a; color: #fff; }}
    .st-key-search_btn button {{ background: #111; color: #fff; border: none; border-radius: 50%; width: 52px;
        height: 52px; padding: 0; }}
    .st-key-search_btn button:hover {{ background: #333; color: #fff; }}
    .st-key-panel [data-testid="stExpander"] details {{ border: none; border-bottom: 1px solid #e3e6ea;
        border-radius: 0; background: transparent; }}
    .st-key-panel [data-testid="stExpander"] summary {{ padding: 18px 4px 10px; font-weight: 700; font-size: 16px;
        background: transparent !important; flex-direction: row-reverse; justify-content: space-between; }}
    .st-key-panel [data-testid="stExpander"] summary:hover {{ color: inherit; }}
    .st-key-panel [data-testid="stExpander"] summary > span {{ flex-direction: row-reverse; justify-content: space-between;
        width: 100%; }}
    .st-key-panel [data-testid="stExpanderDetails"] {{ padding: 0 0 12px; }}
    .st-key-panel [class*="st-key-chat_"] button {{ width: 100%; justify-content: flex-start; padding: 9px 4px;
        min-height: 0; color: #22262c; font-size: 16px; }}
    .st-key-panel [class*="st-key-chat_"] button div {{ justify-content: flex-start; overflow: hidden; }}
    .st-key-panel [class*="st-key-chat_"] button p {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
        font-size: 16px; text-align: left; }}
    .st-key-chat_{st.session_state.current} button {{ background: #eceef1; border-radius: 10px; }}

    /* 빈 화면 예시 질문 카드 */
    .st-key-examples {{ gap: 20px; margin-top: 22vh; }}
    .st-key-examples button {{ background: #f7f8fa; border: none; border-radius: 18px; height: 80px; color: #555;
        font-size: 18px; box-shadow: none; }}
    .st-key-examples button p {{ font-size: 18px; }}
    .st-key-examples button:hover {{ background: #ffffff; color: #111; }}

    /* 입력창: 흰 알약 모양 */
    [data-testid="stChatInput"] {{ border-radius: 999px !important; background: #fff !important; border: none !important;
        box-shadow: 0 2px 14px rgba(20, 24, 31, .06); min-height: 84px; padding: 10px 14px 10px 32px; align-items: center; }}
    [data-testid="stChatInput"] div {{ border: none !important; background: transparent !important; box-shadow: none !important; }}
    [data-testid="stChatInput"] > div {{ display: flex; align-items: center; }}
    [data-testid="stChatInputTextArea"] {{ font-size: 18px !important; background: transparent !important;
        padding-top: 14px !important; padding-bottom: 14px !important; }}
    [data-testid="stChatInputSubmitButton"] {{ width: 42px; height: 42px; border-radius: 50% !important;
        background: #9e9e9e !important; color: #fff !important; }}
    [data-testid="stChatInputSubmitButton"] svg {{ fill: #fff; color: #fff; }}

    /* 대화: 내 질문은 오른쪽 말풍선, 답변은 흰 카드 */
    [data-testid="stChatMessage"] {{ background: #fff; border-radius: 18px; padding: 18px 22px; margin-bottom: 14px; }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ background: transparent; padding: 0;
        flex-direction: row-reverse; }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {{
        flex: 0 1 auto; background: #2f343b; color: #fff; border-radius: 18px 18px 4px 18px; padding: 12px 18px;
        margin-left: auto; }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p {{
        color: #fff; }}
    [data-testid="stChatMessageAvatarUser"] {{ display: none; }}
    </style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- 사이드바
def sidebar():
    ss = st.session_state
    with st.sidebar:
        with st.container(key="rail"):
            for view, icon, label in RAIL:
                with st.container(key=f"rail_{view}"):
                    st.button("", icon=icon, key=f"btn_{view}", help=label, type="tertiary",
                              on_click=set_view, args=(view,))
        with st.container(key="rail_bottom"):
            with st.container(key="rail_settings"):
                st.button("", icon=":material/settings:", key="btn_settings", help="설정", type="tertiary",
                          on_click=set_view, args=("settings",))
            with st.container(key="avatar"):
                st.button(ss.role[0], key="btn_avatar", help=f"역할: {ss.role}", on_click=set_view, args=("user",))
        with st.container(key="panel"):
            with st.container(key="toprow"):
                c1, c2 = st.columns(2, vertical_alignment="center")
                with c1:
                    st.button("새 채팅", icon=":material/add:", key="new_chat", on_click=new_chat, width="stretch")
                with c2:
                    st.button("", icon=":material/search:", key="search_btn", help="대화 검색", on_click=toggle_search)
            q = (st.text_input("대화 검색", placeholder="대화 제목으로 찾기", label_visibility="collapsed")
                 if ss.searching else "")
            chats = sorted(ss.chats.items(), key=lambda kv: kv[1]["created"], reverse=True)
            if q:
                chats = [kv for kv in chats if q in kv[1]["title"]]
            today = dt.date.today()
            groups = [("오늘", [kv for kv in chats if kv[1]["created"].date() == today]),
                      ("이전", [kv for kv in chats if kv[1]["created"].date() < today])]
            for name, items in groups:
                if name == "이전" and not items:
                    continue
                with st.expander(name, expanded=True):
                    if not items:
                        st.caption("검색 결과가 없어요" if q else "아직 대화가 없어요. 질문을 입력해 보세요")
                    for cid, c in items:
                        st.button(c["title"], key=f"chat_{cid}", type="tertiary", on_click=open_chat, args=(cid,),
                                  width="stretch")
            st.caption("대화 목록은 이 브라우저 세션에만 있어요 (질문 원문을 파일로 남기지 않음)")


# ---------------------------------------------------------------- 화면: 대화
def view_chat():
    cfg = st.session_state.cfg
    turns = current_turns()
    # 입력창은 위치와 상관없이 맨 아래에 붙음 → 먼저 받아서, 질문이 들어오면 예시 카드를 숨김
    query = st.chat_input("질문을 입력하세요 (예: 윤선영 고객 상담 내역)")
    query = query or st.session_state.pop("pending", None)
    if not turns and not query:
        with st.container(key="examples"):
            for i, ex in enumerate(EXAMPLES):
                st.button(ex, key=f"ex_{i}", on_click=ask_example, args=(ex,), width="stretch")
    for i, turn in enumerate(turns):
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant", avatar=":material/lock:"):
            render_answer(turn["res"], f"{st.session_state.current}_{i}")
    if not query:
        return
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant", avatar=":material/lock:"):
        live = st.empty()  # 답변이 오는 대로 표시. 끝나면 출력 검사를 거친 답변으로 바뀜
        with st.spinner("가명처리 → 검색 → 답변 생성 중…"):
            t = time.perf_counter()
            try:
                # 임베딩 API 는 분당 60회 한도(키 공유 시 금방 참) → 대화에선 3번만 재시도, 안 되면 토큰 조건만으로 검색
                res = chat.answer(query, k=cfg["k"], model=cfg["model"], mode=cfg["mode"], embed_retries=3,
                                  on_delta=lambda text: live.markdown(show_tokens(text)))
            except Exception as e:
                st.error(f"오류: {type(e).__name__}: {e}")
                st.stop()
            res["timing"]["total"] = time.perf_counter() - t
        live.empty()
    add_turn(query, res)
    st.rerun()  # 대화 목록·답변 카드를 저장된 상태로 다시 그림


# ---------------------------------------------------------------- 화면: 그 밖
def view_stats():
    st.title("성능 요약")
    st.caption("현재 코드 · HCX-007 추론 끔 · 재순위 HCX-005 · 문서 3개 · 임베딩 bge-m3 (worklog 20장)")
    st.dataframe([{"항목": r[0], "holdout (질의 88)": r[1], "synth2 (질의 200)": r[2], "synth3 (질의 102)": r[3]}
                  for r in FINAL], hide_index=True, width="stretch")
    st.info("holdout·synth2·synth3 모두 실패를 보고 고친 뒤의 값이라 낙관적이에요. 처음 실행 값이 처음 보는 데이터에서의 성능에 가까워요.")
    try:
        s = db_stats()
        c = st.columns(4)
        c[0].metric("지금 DB 문서", s["docs"])
        c[1].metric("청크", s["chunks"])
        c[2].metric("추정 인물 (p_id)", s["persons"])
        c[3].metric("확정 인물", s["resolved"])
    except Exception as e:  # DB 가 꺼져 있을 때
        st.error(f"DB 연결 실패: {e}")


def view_user():
    ss = st.session_state
    st.title("사용자 · 권한")
    role = st.radio("역할", list(ROLES), index=list(ROLES).index(ss.role), horizontal=True,
                    format_func=lambda r: f"{r} — {ROLES[r]}")
    if role != ss.role:
        ss.role = role
        st.rerun()
    st.caption("시연용 역할 전환이에요. 로그인·담당 고객 범위 확인은 종합 단계에서 붙입니다.")
    st.markdown("- 검색·답변: 검색 계정(`rag_user`)만 — 원문이 없는 DB\n"
                "- 실명 보기: 팀장만, 매핑 계정(`vault_user`)으로 복원하고 `vault.restore_log` 에 누가·언제·어떤 토큰을 남김\n"
                "- LLM·임베딩 API 에는 가명처리된 텍스트만 전송")


def view_log():
    st.title("실명 복원 기록")
    st.caption("vault.restore_log 최근 30건 — 값은 남기지 않고 토큰만 기록")
    try:
        with db.vault_conn() as conn:
            rows = conn.execute("SELECT created_at, actor, purpose, tokens FROM vault.restore_log "
                                "ORDER BY created_at DESC LIMIT 30").fetchall()
    except Exception as e:
        st.error(f"기록을 읽지 못했어요: {e}")
        return
    if not rows:
        st.info("아직 복원 기록이 없어요.")
        return
    st.dataframe([{"시각": r[0].strftime("%m-%d %H:%M:%S"), "누가": r[1], "대화": r[2], "토큰 수": len(r[3] or []),
                   "토큰": ", ".join((r[3] or [])[:4]) + (" …" if len(r[3] or []) > 4 else "")} for r in rows],
                 hide_index=True, width="stretch")


def view_alerts():
    st.title("검사 알림")
    st.caption("이번 세션 대화에서 출력 검사·마지막 차단·되묻기가 작동한 경우")
    items = []
    for c in st.session_state.chats.values():
        for t in c["turns"]:
            r, chk = t["res"], t["res"].get("check") or {}
            if r.get("ask_back"):
                items.append(("되묻기", t["query"], "같은 이름의 사람이 여럿이라 LLM 대신 되물음"))
            if r.get("guard"):
                items.append(("마지막 차단", t["query"], f"API 로 보내기 직전 {r['guard']}개를 가림"))
            if chk.get("unverified_tokens"):
                items.append(("확인 안 된 토큰", t["query"], ", ".join(chk["unverified_tokens"])))
            if chk.get("raw_pii"):
                items.append(("원문 개인정보", t["query"], ", ".join(p["label"] for p in chk["raw_pii"])))
    if not items:
        st.info("알림이 없어요. 모든 답변이 검사를 통과했어요.")
        return
    st.dataframe([{"종류": a, "질문": q, "내용": d} for a, q, d in items], hide_index=True, width="stretch")


def view_settings():
    cfg = st.session_state.cfg
    st.title("설정")
    cfg["model"] = st.selectbox("LLM 모델", MODELS, index=MODELS.index(cfg["model"]),
                                help="HCX-007: 기본 (추론 끔이면 약 3초). HCX-005: 빠르지만 내용을 놓치기도 함")
    cfg["k"] = st.slider("LLM 에 넘길 문서 수", 1, 5, cfg["k"])
    modes = {"hybrid": "토큰 조건 + 의미 (기본)", "semantic": "의미 검색만 (비교용)"}
    cfg["mode"] = st.radio("검색 방식", list(modes), index=list(modes).index(cfg["mode"]), format_func=modes.get)
    st.caption(f"임베딩 {config.EMBED_MODEL} · 추론 설정 {config.LLM_REASONING_EFFORT or '모델 기본값'} · "
               f"재순위 {'켬' if config.RERANK else '끔'}")


# ---------------------------------------------------------------- 실행
_init()
css()
sidebar()
warm_up()
{"chat": view_chat, "stats": view_stats, "user": view_user, "log": view_log,
 "alerts": view_alerts, "settings": view_settings}[st.session_state.view]()
