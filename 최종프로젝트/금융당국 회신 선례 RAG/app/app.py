"""금융당국 회신 선례 검색 데모 (Streamlit, 92번에서 새로 작성).
실행: streamlit run app/app.py   (DB 컨테이너 `interp-rag-db`가 떠 있어야 하고, .env의 키·접속 값은 읽기만 한다)

처리 로직은 `service.py`에 있고(이전 데모 `demo.py`와 같은 동작, 89번에서 잰 설정), 이 파일은 화면만 그린다.

화면 구성
- 사례 검색: 위쪽 질문 입력 → 답변 → 근거 사례 카드(시점 표시) → 답변에 쓰이지 않은 검색 결과 → 이전 질문
- 조문 조회: "은행법 제27조의2" 같은 조문으로 그 조문을 인용한 회신을 조회 (검색·LLM 없이 DB 조회)
- 이용 안내: 무엇을 하고 무엇을 하지 않는지, 표시 설명, 검증 수치
- 사이드바: 검색 설정(리랭킹, 현장건의, 키워드로 넓혀 찾기, 검색 점수 보기)

화면 원칙 (이전 데모에서 정한 것을 그대로 지킨다)
1. 근거 사례번호 표기는 배지 하나로 통일한다
2. "찾지 못했습니다"면 검색 결과를 근거처럼 보여 주지 않는다(접어서 참고용으로만)
3. 답변이 "직접 일치하는 회신은 찾지 못했다"면서 사례를 인용하면 주의 안내를 먼저 보여 준다
4. 근거 사례 카드는 답변이 실제로 인용한 사례만, 인용 순서대로
5. 시점 표시는 카드 위쪽에 바로 보이게, 조문별 상세는 카드 안에. 아이콘(⚠️ ✅ 🔶 🕒 ❔) 대신 용어로 보여 준다(107번, `MARK_TERM`)
6. 확정 판단을 하지 않는다는 점, 질문을 하나씩 처리한다는 점, 필터가 없다는 점을 화면에 밝힌다
7. 현장건의는 회신과 구분해 보여 준다(배지, 안내 문구)
8. 답변 검증(99번, 토글 기본 끔: 100번): 검증 단계가 인용 사례에서 확인하지 못한 줄에 주황 표시를 붙인다(지우지 않음). 검증 단계가 답변을 거절하면 그 이유를 함께 보여 준다
"""
import html
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag  # noqa: E402
import exmnt_search as es  # noqa: E402
import service as sv  # noqa: E402
from articles_lookup import Lookup  # noqa: E402

st.set_page_config(page_title="금융당국 회신 선례 검색", page_icon="📑", layout="wide")

EXAMPLES = [
    ("특별이익", "적금 가입 고객에게 제휴 카드사 캐시백을 주는 이벤트를 하려는데, 특별이익 제공으로 문제된 회신이 있나요?"),
    ("비대면 실명확인", "비대면으로 법인 고객 계좌를 열어줄 때 실명확인을 어떻게 해야 하는지 당국 의견이 있나요?"),
    ("방카슈랑스", "보험회사가 방카슈랑스 판매 비중 규제를 받는지 알려주세요"),
    ("대주주 신용공여", "저축은행 대주주에 대한 신용공여 한도 관련 회신 알려주세요"),
]
ARTICLE_EXAMPLES = ["은행법 제27조의2", "금소법 시행령 제15조 제2항", "전자금융감독규정 제4-20조"]
MARK_COLOR = {"⚠️": "orange", "✅": "green", "🔶": "orange", "🕒": "blue", "❔": "gray"}
CAUTION = ("⚠️", "🔶", "🕒")
# 시점 판정은 `time_marks.py`가 아이콘으로 돌려준다(평가에 묶인 `rag.py`가 쓰는 모듈이라 고치지 않는다). 화면에서는 헷갈리지 않도록 용어로 바꿔 보여 준다.
MARK_TERM = {"⚠️": "조문 개정·폐지", "✅": "현행 유지", "🔶": "규정 개정", "🕒": "시행 예정 개정", "❔": "확인 불가"}
MARK_RX = re.compile("|".join(map(re.escape, MARK_TERM)))
LEGEND = """| 표시 | 뜻 |
|---|---|
| :orange-badge[조문 개정·폐지] | 회신일 이후 인용 조문이 개정·폐지됨. 현재 적용 여부 확인 필요(회신이 무효라는 뜻은 아닙니다) |
| :green-badge[현행 유지] | 회신일 이후 개정 없음 |
| :orange-badge[규정 개정] | 인용 규정이 개정됨(해당 조문이 바뀌었는지는 미확인) |
| :blue-badge[시행 예정 개정] | 공포됐으나 아직 시행 전인 개정이 있음 |
| :gray-badge[확인 불가] | 확인 불가(법령 연결이 불확실한 인용, 부칙 등이 섞인 경우 포함) |

사례의 대표 표시는 인용한 모든 조문 중 가장 주의가 필요한 것입니다. 판정 기준일은 2026-11-09입니다."""
CSS = """<style>
.block-container {max-width: 1120px; padding-top: 2.4rem; padding-bottom: 4rem;}
h1 {font-size: 1.9rem !important; letter-spacing: -0.02em; margin-bottom: 0 !important;}
h3 {font-size: 1.05rem !important; letter-spacing: -0.01em;}
[data-testid="stVerticalBlockBorderWrapper"] p {line-height: 1.7;}
.q-echo {font-size: 1.12rem; font-weight: 600; line-height: 1.55; letter-spacing: -0.01em; margin: 0.2rem 0 0.1rem;}
.case-title {font-weight: 600; line-height: 1.5;}
.muted {opacity: 0.65; font-size: 0.86rem;}
.ans {padding: 0.2rem 0.1rem 0.4rem;}
.ans-label {font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em; opacity: 0.55; margin: 0.9rem 0 0.35rem;}
.ans-label:first-child {margin-top: 0;}
.ans-row {display: flex; gap: 1rem; align-items: flex-start; justify-content: space-between;}
.ans-text {flex: 1 1 auto; min-width: 0; line-height: 1.75; word-break: keep-all;}
.ans-cites {flex: 0 0 auto; display: flex; flex-wrap: wrap; gap: 0.3rem; justify-content: flex-end; max-width: 11rem; padding-top: 0.22rem;}
.ans-lead .ans-text {font-size: 1.08rem; font-weight: 600; line-height: 1.65; letter-spacing: -0.01em;}
.ans-lead .ans-row + .ans-row {margin-top: 0.4rem;}
.ans-points .ans-row {padding: 0.7rem 0; border-top: 1px solid rgba(128, 128, 128, 0.22);}
.ans-points .ans-row:last-child {padding-bottom: 0.2rem;}
.ans-notes {margin-top: 0.6rem; padding-top: 0.7rem; border-top: 1px solid rgba(128, 128, 128, 0.22);}
.cite {display: inline-block; font-size: 0.78rem; font-weight: 600; line-height: 1.5; padding: 0 0.45rem; border-radius: 0.4rem; white-space: nowrap;
       color: #3b82f6; background: rgba(59, 130, 246, 0.14); cursor: default;}
.ans-text .cite {margin: 0 0.15rem; vertical-align: 0.08em;}
.cite.flag {color: #d97706; background: rgba(245, 158, 11, 0.16);}
@media (max-width: 640px) {.ans-row {flex-direction: column; gap: 0.3rem;} .ans-cites {justify-content: flex-start; max-width: none; padding-top: 0;}}
</style>"""


@st.cache_resource(show_spinner=False)
def warm():
    return sv.warm()


def excerpt(text, n):
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[:n].rstrip() + " …"


def mark_badge(mark):
    """시점 아이콘 → 용어 배지."""
    return f":{MARK_COLOR.get(mark, 'gray')}-badge[{MARK_TERM.get(mark, '확인 불가')}]"


def mark_words(text):
    """글 안의 시점 아이콘을 용어로 바꾼다 ('⚠️ 1 · ✅ 1' → '조문 개정·폐지 1 · 현행 유지 1')."""
    return MARK_RX.sub(lambda m: MARK_TERM[m.group(0)], text or "")


def badges(c):
    if c["doc_class"] == "field_proposal":          # 현장건의: 구분·검토결과·회신일, 수용 계열은 이후 변경 확인 안내
        out = [":orange-badge[현장건의]", f":violet-badge[검토결과 {c['answer_type']}]", f":gray-badge[{c['reply_date']}]"]
        if c.get("changed_after"):
            out.append(":red-badge[이후 제도 변경 여부 확인]")
        if c.get("already_in_place"):
            out.append(":green-badge[회신 당시 이미 운영 중]")
        if c["n_same"]:
            out.append(f":green-badge[같은 건의 {c['n_same']}건]")
        return " ".join(out)
    out = [f":gray-badge[{c['gubun']}]", f":gray-badge[{c['reply_date']}]"]
    if c.get("relaxed"):
        r = c["relaxed"]
        out.insert(0, f":orange-badge[키워드로 넓혀 찾음 {r['matched']}/{r['of']}" + (f" · 없는 개념: {', '.join(r['missing'])}]" if r["missing"] else "]"))
    if c["answer_type"]:
        out.append(f":violet-badge[{c['answer_type']}]")
    if c["answer_type"] == "비조치(무효)":
        out.append(":red-badge[현재 효력 없음]")
    if c["doc_class"] == "extension":
        out.append(":orange-badge[기한 연장 건]")
    if c["doc_class"] == "rule_review_2016":
        out.append(":gray-badge[2016 규제 정비 건]")
    if c["n_same"]:
        out.append(f":green-badge[같은 취지 {c['n_same']}건]")
    return " ".join(out)


def time_line(c):
    """카드 위쪽에 보이는 시점 표시 한 줄."""
    if c["doc_class"] == "field_proposal":
        return ":gray-badge[시점 표시 없음] 현장건의는 업계 건의에 대한 당국 검토 결과이며, 개별 행위에 대한 법령해석·비조치의견이 아닙니다."
    if not c["time_mark"]:
        return ":gray-badge[시점 표시 없음] 인용한 조문이 없습니다."
    return f"{mark_badge(c['time_mark'])} {mark_words(c['time_label'])}"


def render_card(c, uid, open_body=False):
    field = c["doc_class"] == "field_proposal"
    with st.container(border=True):
        head, link = st.columns([5, 1], vertical_alignment="center")
        head.markdown(f":blue-badge[{c['case_key']}] <span class='case-title'>{html.escape(excerpt(c['title'], 110))}</span>", unsafe_allow_html=True)
        if c["url"]:
            link.link_button("포털 원문", c["url"], width="stretch", key=f"url_{uid}")
        st.markdown(badges(c))
        st.markdown(time_line(c))
        with st.expander("건의내용·당국 검토 발췌" if field else "질의요지·회답 발췌", expanded=open_body):
            left, right = st.columns(2)
            left.markdown("**건의내용**" if field else "**질의요지**")
            left.write(excerpt(c["question"], 320) or "-")
            right.markdown("**당국 검토**" if field else "**회답**")
            right.write(excerpt(c["answer"], 420) or "-")
            st.caption("발췌입니다. 전문은 포털 원문에서 확인하세요.")
        if c["citations"]:
            with st.expander(f"인용 조문 {len(c['citations'])}건의 시점 판정"):
                for ct in c["citations"][:8]:
                    d = f" [{', '.join(ct['dates'])}]" if ct["dates"] else ""
                    st.caption(f"{mark_badge(ct['icon'])} {ct['law'] or '법령 미확인'} {ct['raw']} — {ct['label']}{d}")
                if len(c["citations"]) > 8:
                    st.caption(f"… 외 {len(c['citations']) - 8}건")


def summary_badges(res):
    """답변 위 한 줄 요약: 근거 수, 시점 확인이 필요한 근거 수, 현장건의·키워드 완화 여부, 처리 시간."""
    cited = [res["cards"][k] for k in res["cited"]] if not res["check"]["not_found"] else []
    out = [f":blue-badge[근거 사례 {len(cited)}건]"] if cited else [":gray-badge[근거 사례 없음]"]
    n_caution = sum(1 for c in cited if c["time_mark"] in CAUTION)
    if n_caution:
        out.append(f":orange-badge[시점 확인 필요 {n_caution}건]")
    if any(c["doc_class"] == "field_proposal" for c in cited):
        out.append(":orange-badge[현장건의 포함]")
    if res.get("relax") and res["relax"]["extra"]:
        out.append(":orange-badge[키워드로 넓혀 다시 찾음]")
    v = res.get("verify")
    if v and v["flagged"] and not res["check"]["not_found"]:
        out.append(f":orange-badge[확인 필요 문장 {len(v['flagged'])}]")
    out.append(f":gray-badge[{res['seconds']['total']}초]")
    return " ".join(out)


BULLET = re.compile(r"^[-•*]\s+")
TAIL_CITES = re.compile(r"(?:\s*\[[LOF]\d+\])+\s*[.]?\s*$")          # 줄 끝에 몰려 있는 근거 번호


def answer_html(res):
    """답변 글을 화면용 HTML로 (글 내용은 바꾸지 않고 배치만 바꾼다, 106번).
    - 목록 앞의 줄은 '핵심 답변'으로 크게, "- " 항목은 '세부 내용'으로 한 줄씩 나눠 보여 준다
    - 근거 번호가 줄 끝에만 있으면 글 오른쪽으로 빼고, 문장 중간에도 있으면 어느 문장의 근거인지 보이도록 제자리에 둔다
    - 검증 단계가 표시한 줄에는 주황 표시를 붙인다 (줄은 지우지 않는다)"""
    flagged = set((res.get("verify") or {}).get("flagged") or [])

    def chip(k):
        title = excerpt((res["cards"].get(k) or {}).get("title", ""), 80)
        return f"<span class='cite' title='{html.escape(title, quote=True)}'>{k}</span>"

    lead, points, notes = [], [], []
    for line in res["raw"].replace(rag.CLOSING, "").strip().split("\n"):
        t = sv.answer_text(line).strip()
        if not t:
            continue
        is_point = bool(BULLET.match(t))
        t = BULLET.sub("", t)
        side = ""
        if sv.CITE.search(TAIL_CITES.sub("", t)):          # 문장 중간에도 근거 번호가 있다 → 제자리에
            text = sv.CITE.sub(lambda m: chip(m.group(1)), html.escape(t))
        else:
            text = html.escape(re.sub(r"\s+([.,])", r"\1", re.sub(r"\s{2,}", " ", sv.CITE.sub("", t))).strip())
            side = "".join(chip(k) for k in dict.fromkeys(sv.CITE.findall(t)))
        if line.strip() in flagged:
            side += "<span class='cite flag'>사례에서 확인되지 않음</span>"
        row = f"<div class='ans-row'><div class='ans-text'>{text}</div>" + (f"<div class='ans-cites'>{side}</div>" if side else "") + "</div>"
        (points if is_point else notes if points else lead).append(row)
    out = "<div class='ans'>"
    if lead:
        out += "<div class='ans-label'>핵심 답변</div><div class='ans-lead'>" + "".join(lead) + "</div>"
    if points:
        out += "<div class='ans-label'>세부 내용</div><div class='ans-points'>" + "".join(points) + "</div>"
    if notes:
        out += "<div class='ans-notes'>" + "".join(notes) + "</div>"
    return out + "</div>"


def render_result(res, uid, debug, open_first=True):
    chk = res["check"]
    st.markdown(f"<div class='muted'>질문</div><div class='q-echo'>{html.escape(res['question'])}</div>", unsafe_allow_html=True)
    st.markdown(summary_badges(res))
    with st.container(border=True):
        if chk["not_found"]:
            st.markdown("**관련 회신을 찾지 못했습니다.**")
            if (res.get("verify") or {}).get("abstained"):
                st.caption("검색된 사례가 질문의 대상·조건을 직접 다루지 않는다고 판단해 답변을 내지 않았습니다. 아래 검색 결과는 참고용입니다.")
            else:
                st.caption("질문에 직접 맞는 법령해석·비조치의견서 회신이 검색되지 않았습니다. 대상 회사·상품·행위를 더 구체적으로 적어 다시 물어보세요.")
        else:
            if sv.partly_not_found(res):
                st.warning("답변이 \"직접 일치하는 회신은 찾지 못했다\"고 밝혔습니다. 아래 사례는 관련성이 낮을 수 있으니 답변 내용과 함께 확인하세요.")
            st.markdown(answer_html(res), unsafe_allow_html=True)
            if (res.get("verify") or {}).get("flagged"):
                st.caption(":orange-badge[사례에서 확인되지 않음] 표시는 검증 단계가 인용 사례에서 찾지 못한 문장입니다. 표시가 틀릴 수 있고, 표시가 없는 문장도 원문 확인이 필요합니다.")
        st.caption(rag.CLOSING)
    trail = sv.trail_text(res.get("relax"))
    if trail:
        if res["relax"]["extra"] and chk["not_found"]:
            trail += " · 넓혀 찾은 사례에서도 질문에 직접 맞는 회신은 없었습니다(아래 참고 목록에 주황 배지로 표시)."
        elif res["relax"]["extra"]:
            trail += " · 키워드로 넓혀 찾은 사례에는 주황 배지를 붙였습니다."
        st.caption(trail)
    if res["cited"] and not chk["not_found"]:
        st.subheader(f"답변이 근거로 쓴 사례 {len(res['cited'])}건")
        for n, k in enumerate(res["cited"]):
            render_card(res["cards"][k], f"{uid}_{k}", open_body=open_first and n == 0)
    rest = [k for k in res["order"] if k not in res["cited"]] if not chk["not_found"] else res["order"]
    if rest:
        label = "검색은 됐지만 답변에 쓰이지 않은 사례" if not chk["not_found"] else "검색 결과 (관련성이 낮아 답변에 쓰지 않음)"
        with st.expander(f"{label} {len(rest)}건"):
            for k in rest:
                c = res["cards"][k]
                st.markdown(f":gray-badge[{k}] {excerpt(c['title'], 90)}  \n{badges(c)}" + (f" {mark_badge(c['time_mark'])}" if c["time_mark"] else ""))
    if debug:
        with st.expander("검색 점수·검사 결과"):
            st.write({"마스킹된 질문": res["masked"], "처리 시간(초)": res["seconds"], "리랭킹으로 순서 바뀜": res.get("reranked"), "현장건의": res.get("exmnt"), "답변 검증": res.get("verify"), "토큰": res["usage"], "출력 검사": chk})
            st.dataframe(pd.DataFrame([{"사례": k, "점수": round(res["cards"][k]["score"], 3), "청크 유형": res["cards"][k]["chunk_type"], "제목": res["cards"][k]["title"][:60], "인용됨": k in res["cited"]} for k in res["order"]]),
                         hide_index=True)


def use_example(text):
    st.session_state.q_input = text
    st.session_state.pending = text


def search_tab(opts):
    with st.form("ask", border=False):
        box, btn = st.columns([6, 1], vertical_alignment="bottom")
        box.text_area("질문", key="q_input", height=80, label_visibility="collapsed",
                      placeholder="검토하려는 상황을 적어 주세요. 예: 은행 앱에 제휴사 포인트 상품몰을 연동해도 되는지 문제된 회신이 있나요?")
        submitted = btn.form_submit_button("선례 찾기", type="primary", width="stretch")
    prompt = (st.session_state.q_input or "").strip() if submitted else st.session_state.pop("pending", None)
    if submitted and not prompt:
        st.info("질문을 입력해 주세요.")
    if prompt:
        try:
            with st.spinner("유사 사례를 찾고 답변을 만드는 중입니다 (보통 5~10초)"):
                st.session_state.messages.append(sv.ask(prompt, opts["relax"], opts["rerank"], opts["exmnt"], opts["verify"]))
        except Exception as e:          # 키·주소가 들어 있을 수 있어 오류 메시지는 보여 주지 않는다
            st.error(f"답변을 만들지 못했습니다 ({type(e).__name__}). 잠시 후 다시 시도해 주세요.")
    msgs = st.session_state.messages
    if not msgs:
        st.markdown("##### 이렇게 물어보세요")
        cols = st.columns(2)
        for n, (tag, ex) in enumerate(EXAMPLES):
            with cols[n % 2].container(border=True):
                st.markdown(f":blue-badge[{tag}]  \n{ex}")
                st.button("이 질문으로 찾기", key=f"ex_{n}", on_click=use_example, args=(ex,))
        st.caption("대상 회사·상품·행위·조건을 구체적으로 적을수록 맞는 선례를 찾기 쉽습니다. 질문은 한 번에 하나씩 처리하며, 이전 질문을 이어 받는 후속 질문과 구분·기간·분야 필터는 아직 지원하지 않습니다.")
        return
    render_result(msgs[-1], len(msgs) - 1, opts["debug"])
    if len(msgs) > 1:
        st.divider()
        st.subheader(f"이전 질문 {len(msgs) - 1}건")
        for i in range(len(msgs) - 2, -1, -1):
            with st.expander(excerpt(msgs[i]["question"], 100)):
                render_result(msgs[i], i, opts["debug"], open_first=False)


def use_article(text):
    st.session_state.art_input = text


def lookup_tab():
    st.markdown("법령과 조문을 입력하면 그 조문을 인용한 회신을 날짜순으로 보여 줍니다. 검색·LLM 없이 DB를 직접 조회합니다.")
    c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
    text = c1.text_input("조문", key="art_input", placeholder="예: 은행법 제27조의2", label_visibility="collapsed")
    order = c2.segmented_control("정렬", ["최신순", "오래된 순"], default="최신순", label_visibility="collapsed") or "최신순"
    ex = st.columns([1, 1, 1, 2])
    for n, a in enumerate(ARTICLE_EXAMPLES):
        ex[n].button(a, key=f"art_{n}", on_click=use_article, args=(a,), width="stretch")
    if not text.strip():
        return
    try:
        with rag.connect() as conn:
            res = Lookup(conn).lookup(text, order="desc" if order == "최신순" else "asc")
    except Exception as e:
        st.error(f"조회하지 못했습니다 ({type(e).__name__}). 잠시 후 다시 시도해 주세요.")
        return
    for p in res["parsed"]:
        label = f"{p.get('law_name') or p['law_name_text'] or '법령 없음'} {p['span']}"
        if p["status"] == "ok":
            st.success(f"{label}: {p['n_cases']}건")
        elif p["status"] == "need_law":
            names = ", ".join(c["law_name"] for c in p.get("candidates", []))
            st.warning(f"{label}: 법령 이름을 확인하지 못했습니다. 법령명을 함께 입력해 주세요." + (f" (이 조문이 인용된 법령 후보: {names})" if names else ""))
        else:
            st.info(f"{label}: 연결된 회신이 없습니다.")
    if res["status"] == "no_article":
        st.info(res["message"])
        return
    main = [r for r in res["results"] if not r["reference"]]
    ref = [r for r in res["results"] if r["reference"]]
    if main:
        df = pd.DataFrame([{"시점": MARK_TERM.get(r.get("time_mark"), "-"), "사례": r["case_key"], "회신일": str(r["reply_date"] or "미상"), "구분": r["gubun"], "답변유형": r["answer_type"] or "-",
                            "제목": (r["title"] or "").strip(), "연결": r["method"], "같은 취지": r["n_same"], "원문": r["url"]} for r in main])
        st.dataframe(df, hide_index=True, width="stretch", column_config={"원문": st.column_config.LinkColumn("원문", display_text="열기")})
        st.caption("시점 표시는 그 사례가 조회한 조문을 인용한 부분의 판정입니다. '조문 개정·폐지'는 회신이 무효라는 뜻이 아니라 현재 적용 여부 확인이 필요하다는 뜻입니다.")
    if ref:
        with st.expander(f"참고: 법령을 추정해 연결한 사례 {len(ref)}건 (정확도가 낮을 수 있음)"):
            st.dataframe(pd.DataFrame([{"사례": r["case_key"], "회신일": str(r["reply_date"] or "미상"), "제목": (r["title"] or "").strip()} for r in ref]), hide_index=True)


def guide_tab():
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("##### 하는 일")
        st.markdown("- 질문과 비슷한 금융위·금감원 **법령해석·비조치의견서 회신**(2014년 이후, 4,263건)을 찾아 요지를 정리합니다.\n"
                    "- 답변의 모든 문장에 근거 사례번호를 붙이고, 사례 카드에서 포털 원문으로 이동할 수 있습니다.\n"
                    "- 회신 이후 인용 조문이 바뀌었는지 **시점 표시**로 알려 줍니다.\n"
                    "- 관련도가 높을 때만 **현장건의**(업계 건의에 대한 당국 검토)를 함께 보여 줍니다.\n"
                    "- 조문으로 그 조문을 인용한 회신을 조회합니다.")
    with b.container(border=True):
        st.markdown("##### 하지 않는 일")
        st.markdown("- **확정 판단을 하지 않습니다.** 회신은 개별 사안에 대한 당국 의견이며, 최종 판단은 준법감시부서 확인이 필요합니다.\n"
                    "- 맞는 회신이 없으면 추측하지 않고 \"찾지 못했습니다\"라고 답합니다.\n"
                    "- 이전 질문을 이어 받는 후속 질문은 아직 지원하지 않습니다(질문을 하나씩 처리).\n"
                    "- 구분·기간·분야 필터는 아직 없습니다.\n"
                    "- 현장건의에는 시점 표시가 없습니다.")
    st.markdown("##### 시점 표시")
    st.markdown(LEGEND)
    st.markdown("##### 사례번호")
    st.markdown(":blue-badge[L123] 법령해석 · :blue-badge[O123] 비조치의견서 · :blue-badge[F123] 현장건의")
    st.markdown("##### 검증 수치")
    st.dataframe(pd.DataFrame([
        {"항목": "정답 회신이 최종 근거에 포함", "개발용 50문항": "100%", "시험용 50문항": "94%"},
        {"항목": "답변 결론이 정답 회신과 일치", "개발용 50문항": "96%", "시험용 50문항": "92%"},
        {"항목": "└ 회신 질문 35문항", "개발용 50문항": "97.1%", "시험용 50문항": "97.1%"},
        {"항목": "└ 현장건의 질문 15문항", "개발용 50문항": "93.3%", "시험용 50문항": "80%"},
        {"항목": "답변 내용이 인용 사례에 충실", "개발용 50문항": "78%", "시험용 50문항": "88%"},
    ]), hide_index=True, width="stretch")
    st.caption("회신 35 + 현장건의 15문항 세트 2개로 2026-11-12에 잰 값입니다(현재 화면 기본 설정, 답변 검증 끔). 질문은 정답 회신을 보고 LLM이 만든 합성 질문이고 채점도 LLM이 했으므로, 실제 업무 질문에서는 다를 수 있습니다.")


# ---------- 앱
st.markdown(CSS, unsafe_allow_html=True)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("q_input", "")

with st.sidebar:
    st.header("검색 설정")
    opts = {
        "rerank": st.toggle("검색 결과 다시 정렬(리랭킹)", value=True, help="의미 검색 상위 20건을 LLM이 다시 읽고 질문의 쟁점·조건에 직접 맞는 순서로 5건을 고릅니다. 질문마다 LLM 호출이 한 번 더 듭니다."),
        "exmnt": st.toggle("현장건의도 함께 찾기", value=True, help=f"업계 건의에 대한 당국 검토 결과(현장건의 과제)를 회신과 따로 찾아, 관련도가 높고(점수 {es.THRESH} 이상) 가장 비슷한 회신보다 더 질문에 가까울 때만 최대 {es.MAX_N}건을 회신 뒤에 붙입니다."),
        "relax": st.toggle("못 찾으면 키워드로 넓혀 찾기", value=True, help="의미 검색 상위 5건으로 답을 찾지 못했을 때만, 질문의 핵심 키워드를 모두 포함한 회신부터 하나씩 조건을 풀어 가며 최대 3건을 더 찾아 다시 답합니다."),
        "verify": st.toggle("답변 검증", value=False, help="필요할 때만 켭니다(기본 끔). 답변을 만든 뒤 한 번 더 읽어, 검색된 사례가 질문의 대상·조건을 직접 다루지 않으면 답변을 내지 않고, 인용 사례에서 확인되지 않은 문장에 표시를 붙입니다. 질문마다 LLM 호출이 한 번 더 듭니다(약 2~3초)."),
        "debug": st.toggle("검색 점수·검사 결과 보기", value=False),
    }
    st.divider()
    if st.button("질문 기록 지우기", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.caption("대상: 금융위·금감원 법령해석·비조치의견서 회신(2014년 이후)과 현장건의 과제. 확정 판단은 하지 않습니다.")

st.title("금융당국 회신 선례 검색")
st.caption("법령해석·비조치의견서 회신에서 유사 선례를 찾아 근거와 함께 보여 줍니다. 확정 판단은 하지 않으며, 최종 판단은 준법감시부서 확인이 필요합니다.")

with st.spinner("임베딩 모델을 불러오는 중입니다 (처음 한 번, 약 30~80초)"):
    warm()

tab_search, tab_art, tab_guide = st.tabs(["사례 검색", "조문 조회", "이용 안내"])
with tab_search:
    search_tab(opts)
with tab_art:
    lookup_tab()
with tab_guide:
    guide_tab()
