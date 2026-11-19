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
from datetime import date
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

# 예시 질문 (111번): 시험 세트(test7·test10)에서 정답 회신과 일치·충실로 판정된 문항. 순서대로 TEST7-033(L3563), TEST10-008(L2326), TEST10-032(O1619), TEST7-094(L2244)
EXAMPLES = [
    ("대출모집인 업무위탁", "대출모집인이 고객 서류의 기본 정보를 전산에 입력하고 확인·심사는 은행이 맡으면 업무의 핵심을 외주 준 걸로 보나요?"),
    ("법인카드 한도", "지주 자회사에 법인카드 한도를 크게 주려는데, 구매전용이나 자금융통이 아닌 카드채권도 이사회 사전 승인이 필요한 건가요?"),
    ("해외 지점 클라우드", "국내 금융사 해외 지점에서 현지인 정보만 다루는 뱅킹 전산을 해외 서버실·복구센터에 두고 현지 클라우드로 돌려도 되나요?"),
    ("전화 모집", "보험대리점이 계약 맺으려고 고객한테 전화로 상품 설명하고 청약까지 받아 체결하면, 이거 광고인지 통신수단 모집인지?"),
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
[class*="st-key-answer_"] h3 {font-size: 0.82rem !important; font-weight: 600; letter-spacing: 0.04em; opacity: 0.6; padding: 0.9rem 0 0.2rem;}
[class*="st-key-answer_"] h3:first-child {padding-top: 0;}
[class*="st-key-answer_"] li {margin: 0.35rem 0; line-height: 1.75;}
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
    if c.get("in_period") is False:          # 질문의 기간 조건 밖 회신 (117번)
        out.append(":red-badge[기간 조건 밖]")
    elif c.get("in_period"):
        out.append(":green-badge[기간 조건 안]")
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


def period_note(res):
    """질문의 기간 조건을 어떻게 처리했는지 한두 문장 (117번). 조건이 없으면 None. 반환 (글, 기간을 풀었는지)."""
    p = res.get("period")
    if not p:
        return None
    cited = [res["cards"][k] for k in res["cited"]] if not res["check"]["not_found"] else []
    n_in = sum(1 for c in cited if c.get("in_period"))
    basis = " 시행일 기준 조건은 '그 날짜 이후에 나온 회신'으로 바꿔 적용했습니다. 조문이 그 뒤에 바뀌었는지는 사례 카드의 시점 표시로 확인하세요." if p["kind"] == "basis" else ""
    if not p["fallback"]:
        return f"질문의 기간 조건(회신일 {p['label']})에 맞는 회신만으로 찾았습니다.{basis}", False
    if not cited:
        return f"질문의 기간 조건(회신일 {p['label']})에 맞는 회신에서 답을 찾지 못했고, 기간을 풀어 다시 찾아도 없었습니다.{basis}", True
    return (f"질문의 기간 조건(회신일 {p['label']})에 맞는 회신에서는 답을 찾지 못해, 기간을 풀어 다시 찾았습니다. "
            f"아래 근거 사례 {len(cited)}건 중 기간 안 {n_in}건, 기간 밖 {len(cited) - n_in}건입니다. 사례 카드의 회신일을 확인하세요.{basis}"), True


def answer_markdown(res):
    """모델이 마크다운으로 쓴 답변(`APP_SYSTEM_V3`: ### 제목, - 목록, **굵게**)을 그대로 렌더링할 글로 만든다 (110번).
    근거 번호는 배지로 바꾸고, 검증 단계가 표시한 줄에는 주황 배지를 붙인다 (줄은 지우지 않는다)."""
    flagged = set((res.get("verify") or {}).get("flagged") or [])
    valid = set(res["cards"])          # 검색해서 모델에 준 사례만 근거 배지로 그린다 (114번). 그 밖의 번호는 빨간 배지
    lines = []
    for line in res["raw"].replace(rag.CLOSING, "").strip().split("\n"):
        t = sv.CITE.sub(lambda m: f" :blue-badge[{m.group(1)}]" if m.group(1) in valid else f" :red-badge[{m.group(1)} 검색된 사례 아님]", sv.answer_text(line, keep_md=True))
        lines.append(t + (" :orange-badge[사례에서 확인되지 않음]" if line.strip() in flagged else ""))
    return re.sub(r"[ ]{2,}", " ", "\n".join(lines))


@st.cache_data(show_spinner=False, ttl=3600)
def data_basis():
    """데이터 기준(112번): 검색 대상 건수, 마지막 회신일, 시점 판정 기준일을 DB에서 읽는다. 읽지 못하면 None."""
    try:
        with rag.connect() as conn:
            c = conn.execute("select count(*) filter (where indexable) n, max(reply_date) last from interp.cases").fetchone()
            f = conn.execute("select count(*) n, max(reply_date) last from interp.exmnt_cases").fetchone()
            t = conn.execute("select max(checked_on) d from interp.case_article_status").fetchone()
        return {"n": c["n"], "last": str(c["last"]), "f_n": f["n"], "f_last": str(f["last"]), "checked": str(t["d"])}
    except Exception:
        return None


def memo_markdown(res):
    """검토 메모 초안(마크다운, 112번). 새 판단을 쓰지 않는다: 이미 나온 답변 글과 사례 카드의 값(구분·회신일·시점 판정)만 다시 배열한다."""
    cited = [res["cards"][k] for k in res["cited"]]
    basis = data_basis()
    body = re.sub(r"(?m)^#{1,6}\s+", "#### ", sv.answer_text(res["raw"], keep_md=True)).strip()
    out = ["# 선례 검토 메모 (초안)", "",
           f"- 작성일: {date.today().isoformat()} (금융당국 회신 선례 검색으로 생성)",
           f"- 질문: {res['question'].strip()}",
           "- 성격: 중간 검토 자료이며 확정 판단이 아닙니다. 인용 전 포털 원문을 확인하세요.", "",
           "## 1. 검토 요지", "", body, "",
           "## 2. 관련 선례", "",
           "| 사례 | 구분 | 회신일 | 답변유형 | 시점 | 제목 | 원문 |", "|---|---|---|---|---|---|---|"]
    for c in cited:
        mark = "-" if c["doc_class"] == "field_proposal" else MARK_TERM.get(c["time_mark"], "확인 불가") if c["time_mark"] else "인용 조문 없음"
        title = excerpt(c["title"], 70).replace("|", "/")
        out.append(f"| {c['case_key']} | {c['gubun']} | {c['reply_date']} | {c['answer_type'] or '-'} | {mark} | {title} | {c['url'] or '-'} |")
    notes = []
    if period_note(res):
        notes.append(f"- {period_note(res)[0]}")
    if sv.partly_not_found(res):
        notes.append("- 답변이 \"직접 일치하는 회신은 찾지 못했다\"고 밝혔습니다. 아래 사례는 관련성이 낮을 수 있습니다.")
    for c in cited:
        k = c["case_key"]
        if c["doc_class"] == "field_proposal":
            notes.append(f"- {k}: 현장건의(검토결과 {c['answer_type']})입니다. 업계 건의에 대한 당국 검토 결과이며 개별 행위에 대한 법령해석·비조치의견이 아닙니다."
                         + (" 이후 제도가 실제로 바뀌었는지 확인이 필요합니다." if c.get("changed_after") else ""))
            continue
        if c["answer_type"] == "비조치(무효)":
            notes.append(f"- {k}: 현재 효력이 없는 비조치의견서입니다.")
        if c["doc_class"] == "extension":
            notes.append(f"- {k}: 기한 연장 건입니다. 효력 기간을 확인하세요.")
        for ct in c["citations"]:
            if ct["icon"] != "✅":
                d = f" [{', '.join(ct['dates'])}]" if ct["dates"] else ""
                notes.append(f"- {k}: {ct['law'] or '법령 미확인'} {ct['raw']} — {MARK_TERM.get(ct['icon'], '확인 불가')}: {ct['label']}{d}")
    if res.get("invalid"):
        notes.append(f"- 검토 요지에 검색된 사례가 아닌 번호({', '.join(res['invalid'])})가 있습니다. 이 번호가 붙은 문장은 근거를 확인할 수 없습니다.")
    for line in (res.get("verify") or {}).get("flagged") or []:
        notes.append(f"- 답변 검증에서 인용 사례로 확인되지 않은 문장: {sv.answer_text(line).lstrip('- ').strip()}")
    out += ["", "## 3. 확인 필요 사항", ""] + (notes or ["- 시점 판정에서 따로 확인이 필요한 조문은 없었습니다(인용 조문 모두 회신일 이후 개정 없음, 또는 인용 조문 없음)."])
    out += ["", "## 4. 유의", "",
            f"- {rag.CLOSING}",
            "- 검토 요지는 검색된 회신을 LLM이 요약한 것입니다. 문장 끝의 [사례번호]로 원문과 대조하세요.",
            "- 시점 표시는 회신일 이후 인용 조문의 개정 여부이며, 회신이 무효라는 뜻이 아닙니다."
            + (f" 시점 판정 기준일 {basis['checked']}, 수록된 마지막 회신일 {basis['last']}." if basis else "")]
    return "\n".join(out) + "\n"


def render_result(res, uid, debug, open_first=True):
    chk = res["check"]
    st.markdown(f"<div class='muted'>질문</div><div class='q-echo'>{html.escape(res['question'])}</div>", unsafe_allow_html=True)
    st.markdown(summary_badges(res))
    note = period_note(res)
    if note:
        (st.warning if note[1] else st.info)(note[0])
        st.caption(f"검색과 답변에는 기간 조건을 뺀 질문을 썼습니다: “{res['period']['topic']}”")
    with st.container(border=True):
        if chk["not_found"]:
            st.markdown("**관련 회신을 찾지 못했습니다.**")
            if (res.get("verify") or {}).get("abstained"):
                st.caption("검색된 사례가 질문의 대상·조건을 직접 다루지 않는다고 판단해 답변을 내지 않았습니다. 아래 검색 결과는 참고용입니다.")
            elif (res.get("gate") or {}).get("verdicts"):
                st.caption("비슷한 사례는 검색됐지만, 질문과 대상·행위·쟁점이 모두 같은 회신은 없었습니다. 비슷한 사례로 답하지 않았습니다. 아래 검색 결과는 참고용이며 질문에 대한 답이 아닙니다.")
            else:
                st.caption("질문에 직접 맞는 법령해석·비조치의견서 회신이 검색되지 않았습니다. 대상 회사·상품·행위를 더 구체적으로 적어 다시 물어보세요.")
        else:
            if sv.partly_not_found(res):
                st.warning("답변이 \"직접 일치하는 회신은 찾지 못했다\"고 밝혔습니다. 아래 사례는 관련성이 낮을 수 있으니 답변 내용과 함께 확인하세요.")
            if res.get("invalid"):
                st.warning(f"답변에 검색된 사례가 아닌 번호({', '.join(res['invalid'])})가 있습니다. 이 번호가 붙은 문장은 근거를 확인할 수 없으니 쓰지 마세요.")
            with st.container(key=f"answer_{uid}"):
                st.markdown(answer_markdown(res))
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
        memo = memo_markdown(res)
        with st.expander("검토 메모 초안 (복사·내려받기)"):
            st.caption("답변과 근거 사례를 검토 의견서·품의서에 옮겨 적기 쉽게 묶은 초안입니다. 새 판단을 더하지 않고 위 답변과 사례 카드의 내용만 정리합니다. 아래 상자 오른쪽 위 버튼으로 복사할 수 있습니다.")
            st.download_button("내려받기 (.md)", memo, file_name=f"선례검토메모_{date.today():%Y%m%d}_{res['cited'][0]}.md", mime="text/markdown", key=f"memo_dl_{uid}", on_click="ignore")
            st.code(memo, language="markdown", wrap_lines=True)
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
            st.write({"마스킹된 질문": res["masked"], "처리 시간(초)": res["seconds"], "리랭킹으로 순서 바뀜": res.get("reranked"), "현장건의": res.get("exmnt"), "답변 검증": res.get("verify"), "토큰": res["usage"], "출력 검사": chk, "검색된 사례가 아닌 번호": res.get("invalid"), "기간 조건": res.get("period"), "직접성 판정": res.get("gate")})
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
                st.session_state.messages.append(sv.ask(prompt, opts["relax"], opts["rerank"], opts["exmnt"], opts["verify"], opts["period"], opts["gate"]))
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
        st.caption("대상 회사·상품·행위·조건을 구체적으로 적을수록 맞는 선례를 찾기 쉽습니다. 질문은 한 번에 하나씩 처리하며, 이전 질문을 이어 받는 후속 질문과 구분·분야 필터는 아직 지원하지 않습니다. 회신 시기는 \"2023년 이후에 나온 회신\"처럼 질문에 적으면 회신일 조건으로 적용합니다.")
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
                    "- 조문으로 그 조문을 인용한 회신을 조회합니다.\n"
                    "- 답변과 근거 사례를 **검토 메모 초안**(.md)으로 복사하거나 내려받을 수 있습니다.")
    with b.container(border=True):
        st.markdown("##### 하지 않는 일")
        st.markdown("- **확정 판단을 하지 않습니다.** 회신은 개별 사안에 대한 당국 의견이며, 최종 판단은 준법감시부서 확인이 필요합니다.\n"
                    "- 맞는 회신이 없으면 추측하지 않고 \"찾지 못했습니다\"라고 답합니다.\n"
                    "- 이전 질문을 이어 받는 후속 질문은 아직 지원하지 않습니다(질문을 하나씩 처리).\n"
                    "- 구분·분야 필터는 아직 없습니다. 기간은 질문에 적은 회신 시기(\"2023년 이후에 나온 회신\" 등)만 회신일 조건으로 적용합니다.\n"
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
    st.caption("회신 35 + 현장건의 15문항 세트 2개로 2026-11-12에 잰 값입니다(답변 검증 끔). 이후 답변 형식만 마크다운으로 바꿨고(규칙 1~8은 같음), 바꾼 뒤의 수치는 아직 재지 않았습니다.질문은 정답 회신을 보고 LLM이 만든 합성 질문이고 채점도 LLM이 했으므로, 실제 업무 질문에서는 다를 수 있습니다.")


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
        "gate": st.toggle("직접 맞는 사례만으로 답하기", value=True, help="검색된 사례마다 질문과 대상·행위·쟁점이 같은지 먼저 판정해, 같은 사례만으로 답합니다. 직접 맞는 사례가 없으면 비슷한 사례로 답하지 않고 \"찾지 못했습니다\"라고 답합니다. 질문마다 LLM 호출이 한 번 더 들고 시간이 약 2배가 됩니다(개발셋 중앙값 5초 → 10초)."),
        "period": st.toggle("질문의 기간 조건 적용", value=True, help="질문에 회신 시기를 가리키는 말(예: '2024년 이후에 나온 회신', '최근 3년간 회신', '2024-01-15 시행된 개정 기준으로')이 있으면 떼어 내 회신일 조건으로 씁니다. 그 기간의 회신만으로 먼저 찾고, 답이 없으면 기간을 풀어 다시 찾은 뒤 기간 밖 사례임을 표시합니다."),
        "verify": st.toggle("답변 검증", value=False, help="기본 끔(응답이 약 3~4초 빨라짐). 켜면 직접 맞는 사례가 없을 때 틀린 답이 더 줄어듭니다(개발셋 정답 제외 시험 불일치 20 → 8). 답변을 만든 뒤 한 번 더 읽어, 검색된 사례가 질문의 대상·조건을 직접 다루지 않으면 답변을 내지 않고, 인용 사례에서 확인되지 않은 문장에 표시를 붙입니다. 질문마다 LLM 호출이 한 번 더 듭니다(약 2~3초)."),
        "debug": st.toggle("검색 점수·검사 결과 보기", value=False),
    }
    st.divider()
    if st.button("질문 기록 지우기", width="stretch"):
        st.session_state.messages = []
        st.rerun()
    st.caption("대상: 금융위·금감원 법령해석·비조치의견서 회신(2014년 이후)과 현장건의 과제. 확정 판단은 하지 않습니다.")
    _b = data_basis()
    if _b:
        st.caption(f"데이터 기준: 회신 {_b['n']:,}건(마지막 회신일 {_b['last']}), 현장건의 {_b['f_n']:,}건(마지막 {_b['f_last']}). 시점 판정 기준일 {_b['checked']}. 이후 나온 회신·개정은 반영되지 않았습니다.")

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
