"""데모 처리 로직 (화면과 분리). `demo.py`(92번 이전 데모)의 처리 부분을 동작 그대로 옮긴 것이다. Streamlit을 들여오지 않으므로 화면(`app.py`)과 다른 앱에서 함께 쓸 수 있다.

흐름 (89번에서 dev7·test7로 잰 설정 그대로)
1. 질문의 개인정보를 가린다
2. 기존 회신: 벡터 후보 20 → LLM 리랭킹 5 (`rerank.py`, 실패하면 벡터 상위 5)
3. 현장건의: 별도 후보군에서 점수 ≥ 0.65 이면서 회신 1위 벡터 점수보다 높은 묶음만 최대 2건 (`exmnt_search.py`)
4. 답변: `APP_SYSTEM_V3`(110번: V2와 규칙은 같고 형식만 마크다운, 현장건의가 붙은 질문에만 현장건의 규칙 추가) → 출력 검사
5. 첫 답변이 "찾지 못했습니다"(또는 인용 없음)일 때만 키워드 순차 완화로 최대 3건을 더해 한 번 더 답한다 (`keyword_relax.py`)
6. (선택, 기본 끔) 답변 검증(`verify.check`, 99번): 인용 사례가 질문의 대상·조건을 직접 다루지 않으면 "찾지 못했습니다"로 바꾸고, 인용 사례에 없다고 본 줄은 지우지 않고 표시만 한다
`rag.py`는 평가 지문에 들어 있어 고치지 않는다.
"""
import calendar
import functools
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag  # noqa: E402
import exmnt_search as es  # noqa: E402
import keyword_relax as kr  # noqa: E402
import rerank as rr  # noqa: E402
import verify as vf  # noqa: E402
import gate  # noqa: E402
from prompts import APP_SYSTEM_V3 as APP_SYSTEM, REPLY_DATE  # noqa: E402

CITE = re.compile(r"\[([LOF]\d+)\]")          # F = 현장건의
_POOL = ThreadPoolExecutor(8)          # 124번: 판정을 답변과 동시에 돌리는 데 쓴다

# 124번: 같은 질문 글의 임베딩을 한 요청 안에서 여러 번 계산하던 것(리랭킹 후보, 현장건의, 키워드 완화)을 한 번만 계산한다.
# 같은 글이면 같은 벡터라 결과는 바뀌지 않는다(101번에서 같은 결과임을 확인한 변경). 최근 256개 글만 기억한다.
_embed_raw = rag.embed


@functools.lru_cache(maxsize=256)
def _embed_cached(text):
    return tuple(_embed_raw(text))


def _embed(text):
    return list(_embed_cached(text))


rag.embed = _embed
ASIDE = re.compile(r"\[(\d{4}-\d{2}-\d{2} 회신에서[^\[\]]*?)\]")          # 문장 전체를 대괄호로 감싼 경우
HEADING = re.compile(r"^\s*#{1,6}\s.*$\n?", re.M)          # 마크다운 제목 줄 (V3). 검증 단계에는 넘기지 않는다


# 114번: `rag.mask_pii`의 계좌번호 규칙(숫자-숫자-숫자)이 날짜·문서번호까지 가린다. `rag.py`는 고치지 않으므로, 가려지지 않는 표기로 먼저 바꾼 뒤 넘긴다.
# 아래 단계(리랭킹·키워드 추출·검증)가 `rag.mask_pii`를 다시 불러도 바뀐 표기는 걸리지 않는다.
DATE = re.compile(r"(?<![\d-])((?:19|20)\d{2})-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])(?![\d-])")          # 2024-01-15 → 2024년 1월 15일
DOC_NO = re.compile(r"(제\s?)(\d{1,6}(?:-\d{1,6}){2})(\s?호)")          # 제2023-12-45호 → 제2023–12–45호 (붙임표를 줄표로)
MULTI_CITE = re.compile(r"\[\s*([LOF]\d+(?:\s*[,·/]\s*[LOF]\d+)+)\s*\]")          # [L123, O45] → [L123][O45]


def mask(question):
    """질문의 개인정보를 가린다. 날짜와 '제…호' 문서번호는 계좌번호로 잘못 가려지지 않게 표기를 먼저 바꾼다."""
    t = DATE.sub(lambda m: f"{m.group(1)}년 {int(m.group(2))}월 {int(m.group(3))}일", question)
    t = DOC_NO.sub(lambda m: m.group(1) + m.group(2).replace("-", "–") + m.group(3), t)
    return rag.mask_pii(t)


def split_cites(raw):
    """한 대괄호에 사례번호를 여러 개 쓴 것을 하나씩으로 나눈다 (프롬프트 규칙 2의 형식으로 맞춤)."""
    return MULTI_CITE.sub(lambda m: "".join(f"[{k}]" for k in re.findall(r"[LOF]\d+", m.group(1))), raw)


# ---------- 기간 조건 분리 (117번). 116번: 질문에 날짜 조건이 있으면 검색이 날짜 글자에 끌려가고, 답변 모델은 조건에 맞는 사례가 없다며 거절한다.
# 질문에서 "회신 시기를 가리키는 말"만 규칙으로 떼어 회신일 필터로 쓰고, 검색·리랭킹·답변에는 조건을 뺀 질문을 쓴다. LLM을 부르지 않는다.
# 사실관계 속 날짜("2021년 6월 30일까지 유예된 …")는 건드리지 않도록, 날짜 뒤에 회신을 가리키는 말(나온·회신된 / 회신·사례 등)이나 "시행"이 올 때만 조건으로 본다.
_D = r"(?:(?:19|20)\d{2}\s*[-./]\s*\d{1,2}\s*[-./]\s*\d{1,2}\.?|(?:19|20)\d{2}\s*년(?:도)?(?:\s*\d{1,2}\s*월(?:\s*\d{1,2}\s*일)?)?)"
_VERB = r"\s*(?:나온|나왔던|회신된|회신한|발표된|있었던)"
_NOUN = r"(?=\s*(?:회신|사례|선례|법령해석|비조치|해석))"
_TAIL = rf"(?:(?:에|의|로)?{_VERB}|의?{_NOUN})"
_LAW = r"(?:\s*(?:개정(?:된|한)?|감독규정|시행령|시행규칙|법령|규정|세칙|고시|법))*"
PERIOD_RULES = [          # (종류, 규칙) 위에서부터 처음 맞는 것 하나만 쓴다
    ("basis", re.compile(rf"({_D})\s*(?:자|부터|부)?\s*시행(?:된|되는|한|하는)?(?:{_LAW}\s*기준(?:으로|에서)?)?")),          # 2024-01-15 시행된 개정 감독규정 기준으로
    ("range", re.compile(rf"({_D})\s*(?:부터|에서|~|∼)\s*({_D})\s*(?:까지|사이)(?:에)?{_TAIL}")),          # 2022년부터 2023년까지 나온
    ("from", re.compile(rf"({_D})\s*(?:이후|부터){_TAIL}")),          # 2024-01-15 이후에 나온
    ("to", re.compile(rf"({_D})\s*(?:이전|까지){_TAIL}")),          # 2020년 이전 회신
    ("recent", re.compile(rf"최근\s*(\d{{1,2}})\s*(년|개월)(?:간|\s*동안)?{_TAIL}")),          # 최근 3년간 나온
    ("in", re.compile(rf"({_D})(?:(?:에|\s*중에?)?{_VERB}|의?{_NOUN})")),          # 2023년에 나온 / 2023년 회신
]


def _bound(token, end):
    """날짜 표기 → date. 월·일이 없으면 기간의 처음(end=False)이나 끝(end=True)으로 채운다."""
    n = [int(x) for x in re.findall(r"\d+", token)]
    y, m, d = n[0], (n[1] if len(n) > 1 else None), (n[2] if len(n) > 2 else None)
    m2 = m or (12 if end else 1)
    return date(y, m2, d or (calendar.monthrange(y, m2)[1] if end else 1))


def split_period(question, today=None):
    """질문 → (조건을 뺀 질문, 기간 조건 또는 None). 기간 조건: {"from", "to"(date 또는 None), "kind", "text"(떼어 낸 글), "label"}.
    kind "basis"는 '○○ 시행된 개정 … 기준'이다. 시행일을 '그 날짜 이후에 나온 회신'으로 바꿔 적용한다(근사)."""
    for kind, rx in PERIOD_RULES:
        m = rx.search(question)
        if not m:
            continue
        try:
            if kind == "recent":
                t = today or date.today()
                k = int(m.group(1)) * (12 if m.group(2) == "년" else 1)
                y, mo = divmod(t.year * 12 + t.month - 1 - k, 12)
                a, b = date(y, mo + 1, min(t.day, calendar.monthrange(y, mo + 1)[1])), None
            elif kind == "range":
                a, b = _bound(m.group(1), False), _bound(m.group(2), True)
            elif kind in ("from", "basis"):
                a, b = _bound(m.group(1), False), None
            elif kind == "to":
                a, b = None, _bound(m.group(1), True)
            else:
                a, b = _bound(m.group(1), False), _bound(m.group(1), True)
        except ValueError:                         # 없는 날짜(2024-13-40 등)는 조건으로 보지 않는다
            continue
        topic = re.sub(r"\s{2,}", " ", question[:m.start()] + " " + question[m.end():]).strip(" ,·")
        if len(re.sub(r"\W", "", topic)) < 8 or (a and b and a > b):          # 조건을 빼면 찾을 말이 남지 않는 질문은 나누지 않는다
            return question, None
        label = f"{a} ~ {b}" if a and b else f"{a} 이후" if a else f"{b} 이전"
        return topic, {"from": a, "to": b, "kind": kind, "text": m.group(0).strip(), "label": label}
    return question, None


def in_range(d, rng):
    """회신일(date 또는 'YYYY-MM-DD' 글)이 기간 안인지. 날짜를 모르면 None."""
    if isinstance(d, str):
        try:
            d = date.fromisoformat(d)
        except ValueError:
            return None
    if d is None:
        return None
    return (rng[0] is None or d >= rng[0]) and (rng[1] is None or d <= rng[1])


def candidates_in(conn, masked, rng, n=rr.N_CAND):
    """회신일이 기간 안인 묶음만으로 벡터 상위 n개 (`rr.candidates`와 같은 모양). 묶음의 회신일은 대표 사례의 것으로 본다(화면 카드가 대표 사례라서).
    조건을 건 검색은 HNSW 색인으로 하면 개수가 모자랄 수 있어, 색인을 타지 않게 점수로 정렬한다(청크 약 9천 개라 전수 비교로 충분)."""
    v = "[" + ",".join(map(str, rag.embed(masked))) + "]"
    rows = conn.execute("""
        select x.group_id, x.case_key, x.chunk_type, x.text, 1 - (x.embedding <=> %(v)s::vector) as score
        from interp.chunks x join interp.case_groups g on g.group_id = x.group_id join interp.cases c on c.case_key = g.representative_case_key
        where c.reply_date is not null and (%(a)s::date is null or c.reply_date >= %(a)s::date) and (%(b)s::date is null or c.reply_date <= %(b)s::date)
        order by score desc limit %(n)s""", {"v": v, "a": rng[0], "b": rng[1], "n": rr.FETCH}).fetchall()
    best = {}
    for r in rows:
        best.setdefault(r["group_id"], r)
    top = list(best.values())[:n]
    groups = {g["group_id"]: g for g in conn.execute(
        "select g.group_id, g.n_cases, g.representative_case_key, c.title from interp.case_groups g join interp.cases c on c.case_key = g.representative_case_key where g.group_id = any(%s)",
        ([t["group_id"] for t in top],)).fetchall()}
    return [{"doc_id": groups[t["group_id"]]["representative_case_key"], "group_id": t["group_id"], "title": groups[t["group_id"]]["title"], "text": t["text"],
             "score": float(t["score"]), "chunk_type": t["chunk_type"], "hit_case": t["case_key"], "n_same": groups[t["group_id"]]["n_cases"] - 1} for t in top]


def rank(conn, masked, use_rerank=True, rng=None):
    """후보 20 → (리랭킹) 5. 반환 (후보, 최종 5건, 순서가 바뀌었는지). 리랭킹을 껐거나 호출이 실패하면 벡터 순서 상위 5개."""
    cands = candidates_in(conn, masked, rng) if rng else rr.candidates(conn, masked)
    if use_rerank and cands:
        try:
            ids = [c["doc_id"] for c in cands]
            order, _, _ = rr.rerank(masked, ids, rr.load_brief(conn, ids))
            by_id = {c["doc_id"]: c for c in cands}
            return cands, [by_id[i] for i in order], order != ids[:5]
        except Exception:
            pass
    return cands, cands[:5], False


def ask(question, relax=True, use_rerank=True, use_exmnt=True, use_verify=False, use_period=True, use_gate=True):          # use_gate 기본 켬(120번). use_verify는 125번 사용자 결정으로 기본 끔(시간 약 −3.5초)
    """질문 하나를 처리한다. 질문에 회신 시기 조건이 있으면(117번) 조건을 떼어 회신일 조건으로 쓴다. 검색·답변에는 조건을 뺀 질문을 쓴다.
    1. 기간 없이 찾은 1위 회신(리랭킹 뒤)이 기간 안이면 → 그 기간의 회신만으로 다시 찾아 답한다
    2. 1위가 기간 밖이거나, 기간 안에서 답이 나오지 않으면 → 기간을 풀고 답한다(`period.fallback`). 화면이 기간 밖 사례임을 표시한다
    1의 확인을 두는 이유: 기간 안에 맞는 회신이 없는데 기간으로만 좁히면, 남은 것 중 가까운 사례로 답해 버린다(117번 첫 시험).
    사례 카드마다 기간 안인지(`in_period`)를 붙인다."""
    topic, period = split_period(question) if use_period else (question, None)
    if not period:
        return _ask(question, relax, use_rerank, use_exmnt, use_verify, use_gate=use_gate)
    t0 = time.time()
    rng = (period["from"], period["to"])
    with rag.connect() as conn:
        pre = rank(conn, mask(topic), use_rerank)
        top = pre[1][0]["doc_id"] if pre[1] else None
        row = conn.execute("select reply_date from interp.cases where case_key = %s", (top,)).fetchone() if top else None
    top_in = bool(row and in_range(row["reply_date"], rng))
    first = _ask(topic, False, use_rerank, use_exmnt, use_verify, rng, use_gate=use_gate) if top_in else None          # 기간 안에서만. 키워드 완화는 기간을 푼 뒤에만 한다
    ok = bool(first and first["cited"] and not first["check"]["not_found"])
    res = first if ok else _ask(topic, relax, use_rerank, use_exmnt, use_verify, pre=pre, use_gate=use_gate)
    if not ok and first:
        res["usage"] = {k: v + first["usage"].get(k, 0) for k, v in res["usage"].items()}
    res["seconds"]["total"] = round(time.time() - t0, 1)
    for c in res["cards"].values():
        c["in_period"] = in_range(c["reply_date"], rng)
    res["question"] = question
    res["period"] = {"label": period["label"], "kind": period["kind"], "text": period["text"], "topic": topic, "fallback": not ok,
                     "why": None if ok else ("기간 없이 찾은 1위 회신이 기간 밖" if not top_in else "기간 안 회신으로는 답이 나오지 않음"),
                     "top": top, "top_date": str(row["reply_date"]) if row else "", "from": str(period["from"] or ""), "to": str(period["to"] or "")}
    return res


def warm():
    """임베딩 모델을 불러온다 (처음 30~80초)."""
    rag.embed("준비")
    return True


def _answer(masked, context, ids, system=APP_SYSTEM):
    raw, usage = rag.llm([{"role": "system", "content": system}, {"role": "user", "content": f"[질문]\n{masked}\n\n[사례]\n{context}"}])
    raw = split_cites(raw.strip())
    check = rag.check_output(raw, ids)
    cited = [k for k in dict.fromkeys(CITE.findall(raw)) if k in ids]
    return raw, usage, check, cited


def _answer_merged(masked, context, ids, system, log):
    """판정 + 답변을 한 번의 호출로(123번, use_gate="merged"). 규칙(결과를 보기 전에 정함):
    - '직접' 사례가 없거나 답변이 "찾지 못했습니다"면 → 찾지 못했습니다
    - 답변이 '간접'으로 판정한 사례를 인용하면 → 찾지 못했습니다(자기 판정과 어긋난 답은 쓰지 않는다, log["inconsistent"]에 셈)
    - 출력 형식이 깨지면 → 판정 없이 예전처럼 답한다(`_answer`, 호출 1회 추가, log["error"]에 남김)"""
    try:
        v, answer, usage = gate.merged(system, masked, context)
    except Exception as e:                         # 키·주소가 담길 수 있어 종류만 남긴다
        log["error"] = type(e).__name__
        return _answer(masked, context, ids, system)
    log["verdicts"].update(v)
    log["unlisted"] = log.get("unlisted", 0) + usage.pop("unlisted", 0)          # 모델이 판정 목록에서 뺀 사례 수('간접'으로 봄)
    raw = split_cites(answer)
    cited = [k for k in dict.fromkeys(CITE.findall(raw)) if k in ids]
    direct = {k for k, x in v.items() if x == "직접"}
    if cited and not set(cited) <= direct:
        log["inconsistent"] = log.get("inconsistent", 0) + 1
    if not direct or rag.NOT_FOUND in raw or not set(cited) <= direct:
        return rag.NOT_FOUND, usage, rag.check_output(rag.NOT_FOUND, ids), []
    return raw, usage, rag.check_output(raw, ids), cited


def _answer_parallel(masked, context, ids, system, log):
    """판정과 답변을 동시에(124번, use_gate="parallel"). 판정 결과로 아래처럼 정한다(결과를 보기 전에 정함).
    - '직접' 사례가 없으면 → 찾지 못했습니다(먼저 만든 답은 버린다)
    - 먼저 만든 답이 '직접' 사례만 인용했으면 → 그 답을 쓴다(판정 시간이 답변 시간 안에 숨는다)
    - 그 밖(간접 사례를 인용했거나, 찾지 못했다고 했거나, 인용이 없음) → '직접' 사례만으로 다시 답한다(판정만 쓰는 방식과 같은 답, 이때만 시간이 늘어난다)
    - 판정 호출이 실패하면 → 먼저 만든 답을 쓴다(판정 실패 처리와 같음, log["error"])
    답변은 지금 스레드에서 부른다(평가 스크립트가 스레드별로 답변 문맥을 잡기 때문)."""
    new = [i for i in ids if i not in log["verdicts"]]
    fut = _POOL.submit(gate.judge, masked, gate.keep(context, new)) if new else None
    first = _answer(masked, context, ids, system)
    if fut:
        try:
            v, u = fut.result()
            log["verdicts"].update(v)
            log["usage"] = {k: log["usage"][k] + u[k] for k in ("input", "output")}
        except Exception as e:                     # 키·주소가 담길 수 있어 종류만 남긴다
            log["error"] = type(e).__name__
            return first
    direct = [i for i in ids if log["verdicts"].get(i) == "직접"]
    if not direct:
        return rag.NOT_FOUND, first[1], rag.check_output(rag.NOT_FOUND, ids), []
    raw, usage, check, cited = first
    if cited and not check["not_found"] and set(cited) <= set(direct):
        log["kept_first"] = log.get("kept_first", 0) + 1
        return first
    log["reanswered"] = log.get("reanswered", 0) + 1
    system = APP_SYSTEM + es.EXMNT_RULES if any(i.startswith("F") for i in direct) else APP_SYSTEM
    raw2, u2, check2, cited2 = _answer(masked, gate.keep(context, direct), direct, system)
    return raw2, {k: usage.get(k, 0) + u2.get(k, 0) for k in u2}, check2, cited2


def _gated(masked, context, ids, use_gate, log):
    """직접성 판정(120번). 끄면 `_answer`와 같다. 켜면 '직접'인 사례만으로 답하고, 없으면 답변 모델을 부르지 않고 "찾지 못했습니다".
    log: 이 질문의 판정 기록 {"verdicts": {...}, "usage": {...}, "error": ...}. 이미 판정한 사례는 다시 판정하지 않는다(키워드 완화로 늘어난 사례만 새로 본다).
    판정 호출이 실패하면 판정 없이 예전처럼 답한다(리랭킹 실패와 같은 처리, log["error"]에 남김)."""
    system = APP_SYSTEM + es.EXMNT_RULES if any(i.startswith("F") for i in ids) else APP_SYSTEM
    if not use_gate:
        return _answer(masked, context, ids, system)
    if use_gate == "merged":
        return _answer_merged(masked, context, ids, system, log)
    if use_gate == "parallel":
        return _answer_parallel(masked, context, ids, system, log)
    new = [i for i in ids if i not in log["verdicts"]]
    if new:
        try:
            v, u = gate.judge(masked, gate.keep(context, new))
            log["verdicts"].update(v)
            log["usage"] = {k: log["usage"][k] + u[k] for k in ("input", "output")}
        except Exception as e:                     # 키·주소가 담길 수 있어 종류만 남긴다
            log["error"] = type(e).__name__
            return _answer(masked, context, ids, system)
    direct = [i for i in ids if log["verdicts"].get(i) == "직접"]
    if not direct:
        return rag.NOT_FOUND, {"input": 0, "output": 0, "reasoning": 0}, rag.check_output(rag.NOT_FOUND, ids), []
    system = APP_SYSTEM + es.EXMNT_RULES if any(i.startswith("F") for i in direct) else APP_SYSTEM
    return _answer(masked, gate.keep(context, direct), direct, system)


def _ask(question, relax=True, use_rerank=True, use_exmnt=True, use_verify=False, rng=None, pre=None, use_gate=False):
    """질문 하나를 처리해 화면에 필요한 값(답변, 인용 순서, 사례 카드)을 돌려준다.
    rng=(시작일, 끝일)을 주면 회신일이 그 안인 사례만 쓴다. 그 기간에 후보가 하나도 없으면 None.
    pre=`rank()`의 결과를 주면 검색·리랭킹을 다시 하지 않고 그것을 쓴다.
    use_gate: 직접성 판정(120번). 검색된 사례 중 질문에 직접 답하는 것만으로 답한다."""
    t0 = time.time()
    masked = mask(question)
    relax_info = None
    gate_log = {"verdicts": {}, "usage": {"input": 0, "output": 0}, "error": None}
    with rag.connect() as conn:
        if pre or rng or use_rerank:
            cands, hits, reranked = pre or rank(conn, masked, use_rerank, rng)
            if not cands:
                return None
        else:
            cands, hits, reranked = None, rag.search(masked, 5, conn), False
        ids = [h["doc_id"] for h in hits]
        cases = rag.load_cases(ids, conn)
        ref = max(c["score"] for c in cands) if cands else (hits[0]["score"] if hits else None)     # 기존 회신 1위 벡터 점수 (리랭킹 전)
        fields, f_top1 = es.search(conn, rag.embed(masked), ref) if use_exmnt else ([], None)
        if rng:
            fields = [f for f in fields if in_range(f["reply_date"], rng)]
    t_search = time.time() - t0
    f_ids = [f["case_key"] for f in fields]
    f_ctx = "".join("\n\n" + es.block(f) for f in fields)
    context = "\n\n".join(rag.case_block(cases[i]) for i in ids)          # 현장건의가 붙은 질문에만 현장건의 규칙을 더한다(`_gated`)
    raw, usage, check, cited = _gated(masked, context + f_ctx, ids + f_ids, use_gate, gate_log)
    used_ctx = context + f_ctx
    extra = []
    if relax and (check["not_found"] or not cited):
        with rag.connect() as conn:
            concepts, _ = kr.extract_concepts(masked)
            groups = {r["group_id"] for r in conn.execute("select group_id from interp.case_groups where representative_case_key = any(%s)", (ids,)).fetchall()}
            found = kr.relax_search(conn, concepts, rag.embed(masked), groups)
            extra = found["extra"]
            if extra:
                cases.update(rag.load_cases([e["doc_id"] for e in extra], conn))
        relax_info = {"concepts": concepts, "trail": found["trail"], "extra": [e["doc_id"] for e in extra], "retried": bool(extra)}
        if extra:
            context2 = context + "".join("\n\n" + kr.extra_block(cases[e["doc_id"]], e) for e in extra) + f_ctx
            raw, u2, check, cited = _gated(masked, context2, ids + [e["doc_id"] for e in extra] + f_ids, use_gate, gate_log)
            usage = {k: usage[k] + u2[k] for k in usage}
            used_ctx = context2
    if use_gate:
        usage = {**usage, "input": usage["input"] + gate_log["usage"]["input"], "output": usage["output"] + gate_log["usage"]["output"]}
    verify_info = None
    if use_verify:                                 # 답변 검증: 직접 다루지 않으면 거절, 없는 줄은 표시만 (답변 글은 거절할 때만 바뀐다)
        new_raw, verify_info = vf.check(masked, HEADING.sub("", raw), used_ctx)          # 제목 줄은 판정 대상이 아니다
        usage = {**usage, "input": usage["input"] + verify_info["usage"]["input"], "output": usage["output"] + verify_info["usage"]["output"]}
        if verify_info["abstained"]:
            raw, cited = new_raw, []
            check = rag.check_output(raw, ids + [e["doc_id"] for e in extra] + f_ids)
    by_extra = {e["doc_id"]: e for e in extra}
    all_ids = ids + [e["doc_id"] for e in extra]
    by_hit = {h["doc_id"]: h for h in hits}
    cards = {}
    for i in all_ids:
        c = cases[i]
        src = by_hit.get(i) or by_extra[i]
        cards[i] = {"case_key": i, "title": (c["title"] or "").strip(), "gubun": c["gubun"], "reply_date": str(c["reply_date"] or "미상"), "answer_type": c["answer_type"], "doc_class": c["doc_class"],
                    "question": (c["question"] or "").strip(), "answer": (c["answer"] or "").strip(), "url": c["url"], "time_mark": c["time_mark"], "time_label": c["time_label"],
                    "citations": c["citations"], "n_same": src["n_same"], "score": src["score"], "chunk_type": by_hit[i]["chunk_type"] if i in by_hit else "키워드 완화",
                    "relaxed": {"matched": by_extra[i]["matched"], "of": by_extra[i]["of"], "missing": by_extra[i]["missing_terms"]} if i in by_extra else None}
    for f in fields:                               # 현장건의 카드 (시점 판정은 아직 없다)
        cards[f["case_key"]] = {"case_key": f["case_key"], "title": (f["title"] or "").strip(), "gubun": "현장건의", "reply_date": str(f["reply_date"] or "미상"), "answer_type": f["answer_type"],
                                "doc_class": "field_proposal", "question": (f["question"] or "").strip(), "answer": (f["answer"] or "").strip(), "url": f["url"], "time_mark": None,
                                "time_label": "", "citations": [], "n_same": f["n_same"], "score": f["score"], "chunk_type": "현장건의", "relaxed": None,
                                "changed_after": f["changed_after"], "already_in_place": f["already_in_place"]}
    invalid = [k for k in dict.fromkeys(CITE.findall(raw)) if k not in cards]          # 문맥에 준 적 없는 사례번호 (L·O·F 모두. `rag.check_output`은 L·O만 본다)
    return {"question": question, "masked": masked, "raw": raw, "cited": cited, "invalid": invalid, "cards": cards, "order": all_ids + f_ids, "check": check, "usage": usage, "relax": relax_info, "reranked": reranked, "verify": verify_info, "gate": gate_log if use_gate else None,
            "exmnt": {"top1": f_top1, "ref_score": ref, "attached": f_ids, "threshold": es.THRESH} if use_exmnt else None,
            "seconds": {"search": round(t_search, 1), "total": round(time.time() - t0, 1)}}


def trail_text(info):
    """키워드 완화 과정: '적금·캐시백·특별이익 3개 모두 0건 → 2개 이상 2건'. 처음으로 결과가 나온 단계까지만."""
    if not info:
        return None
    if not info["concepts"]:
        return "범위를 넓혀 다시 찾으려 했으나 질문에서 회신 검색용 핵심어를 뽑지 못했습니다."
    names = "·".join(c[0] for c in info["concepts"])
    steps = []
    for t in info["trail"]:
        label = f"{t['need']}개 모두" if t["need"] == len(info["concepts"]) else f"{t['need']}개 이상"
        steps.append(f"{label} 일치 {t['groups']}건")
        if t["groups"]:
            break
    return f"상위 5건에서 찾지 못해 키워드로 넓혀 다시 찾았습니다. 키워드 검색({names}): " + " → ".join(steps)


def answer_text(raw, keep_md=False):
    """고정 안내 문구·회신 날짜 머리말·굵은 글씨를 떼어 낸 답변 글. 근거 번호 `[L123]`는 그대로 둔다(화면이 배지로 바꾼다).
    keep_md=True면 굵은 글씨 표시(`**`)를 남긴다(마크다운으로 렌더링할 때)."""
    t = raw.replace(rag.CLOSING, "").strip()
    t = ASIDE.sub(r"\1", t)
    for rx in REPLY_DATE:
        t = rx.sub("", t)
    return re.sub(r"[ \t]{2,}", " ", t if keep_md else t.replace("**", ""))


def partly_not_found(res):
    """답변 앞부분에서 '직접 일치하는 회신은 찾지 못했다'고 하면서 사례를 인용한 경우."""
    head = res["raw"][:140]
    return bool(res["cited"]) and "찾지 못" in head and not res["check"]["not_found"]
