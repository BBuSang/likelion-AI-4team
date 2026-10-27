"""LLM 탐지기 v2 — HCX(HyperCLOVA X)만으로 개인정보를 찾음 (규칙 탐지기와 합치지 않음, 표준 라이브러리만 사용).

사용법
    python detectors/llm_hcx.py <입력.jsonl> <출력.jsonl> [옵션]

옵션
    --model HCX-DASH-002 | HCX-007     (기본 HCX-DASH-002)
    --effort none|low|medium|high      HCX-007의 추론(생각) 강도 (기본 low). DASH에는 없음
    --strategy base|review             base: 한 번 묻기 / review: 한 번 묻고, 답을 보여주며 "빠뜨린 것 더 찾아라"로 한 번 더
    --runs N                           같은 문서를 N번 묻고 결과를 합침 (기본 1)
    --post-out <파일>                  형식 정리(라벨 이름·직함 떼기)를 한 결과를 따로 저장
    --ids D0001,D0002 / --limit N      일부 문서만

동작 순서
    1) 지시문 + 가짜 예시 문서 1개(풀이 포함) + 실제 문서를 HCX에 보냄
    2) HCX가 {"entities": [{"label", "text"}]} JSON으로 답함 (깨지면 고쳐 달라고 다시 요청, 그래도 안 되면 읽을 수 있는 부분만 건짐)
    3) LLM은 글자 위치를 못 세므로, 답한 text를 문서에서 직접 찾아 위치를 붙임
    4) run_compare.py 형식으로 저장. ms에는 API 왕복·대기 시간이 모두 포함됨

주의
    - 가상 데이터라 원문을 클라우드로 보내도 되는 '탐지 실험'에서만 사용. 실제 고객 문서에는 쓰면 안 됨.
    - 키는 .env의 HCX_API_KEY에서 읽고, 화면·파일에 출력하지 않음.
    - 예시 문서의 이름·번호는 데이터셋에 없는 가짜 값임 (데이터 값을 쓰면 LLM이 그대로 베껴 냄).
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.join(os.path.dirname(__file__), "..")
API_URL = "https://clovastudio.stream.ntruss.com/v1/openai/chat/completions"
LABELS = ["person", "phone_number", "email", "address",
          "resident_registration_number", "account_number", "card_number"]

# 지시문을 더 길게 보강(민원인·고객센터 제외, 숫자 라벨 확인)해 봤지만 dev 재현율이 0.987 → 0.976으로 내려가서 되돌림.
# 같은 효과는 post_process의 '숫자 라벨인데 숫자가 없으면 바로잡기'로 대신함.
SYSTEM_PROMPT = """너는 금융사 사내 문서에서 개인정보를 찾는 탐지기다.
하나 놓치는 것이 하나 더 찾는 것보다 훨씬 위험하다. 문서를 첫 줄부터 마지막 줄까지 한 줄씩 빠짐없이 확인한다.

찾을 라벨 7종 (label에는 아래 영어 이름만 쓴다)
- person: 사람 이름. 고객뿐 아니라 직원(상담원·담당자·작성자·참석자·결재자·수신자·발신자)도 모두 포함.
  · 메신저·대화 형식이면 줄 앞 "이름:"의 말하는 사람, 대화 중 부르거나 언급한 사람도 포함
  · 성 없이 이름만 쓴 것도 포함 (예: "민수야"의 "민수", "영희 씨"의 "영희")
  · 띄어 쓴 이름도 포함 (예: "이 영희")
- phone_number: 개인 전화번호. 하이픈·점·공백이 없거나 달라도 포함 (예: 01098765432, 010.2345.6789)
- email: 이메일 주소
- address: 집·회사 주소 (시·도부터 동·호수·층까지 한 덩어리)
- resident_registration_number: 주민등록번호
- account_number: 계좌번호
- card_number: 카드번호

찾지 말 것
- 회사 대표번호(1588-0000처럼 4자리-4자리), 내선번호, 민원번호·접수번호·날짜·금액
- 회사·부서·팀·상품 이름
- 성과 직함만 있는 표현 (예: "허 과장님", "김 대리") — 이름이 없으므로 개인정보 아님

text 작성 규칙
- 문서에 적힌 글자를 그대로 복사한다. 띄어쓰기·하이픈·점을 바꾸지 않는다.
- 이름 앞뒤의 소속·직함·호칭·조사는 뺀다 ("오태식 대리" → "오태식", "이영희는" → "이영희", "민수야" → "민수")
- 같은 사람이라도 문서에 적힌 모양이 다르면 모양마다 따로 쓴다 ("이 영희"와 "이영희" 둘 다)

출력: 설명 없이 JSON 하나만 출력한다. {"entities": [{"label": "...", "text": "..."}]}
개인정보가 없으면 {"entities": []}"""

# 가짜 예시 문서와 모범 답 — 데이터셋에 없는 값만 사용 (tools로 확인함)
EXAMPLE_DOC = """[상담 기록] 2026-03-02
상담원: 박철수 대리 / 내선 4412

이 영희 고객(01098765432)이 카드 분실을 신고함. 주민등록번호 880515-1234567로 본인 확인.
재발급 카드는 등록 주소(세종특별자치시 한솔동 가온로 12, 501호)로 발송. 결제 계좌 333-222111-09.
분실 카드 5555-6666-7777-8888 은 즉시 정지함. 안내 메일은 hgd.hong@example.org 로 보냄.
이영희는 추가 문의는 고객센터 1588-0000으로 하겠다고 함. 처리 결과는 허 과장님께 보고.

[사내 메신저]
오태식: 민수야, 홍길동 고객 건 정리됐어?
강민수: 응, 연락처 010-2345-6789로 통화 완료"""

EXAMPLE_ANSWER = json.dumps({"entities": [
    {"label": "person", "text": "박철수"}, {"label": "person", "text": "이 영희"},
    {"label": "phone_number", "text": "01098765432"},
    {"label": "resident_registration_number", "text": "880515-1234567"},
    {"label": "address", "text": "세종특별자치시 한솔동 가온로 12, 501호"},
    {"label": "account_number", "text": "333-222111-09"},
    {"label": "card_number", "text": "5555-6666-7777-8888"},
    {"label": "email", "text": "hgd.hong@example.org"},
    {"label": "person", "text": "이영희"},
    {"label": "person", "text": "오태식"}, {"label": "person", "text": "민수"},
    {"label": "person", "text": "홍길동"}, {"label": "person", "text": "강민수"},
    {"label": "phone_number", "text": "010-2345-6789"}]}, ensure_ascii=False)

REVIEW_PROMPT = ("위 답을 검토해라. 문서를 첫 줄부터 마지막 줄까지 다시 한 줄씩 보면서, 앞의 답에서 빠뜨린 개인정보만 "
                 "추가로 출력해라. 특히 직원 이름, 메신저에서 말하는 사람, 성 없이 쓴 이름, 띄어 쓴 이름, "
                 "형식이 다른 번호를 확인해라. 빠뜨린 것이 없으면 {\"entities\": []}")
FIX_JSON = ("앞의 답은 JSON 형식이 잘못되어 읽을 수 없다. 같은 내용을 올바른 JSON 하나로만 다시 출력해라. "
            '형식: {"entities": [{"label": "...", "text": "..."}]}')


def load_key():
    for line in open(os.path.join(ROOT, ".env"), encoding="utf-8-sig"):
        if line.startswith("HCX_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(".env에 HCX_API_KEY가 없습니다")


# ------------------------------------------------------------------ API 호출
def call_hcx(key, model, messages, effort, tries=8):
    body = {"model": model, "messages": messages}
    if model.startswith("HCX-007"):  # 추론 모델: 생각하는 데 쓰는 토큰까지 포함한 한도
        # 한도를 크게 잡으면 분당 토큰 한도(6만)를 미리 차지해 요청이 느려지므로, 추론 강도에 맞춰 필요한 만큼만
        limit = {"none": 1500, "low": 4000, "medium": 6000, "high": 8000}.get(effort, 4000)
        body.update(max_completion_tokens=limit, reasoning_effort=effort)
    else:
        body.update(max_tokens=2048, temperature=0)
    data = json.dumps(body).encode("utf-8")
    for attempt in range(tries):
        req = urllib.request.Request(API_URL, data=data, method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            if e.code == 429:  # 분당 요청·토큰 한도 초과 → 서버가 알려준 시간만큼 기다림
                wait = re.match(r"(\d+)", e.headers.get("x-ratelimit-reset-tokens")
                                or e.headers.get("x-ratelimit-reset-requests") or "10")
                time.sleep(int(wait.group(1)) + 1 if wait else 10)
                continue
            if e.code in (500, 502, 503, 504) and attempt < tries - 1:
                time.sleep(2 ** min(attempt, 4))
                continue
            raise RuntimeError(f"HCX 오류 {e.code}: {e.read().decode('utf-8')[:200]}")
        except (urllib.error.URLError, TimeoutError):
            if attempt < tries - 1:
                time.sleep(2 ** min(attempt, 4))
                continue
            raise
    raise RuntimeError("HCX 요청 한도 대기 후에도 실패")


def parse_entities(content):
    """답에서 JSON을 꺼냄. 깨졌으면 {"label": .., "text": ..} 짝만이라도 건짐."""
    m = re.search(r"\{.*\}", content, re.S)
    if m:
        try:
            items = json.loads(m.group(0)).get("entities", [])
            return [x for x in items if isinstance(x, dict) and x.get("text")], None
        except (json.JSONDecodeError, AttributeError):
            pass
    pairs = re.findall(r'"label"\s*:\s*"([^"]+)"\s*,\s*"text"\s*:\s*"([^"]+)"', content)
    return [{"label": l, "text": t} for l, t in pairs], ("JSON 깨짐(부분 복구)" if pairs else "JSON 없음")


def ask(key, model, effort, messages):
    """묻고, JSON이 깨졌으면 틀린 답을 보여주며 다시 쓰게 함 (최대 2번)."""
    answers = [call_hcx(key, model, messages, effort)]
    items, err = parse_entities(answers[-1])
    while err and len(answers) <= 2:
        messages = messages + [{"role": "assistant", "content": answers[-1]}, {"role": "user", "content": FIX_JSON}]
        answers.append(call_hcx(key, model, messages, effort))
        items, err = parse_entities(answers[-1])
    return answers, items, err


def detect_once(key, model, effort, strategy, text):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": EXAMPLE_DOC}, {"role": "assistant", "content": EXAMPLE_ANSWER},
                {"role": "user", "content": text}]
    answers, items, err = ask(key, model, effort, messages)
    if strategy == "review":
        first = json.dumps({"entities": items}, ensure_ascii=False)
        more_answers, more, err2 = ask(key, model, effort, messages + [
            {"role": "assistant", "content": first}, {"role": "user", "content": REVIEW_PROMPT}])
        answers += ["----- 검토 -----"] + more_answers
        items = items + more
        err = err or err2
    return answers, items, err


# ------------------------------------------------------------------ 위치 찾기
def variants(label, value):
    """이름은 붙여 쓴 형태·띄어 쓴 형태를 모두 찾음 ('이영희' ↔ '이 영희')."""
    forms = {value}
    if label == "person":
        bare = value.replace(" ", "")
        forms.add(bare)
        if len(bare) == 3:
            forms.add(bare[0] + " " + bare[1:])
    return forms


def locate(text, items):
    """LLM이 준 text를 문서에서 모두 찾아 위치를 붙임. 겹치면 긴 것을 남김."""
    found, missing = [], []
    for it in items:
        label, value = str(it.get("label", "")).strip(), str(it["text"]).strip()
        # 이름은 '김은영' 안의 '은영'처럼 다른 글자에 이어진 곳을 잡지 않도록 한글 경계 확인
        edge = r"(?<![가-힣])" if label == "person" else ""
        hits = [(m.start(), m.end()) for v in variants(label, value) if v
                for m in re.finditer(edge + re.escape(v), text)]
        if not hits:
            missing.append(value)
        found += [{"label": label, "start": s, "end": e} for s, e in hits]
    found.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    out, last_end = [], -1
    for e in found:
        if e["start"] >= last_end:
            out.append(e)
            last_end = e["end"]
    return out, missing


# ------------------------------------------------------------------ 형식 정리 (--post-out 일 때만)
# 탐지를 새로 하지 않고, LLM 답의 형식만 바로잡음: 라벨 이름 통일, 이름에 붙은 소속·직함 떼기
LABEL_MAP = {"name": "person", "이름": "person", "사람": "person", "per": "person", "people": "person",
             "phone": "phone_number", "전화번호": "phone_number", "휴대전화": "phone_number", "mobile": "phone_number",
             "주민등록번호": "resident_registration_number", "rrn": "resident_registration_number",
             "account": "account_number", "계좌번호": "account_number", "계좌": "account_number",
             "card": "card_number", "카드번호": "card_number",
             "주소": "address", "이메일": "email", "mail": "email", "e-mail": "email"}
TITLES = ("고객님", "고객", "팀장님", "팀장", "부장님", "부장", "과장님", "과장", "차장님", "차장", "대리님", "대리",
          "사원", "주임", "선임", "책임", "매니저", "이사", "대표", "실장", "본부장", "지점장", "상담원", "님", "씨")
TITLE_TAIL = re.compile(r"\s?(?:" + "|".join(TITLES) + r")$")
ORG_HEAD = re.compile(r"^\S*(?:팀|센터|본부|지점|부서)\s")


def post_process(items):
    """1) 라벨 이름 맞추기 (NAME·이름 → person 등)
    2) 숫자 라벨(전화·주민번호·계좌·카드)인데 값에 숫자가 없으면 라벨 실수로 보고 바로잡기
    3) 이름에 붙은 소속·직함 떼기 ('채권관리팀 이상훈' → '이상훈', '민준씨' → '민준')
    4) 떼고 나니 성 한 글자만 남으면 버리기 ('김 대리' → 성+직함만이므로 개인정보 아님)"""
    out = []
    for it in items:
        label = str(it.get("label", "")).strip()
        label = LABEL_MAP.get(label.lower(), label.lower() if label.lower() in LABELS else label)
        if label not in LABELS:  # 지어낸 라벨: 사람을 가리키는 이름이면 person, 아니면 버림 (직업·날짜 등)
            if re.search(r"leader|head|manager|staff|employee|customer|agent|담당|직원|고객|상담|팀장|성명", label, re.I):
                label = "person"
            else:
                continue
        value = str(it["text"]).strip()
        if label in LABELS[1:] and label not in ("email", "address") and not re.search(r"\d", value):
            # 숫자 라벨인데 숫자가 하나도 없으면 라벨 실수 → 한글 이름 모양이면 person, 아니면 버림
            if re.fullmatch(r"[가-힣]{1,2} ?[가-힣]{1,3}", value):
                label = "person"
            else:
                continue
        if label == "person":
            value = ORG_HEAD.sub("", value)
            while TITLE_TAIL.search(value):
                value = TITLE_TAIL.sub("", value).strip()
            if len(value.replace(" ", "")) <= 1:
                continue
        out.append({"label": label, "text": value})
    return out


# ------------------------------------------------------------------ 실행
def main():
    args = sys.argv[1:]
    opt = {k: args[i + 1] for i, k in enumerate(args) if k.startswith("--") and i + 1 < len(args)}
    in_path, out_path = [a for i, a in enumerate(args)
                         if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))][:2]
    model = opt.get("--model", "HCX-DASH-002")
    effort = opt.get("--effort", "low")
    strategy = opt.get("--strategy", "base")
    runs = int(opt.get("--runs", 1))
    post_path = opt.get("--post-out")
    docs = [json.loads(l) for l in open(in_path, encoding="utf-8") if l.strip()]
    if "--ids" in opt:
        ids = set(opt["--ids"].split(","))
        docs = [d for d in docs if d["id"] in ids]
    if "--limit" in opt:
        docs = docs[:int(opt["--limit"])]

    key = load_key()
    # HCX 원본 답변 (디버깅용, outputs/는 .gitignore 대상). 결과 파일 이름별 폴더에 저장
    raw_dir = os.path.join(ROOT, "outputs", "llm_raw", os.path.splitext(os.path.basename(out_path))[0])
    os.makedirs(raw_dir, exist_ok=True)
    stats = {"JSON 실패": 0, "못 찾은 값": 0}
    out = open(out_path, "w", encoding="utf-8")
    post = open(post_path, "w", encoding="utf-8") if post_path else None
    t_all = time.perf_counter()
    for i, d in enumerate(docs, 1):
        t0 = time.perf_counter()
        items, raws = [], []
        for r in range(runs):
            answers, got, err = detect_once(key, model, effort, strategy, d["text"])
            items += got
            raws += [f"===== {r + 1}번째 =====", *answers]
            stats["JSON 실패"] += bool(err)
        ms = (time.perf_counter() - t0) * 1000
        with open(os.path.join(raw_dir, f"{d['id']}.txt"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(raws))
        ents, missing = locate(d["text"], items)
        out.write(json.dumps({"id": d["id"], "entities": ents, "ms": ms}, ensure_ascii=False) + "\n")
        note = ""
        if post:
            p_ents, missing = locate(d["text"], post_process(items))
            post.write(json.dumps({"id": d["id"], "entities": p_ents, "ms": ms}, ensure_ascii=False) + "\n")
        stats["못 찾은 값"] += len(missing)
        note += f" ⚠ 못 찾은 값 {missing[:3]}" if missing else ""
        print(f"[{i}/{len(docs)}] {d['id']} {len(ents)}개 {ms / 1000:.1f}초{note}", flush=True)
    out.close()
    if post:
        post.close()
    print(f"저장: {out_path}" + (f", {post_path}" if post_path else "")
          + f"  ({model}, effort={effort if model.startswith('HCX-007') else '-'}, {strategy}, runs={runs}, "
          + f"{stats}, 총 {time.perf_counter() - t_all:.0f}초)")


if __name__ == "__main__":
    main()
