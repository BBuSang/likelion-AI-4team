"""LLM 탐지기 — OpenAI 모델로 개인정보를 찾음 (표준 라이브러리만 사용).

지시문·가짜 예시 문서·위치 찾기·형식 정리는 llm_hcx.py와 똑같은 것을 그대로 가져다 씀 → 모델만 바꿔 공정하게 비교.
HCX와 다른 점: OpenAI의 'JSON 형식 강제(structured outputs)'를 써서 답이 항상 올바른 JSON이고,
label도 7종 중 하나만 나올 수 있음 (HCX에서 겪은 JSON 깨짐·엉뚱한 라벨이 원천적으로 없음).

사용법
    python detectors/llm_openai.py <입력.jsonl> <출력.jsonl> --model gpt-4.1-mini [옵션]

옵션
    --model gpt-4.1-mini | gpt-5.4-mini | gpt-5.5 ...
    --effort none|low|medium|high       추론 모델(gpt-5.x)의 생각 강도 (기본 none). gpt-4.1 계열에는 없음
    --post-out <파일>                   형식 정리(직함 떼기 등)를 한 결과를 따로 저장
    --workers N                         동시에 보낼 문서 수 (기본 8). 문서별 ms는 각자 잰 값
    --ids D0001,D0002 / --limit N       일부 문서만

주의
    - 가상 데이터라 원문을 클라우드로 보내도 되는 '탐지 실험'에서만 사용. 실제 고객 문서에는 쓰면 안 됨.
    - 키는 .env의 openai-api-key에서 읽고, 화면·파일에 출력하지 않음.
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
import llm_hcx  # noqa: E402  (지시문·예시·위치 찾기·형식 정리를 같이 씀)

API_URL = "https://api.openai.com/v1/chat/completions"
SCHEMA = {"type": "json_schema", "json_schema": {"name": "pii_entities", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["entities"],
    "properties": {"entities": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["label", "text"],
        "properties": {"label": {"type": "string", "enum": llm_hcx.LABELS}, "text": {"type": "string"}}}}}}}}


def load_key():
    for line in open(os.path.join(llm_hcx.ROOT, ".env"), encoding="utf-8-sig"):
        if line.startswith("openai-api-key="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(".env에 openai-api-key가 없습니다")


def call_openai(key, model, effort, text, tries=6):
    messages = [{"role": "system", "content": llm_hcx.SYSTEM_PROMPT},
                {"role": "user", "content": llm_hcx.EXAMPLE_DOC},
                {"role": "assistant", "content": llm_hcx.EXAMPLE_ANSWER},
                {"role": "user", "content": text}]
    body = {"model": model, "messages": messages, "response_format": SCHEMA}
    if body_has_effort(model):  # 추론 모델은 temperature 대신 생각 강도
        body["reasoning_effort"] = effort
    else:
        body["temperature"] = 0
    data = json.dumps(body).encode("utf-8")
    for attempt in range(tries):
        req = urllib.request.Request(API_URL, data=data, method="POST",
                                     headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < tries - 1:
                time.sleep(float(e.headers.get("retry-after") or 2 ** attempt))
                continue
            raise RuntimeError(f"OpenAI 오류 {e.code}: {e.read().decode('utf-8')[:200]}")
        except (urllib.error.URLError, TimeoutError):
            if attempt < tries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def main():
    args = sys.argv[1:]
    opt = {k: args[i + 1] for i, k in enumerate(args) if k.startswith("--") and i + 1 < len(args)}
    in_path, out_path = [a for i, a in enumerate(args)
                         if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))][:2]
    model = opt.get("--model", "gpt-4.1-mini")
    effort = opt.get("--effort", "none")
    post_path = opt.get("--post-out")
    workers = int(opt.get("--workers", 8))
    docs = [json.loads(l) for l in open(in_path, encoding="utf-8") if l.strip()]
    if "--ids" in opt:
        ids = set(opt["--ids"].split(","))
        docs = [d for d in docs if d["id"] in ids]
    if "--limit" in opt:
        docs = docs[:int(opt["--limit"])]

    key = load_key()
    raw_dir = os.path.join(llm_hcx.ROOT, "outputs", "llm_raw", os.path.splitext(os.path.basename(out_path))[0])
    os.makedirs(raw_dir, exist_ok=True)

    def work(d):
        t0 = time.perf_counter()
        content = call_openai(key, model, effort, d["text"])
        ms = (time.perf_counter() - t0) * 1000
        with open(os.path.join(raw_dir, f"{d['id']}.txt"), "w", encoding="utf-8") as f:
            f.write("===== 1번째 =====\n\n" + content)
        items, err = llm_hcx.parse_entities(content)
        return d, items, err, ms

    t_all = time.perf_counter()
    stats = {"JSON 실패": 0, "못 찾은 값": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(work, docs))  # 결과는 입력 순서대로
    with open(out_path, "w", encoding="utf-8") as out:
        post = open(post_path, "w", encoding="utf-8") if post_path else None
        for d, items, err, ms in results:
            stats["JSON 실패"] += bool(err)
            ents, missing = llm_hcx.locate(d["text"], items)
            out.write(json.dumps({"id": d["id"], "entities": ents, "ms": ms}, ensure_ascii=False) + "\n")
            if post:
                ents, missing = llm_hcx.locate(d["text"], llm_hcx.post_process(items))
                post.write(json.dumps({"id": d["id"], "entities": ents, "ms": ms}, ensure_ascii=False) + "\n")
            stats["못 찾은 값"] += len(missing)
        if post:
            post.close()
    print(f"저장: {out_path}" + (f", {post_path}" if post_path else "")
          + f"  ({model}, effort={effort if body_has_effort(model) else '-'}, {len(docs)}건, {stats}, "
          + f"총 {time.perf_counter() - t_all:.0f}초)")


def body_has_effort(model):
    """추론 모델인가: gpt-4 계열(4.1·4o)만 추론이 없고, gpt-5·gpt-6 이후와 o 계열은 추론 모델."""
    return not model.startswith("gpt-4") and not model.startswith("gpt-3")


if __name__ == "__main__":
    main()
