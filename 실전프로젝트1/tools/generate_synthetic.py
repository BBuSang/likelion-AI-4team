"""일상 카테고리의 가상 문서를 OpenAI 모델로 대량 생성한다 (규칙과 무관한 시험 데이터).

편향 줄이기
- 프롬프트에는 탐지 규칙을 전혀 넣지 않는다. 카테고리·문체와 라벨 표시 기준(데이터셋 정답 기준)만 준다.
- 값의 형식(전화번호 모양, 주소 형식, 이름 등)도 모델이 정한다.
- 생성 결과는 사람이(그리고 규칙 작성자가) 읽지 않고 parse_synthetic.py 로 형식만 검사한다.

카테고리 20 × 문체 5 = 100묶음, 묶음당 20건 → 2,000건. 이미 만든 묶음은 건너뛴다(이어하기).

사용법
  python -X utf8 tools/generate_synthetic.py --batches 1      # 시범 1묶음
  python -X utf8 tools/generate_synthetic.py                  # 전체 100묶음
결과: data/generated/raw/<모델>_<묶음>.txt, data/generated/usage.jsonl
"""
import argparse
import json
import os
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 맨 위 폴더의 common·run_compare·detectors 사용

from concurrent.futures import ThreadPoolExecutor

from common import DATA, load_env

CATEGORIES = ["상담 기록", "환불 요청", "배송·택배 문의", "병원·식당·숙소 예약", "민원 접수", "중고거래",
              "이사·부동산 계약", "보험금 청구", "채용·입사 지원", "학교·학원 공지", "카드 분실·정지", "대출·금융 상담",
              "회사 업무 메일", "단체 대화방", "고객 문의 게시판", "택배·퀵 접수", "경조사 안내", "관리비·공과금 안내",
              "동호회·모임 공지", "항공·여행 예약"]
STYLES = ["공식 문서·기록", "이메일", "메신저 대화", "짧은 메모", "게시판 글"]
DOCS_PER_BATCH = 20

PROMPT = """당신은 한국어 문서 작성자입니다. 개인정보 탐지기를 시험하기 위한 "가상의" 한국어 문서를 만들어 주세요.

[이번 요청]
- 카테고리: {category}
- 문체: {style}
- 개수: {n}건
- 묶음 번호: {batch}

[작성 원칙]
1. 실제 사람이 일상이나 업무에서 쓰는 글처럼 자연스럽게 써 주세요. 사람마다 쓰는 습관이 다르듯 표기·형식·길이·말투를 문서마다 다양하게 해 주세요.
2. 문서 길이는 2줄짜리 짧은 메모부터 15줄 정도의 긴 기록까지 섞어 주세요.
3. {n}건 중 2~3건은 개인정보가 전혀 없는 문서로 써 주세요.
4. 개인정보가 아닌 숫자·명칭(날짜, 시간, 금액, 주문번호, 접수번호, 회사 대표번호, 회사·기관·상품 이름 등)도 자연스럽게 섞어 주세요.
5. 모든 개인정보는 실존하지 않는 가상 값으로 써 주세요.
   - 이메일 도메인은 example.com, example.net, example.co.kr 같은 예시 도메인만 사용
   - 전화번호는 실제 번호와 겹치지 않도록 국번(가운데 자리)을 0으로 시작
   - 주민등록번호·카드번호·계좌번호는 실제로 유효하지 않은 임의의 숫자

[개인정보 표시 방법]
아래 7종류의 개인정보는 반드시 [[값|라벨]] 형식으로 표시해 주세요. 표시를 빠뜨리면 안 됩니다.
- person: 사람 이름 (고객, 직원, 지인 등 모든 사람)
- phone_number: 전화번호
- email: 이메일 주소
- address: 주소 (상세 주소까지 포함)
- resident_registration_number: 주민등록번호 또는 외국인등록번호
- account_number: 은행 계좌번호
- card_number: 카드번호

표시 기준
- 문서에 적힌 글자 그대로 표시합니다. 예: [[윤 선영|person]], [[010.0349.2063|phone_number]]
- 이름 뒤의 조사·호칭·직함은 표시 밖에 둡니다. 예: [[김민수|person]]님은, [[이소율|person]] 과장
- 같은 사람이 여러 번 나오면 나올 때마다 표시합니다.
- 다음은 개인정보가 아니므로 표시하지 않습니다: 성과 직함만 있는 표현(예: 김 과장), 회사 대표번호, 회사·기관·은행·상품 이름, 날짜, 금액, 주문·접수번호

[출력 형식]
설명이나 인사말 없이 아래 형식만 출력하세요. 각 문서는 "=== " 줄로 시작합니다.

=== {batch}-01 | {category} | {style}
(문서 본문)

=== {batch}-02 | {category} | {style}
(문서 본문)

... {n}건까지"""


def batches():
    for i in range(len(CATEGORIES) * len(STYLES)):
        yield f"{i + 1:03d}", CATEGORIES[i // len(STYLES)], STYLES[i % len(STYLES)]


def generate(client, model, temperature, batch, category, style, out_dir, retries=3):
    path = out_dir / f"{model}_{batch}.txt"
    if path.exists() and path.stat().st_size > 0:
        return None  # 이미 만든 묶음
    prompt = PROMPT.format(category=category, style=style, n=DOCS_PER_BATCH, batch=batch)
    for attempt in range(retries):
        try:
            t0 = time.time()
            res = client.chat.completions.create(model=model, temperature=temperature, max_tokens=16000,
                                                 messages=[{"role": "user", "content": prompt}])
            path.write_text(res.choices[0].message.content, encoding="utf-8")
            return {"batch": batch, "model": model, "category": category, "style": style,
                    "prompt_tokens": res.usage.prompt_tokens, "completion_tokens": res.usage.completion_tokens,
                    "finish_reason": res.choices[0].finish_reason, "sec": round(time.time() - t0, 1)}
        except Exception as e:  # 일시 오류(속도 제한 등)는 잠시 쉬고 재시도
            err = str(e)[:200]
            time.sleep(5 * (attempt + 1))
    return {"batch": batch, "model": model, "error": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--batches", type=int, default=100, help="앞에서부터 몇 묶음 만들지")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    load_env()
    from openai import OpenAI
    client = OpenAI()

    out_dir = DATA / "generated" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = list(batches())[:a.batches]
    usage_path = DATA / "generated" / "usage.jsonl"
    done = 0
    with ThreadPoolExecutor(a.workers) as ex, open(usage_path, "a", encoding="utf-8") as log:
        for r in ex.map(lambda b: generate(client, a.model, a.temperature, *b, out_dir), todo):
            if r is None:
                continue
            log.write(json.dumps(r, ensure_ascii=False) + "\n")
            log.flush()
            done += 1
            msg = r.get("error") or f"{r['completion_tokens']} 출력 토큰, {r['finish_reason']}, {r['sec']}초"
            print(f"[{done}] 묶음 {r['batch']}: {msg}", flush=True)
    print("완료")


if __name__ == "__main__":
    main()
