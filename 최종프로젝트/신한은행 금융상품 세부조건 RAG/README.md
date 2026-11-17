# 신한은행 금융상품 세부조건 RAG

은행 직원용 AI Agent 중 금융상품 담당 RAG예요. 직원이 묻는 상품 조건·우대조건·금리·판매 여부를 **근거(문서·쪽·기준일)와 함께** 답해요.
신한은행 상품공시의 판매 중 상품 391개와 설명서·약관·특약 247개를 다뤄요.

## 이 폴더에 있는 것과 없는 것

| 구분 | 파일 |
|------|---|
| 있음 — 코드 | `analysis/s12_retrieval.py` (상품명 찾기·판매 상태·검색·재순위), `analysis/s15_answer_eval.py` (근거 고르기·답변 원칙·대상 상품 줄·근거 문서), `analysis/s10_chunks.py` (형태소 분석·임베딩), `analysis/s8_extract.py` (`.env` 읽기), `analysis/s16_rerank_variants.py` (재순위 지시문), `models/models.py` |
| 있음 — 목록 | `data/meta/` 상품 391 · 문서 247 · 상품↔문서 연결 1,056 · 문서↔공통 조항 연결 · 판매 중지 목록 1,338 |
| 있음 — 설명 | `docs/답변원칙.md` (답변 규칙), `docs/답변_예시.txt` (질문 유형별 실제 답변 28개) |
| **없음 — 따로 받기** | `data/chunks/chunks.jsonl` · `clauses.jsonl` · `glossary.jsonl` — 은행 문서 원문이 들어 있어 공개 저장소에 올리지 않아요. 담당자에게 따로 받아 `data/chunks/`에 넣어요 |
| 없음 — 다시 만들기 | `data/chunks/embed_cache.jsonl` (임베딩, 140MB) — 위 파일을 넣은 뒤 아래 명령으로 만들어요 |

## 준비

```bash
pip install -r requirements.txt
cp .env.example .env    # OPENAI_API_KEY, CHAT_MODEL 등 채우기
python analysis/s10_chunks.py embed-local    # 임베딩 만들기 (KURE-v1, 처음 한 번)
```

## 답변 흐름

1. 상품명 찾기 — 별칭 일치, 애매하면 LLM 확인, 못 찾으면 이름 후보 중 LLM이 고름 (판매 중지 이름도 후보)
2. 판매 상태 — 판매 중·판매 중지 목록과 대조
3. 질문 나누기 — 한 메시지에 질문이 여러 개면 질문마다, 두 상품 비교면 상품마다 근거
4. 검색 — 찾은 상품의 문서로 좁힌 뒤 벡터(KURE-v1) + 키워드(BM25) 결합 → LLM 점수형 재순위 → 근거 5개
5. 답변 — 근거에 있는 것만, 근거 번호·기준일, 없으면 확인할 곳 안내, 추정한 상품은 앞에 표시
6. 표시 — 맨 위 '대상 상품: 이름 (판매 중/판매 중지/추정)', 아래 '근거 문서'·'함께 볼 문서'

## 질문 하나에 답하기 (예시)

```python
import csv, sys
sys.path.insert(0, "analysis")
import s12_retrieval as R
import s15_answer_eval as A
from s8_extract import load_env
from openai import OpenAI

env = load_env()
client = OpenAI(api_key=env["OPENAI_API_KEY"], base_url=env.get("OPENAI_BASE_URL") or None)
idx = R.Index()                      # 서버를 띄울 때 한 번 (약 15초)
docs = {r["doc_id"]: r for r in csv.DictReader(open("data/meta/documents.csv"))}
names = {r["product_id"]: r["name"] for r in csv.DictReader(open("data/meta/products.csv"))}

def ask(q, version="v9"):
    ctx_ids, groups = A.context_ids(idx, q, idx.text_vec(q), version, k=5)
    pids = set().union(*groups) if len(groups) >= 2 else idx.detect_products(q)
    linked = set().union(*(idx.links[p] for p in pids)) if pids else set()
    lcl = set().union(*(idx.doc_clause[d] for d in linked)) if linked else set()
    pn = ", ".join(sorted(names[p] for p in pids))[:120]
    if len(groups) >= 2:
        ctx = "\n\n".join(f"[{n}] {A.multi_header(idx, i, docs, groups, names)}\n{idx.items[i]['text'][:2000]}" for n, i in enumerate(ctx_ids, 1))
    else:
        ctx = "\n\n".join(f"[{n}] {A.context_header(idx, i, docs, linked, lcl, pn)}\n{idx.items[i]['text'][:2000]}" for n, i in enumerate(ctx_ids, 1))
    how = idx.detect_how(q)
    inp = A.extra_context(q, idx) + "\n\n" + A.answer_input(version, ctx, q, pn, inferred=bool(pids) and how != "exact")
    ans = client.responses.create(model=env["CHAT_MODEL"], instructions=A.prompt_of(version), input=inp).output_text
    if len(groups) < 2:
        ans = A.finalize_answer(version, ans, how, pn)
    return A.product_line(idx, q) + "\n" + ans + "\n" + A.source_lines(idx, q, ans, ctx_ids)

print(ask("신한 알.쏠 적금 중도해지하면 이자 어떻게 돼요?"))
```

## 성능 (잠근 평가 세트, 방식을 정한 뒤 한 번만 채점)

| 항목 | 결과 |
|---|---|
| 상품 사실 질문 120 (절반은 실제 말투) | 검색 Hit@5 1.000, 답변 정답 113/120 |
| 답이 공통 약관·공용 설명서에만 있는 질문 40 | 검색 Hit@5 0.95, 답변 38/40 |
| 문서에 없는 값 40 | 지어냄 0 (확인할 곳 안내) |
| 판매 중지 상품 20 · 판매 중지 줄임말 10 | 판매 중지 안내 20/20 · 10/10 |
| 두 상품 비교 20 | 두 상품 값 모두 정확 20/20 |
| 응답 시간 (캐시 없이) | 중앙값 10.3초 · p90 16.1초 |

## 지켜야 할 것
- `.env`(API 키)와 `data/chunks/`의 원문 데이터는 이 저장소에 올리지 않아요 (`.gitignore`에 있음)
- 재순위·상품명 고르기 결과는 `data/eval/`·`data/chunks/`에 캐시로 쌓여요. 같은 질문은 LLM을 다시 부르지 않아요
