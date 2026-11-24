# 최종 프로젝트 — 신한은행 금융상품 세부조건 RAG

작업 과정과 모든 결과는 [worklog.md](worklog.md)에 있어요. 이 문서는 폴더 안내와 자주 쓰는 명령만 담아요.

은행 직원용 AI Agent 중 금융상품 담당 RAG예요. 직원이 모든 상품의 세부 조건을 알 수 없으니, 상품 조건·우대조건·금리·판매 여부를 **근거(문서·쪽·기준일)와 함께** 답하도록 돕습니다.

## 이 저장소에 없는 것

팀 저장소는 공개라 **은행 문서 원문이 담긴 파일은 올리지 않아요.** 아래 경로는 이 폴더에 없고, 필요하면 담당자에게 따로 받아요.

| 없는 것 | 이유 |
|---|---|
| `data/docs/` 은행 PDF 247개, `data/text/` · `data/text_v2/` · `data/ocr/` · `data/stage/` · `data/sections/` · `data/clean/` | 문서 원문과 그 전처리 결과 (재배포 금지) |
| `data/chunks/*.jsonl` (청크·공통 조항·용어), `data/extract/` (추출 근거 문장) | 원문이 들어 있음. 받아서 `data/chunks/`에 넣은 뒤 `python analysis/s10_chunks.py embed-local`로 임베딩을 만들어요 |
| `data/eval/` 평가 세트·채점 결과 | 질문마다 근거 원문 문장이 들어 있음 |
| `data/meta/common_clause.csv` · `glossary.csv` · `s2_removed.csv` | 공통 조항·용어 본문, 지운 반복 문구 원문 |
| `docs/5-2` · `docs/6-4` 질문별 검색 결과, `docs/13-1_검토용_6개.md` | 질문마다 근거 원문을 옮겨 둔 문서 |

그래서 아래 폴더 구성과 worklog에 나오는 일부 파일·링크는 이 저장소에서 열리지 않아요.

## 준비

```bash
pip install -r requirements.txt
cp .env.example .env    # OPENAI_API_KEY, CHAT_MODEL 등 채우기
python analysis/s10_chunks.py embed-local    # 임베딩 만들기 (KURE-v1, 처음 한 번)
```

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

## 현재 상태

검색·답변은 완성하고 잠근 세트로 검증까지 마쳤어요.

| 항목 | 결과 |
|---|---|
| 최종 검증 (잠금 280문항, 모든 질문 유형, 실제 말투 포함, 표 재추출 데이터) | 상품 사실 검색 Hit@5 1.000 · 답변 115/120 (원문 대조), 없는 값 지어냄 1/40, 판매 중지 안내 20/20, 두 상품 비교 15/20 |
| 표 값 질문 (표 재추출 전 → 후) | 검색 Hit@5 0.85 → 1.00 |
| 처음 보는 상품 test2 (98문항) | 검색 Hit@5 0.969, 답변 94/98 |
| 질문 두 개 (잠금 30문항) | 두 질문 모두 맞음 27/30 |
| 응답 시간 (캐시 없이, 질문 하나씩) | 중앙값 10.3초 · p90 16.1초 |

- 남은 약점: 목록에 없는 가짜 상품 이름에 추정 표시 없이 답하는 경우(2/20), 문서에 없는 값을 "정해져 있지 않다"고 단정한 경우(1/40), 평가 질문이 모두 LLM이 만든 질문이라는 점
- 단계별 결정·근거·수치는 [worklog.md](worklog.md), 답변 모양은 [docs/답변_예시.txt](docs/답변_예시.txt)

## 답변 흐름

1. 상품명 찾기 — 별칭 일치, 애매하면 LLM 확인, 못 찾으면 이름 후보 중 LLM이 고름 (판매 중지 이름도 후보)
2. 판매 상태 — 판매 중·판매 중지 목록과 대조
3. 질문 나누기 — 한 메시지에 질문이 여러 개면 나눠서 질문마다 근거를 찾음 (두 상품 비교는 상품마다)
4. 검색 — 찾은 상품의 문서로 좁힌 뒤 벡터(KURE-v1) + 키워드(BM25) 결합, 재순위 후보는 '지금 순위 7 + 상품 이름을 뺀 주제 순위 3'으로 섞음
5. 점수형 재순위 (LLM, 후보마다 0~3점) → 근거 5개
6. 답변 — 답변 원칙 v9 ([docs/답변원칙.md](docs/답변원칙.md)): 근거에 있는 것만, 근거 번호·기준일, 없으면 확인할 곳 안내, 추정한 상품은 앞에 표시(코드로 보장)
7. 답변 위아래 표시 (코드) — 대상 상품과 판매 상태 한 줄, 실제 인용한 근거 문서, 함께 볼 문서(재순위 1점 이상). 함수(`s15` `product_line` · `source_lines`)로 준비돼 있고, 지금은 답변 예시 파일에서만 붙여요

## 폴더 구성

```
최종프로젝트/
├── README.md               폴더 안내 (이 문서)
├── worklog.md              작업 기록 (결정·근거·결과)
├── analysis/               단계별 스크립트 (파일 이름 앞 번호 = 단계)
│   ├── build_doc_map.py      상품공시 연결표 → 상품·문서·연결 CSV
│   ├── extract_pdf.py        PDF → 쪽별 텍스트
│   ├── ocr_pdf.py            이미지뿐인 PDF OCR (macOS Vision, 워터마크 제거)
│   ├── build_rate_notice.py  고시 금리 정리·상품 연결
│   ├── build_discontinued.py 판매 중지 목록 정리
│   ├── survey_formats.py     문서 형식 조사
│   ├── s2 ~ s4               반복 문구 제거 · 사본·요약 쪽 분리 · 메타(심의필·기준일)
│   ├── s5_*.py               섹션 분할 (행 이름 사전 · 분할 · 정답 경계 채점)
│   ├── s6_linebreaks.py      끊긴 줄 복원 (어절 빈도 + Kiwi)
│   ├── s7_common.py          공통 조항·정형 문구·용어집 분리
│   ├── s8_extract.py         LLM 정형 추출 (입력 만들기·근거 검사·실행)
│   ├── s9_load_db.py         PostgreSQL + pgvector 적재
│   ├── s10_chunks.py         청크·임베딩 (OpenAI 또는 KURE, 코랩·맥)
│   ├── s11 · s17 · s19       검색 평가 세트 (dev·test · 추가 정답 · test2)
│   ├── s12_retrieval.py      검색 (벡터·키워드·결합·상품명 좁히기·재순위) + 상품명 찾기
│   ├── s13 · s16             재순위 흔들림 측정 · 재순위 방식 비교
│   ├── s14_report_retrieval.py  질문별 검색 결과 보고서 (md)
│   ├── s15_answer_eval.py    답변 생성·채점 (답변 원칙 v1~v9, 질문 나누기, 대상 상품 줄·근거 문서 만들기)
│   ├── s18 · s22 · s23 · s24 상품명 찾기 확인 세트·채점 (줄임말 · 별칭 오탐 · 가짜 상품)
│   ├── s20 · s21             '없는 값' 평가 세트 · 채점
│   ├── s25 · s32             공통 약관에만 답이 있는 질문 세트 · 잘못된 질문 교체
│   ├── s26 · s27             판매 중지·상품 비교 세트 · 채점 (빈틈·가짜 상품 세트도 채점)
│   ├── s28 · s29 · s30       빈틈 확인 세트 (판매 중지 줄임말 · 정식 이름을 품은 가짜 상품)
│   ├── s31_realistic_set.py  실제 말투 짝 비교 세트
│   ├── s33_final_set.py      최종 검증 세트 (모든 유형 280문항)
│   ├── s34 · s35             응답 시간 측정 · 재순위 시간 줄이기 시험
│   ├── s36_answer_examples.py  질문 유형별 답변 예시 (docs/답변_예시.txt)
│   ├── s37 · s38             질문 두 개 세트 · 채점
│   └── archive/              비교용 옛 코드
├── models/
│   ├── schema.sql            DB 스키마 (상품 ↔ 문서 N:M, 고시 금리, 판매 중지, 청크)
│   ├── models.py             LLM 추출 출력 형식 (pydantic)
│   └── extract_prompt.md     LLM 추출 지시문
├── notebooks/
│   └── kure_embed_colab.ipynb  KURE-v1 임베딩 (코랩 GPU)
├── docs/                   단계별 조사·결정·평가 기록 (번호 = 단계), 답변원칙.md, 답변_예시.txt
└── data/
    ├── docs/                 은행 PDF 247개 (재배포 금지, 저장소 비공개 유지)
    ├── text/ stage/ ocr/     추출 텍스트와 전처리 중간 결과
    ├── sections/ clean/      섹션 분할·줄바꿈 복원 결과
    ├── extract/              LLM 정형 추출 결과 (full/ 130개)
    ├── chunks/               청크·공통 조항·용어, 상품명 고르기 기록 (임베딩 캐시는 저장소 제외)
    ├── colab/                코랩 임베딩 입력
    ├── eval/                 평가 세트·채점 결과 (잠근 세트는 아래 '지켜야 할 것')
    ├── meta/                 상품·문서·연결·공통 조항·용어·판매 중지 (CSV), raw/ 수집 원본
    └── rates/                고시 금리 (CSV), raw/ 수집 원본
```

## 설치

```bash
pip install pymupdf openpyxl kiwipiepy openai pydantic "psycopg[binary]" numpy sentence-transformers torch
```

OCR을 다시 할 때만 `pyobjc-framework-Vision pyobjc-framework-Quartz`가 더 필요해요 (macOS).
API 키와 모델 설정은 이 폴더의 `.env`에 둬요 (깃에 올라가지 않음): `OPENAI_API_KEY`, `CHAT_MODEL`, `EMBED_MODEL`, `EMBED_DIM`.

DB (pgvector, 로컬 5433 포트)
```bash
docker run -d --name fss_rag_db -e POSTGRES_PASSWORD=fss_local -e POSTGRES_DB=fss_rag -p 127.0.0.1:5433:5432 -v fss_rag_pgdata:/var/lib/postgresql/data pgvector/pgvector:pg16
```

## 자주 쓰는 명령 (이 폴더에서 실행)

데이터 수집·전처리 전체 순서는 [docs/데이터수집_요약.md](docs/데이터수집_요약.md) 3절에 있어요.

섹션 분할 → 줄바꿈 복원 → 공통 조항 분리
```bash
python analysis/s5_split.py && python analysis/s6_linebreaks.py && python analysis/s7_common.py
```

섹션 분할 채점 (기본은 개발용 정답, `--gold=`로 다른 정답 세트)
```bash
python analysis/s5_eval.py
```

LLM 정형 추출 (130개 문서, 이어하기·병렬)
```bash
python analysis/s8_extract.py run-all
```

DB 적재 → 청크 만들기 → 청크 적재
```bash
python analysis/s9_load_db.py && python analysis/s10_chunks.py build && python analysis/s10_chunks.py load
```

KURE 임베딩을 맥에서 직접 (캐시에 없는 것만)
```bash
python analysis/s10_chunks.py embed-local
```

검색 방식 비교 (dev)
```bash
python analysis/s12_retrieval.py dev routed rerank_score
```

답변 생성·채점 (dev, 상위 5개 근거)
```bash
python analysis/s15_answer_eval.py dev 5 --version=v9 --retrieval=rerank_score
```

'없는 값' 답변 채점 (dev)
```bash
python analysis/s21_absent_eval.py dev --version=v9
```

상품명 찾기(줄임말) 채점 (고치기용 세트)
```bash
python analysis/s22_alias_check2.py score-dev
```

판매 중지·비교·빈틈 답변 채점 (dev)
```bash
python analysis/s27_status_compare_eval.py dev --version=v9
```

질문 두 개 채점 (dev)
```bash
python analysis/s38_double_eval.py eval dev
```

응답 시간 측정 (캐시 없이, 질문 40개)
```bash
python analysis/s34_latency.py 40
```

질문 유형별 답변 예시 다시 만들기 (LLM 호출 없음)
```bash
python analysis/s36_answer_examples.py
```

방식 켜고 끄기 (환경변수, 비교 시험용)

| 변수 | 기본 | 뜻 |
|---|---|---|
| `ROUTED_TOPIC` | `mix` | 재순위 후보 섞기 (`off` 이전 방식 · `on` 주제만) |
| `SPLIT_QUESTIONS` | `1` | 질문 나누기 |
| `PRODUCT_FALLBACK` · `PRODUCT_VERIFY` · `PICK_DISC` | `1` | 상품명 후보 고르기 · 애매한 별칭 LLM 확인 · 판매 중지 이름 후보 |
| `RERANK_CHARS` · `RERANK_EFFORT` | `1500` · 모델 기본 | 재순위 후보 글자 수 · 추론 강도 (시험 결과 기본 유지) |
| `LIST_RULE` | `0` | 목록형 답 규칙 (시험 결과 쓰지 않음) |

**최종 방식: KURE-v1 임베딩 + 상품명으로 좁히기(별칭, 못 찾으면 후보 → LLM 고르기) + 벡터·키워드 결합 + 재순위 후보 섞기 + 점수형 재순위 + 판매 상태 목록 대조(판매 중지 이름도 상품명 후보) + 질문 나누기 + 답변 원칙 v9** (worklog 5장).

## 지켜야 할 것
- 방식은 **dev로만** 고르고, 잠근 세트는 다 정한 뒤 `--final`로 **한 번만** 채점해요. 잠근 세트를 보고 고치면 새 세트를 만들어요
  - 잠근 세트 (`data/eval/`): `retrieval_test` · `retrieval_test2` · `absent_test` · `alias_check2~4` · `retrieval_common_fix_test` · `status_test` · `compare_test` · `gap_test` · `gap2_test` · `gap3_test` · `*_real_test` · `*_final` (최종 검증) · `double_test`
  - 평가 세트는 OpenAI로 만들고, 결과를 보기 전에 통과 기준을 문서에 먼저 적어요
- 은행 PDF와 전문 텍스트가 들어 있어 **저장소는 비공개로 유지**해요 (재배포 권한 불명확, PDF에 다운로드 워터마크)
- API 키는 `.env`에만 (깃 업로드 제외)
- 임베딩 캐시(`data/chunks/embed_cache.jsonl`)는 GitHub 용량 한도를 넘어 올리지 않아요. 다시 만드는 방법은 `.gitignore`에 있어요
