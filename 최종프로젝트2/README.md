# ⚖️ 금융권 법령 RAG

금융 관련 법령 380개(법률·시행령·시행규칙·감독규정)에서 질문에 맞는 조문을 찾아, 그 조문만 근거로 출처와 함께 답하는 RAG입니다.

> 질문: 자금세탁이 의심되는 거래는 어디에 언제까지 보고해야 해?
> 답변: 금융회사등은 의심되는 거래를 **금융정보분석원장에게** 보고해야 합니다(특정 금융거래정보의 보고 및 이용 등에 관한 법률 제4조). 보고대상 거래로 **결정한 날부터 3영업일 이내**에 보고해야 합니다(특정 금융거래정보 보고 및 감독규정 제3조).

자세한 설계와 판단 기준은 [기획안.md](기획안.md), 평가 결과는 [reports/5-6_평가_최종.md](reports/5-6_평가_최종.md)에 있습니다. 이 브랜치에는 데모를 처음부터 만들고 실행하는 데 필요한 파일만 있습니다.

## 구조

```
질문
 ① 질문 임베딩 — KURE-v1 (로컬 CPU, 약 0.2초)
 ② 벡터 검색 — PostgreSQL + pgvector(HNSW), 조각 50개 → 조문 단위로 합침
 ③ 후보 — 1~10위 조문 + 1~5위와 위임·준용으로 연결된 조문
 ④ 조문 글 — 여러 조각으로 나뉜 조문은 2,000자 이하면 전체를 넘김
 ⑤ 답변불가 — 검색 1위 점수가 0.5338보다 낮으면 LLM 없이 "관련 조문을 찾지 못했습니다"
 ⑥ LLM 한 번 — 근거 조문 번호 + 답변 + 핵심 구절
답변 + 근거 조문(출처 · 시행일자 · 본문 · 국가법령정보센터 링크)
```

| 구분 | 사용한 것 |
|---|---|
| 데이터 | 국가법령정보센터 Open API — 법령 380개, 검색 대상 조문 26,603개, 조각 35,321개, 위임·준용 연결 11,477건 |
| 임베딩 | `nlpai-lab/KURE-v1` (1024차원) |
| 벡터 DB | PostgreSQL 16 + pgvector (Docker) |
| 답변 | OpenAI 호환 API, 질문당 1회 호출 |
| 데모 | Streamlit |

## 성능

| 측정 | 1위 근거 (Hit@1) | 5위 안 (Hit@5) | MRR | 답변 정확 | 최대 시간 |
|---|---|---|---|---|---|
| test10 최종 (처음 보는 50문항, 한 번만 잼) | 0.94 | 0.98 | 0.957 | 46 / 50 | 5.25초 |
| 지금 데모 설정 (dev 200문항) | 0.95 | 0.985 | 0.968 | 186 / 200 | 8.15초 |
| (비교) 검색만, test10 | 0.88 | 0.96 | 0.919 | — | — |

- test10 점수는 답변 지시문을 개선하기 전 설정으로 잰 값입니다. 지금 데모 설정은 test가 모두 잠겨 dev 세트로만 쟀습니다.
- 평가 질문 1,000개(dev1~10, test1~10)와 정답은 AI가 조문을 보고 만든 초안이고, 답변 정확도도 AI 채점입니다. 사람 확인 전의 잠정 점수입니다.

## 실행 방법

이 폴더(`최종프로젝트2/`)에서 실행합니다. Docker Desktop과 파이썬 3.12가 필요합니다.

**1. 패키지 설치**

```bash
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**2. `.env` 만들기**

`.env.example`을 `.env`로 복사하고 빈 값(`POSTGRES_PASSWORD`, `LAW_PASSWORD`, `LLM_API_KEY`)을 채웁니다. 값은 저장소에 없으니 따로 받으세요.

**3. DB 띄우기**

```bash
docker compose up -d
```

컨테이너 `team4-lawrag2-db`가 `127.0.0.1:5435`에 뜹니다. 처음 뜰 때 `db/init/01_init.sh`가 확장(vector), 스키마(`lawrag`), 검색 전용 계정(`law_reader`)을 만듭니다.

**4. 데이터 넣기 (DB 덤프 복원)**

DB 덤프 `share/law_rag.dump`(193MB, 법령 380개 · 조각 35,321개 · 벡터 · 색인)가 Git LFS로 함께 올라가 있습니다. 브랜치를 받으면 같이 내려옵니다. 파일 크기가 1KB 정도로 작으면 LFS 파일이 아직 안 받아진 것이니 `git lfs pull`을 한 번 실행하세요. 복원은 색인을 다시 만드느라 몇 분 걸립니다.

```bash
docker cp share/law_rag.dump team4-lawrag2-db:/tmp/law_rag.dump
docker exec team4-lawrag2-db pg_restore -U admin -d law_rag --clean --if-exists --no-owner /tmp/law_rag.dump
```

원본 법령·청크·벡터 파일(약 790MB)은 저장소에 없습니다. 덤프만 있으면 실행에는 충분합니다.

(`admin`, `law_rag`는 `.env`의 `POSTGRES_USER`, `POSTGRES_DB`와 같게)

**5. 실행**

```bash
python retriever/answer.py "예금자보호 한도는 1인당 얼마야?"
python -m streamlit run demo/app.py --server.port 8504
```

처음 실행할 때 임베딩 모델 `nlpai-lab/KURE-v1`(약 2.2GB)을 자동으로 받습니다. 데모는 http://localhost:8504 에서 열리고, 첫 화면은 모델을 올리느라 30~50초 걸립니다.

## 데이터를 처음부터 만들기 (덤프 대신)

4단계의 덤프 복원 대신, 법령을 직접 받아 DB를 만들 수도 있습니다. `.env`에 국가법령정보센터 Open API 인증값(`LAW_API_OC`)이 필요하고, 임베딩은 코랩 GPU를 씁니다. 수집 날짜의 현행 법령을 받으므로 덤프(2026-11-13 기준)와 조금 다를 수 있습니다. 2026-11-19에 이 폴더만으로 다시 만들어 본 결과, 법령 380개와 판은 모두 같았고, API가 표·수식 안에 줄바꿈을 새로 넣어 보내면서 5개 법령의 조문 57개 글자 배치와 조각 수(35,317개)가 조금 달라졌습니다. 내용은 같습니다.

| 순서 | 명령 | 하는 일 | 만드는 것 |
|---|---|---|---|
| 1 | `python pipeline/fetch_01.py all` | `data/source/target_laws.json`의 법령과 하위 법령을 API로 받음 (1초 간격, 약 20분) | `data/work/01_fetch/` |
| 2 | `python pipeline/clean_02.py` | 조문·항·호·목 구조로 정리, 삭제 이력 태그 제거 | `data/work/02_clean/` |
| 3 | `python pipeline/chunk_03.py` | 조문을 검색 조각으로 나눔 (정의 조문은 정의 하나 = 조각 하나) | `data/work/04_chunks/` |
| 4 | `python pipeline/colab_bundle_04.py` | 코랩에 올릴 입력 묶음을 만듦 | `data/work/05_embed/kure_input.zip` |
| 5 | 코랩에서 `colab/kure_embed.ipynb` 실행 | 4의 zip을 올리고 전부 실행 → `kure_vectors.npz`를 받아 `data/work/05_embed/`에 둠 (T4 GPU 약 6분) | `kure_vectors.npz` |
| 6 | `python pipeline/links_06.py` | 위임·준용 연결 만들기 | `data/work/06_links/` |
| 7 | `python pipeline/load_05.py` | DB(`POSTGRES_DB`)를 **지우고 새로 만든 뒤** 적재·색인 (약 7분) | DB |

그다음 5단계(실행)는 같습니다.

## 폴더

```
├─ README.md
├─ 기획안.md                  기준 문서 (설계, 단계별 판단 기준, 시험 기록)
├─ reports/5-6_평가_최종.md   최종 평가 보고서
├─ retriever/
│  ├─ search.py               검색 (임베딩, 벡터 검색, 연결 조문, 조문 글)
│  └─ answer.py               답변 (후보 → LLM 1회 → 답변·근거·핵심 구절)
├─ demo/
│  ├─ app.py                  Streamlit 데모
│  └─ answer_format.py        답변 표시 (조문별 문단, 목록, 핵심 구절 빨간색)
├─ pipeline/                  수집 → 전처리 → 청크 → 임베딩 입력 → 연결 → 적재
├─ colab/kure_embed.ipynb     조각 임베딩용 코랩 노트북 (GPU)
├─ data/source/target_laws.json  수집할 법령 목록
├─ share/law_rag.dump         DB 덤프 (Git LFS, 193MB)
├─ db/init/01_init.sh         DB 초기화 (확장·스키마·검색 전용 계정)
├─ docker-compose.yml         DB 컨테이너
├─ requirements.txt
└─ .env.example
```

평가 질문·정답·측정 결과, 단계별 점검 보고서, 작업 기록 등 과정 자료는 이 브랜치의 첫 커밋(`5902a1e`)에 남아 있습니다.
