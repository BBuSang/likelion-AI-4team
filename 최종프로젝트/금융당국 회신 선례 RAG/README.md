# 금융당국 회신 선례 RAG

**▶ 데모: http://15.165.132.182**
(AWS EC2 서울. 데모 시간 외에는 서버를 꺼 둬서 열리지 않을 수 있습니다. 처음 열 때 모델을 불러오느라 30~80초 걸립니다. 배포 방법은 [deploy/README.md](deploy/README.md))

은행 직원이 업무 중 "이거 해도 되나?" 싶을 때, **금융위원회·금융감독원이 비슷한 질의에 실제로 어떻게 회신했는지** 찾아 근거와 함께 보여주는 RAG 챗봇입니다.
조문 해설이나 "가능/불가" 같은 확정 판단은 하지 않습니다. "당국이 이렇게 적용한 선례가 있다"까지만 보여 주고, 최종 판단은 준법감시·법무 부서에 맡깁니다.

- **사용자**: 은행 상품기획, 준법감시, 영업점 지원, 여신·외환 담당 직원
- **데이터**: 금융규제·법령해석포털(better.fsc.go.kr)의 법령해석·비조치의견서 회신(2014년 이후)과 현장건의 과제, 국가법령정보 OPEN API의 조문 개정 이력

### 주요 기능

| 기능 | 설명 |
|---|---|
| 사례 검색 | 상황을 문장으로 설명하면 비슷한 회신을 의미로 찾아, 질의요지·회답 요지·회신일·원문 링크와 함께 답합니다 |
| 조문 조회 | 조문(예: 은행법 제27조의2)을 넣으면 그 조문을 다룬 회신을 날짜순으로 모아 줍니다 |
| 시점 표시 | 모든 답변에 "회신일 이후 관련 조문이 개정됨" 같은 표시를 자동으로 붙여, 오래된 회신을 그대로 믿지 않게 합니다 |

### 구성

- DB: PostgreSQL 17 + pgvector (Docker)
- 임베딩: KURE-v1 (로컬, 1024차원)
- 검색 → LLM 리랭킹 → 직접성 판정 → 답변 생성
- 화면: Streamlit (`streamlit run app/app.py`)

자세한 설계는 [기획안.md](기획안.md)를 참고하세요.

> 회신 원문의 저작권은 금융규제·법령해석포털에 있습니다(포털 저작권정책: 무단 복제·배포 금지, 출처 표시 필요).

## DB 복원

`rag_db.dump`는 이 RAG가 쓰는 PostgreSQL DB(스키마 `interp`, 청크·임베딩 포함)를 통째로 백업한 파일입니다. 복원하면 수집·적재·임베딩을 다시 하지 않고 바로 데모를 돌릴 수 있습니다.

- 형식: `pg_dump -Fc`(custom 형식), **PostgreSQL 17**. 16 이하에는 바로 복원되지 않습니다.
- 필요한 것: Docker Desktop, Git LFS, Python, 이 폴더의 `.env`(`RAG_DSN`)
- 아래 명령은 모두 이 폴더(`최종프로젝트/금융당국 회신 선례 RAG`)에서 실행합니다.

### 1. dump 파일 받기 (Git LFS)

저장소를 받으면 `rag_db.dump`는 LFS 포인터(작은 텍스트 파일)로만 들어옵니다. 실제 파일(약 111MB)을 받습니다.

```powershell
git lfs install
git lfs pull --include "최종프로젝트/금융당국 회신 선례 RAG/rag_db.dump"
```

파일 크기가 약 111MB면 정상입니다.

### 2. DB 컨테이너 띄우기

`.env`의 `RAG_DSN`을 읽어 pgvector(pg17) 컨테이너 `interp-rag-db`를 띄웁니다. `RAG_DSN` 형식은 `postgresql://<사용자>:<비밀번호>@localhost:<포트>/<DB이름>`입니다.

```powershell
python pipeline/db_up.py
```

`[OK] 컨테이너 실행`과 `PostgreSQL 17...`이 나오면 됩니다.

### 3. 복원

dump를 컨테이너 안으로 복사한 뒤 컨테이너 안의 `pg_restore`로 넣습니다(내 PC에 PostgreSQL을 설치하지 않아도 됩니다). `<사용자>`, `<DB이름>`은 `RAG_DSN`의 값으로 바꿉니다.

```powershell
docker cp rag_db.dump interp-rag-db:/tmp/rag_db.dump
docker exec interp-rag-db pg_restore -U <사용자> -d <DB이름> --no-owner --clean --if-exists /tmp/rag_db.dump
docker exec interp-rag-db rm /tmp/rag_db.dump
```

- `--clean --if-exists`: 이미 있는 테이블은 지우고 다시 만듭니다. 처음 복원할 때 "does not exist" 경고가 나와도 괜찮습니다.
- `--no-owner`: 덤프를 만든 계정과 사용자 이름이 달라도 복원되게 합니다.

### 4. 확인

```powershell
docker exec interp-rag-db psql -U <사용자> -d <DB이름> -c "\dt interp.*"
```

`interp` 스키마의 테이블 16개가 보이면 복원이 끝난 것입니다. 이후 `streamlit run app/app.py`로 데모를 실행합니다.
