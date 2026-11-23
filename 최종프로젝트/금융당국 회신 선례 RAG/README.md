# 금융당국 회신 선례 RAG

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
