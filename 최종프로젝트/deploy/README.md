# 배포 (EC2 + GitHub Actions)

서버 한 대(m7i.xlarge, Ubuntu, 미국 오하이오 us-east-2)에 Docker 로 4개를 띄웁니다. 설계·확인 기록은 `worklog.md` 65장.

```
브라우저 ──80──▶ web (Nginx: 화면 파일, /api → backend)
                  └▶ backend (Spring :8080) ──▶ ai (FastAPI :8000) ──▶ db (PostgreSQL 16 + pgvector)
```

- 밖에 여는 포트는 **80** 하나. DB 는 서버 안 `127.0.0.1:5434` 에서만 (노트북에서 볼 땐 SSH 터널)
- 비밀값·데이터는 저장소 밖 `/srv/saebom` 에 두고 git 에 올리지 않습니다

| 파일 | 내용 |
|---|---|
| `docker-compose.yml` | 운영 구성 (db·ai·backend·web, 재시작 자동, 로그 20MB×5) |
| `deploy.sh` | 서버에서: `git reset --hard origin/main` → `docker compose up -d --build` → 화면·API 응답 확인 |
| `env.example/*.env` | 서버 설정 파일 틀 (값은 비어 있음) |
| `../.github/workflows/deploy.yml` (저장소 맨 위) | main 에 올라오면 SSH 로 `deploy.sh` 실행. Actions 탭에서 손으로도 |
| `../AI/Dockerfile`, `../Backend/Dockerfile`, `../Frontend/Dockerfile`, `../Frontend/nginx.conf` | 이미지 |

## 처음 한 번 (서버 준비)

1. **코드 받기** — 저장소가 공개라 인증 없이
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/BBuSang/likelion-AI-4team.git ~/likelion-AI-4team
   ```
2. **설정 파일** — `/srv/saebom/env/{db,ai,backend}.env` 를 `env.example` 에서 복사해 채움 (`chmod 600`)
   - 비밀번호·JWT 는 서버에서 새로 만듦 (`openssl rand -base64 32`)
   - **HMAC_KEY 는 로컬과 같은 값** (옮겨 온 DB 의 가명처리 토큰과 맞아야 함), OPENAI_API_KEY 도 같은 키
3. **데이터** — git 에 없는 것만 노트북에서 복사
   - `AI/data/public_rag/product/` → `/srv/saebom/ai/data/public_rag/product/` (금융상품 청크·벡터)
   - `AI/outputs/models/atonlee-ft-v6/` → `/srv/saebom/ai/outputs/models/atonlee-ft-v6/` (이름 보태기 모델)
4. **DB 옮기기** — 로컬 DB 를 통째로
   - 로컬: `pg_dumpall --roles-only` + `pg_dump -Fc saebom` → 서버로 복사
   - 서버: `docker compose up -d db` → 계정 → `pg_restore --clean --if-exists` → 계정 비밀번호를 `db.env` 값으로 `ALTER ROLE`
5. **GitHub 비밀값** — 저장소 Settings → Secrets and variables → Actions
   - `EC2_HOST` = 고정 IP, `EC2_USER` = `ubuntu`, `EC2_SSH_KEY` = `.pem` 내용 전체
6. 첫 배포: `bash ~/likelion-AI-4team/최종프로젝트/deploy/deploy.sh` (또는 Actions → Deploy → Run workflow)
   - 첫 실행은 이미지 빌드(AI 2.2GB)·모델 내려받기로 10분 남짓

## 그다음부터

- `main` 에 `최종프로젝트/AI|Backend|Frontend|infra|deploy` 가 바뀌어 올라오면 자동 배포
- 배포가 겹치면 앞 것이 끝난 뒤 다음 것이 돎
- 로그: `docker compose -f ~/likelion-AI-4team/최종프로젝트/deploy/docker-compose.yml logs -f backend ai`

## 주의

- 서버의 저장소 폴더는 배포 전용입니다 — `deploy.sh` 가 `git reset --hard` 로 맞추므로 서버에서 고친 코드는 지워집니다
- 보안 그룹의 22번이 팀원 IP 로만 열려 있으면 GitHub Actions 가 접속하지 못합니다 (Actions 서버 IP 는 매번 바뀜)
- 인스턴스를 중지했다 켜도 고정 IP 라 주소가 그대로입니다
