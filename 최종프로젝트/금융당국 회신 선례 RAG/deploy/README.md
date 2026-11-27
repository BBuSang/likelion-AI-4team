# AWS 배포 (EC2 한 대 + Docker)

금융당국 회신 선례 RAG 데모를 **내 AWS 계정의 EC2 서버 한 대**에 올려 `http://<고정 IP>` 링크로 여는 방법과 비용입니다.
팀 AI 에이전트(`최종프로젝트/deploy/`)와는 별개 서버이고, 같은 방식(EC2 + Docker)을 씁니다.

**현재 배포**: http://15.165.132.182 (서울, m7i-flex.large, 2026-11-27). 서버를 중지해 두면 열리지 않습니다.

```
브라우저 ──80──▶ EC2 (Ubuntu) ──▶ 컨테이너 interp-rag
                                   ├ Streamlit 화면 (:8080)
                                   └ PostgreSQL 17 + pgvector (컨테이너 안, rag_db.dump를 빌드 때 복원)
```

| 파일 | 내용 |
|---|---|
| `Dockerfile` | 이미지: pgvector:pg17 + Python. 빌드 때 KURE-v1 모델을 받고 `rag_db.dump`를 복원 |
| `start.sh` | 컨테이너 시작: DB 켜기 → Streamlit |
| `requirements.txt` | 파이썬 패키지 (로컬과 같은 버전) |
| `../.dockerignore` | 이미지에 넣을 파일만 허용 (`data/`, `.env`, 평가 스크립트는 빠짐) |

이 파일들은 깃에 올리지 않고 **내 PC에서 서버로 `scp`로 복사**합니다. 앱 코드(`app/`, `pipeline/`)와 `rag_db.dump`는 서버가 `kloid_rag` 브랜치에서 받습니다.

> **명령을 넣을 창 구분**: 줄 앞이 `PS C:\...>`이면 **내 PC**(PowerShell), `ubuntu@ip-...:~$`이면 **서버 안**(bash)입니다. 아래 각 명령 위에 어디서 실행하는지 적어 두었습니다. 서버용 명령을 PowerShell에 넣으면 `'&&' 토큰은 ... 올바른 문 구분 기호가 아닙니다` 오류가 납니다.

## 1. 비용 (서울 리전 기준)

### 무료로 쓸 수 있는 범위

2025-07-15 이후 만든 AWS 계정은 **무료 플랜**입니다.

- 가입 때 크레딧 **$100**, 콘솔에서 안내하는 활동(EC2 띄우기 등)을 하면 최대 **$100 추가**
- **6개월 또는 크레딧을 다 쓸 때까지** 중 먼저 오는 날까지. 그 뒤에도 쓰려면 유료 플랜으로 바꿔야 합니다
- 무료 플랜에서는 정해진 서버 종류만 쓸 수 있고, 사용 요금은 크레딧에서 빠집니다

### 항목별 요금

| 항목 | 단가 | 이 배포에서 |
|---|---|---|
| 서버 m7i-flex.large (CPU 2, 메모리 8GB) | 시간당 약 **$0.118** | **켜 둔 시간만** 나감 |
| 디스크 gp3 30GB | GB·월 약 $0.08~0.09 | 월 약 **$2.7**, 서버를 꺼도 나감 |
| 고정 IP (Elastic IP) | 시간당 $0.005 | 월 약 **$3.65**, 서버를 꺼도 나감 |
| 밖으로 나가는 데이터 | 월 100GB까지 무료 | 화면만 오가서 거의 0 |

로컬에서 잰 컨테이너 메모리는 모델을 불러온 뒤 약 1.4GB라서 8GB면 넉넉합니다. 메모리 2GB짜리(t3.small)는 빌드와 모델 로딩에 모자랍니다.

### 사용 방식별 한 달 비용

| 방식 | 서버 시간 | 한 달 | $100 크레딧으로 |
|---|---|---|---|
| 24시간 계속 켜 둠 | 730시간 | 약 **$92** | 약 1개월 |
| 데모·발표 때만 켬 (예: 주 2회 × 5시간) | 약 40시간 | 약 **$11** | 6개월 플랜 기간 내내 충분 (약 $66) |
| 거의 꺼 둠 (월 10시간) | 10시간 | 약 **$8** | 충분 |

- **꺼 둬도 디스크와 고정 IP로 월 약 $6.4가 나갑니다.** 오래 안 쓸 거면 7장 정리하기를 따르세요
- 답변 생성에 쓰는 LLM API 요금은 AWS와 별개입니다(`.env`의 키 계정에서 나감)
- 단가는 2026-11 기준 공개 자료입니다. 실제 요금은 [EC2 요금](https://aws.amazon.com/ec2/pricing/on-demand/)에서 서울 리전으로 확인하세요

## 2. 처음 한 번: AWS 준비 (웹 콘솔)

1. **가입**: https://aws.amazon.com → 계정 만들기 → 플랜은 **무료(Free)** 선택, 카드 등록
   - 진행 중에 유료 플랜으로 업그레이드하라고 나올 수 있습니다. **업그레이드하면 무료 플랜으로 되돌릴 수 없고**, 남은 크레딧은 계속 쓰이지만 크레딧을 넘는 사용량은 카드로 바로 청구됩니다. 그래서 3번 예산 알림이 꼭 필요합니다
2. **리전**: 콘솔 오른쪽 위에서 **아시아 태평양(서울) ap-northeast-2** 선택
3. **예산 알림**: 결제 및 비용 관리 → 예산 → 예산 생성 → 월 $20, 80%·100%에서 메일. 알림일 뿐 자동으로 멈추지는 않습니다
4. **서버 만들기**: EC2 → 인스턴스 시작
   - 이름: `interp-rag-demo`
   - AMI: **Ubuntu Server 24.04 LTS (x86_64)**
   - 인스턴스 유형: **m7i-flex.large**
   - 키 페어: 새 키 페어 생성 → 이름 `interp-rag`, RSA, `.pem` → 내려받은 파일을 `C:\Users\<내 이름>\.ssh\interp-rag.pem`으로 옮김
   - 네트워크 설정: **SSH 트래픽 허용 → 내 IP**, **인터넷에서 HTTP 트래픽 허용** 체크
   - 스토리지: **30 GiB, gp3**
   - 인스턴스 시작
5. **고정 IP**: EC2 → 탄력적 IP → 탄력적 IP 주소 할당 → 할당된 주소 선택 → 작업 → 탄력적 IP 주소 연결 → 인스턴스 `interp-rag-demo` 선택. 이 주소가 데모 링크가 됩니다

## 3. 처음 한 번: 서버 접속 (내 PC PowerShell)

키 파일 권한을 나만 읽게 바꿉니다(Windows SSH가 권한이 넓으면 거부함).

```powershell
icacls "$HOME\.ssh\interp-rag.pem" /inheritance:r
icacls "$HOME\.ssh\interp-rag.pem" /grant:r "$($env:USERNAME):(R)"
```

접속합니다. `<고정IP>`는 2-5의 주소입니다. 처음 접속 때 `yes`를 입력합니다.

```powershell
ssh -i "$HOME\.ssh\interp-rag.pem" ubuntu@<고정IP>
```

> 집·카페 등 내 IP가 바뀌면 접속이 막힙니다. EC2 → 보안 그룹 → 인바운드 규칙에서 SSH의 소스를 다시 "내 IP"로 바꾸세요.

## 4. 처음 한 번: 서버에 Docker 설치 (서버 안에서)

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-buildx git git-lfs
sudo usermod -aG docker ubuntu
exit
```

`exit`로 나간 뒤 3장의 `ssh` 명령으로 **다시 접속**해야 docker 권한이 적용됩니다.

## 5. 코드·덤프 받기 → 배포 파일 복사 → 빌드 → 실행

### 5-1. 코드와 DB 덤프 받기 (서버 안)

```bash
# 코드 받기 (큰 LFS 파일은 건너뛰고, 이 RAG의 덤프만 받음)
GIT_LFS_SKIP_SMUDGE=1 git clone --branch kloid_rag --single-branch https://github.com/BBuSang/likelion-AI-4team.git ~/repo
cd ~/repo
git lfs install
git lfs pull --include "최종프로젝트/금융당국 회신 선례 RAG/rag_db.dump"
cd "$HOME/repo/최종프로젝트/금융당국 회신 선례 RAG"
ls -lh rag_db.dump        # 111M~112M 이면 정상 (ls가 올림해서 보여 줌)
```

### 5-2. 배포 파일 4개를 서버로 복사 (내 PC PowerShell)

서버 창은 그대로 두고, 내 PC PowerShell 창에서 실행합니다. `<고정IP>`는 2-5의 주소입니다.

```powershell
cd "C:\Users\USER\Desktop\likelion-AI-4team\최종프로젝트\금융당국 회신 선례 RAG"
scp -i "$HOME\.ssh\interp-rag.pem" deploy\Dockerfile deploy\start.sh deploy\requirements.txt .dockerignore ubuntu@<고정IP>:~/
```

파일 4개가 `100%`로 찍히면 성공입니다. `deploy\build`(덤프 복사본)처럼 필요 없는 파일은 보내지 않습니다.

### 5-3. 복사한 파일 제자리에 두기 (서버 안)

```bash
cd "$HOME/repo/최종프로젝트/금융당국 회신 선례 RAG"
mkdir -p deploy && mv ~/Dockerfile ~/start.sh ~/requirements.txt deploy/ && mv ~/.dockerignore . && ls deploy
```

`Dockerfile  requirements.txt  start.sh`가 보이면 됩니다.

### 5-4. API 키 파일 만들기 (서버 안)

API 키 파일을 만듭니다. `nano`가 열리면 아래 세 줄을 값과 함께 적고 `Ctrl+O` → Enter → `Ctrl+X`로 저장합니다. 값은 로컬 `.env`와 같습니다. **`RAG_DSN`은 넣지 않습니다**(컨테이너 안 DB 주소가 따로 정해져 있음).

```bash
nano ~/interp-rag.env
#   OPENAI_API_KEY=...
#   OPENAI_BASE_URL=...
#   CHAT_MODEL=...
chmod 600 ~/interp-rag.env && cut -d= -f1 ~/interp-rag.env   # 키 이름 세 줄만 보이면 정상 (값은 안 나옴)
```

붙여 넣기는 `Ctrl+Shift+V` 또는 마우스 오른쪽 클릭입니다.

### 5-5. 빌드 (서버 안)

처음에는 패키지와 모델(약 2GB)을 받느라 **10~20분** 걸립니다. `[10/14]` DB 복원 단계는 복원된 데이터가 0건이면 실패하게 되어 있어서, 빌드가 끝까지 가면 DB도 정상입니다.

```bash
docker build -f deploy/Dockerfile -t interp-rag .
docker builder prune -f   # 빌드 중간 파일 정리 (디스크 확보)
```

### 5-6. 실행 (서버 안)

서버가 재부팅돼도 컨테이너가 다시 켜집니다.

```bash
docker run -d --name interp-rag --restart unless-stopped -p 80:8080 \
  --env-file ~/interp-rag.env -e RAG_DSN=postgresql://rag@localhost:5432/rag interp-rag
docker logs -f interp-rag   # "You can now view your Streamlit app" 이 보이면 Ctrl+C
```

브라우저에서 **`http://<고정IP>`** 를 엽니다. 처음에는 임베딩 모델을 불러오느라 30~80초 걸립니다. 이 주소를 GitHub README에 데모 링크로 넣으면 됩니다.

> 주소가 `http://`라서 브라우저에 "주의 요함"이 뜹니다. `https://`가 필요하면 팀 배포처럼 무료 도메인 + certbot을 붙여야 합니다(`최종프로젝트/deploy/README.md` 6번).

## 6. 평소 쓰기

| 할 일 | 방법 |
|---|---|
| **끄기** (요금 아끼기) | EC2 → 인스턴스 선택 → 인스턴스 상태 → **중지** |
| **켜기** | 인스턴스 상태 → **시작**. 1~2분 뒤 같은 주소로 열림(컨테이너 자동 시작) |
| 로그 보기 | 서버 접속 후 `docker logs --tail 50 interp-rag` |
| 앱 코드·DB 바꾼 뒤 다시 배포 | 바꾼 `app/`·`pipeline/`·`rag_db.dump`를 `kloid_rag`에 푸시한 뒤, 서버에서 아래 명령 |
| 배포 파일(`deploy/`, `.dockerignore`)을 바꾼 뒤 | 깃에 없으므로 `git pull`로는 안 바뀝니다. **5-2·5-3을 다시** 한 뒤 아래 명령의 `docker build`부터 |

```bash
cd "$HOME/repo/최종프로젝트/금융당국 회신 선례 RAG"
git pull
git lfs pull --include "최종프로젝트/금융당국 회신 선례 RAG/rag_db.dump"
docker build -f deploy/Dockerfile -t interp-rag . && docker builder prune -f
docker rm -f interp-rag
docker run -d --name interp-rag --restart unless-stopped -p 80:8080 \
  --env-file ~/interp-rag.env -e RAG_DSN=postgresql://rag@localhost:5432/rag interp-rag
```

## 7. 다 쓰고 정리하기 (요금 0으로)

순서대로 해야 남는 요금이 없습니다.

1. EC2 → 인스턴스 → 인스턴스 상태 → **종료(삭제)**. 디스크도 함께 지워집니다
2. EC2 → 탄력적 IP → 주소 선택 → 작업 → **탄력적 IP 주소 릴리스**. 연결 안 된 고정 IP도 요금이 나갑니다
3. 결제 및 비용 관리 → 청구서에서 다음 달에 남는 항목이 없는지 확인

## 주의

- 링크가 공개라 누구나 질문할 수 있고, 질문마다 LLM API 요금이 나갑니다. API 쪽에도 사용 한도를 걸어 두세요
- 회신 원문이 화면에 그대로 보입니다. 원문 저작권은 금융규제·법령해석포털에 있습니다
- 답변에서 `RateLimitError`가 나면 API 키 잔액부터 확인하세요(429 `credit_balance_exhausted`, 2026-11-26에 한 번 있었음)
