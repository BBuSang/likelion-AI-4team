#!/usr/bin/env bash
# 서버(EC2)에서 실행: 최신 코드 받기 → 이미지 빌드 → 바뀐 서비스만 다시 켜기 → 상태 확인 (worklog 65장)
# GitHub Actions(.github/workflows/deploy.yml)가 SSH 로 부름. 손으로: bash ~/likelion-AI-4team/최종프로젝트/deploy/deploy.sh
set -euo pipefail

REPO="${REPO_DIR:-$HOME/likelion-AI-4team}"
BRANCH="${DEPLOY_BRANCH:-main}"
export SAEBOM_HOME="${SAEBOM_HOME:-/srv/saebom}"
COMPOSE=(docker compose -f "$REPO/최종프로젝트/deploy/docker-compose.yml")

for f in db ai backend; do
  [ -f "$SAEBOM_HOME/env/$f.env" ] || { echo "설정 파일 없음: $SAEBOM_HOME/env/$f.env (deploy/env.example 참고)"; exit 1; }
done

echo "== 코드 받기 ($BRANCH)"
cd "$REPO"
export GIT_LFS_SKIP_SMUDGE=1   # 법령 덤프(LFS 193MB)는 서버에 필요 없음 (DB 에 이미 있음)
git fetch --prune origin "$BRANCH"
git checkout -q "$BRANCH"
git reset --hard -q "origin/$BRANCH"   # 서버 checkout 은 배포 전용 — 손으로 고친 건 지워짐
echo "   $(git log -1 --format='%h %s')"

echo "== 빌드·실행"
"${COMPOSE[@]}" up -d --build --remove-orphans

echo "== 상태 확인"
for i in $(seq 1 60); do
  # https 를 켰으면 http 는 301 로 넘기므로 https(localhost, 인증서 이름 확인 생략)로 확인
  code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost/api/auth/options || true)
  [ "$code" = 301 ] && code=$(curl -sk -o /dev/null -w '%{http_code}' https://localhost/api/auth/options || true)
  if [ "$code" = 200 ]; then
    echo "   화면·API 응답 OK"
    break
  fi
  [ "$i" = 60 ] && { echo "   5분 안에 응답 없음"; "${COMPOSE[@]}" ps; "${COMPOSE[@]}" logs --tail 80 backend ai; exit 1; }
  sleep 5
done
"${COMPOSE[@]}" ps

docker image prune -f >/dev/null   # 이전 빌드에서 남은 이미지 정리
