#!/usr/bin/env bash
# HTTPS 인증서 (Let's Encrypt) — 서버에서 실행
#   bash certbot.sh issue    처음 발급 (도메인이 이 서버를 가리키고 web 이 80 번에 떠 있어야 함) → web 다시 켜서 https 로
#   bash certbot.sh renew    갱신 (만료 30일 전부터만 실제로 바꿈) → 바뀌었으면 Nginx 설정 다시 읽기
#   bash certbot.sh cron     매일 03:17 갱신을 crontab 에 등록
# 인증서는 /srv/saebom/certs (= 컨테이너 /etc/letsencrypt), 확인 파일은 /srv/saebom/certbot-www
set -euo pipefail

DOMAIN="${SITE_DOMAIN:-403.p-e.kr}"
export SAEBOM_HOME="${SAEBOM_HOME:-/srv/saebom}"
REPO="${REPO_DIR:-$HOME/likelion-AI-4team}"
COMPOSE=(docker compose -f "$REPO/최종프로젝트/deploy/docker-compose.yml")
mkdir -p "$SAEBOM_HOME/certs" "$SAEBOM_HOME/certbot-www"

certbot() {
  docker run --rm \
    -v "$SAEBOM_HOME/certs:/etc/letsencrypt" \
    -v "$SAEBOM_HOME/certbot-www:/var/www/certbot" \
    certbot/certbot "$@"
}

case "${1:-}" in
  issue)
    certbot certonly --webroot -w /var/www/certbot -d "$DOMAIN" \
      --agree-tos --register-unsafely-without-email --non-interactive --keep-until-expiring
    "${COMPOSE[@]}" up -d --force-recreate web   # 시작 때 인증서를 보고 https 설정을 고름
    ;;
  renew)
    certbot renew --webroot -w /var/www/certbot --quiet
    docker exec saebom-web-1 nginx -s reload
    ;;
  cron)
    line="17 3 * * * bash $REPO/최종프로젝트/deploy/certbot.sh renew >> $SAEBOM_HOME/certbot-renew.log 2>&1"
    # crontab 이 비어 있으면 'crontab -l' 이 실패하므로 || true (set -e·pipefail)
    { { crontab -l 2>/dev/null || true; } | grep -v 'certbot.sh renew' || true; echo "$line"; } | crontab -
    crontab -l | grep certbot
    ;;
  *)
    echo "사용: bash certbot.sh issue|renew|cron"; exit 1 ;;
esac
