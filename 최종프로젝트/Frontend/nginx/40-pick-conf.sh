#!/bin/sh
# Nginx 시작 때 설정 고르기 (worklog 66장): SITE_DOMAIN 의 인증서가 있으면 https, 없으면 http 만
set -e
D="${SITE_DOMAIN:-}"
if [ -n "$D" ] && [ -f "/etc/letsencrypt/live/$D/fullchain.pem" ]; then
  sed "s/__DOMAIN__/$D/g" /etc/nginx/saebom/https.conf > /etc/nginx/conf.d/default.conf
  echo "saebom: https ($D)"
else
  cp /etc/nginx/saebom/http.conf /etc/nginx/conf.d/default.conf
  echo "saebom: http (인증서 없음${D:+: $D})"
fi
