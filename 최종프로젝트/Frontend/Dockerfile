# 화면 (React 빌드 → Nginx) — 배포용
# Nginx 가 80(·443) 번에서 화면 파일을 주고, /api 는 Spring(backend:8080)으로 넘김 (같은 주소라 CORS 없음)
FROM node:20-alpine AS build
WORKDIR /src
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
ENV TZ=Asia/Seoul
# 설정은 시작 때 고름: 인증서가 있으면 https, 없으면 http (nginx/40-pick-conf.sh)
COPY nginx/app.conf /etc/nginx/snippets/app.conf
COPY nginx/http.conf nginx/https.conf /etc/nginx/saebom/
COPY nginx/40-pick-conf.sh /docker-entrypoint.d/40-pick-conf.sh
# 윈도우에서 저장돼 줄 끝이 CRLF 여도 돌게 (dos2unix)
RUN dos2unix /docker-entrypoint.d/40-pick-conf.sh 2>/dev/null || true; chmod +x /docker-entrypoint.d/40-pick-conf.sh
COPY --from=build /src/dist /usr/share/nginx/html
EXPOSE 80 443
