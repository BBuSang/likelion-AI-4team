# 화면 (React 빌드 → Nginx) — 배포용 (worklog 65장)
# Nginx 가 80 번에서 화면 파일을 주고, /api 는 Spring(backend:8080)으로 넘김 (같은 주소라 CORS 없음)
FROM node:20-alpine AS build
WORKDIR /src
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.27-alpine
ENV TZ=Asia/Seoul
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/dist /usr/share/nginx/html
EXPOSE 80
