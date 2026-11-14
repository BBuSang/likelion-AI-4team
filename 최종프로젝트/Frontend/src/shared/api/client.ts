// ============================================
// Axios 인스턴스 (참고 레포 axios.ts 기반)
// - 만료 5분 전이면 요청 전에 토큰 갱신
// - 401 이면 토큰 지우고 /login 으로. 단, 로그인 실패(401)는 화면에서 처리하도록 제외
// ============================================

import axios, { AxiosError } from 'axios';
import type { InternalAxiosRequestConfig } from 'axios';
import { getToken, setToken, getTokenExpiration, isRememberMe, removeToken } from '@/shared/utils/tokenManager';

// 빈 값 = 같은 주소(/api → Vite 프록시 → 스프링). 배포 때는 백엔드 주소를 넣음
export const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '';

const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 120000, // LLM 답변이 길어질 수 있어 2분. 스트리밍(SSE)은 fetch 로 따로 처리 예정
  headers: { 'Content-Type': 'application/json' },
});

const NO_AUTH_REDIRECT = ['/api/auth/login', '/api/auth/signup', '/api/auth/check-email'];

let refreshPromise: Promise<void> | null = null;

async function refreshTokenIfNeeded(): Promise<void> {
  const token = getToken();
  if (!token) return;
  const exp = getTokenExpiration(token);
  if (!exp) return;
  const remaining = exp - Date.now();
  if (remaining > 5 * 60 * 1000 || remaining <= 0) return;

  if (!refreshPromise) {
    refreshPromise = (async () => {
      try {
        const res = await axios.post(`${BASE_URL}/api/auth/refresh`, null, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.data?.token) setToken(res.data.token, isRememberMe());
      } catch {
        // 갱신 실패 시 기존 토큰으로 계속 (만료되면 401 로 처리됨)
      } finally {
        refreshPromise = null;
      }
    })();
  }
  await refreshPromise;
}

apiClient.interceptors.request.use(async (config: InternalAxiosRequestConfig) => {
  if (!config.url?.includes('/api/auth/refresh')) {
    await refreshTokenIfNeeded();
  }
  const token = getToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    const url = error.config?.url ?? '';
    if (error.response?.status === 401 && !NO_AUTH_REDIRECT.some((p) => url.includes(p))) {
      removeToken();
      if (window.location.pathname !== '/login') window.location.href = '/login';
    }
    return Promise.reject(error);
  },
);

export default apiClient;
