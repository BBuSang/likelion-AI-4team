// ============================================
// JWT 토큰 저장 (참고 레포에서 가져옴, 전화번호 유틸은 뺌)
// rememberMe 면 localStorage(브라우저 닫아도 유지), 아니면 sessionStorage
// ============================================

import type { StoredUser } from '@/shared/types/user.types';

const TOKEN_KEY = 'auth_token';
const USER_INFO_KEY = 'user_info';
const REMEMBER_KEY = 'remember_me';

export const isRememberMe = (): boolean => localStorage.getItem(REMEMBER_KEY) === 'true';

export const setToken = (token: string, rememberMe: boolean = false): void => {
  localStorage.setItem(REMEMBER_KEY, String(rememberMe));
  const storage = rememberMe ? localStorage : sessionStorage;
  storage.setItem(TOKEN_KEY, token);
};

export const getToken = (): string | null =>
  localStorage.getItem(TOKEN_KEY) || sessionStorage.getItem(TOKEN_KEY);

export const removeToken = (): void => {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_INFO_KEY);
  localStorage.removeItem(REMEMBER_KEY);
  sessionStorage.removeItem(TOKEN_KEY);
  sessionStorage.removeItem(USER_INFO_KEY);
};

export const setUserInfo = (user: StoredUser, rememberMe: boolean = false): void => {
  const storage = rememberMe ? localStorage : sessionStorage;
  storage.setItem(USER_INFO_KEY, JSON.stringify(user));
};

export const getUserInfo = (): StoredUser | null => {
  const raw = localStorage.getItem(USER_INFO_KEY) || sessionStorage.getItem(USER_INFO_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as StoredUser;
  } catch {
    return null;
  }
};

/** JWT 의 exp(초)를 ms 로. 서명 검증은 서버가 하고, 여기서는 만료 시각만 읽음 */
export const getTokenExpiration = (token: string): number | null => {
  try {
    const part = token.split('.')[1].replace(/-/g, '+').replace(/_/g, '/');
    const payload = JSON.parse(atob(part));
    return payload.exp ? payload.exp * 1000 : null;
  } catch {
    return null;
  }
};
