// ============================================
// 로그인 상태 (참고 레포 AuthContext 기반, 화면 알림(alert)은 뺌)
// - 새로고침 시 저장소에서 복원, 만료된 토큰은 버림
// - 만료 시각에 자동 로그아웃
// ============================================

import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import * as authApi from '@/features/auth/api/auth.api';
import type { LoginRequest } from '@/features/auth/types/auth.types';
import type { StoredUser } from '@/shared/types/user.types';
import { getToken, getTokenExpiration, getUserInfo, removeToken, setToken, setUserInfo } from '@/shared/utils/tokenManager';

interface AuthContextType {
  user: StoredUser | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (body: LoginRequest) => Promise<StoredUser>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<StoredUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const logout = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = null;
    removeToken();
    setUser(null);
  }, []);

  const scheduleExpiration = useCallback((token: string) => {
    if (timerRef.current) clearTimeout(timerRef.current);
    const exp = getTokenExpiration(token);
    if (!exp) return;
    const remaining = exp - Date.now();
    if (remaining <= 0) {
      logout();
      return;
    }
    timerRef.current = setTimeout(logout, remaining);
  }, [logout]);

  useEffect(() => {
    const token = getToken();
    const saved = getUserInfo();
    if (token && saved) {
      const exp = getTokenExpiration(token);
      if (exp && exp <= Date.now()) {
        removeToken();
      } else {
        setUser(saved);
        scheduleExpiration(token);
      }
    }
    setIsLoading(false);
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [scheduleExpiration]);

  const login = async (body: LoginRequest): Promise<StoredUser> => {
    const res = await authApi.login(body);
    const remember = body.rememberMe ?? false;
    const { email, name, team, position, role, admin } = res.user;
    const u: StoredUser = { email, name, team, position, role, admin };
    setToken(res.token, remember);
    setUserInfo(u, remember);
    setUser(u);
    scheduleExpiration(res.token);
    return u;
  };

  return (
    <AuthContext.Provider value={{ user, isAuthenticated: !!user, isLoading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

// eslint-disable-next-line react-refresh/only-export-components
export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within an AuthProvider');
  return ctx;
};
