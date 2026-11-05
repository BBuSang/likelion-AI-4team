// ============================================
// Auth API. 참고 레포는 비밀번호를 쿼리스트링(params)으로 보냈는데,
// 서버 로그·주소에 남으므로 JSON 본문으로 보냄 (백엔드도 그렇게 고침)
// ============================================

import axios from 'axios';
import apiClient from '@/shared/api/client';
import type { ApiResponse, LoginRequest, LoginResponse, SignupRequest, UserResponse } from '../types/auth.types';

const AUTH_BASE = '/api/auth';

/** 사용 가능하면 true, 이미 있으면(409) false */
export const checkEmail = async (email: string): Promise<boolean> => {
  try {
    await apiClient.post<ApiResponse>(`${AUTH_BASE}/check-email`, { email });
    return true;
  } catch (e) {
    if (axios.isAxiosError(e) && e.response?.status === 409) return false;
    throw e;
  }
};

export const signup = async (body: SignupRequest): Promise<ApiResponse> =>
  (await apiClient.post<ApiResponse>(`${AUTH_BASE}/signup`, body)).data;

/** 토큰 저장은 AuthContext.login 이 함 */
export const login = async (body: LoginRequest): Promise<LoginResponse> =>
  (await apiClient.post<LoginResponse>(`${AUTH_BASE}/login`, body)).data;

export const getMe = async (): Promise<UserResponse> =>
  (await apiClient.get<UserResponse>(`${AUTH_BASE}/me`)).data;
