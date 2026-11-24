// ============================================
// Auth API. 참고 레포는 비밀번호를 쿼리스트링(params)으로 보냈는데,
// 서버 로그·주소에 남으므로 JSON 본문으로 보냄 (백엔드도 그렇게 고침)
// ============================================

import axios from 'axios';
import apiClient from '@/shared/api/client';
import type {
  ApiResponse,
  LoginRequest,
  LoginResponse,
  OptionsResponse,
  SignupRequest,
  UserResponse,
} from '../types/auth.types';

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

/** 가입 이메일 인증 코드 보내기 */
export const sendEmailCode = async (email: string): Promise<ApiResponse> =>
  (await apiClient.post<ApiResponse>(`${AUTH_BASE}/email/send`, { email })).data;

export const verifyEmailCode = async (email: string, code: string): Promise<ApiResponse> =>
  (await apiClient.post<ApiResponse>(`${AUTH_BASE}/email/verify`, { email, code })).data;

/** 비밀번호 재설정. 가입되지 않은 주소여도 같은 응답 */
export const sendResetCode = async (email: string): Promise<ApiResponse> =>
  (await apiClient.post<ApiResponse>(`${AUTH_BASE}/password/send`, { email })).data;

export const verifyResetCode = async (email: string, code: string): Promise<ApiResponse> =>
  (await apiClient.post<ApiResponse>(`${AUTH_BASE}/password/verify`, { email, code })).data;

export const resetPassword = async (email: string, password: string): Promise<ApiResponse> =>
  (await apiClient.post<ApiResponse>(`${AUTH_BASE}/password/reset`, { email, password })).data;

/** 가입 화면 팀·직급 선택칸 (로그인 없이) */
export const getOptions = async (): Promise<OptionsResponse> =>
  (await apiClient.get<OptionsResponse>(`${AUTH_BASE}/options`)).data;

export const signup = async (body: SignupRequest): Promise<ApiResponse> =>
  (await apiClient.post<ApiResponse>(`${AUTH_BASE}/signup`, body)).data;

/** 토큰 저장은 AuthContext.login 이 함 */
export const login = async (body: LoginRequest): Promise<LoginResponse> =>
  (await apiClient.post<LoginResponse>(`${AUTH_BASE}/login`, body)).data;

export const getMe = async (): Promise<UserResponse> =>
  (await apiClient.get<UserResponse>(`${AUTH_BASE}/me`)).data;
