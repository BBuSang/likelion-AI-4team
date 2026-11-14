// 백엔드 AuthDtos.java 와 맞춘 요청·응답 타입
import type { Role, UserStatus } from '@/shared/types/user.types';

/** 팀·직급은 GET /api/auth/options 선택칸에서 고른 id */
export interface SignupRequest {
  email: string;
  password: string; // 8~72자
  name: string;
  teamId: number;
  positionId: number;
}

export interface LoginRequest {
  email: string;
  password: string;
  rememberMe?: boolean;
}

/** 로그인: 401 = 이메일·비밀번호 틀림, 403 = 승인 대기 또는 정지 (message 로 구분) */
export interface LoginResponse {
  token: string;
  type: string; // "Bearer"
  user: UserResponse;
}

export interface UserResponse {
  id: number;
  email: string;
  name: string;
  team: string | null;
  position: string | null;
  role: Role;
  admin: boolean;
  status: UserStatus;
}

export interface Option {
  id: number;
  name: string;
}

export interface OptionsResponse {
  teams: Option[];
  positions: Option[];
}

export interface ApiResponse {
  success: boolean;
  message: string;
}
