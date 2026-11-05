// 백엔드 AuthDtos.java 와 맞춘 요청·응답 타입
import type { Role } from '@/shared/types/user.types';

export interface SignupRequest {
  email: string;
  password: string; // 8~72자
  name: string;
}

export interface LoginRequest {
  email: string;
  password: string;
  rememberMe?: boolean;
}

export interface LoginResponse {
  token: string;
  type: string; // "Bearer"
  email: string;
  name: string;
  role: Role;
}

export interface UserResponse {
  id: number;
  email: string;
  name: string;
  role: Role;
}

export interface ApiResponse {
  success: boolean;
  message: string;
}
