// 백엔드 AdminController·MeController(/api/admin/*) 와 맞춘 타입
import apiClient from '@/shared/api/client';
import type { Role, UserStatus } from '@/shared/types/user.types';

export interface AdminUser {
  id: number; email: string; name: string;
  teamId: number | null; team: string | null; positionId: number | null; position: string | null;
  role: Role; admin: boolean; status: UserStatus; empId: string | null;
  approvedBy: number | null; approvedAt: string | null; createdAt: string;
}
export interface Employee { empId: string; name: string; team: string | null; position: string | null; role: Role; nDocs: number; sameTeam: boolean | null }
export interface Candidate { employee: Employee; linkedTo: number | null }
export interface Team { id: number; name: string; active: boolean; source: string }
export interface AuditLog { id: number; adminId: number; targetUserId: number | null; action: string; changes: string; createdAt: string }
export interface Position { id: number; name: string }

export interface ApproveBody { empId?: string; createEmployee?: boolean; teamId?: number; positionId?: number; role?: Role }
export interface UpdateBody { teamId?: number; positionId?: number; role?: Role; admin?: boolean; status?: UserStatus; empId?: string; unlinkEmployee?: boolean }

const A = '/api/admin';
export const users = async (status?: UserStatus) => (await apiClient.get<AdminUser[]>(`${A}/users`, { params: { status } })).data;
export const candidates = async (id: number) =>
  (await apiClient.get<{ candidates: Candidate[]; positionDefaultRole: Role | null }>(`${A}/users/${id}/candidates`)).data;
export const approve = async (id: number, body: ApproveBody) => (await apiClient.post<AdminUser>(`${A}/users/${id}/approve`, body)).data;
export const update = async (id: number, body: UpdateBody) => (await apiClient.put<AdminUser>(`${A}/users/${id}`, body)).data;
export const employees = async (q?: string) => (await apiClient.get<Employee[]>(`${A}/employees`, { params: { q } })).data;
export const teams = async () => (await apiClient.get<Team[]>(`${A}/teams`)).data;
export const createTeam = async (name: string) => (await apiClient.post<Team>(`${A}/teams`, { name })).data;
export const setTeamActive = async (id: number, active: boolean) => (await apiClient.put<Team>(`${A}/teams/${id}`, { active })).data;
export const syncTeams = async () => (await apiClient.post<{ added: number; total: number }>(`${A}/teams/sync`)).data;
export const auditLog = async (userId?: number) => (await apiClient.get<AuditLog[]>(`${A}/audit-log`, { params: { userId, limit: 200 } })).data;
export const positions = async () => (await apiClient.get<{ teams: Position[]; positions: Position[] }>('/api/auth/options')).data.positions;
