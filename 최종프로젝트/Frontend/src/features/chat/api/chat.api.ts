// 백엔드 ChatController (/api/chat/*) 와 맞춘 요청·응답
import apiClient from '@/shared/api/client';

export interface SourceDoc {
  docId: string;
  docType: string | null;
  score: number;
  snippet: string; // 가명처리된 글 (토큰 상태)
}

export interface AskResult {
  maskedQuery: string;
  answer: string | null; // 토큰 상태. AI 서버에 LLM 키가 없으면 null (검색 결과만)
  askBack: boolean;
  docs: SourceDoc[];
  check: { unverifiedTokens: string[]; rawPii: unknown[] } | null;
  model: string | null;
  timing: Record<string, number>;
  agent: AgentInfo | null; // 에이전트 분류 결과. AI 서버 AGENT_CHAT=0 이면 null
}

export interface AgentInfo {
  route: 'internal' | 'public' | 'mixed' | 'out_of_scope' | 'clarify';
  sources: string[];
  reason: string;
  fallback: boolean;
  pending: string[]; // 필요하지만 아직 준비 안 된 RAG (law·product·noaction)
  model: string | null;
  costUsd: number;
}

export interface RestoreResult {
  text: string; // 실명이 들어간 글 — 화면에만 보여 주고 저장하지 않음
  restored: number;
  denied: string[];
  unknown: string[];
  docs: string[];
}

export const ask = async (query: string, k = 3): Promise<AskResult> =>
  (await apiClient.post<AskResult>('/api/chat/ask', { query, k })).data;

export const restore = async (text: string, docIds: string[], purpose?: string): Promise<RestoreResult> =>
  (await apiClient.post<RestoreResult>('/api/chat/restore', { text, docIds, purpose })).data;
