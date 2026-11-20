// 백엔드 ChatController (/api/chat/*) 와 맞춘 요청·응답
import apiClient, { BASE_URL, refreshTokenIfNeeded } from '@/shared/api/client';
import { ApiMessageError } from '@/shared/utils/error';
import { getToken, removeToken } from '@/shared/utils/tokenManager';

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
  ambiguousPeople?: string[]; // 답변 속 사람 중 같은 이름의 다른 사람이 있는 토큰 (worklog 37장)
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
  warnings?: AccessWarning[]; // 이번 실명 보기가 걸린 이상 접근 규칙 (본인에게 알림, worklog 39장)
}

export interface AccessWarning { rule: string; ruleKo: string; severity: 'high' | 'medium' | 'low'; detail: string }

export const ask = async (query: string, k = 3): Promise<AskResult> =>
  (await apiClient.post<AskResult>('/api/chat/ask', { query, k })).data;

/** 진행 단계 (AI app/agents/progress.py) */
export type StreamStage = 'protect' | 'classify' | 'plan' | 'search' | 'replan' | 'search2' | 'judge' | 'relax' | 'answer' | 'check';

/** 단계에 딸린 값: sources·tools (시작) / route·sources·steps·counts·direct·total (결과) */
export interface StageData {
  sources?: string[];
  tools?: string[];
  route?: string;
  steps?: number;
  counts?: Record<string, number>;
  direct?: number;
  total?: number;
  n?: number;
}

export interface StreamHandlers {
  /** 새 단계 시작 (앞 단계는 끝남) */
  onStage?: (stage: StreamStage, data: StageData) => void;
  /** 단계 결과 (분류된 출처, 검색 건수 등) */
  onStepDone?: (stage: StreamStage, data: StageData) => void;
  /** 출력 검사를 거친 답변 조각 (서버가 문장·줄 단위로 검사한 뒤에만 보냄) */
  onDelta?: (text: string) => void;
}

/**
 * 답변 스트리밍 (worklog 59장): POST /api/chat/ask/stream 의 SSE 를 읽어 단계·조각을 알려 주고, 마지막 결과(ask 와 같은 모양)를 돌려줌.
 * axios 는 응답을 다 받은 뒤에야 넘겨서 fetch 로. 토큰 갱신·401 처리는 axios 쪽과 같게.
 */
export async function askStream(query: string, handlers: StreamHandlers = {},
                                { k = 3, signal, conversationId }: { k?: number; signal?: AbortSignal; conversationId?: number } = {}): Promise<AskResult> {
  await refreshTokenIfNeeded();
  const token = getToken();
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}/api/chat/ask/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      // conversationId 가 있으면 서버가 끝난 답을 그 대화에 저장 (worklog 61장)
      body: JSON.stringify({ query, k, conversationId }),
      signal,
    });
  } catch {
    throw new ApiMessageError('서버에 연결할 수 없어요.');
  }
  if (res.status === 401) {
    removeToken();
    if (window.location.pathname !== '/login') window.location.href = '/login';
    throw new ApiMessageError('로그인이 필요해요.', 401);
  }
  if (!res.ok || !res.body) {
    const body = (await res.json().catch(() => null)) as { message?: string } | null;
    const fallback = res.status === 403 ? '권한이 없어요.' : res.status === 502 ? 'AI 서버가 응답하지 않아요.' : '요청을 처리하지 못했어요.';
    throw new ApiMessageError(body?.message ?? fallback, res.status);
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = '';
  let result: AskResult | null = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let cut: number;
    while ((cut = buf.indexOf('\n\n')) >= 0) {  // 이벤트 하나 = 빈 줄까지
      const block = buf.slice(0, cut);
      buf = buf.slice(cut + 2);
      let event = 'message';
      let data = '';
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data += line.slice(5).trim();
      }
      if (!data) continue;
      const d = JSON.parse(data);
      if (event === 'stage') handlers.onStage?.(d.stage, d);
      else if (event === 'step_done') handlers.onStepDone?.(d.stage, d);
      else if (event === 'delta') handlers.onDelta?.(d.text);
      else if (event === 'done') result = d.result as AskResult;
      else if (event === 'error') throw new ApiMessageError(d.message ?? '검색 또는 답변 생성에 실패했어요.', d.status ?? 502);
    }
  }
  if (!result) throw new ApiMessageError('답변이 중간에 끊겼어요. 다시 시도해 주세요.');
  return result;
}

export const restore = async (text: string, docIds: string[], purpose?: string): Promise<RestoreResult> =>
  (await apiClient.post<RestoreResult>('/api/chat/restore', { text, docIds, purpose })).data;

// ---------------------------------------------------------------- 채팅 내역 (worklog 61장, 계정별 서버 저장)

/** title: 첫 질문의 가명처리된 글 (첫 답 전엔 null) */
export interface ConversationSummary { id: number; title: string | null; messages: number; createdAt: string; updatedAt: string }
/** 질문 하나 + 답변. query = 친 질문 원문 (서버에 암호문으로 저장, 본인 조회 때만 옴. 관리자 열람·못 풀면 null), maskedQuery = 가명처리된 질문 */
export interface StoredMessage { id: number; query: string | null; maskedQuery: string; result: AskResult | null; createdAt: string }
export interface ConversationDetail { id: number; title: string | null; createdAt: string; updatedAt: string; messages: StoredMessage[] }

const C = '/api/chat/conversations';
export const conversations = async () => (await apiClient.get<ConversationSummary[]>(C)).data;
export const createConversation = async () => (await apiClient.post<ConversationSummary>(C)).data;
export const conversation = async (id: number) => (await apiClient.get<ConversationDetail>(`${C}/${id}`)).data;
export const deleteConversation = async (id: number) => { await apiClient.delete(`${C}/${id}`); };
