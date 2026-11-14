// 백엔드 IngestController (/api/ingest/*) → AI /internal/ingest/* (worklog 9장)
import apiClient from '@/shared/api/client';

export interface Span { label: string; start: number; end: number; text?: string }
export interface Job {
  id: number; uploader: string; source: string; filename: string | null; total: number; status: string; created_at: string;
  pending: number; stored: number; rejected: number; failed: number;
}
export interface JobSummary { id: number; status: string; total: number; counts: Record<string, number>; auto_approved: number; filename: string | null }
export interface StagingRow { id: number; seq: number; ext_id: string | null; doc_id: string; doc_type: string | null; status: string; approved_by: string | null; n_detected: number; has_gold: boolean; n_flags: number; n_hints: number }
export interface LinkGroup { id: string; mentions: [number, number][]; basis: string; pid: string | null; candidates: string[]; evidence: string[]; needs_check: boolean }
// 검수 판단 (worklog 29장): 이 구간들 때문에 검수로 왔음 / 놓쳤을 수 있는 곳 / 약한 사람 잇기
export interface ReviewFlag extends Span { flag: 'low_conf' | 'rule_added' | 'disagree'; p: number | null }
export interface ReviewHint extends Span { p: number }
export interface ReviewInfo {
  needs_review: boolean; flags: ReviewFlag[]; hints: ReviewHint[];
  link_check?: { mentions: [number, number][]; basis: string }[];
}
export interface StagingDoc {
  id: number; job_id: number; doc_id: string; doc_type: string | null; text: string; status: string; approved_by: string | null;
  detected: Span[]; spans: Span[]; links: LinkGroup[]; meta: Record<string, unknown>; review: ReviewInfo | null;
}
export const FLAG_KO: Record<ReviewFlag['flag'], string> = {
  low_conf: '모델 확신이 낮아요', rule_added: '보완 규칙만 찾았어요', disagree: '규칙과 모델의 판단이 달라요',
};
export const APPROVED_BY_KO: Record<string, string> = { gold: '정답과 같아 자동 저장', auto: '확인할 곳이 없어 자동 저장' };
export interface ReviewResult { status: string; doc_id?: string; added?: number; removed?: number; relabeled?: number; chunks?: number; links?: LinkGroup[] }

const B = '/api/ingest';
export const upload = async (body: { source: 'jsonl' | 'text'; content: string; filename?: string; docType?: string }) =>
  (await apiClient.post<{ job_id: number; total: number; status: string }>(`${B}/jobs`, body)).data;
export const jobs = async () => (await apiClient.get<Job[]>(`${B}/jobs`)).data;
export const job = async (id: number) => (await apiClient.get<JobSummary>(`${B}/jobs/${id}`)).data;
export const jobDocs = async (id: number) => (await apiClient.get<StagingRow[]>(`${B}/jobs/${id}/docs`)).data;
export const doc = async (id: number) => (await apiClient.get<StagingDoc>(`${B}/docs/${id}`)).data;
export const previewLinks = async (id: number, spans: Span[], choices?: Record<string, string>) =>
  (await apiClient.post<LinkGroup[]>(`${B}/docs/${id}/preview-links`, { spans, choices })).data;
export const review = async (id: number, body: { action: 'approve' | 'reject'; spans?: Span[]; reason?: string; choices?: Record<string, string> }) =>
  (await apiClient.post<ReviewResult>(`${B}/docs/${id}/review`, body)).data;

export const DOC_STATUS: Record<string, { n: string; cls: string }> = {
  pending: { n: '검수 대기', cls: 'badge-warn' }, stored: { n: '저장됨', cls: 'badge-ok' },
  rejected: { n: '반려', cls: 'badge-gray' }, failed: { n: '저장 실패', cls: 'badge-bad' },
};
