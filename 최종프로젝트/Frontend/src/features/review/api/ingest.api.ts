// 백엔드 IngestController (/api/ingest/*) → AI /internal/ingest/*
import apiClient from '@/shared/api/client';

export interface Span { label: string; start: number; end: number; text?: string }
export interface Job {
  id: number; uploader: string; source: string; filename: string | null; total: number; status: string; created_at: string; error: string | null;
  pending: number; stored: number; rejected: number; failed: number;
}
export interface JobSummary { id: number; status: string; total: number; counts: Record<string, number>; auto_approved: number; filename: string | null }
export interface StagingRow { id: number; seq: number; ext_id: string | null; doc_id: string; doc_type: string | null; status: string; approved_by: string | null; n_detected: number; has_gold: boolean; n_flags: number; n_hints: number }
export interface LinkGroup { id: string; mentions: [number, number][]; basis: string; pid: string | null; candidates: string[]; evidence: string[]; needs_check: boolean }
// 검수 판단: 이 구간들 때문에 검수로 왔음 / 놓쳤을 수 있는 곳 / 약한 사람 잇기
export interface ReviewFlag extends Span { flag: 'low_conf' | 'rule_added' | 'disagree'; p: number | null }
export interface ReviewHint extends Span { p: number }
export interface ReviewInfo {
  needs_review: boolean; flags: ReviewFlag[]; hints: ReviewHint[];
  link_check?: { mentions: [number, number][]; basis: string }[];
  doc_type_check?: { pick: string; second: string | null }; // 종류 자동 판단이 애매함
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
  (await apiClient.post<{ job_id: number; total: number; status: string; skipped?: { line: number; id: string | null }[] }>(`${B}/jobs`, body)).data;
// 녹음 (wav·mp3·m4a 등) — 받아쓰기 → 글 1건. 녹음 파일은 저장하지 않음
// kind: call 상담 · meeting 회의 (목소리로 화자 나눔), speakers: 회의 참석자 수 (모르면 비움)
export type AudioKind = 'call' | 'meeting';
export const uploadAudio = async (file: File, opt: { kind?: AudioKind; speakers?: number; docType?: string } = {}) => {
  const fd = new FormData();
  fd.append('file', file);
  fd.append('kind', opt.kind ?? 'call');
  if (opt.speakers) fd.append('speakers', String(opt.speakers));
  if (opt.docType) fd.append('docType', opt.docType);
  return (await apiClient.post<{ job_id: number; total: number; status: string; sec: number; channels: number }>(
    `${B}/audio-jobs`, fd, { headers: { 'Content-Type': 'multipart/form-data' } })).data;
};
// 올리기 화면 드롭다운: 저장된 문서 종류 (많은 순)
export interface DocTypeCount { doc_type: string; docs: number }
export const docTypes = async () => (await apiClient.get<DocTypeCount[]>(`${B}/doc-types`)).data;
export const jobs = async () => (await apiClient.get<Job[]>(`${B}/jobs`)).data;
export const job = async (id: number) => (await apiClient.get<JobSummary>(`${B}/jobs/${id}`)).data;
export const jobDocs = async (id: number) => (await apiClient.get<StagingRow[]>(`${B}/jobs/${id}/docs`)).data;
// 검수 때 본문 고치기: 안 바뀐 곳의 구간은 옮기고 바뀐 곳은 다시 탐지. spans = 지금 화면 구간 (옛 본문 위치)
export const editText = async (id: number, body: { text: string; spans: Span[] }) =>
  (await apiClient.post<{ changed: boolean; kept: number; dropped: number; added: number }>(`${B}/docs/${id}/text`, body)).data;
// 녹음 글 한 줄 '[분:초] 화자: 말' 의 화자 (서버 SPEAKER_LINE 과 같음)
export const SPEAKER_LINE = /^\[\d{2,}:\d{2}\] ([^:\n]{1,30}): /gm;
export const speakersIn = (text: string) => {
  const n = new Map<string, number>();
  for (const m of text.matchAll(SPEAKER_LINE)) n.set(m[1], (n.get(m[1]) ?? 0) + 1);
  return [...n.entries()].map(([name, lines]) => ({ name, lines }));
};
// 화자 이름 바꾸기 {지금 이름: 새 이름}. 다른 화자 이름을 넣으면 한 사람으로 합쳐짐
export const renameSpeakers = (text: string, names: Record<string, string>) =>
  text.replace(SPEAKER_LINE, (line, who: string) => {
    const to = (names[who] ?? who).trim();
    return to && to !== who ? line.slice(0, line.length - who.length - 2) + `${to}: ` : line;
  });
export const doc = async (id: number) => (await apiClient.get<StagingDoc>(`${B}/docs/${id}`)).data;
export const previewLinks = async (id: number, spans: Span[], choices?: Record<string, string>) =>
  (await apiClient.post<LinkGroup[]>(`${B}/docs/${id}/preview-links`, { spans, choices })).data;
export const review = async (id: number, body: { action: 'approve' | 'reject'; spans?: Span[]; reason?: string; choices?: Record<string, string>; docType?: string }) =>
  (await apiClient.post<ReviewResult>(`${B}/docs/${id}/review`, body)).data;

export const DOC_STATUS: Record<string, { n: string; cls: string }> = {
  pending: { n: '검수 대기', cls: 'badge-warn' }, stored: { n: '저장됨', cls: 'badge-ok' },
  rejected: { n: '반려', cls: 'badge-gray' }, failed: { n: '저장 실패', cls: 'badge-bad' },
};
