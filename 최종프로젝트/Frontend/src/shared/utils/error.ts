import axios from 'axios';

/** 백엔드 {success:false, message} 의 message, 없으면 상태별 기본 문구 */
export function errMsg(e: unknown, fallback = '요청을 처리하지 못했어요. 잠시 뒤 다시 시도해 주세요.'): string {
  if (axios.isAxiosError(e)) {
    const m = (e.response?.data as { message?: string } | undefined)?.message;
    if (m) return m;
    if (!e.response) return '서버에 연결할 수 없어요.';
    if (e.response.status === 403) return '권한이 없어요.';
    if (e.response.status === 502) return 'AI 서버가 응답하지 않아요.';
  }
  return fallback;
}

export const fmtTime = (s?: string | null) =>
  s ? new Date(s).toLocaleString('ko-KR', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }) : '-';
