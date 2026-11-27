// 관리자: 사용자 상세의 "채팅 내역"
// 목록은 제목·개수만. 내용은 눌러서 열고, 열 때마다 서버가 변경 기록에 남김(view_chat).
// 보이는 글은 가명처리된 질문·토큰 상태 답변뿐 — 관리자 화면에서는 실명 보기를 하지 않음
import { useEffect, useState } from 'react';
import { FileText, MessageSquare, X } from 'lucide-react';
import { errMsg, fmtTime } from '@/shared/utils/error';
import { docLabel } from '@/shared/utils/pii';
import type { ConversationDetail, ConversationSummary } from '@/features/chat/api/chat.api';
import AnswerMarkdown from '@/features/chat/components/AnswerMarkdown';
import MaskedText from '@/features/chat/components/MaskedText';
import * as api from '../api/admin.api';

const ROUTE_KO: Record<string, string> = { internal: '사내', public: '공개 자료', mixed: '사내+공개', out_of_scope: '범위 밖', clarify: '되묻기' };

/** onViewed: 대화를 연 뒤 (열람 기록이 생겼으니 변경 기록을 다시 불러오게) */
export default function UserChats({ userId, name, onViewed }: { userId: number; name: string; onViewed?: () => void }) {
  const [list, setList] = useState<ConversationSummary[] | null>(null);
  const [error, setError] = useState('');
  const [openId, setOpenId] = useState<number | null>(null);

  useEffect(() => {
    api.userConversations(userId).then(setList).catch((e) => setError(errMsg(e)));
  }, [userId]);

  return (
    <div>
      <h3 style={{ marginBottom: 4 }}>채팅 내역</h3>
      <p className="small muted" style={{ marginBottom: 6 }}>가명처리된 글만 보여요. 대화를 열면 열람 기록이 남아요.</p>
      {error && <p className="small" style={{ color: 'var(--bad)' }}>{error}</p>}
      {list === null && !error && <p className="small muted">불러오는 중…</p>}
      {list?.length === 0 && <p className="small muted">대화가 없어요.</p>}
      {list?.slice(0, 20).map((c) => (
        <button key={c.id} type="button" className="chat-row" onClick={() => setOpenId(c.id)}>
          <MessageSquare size={13} aria-hidden className="muted" />
          <span className="chat-row-title"><MaskedText text={c.title ?? '(질문 없음)'} /></span>
          <span className="small muted nowrap">{c.messages}개 · {fmtTime(c.updatedAt)}</span>
        </button>
      ))}
      {list && list.length > 20 && <p className="small muted">최근 20개만 보여요 (전체 {list.length}개).</p>}
      {openId !== null && <ChatViewer userId={userId} name={name} convId={openId} onClose={() => { setOpenId(null); onViewed?.(); }} />}
    </div>
  );
}

function ChatViewer({ userId, name, convId, onClose }: { userId: number; name: string; convId: number; onClose: () => void }) {
  const [d, setD] = useState<ConversationDetail | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api.userConversation(userId, convId).then(setD).catch((e) => setError(errMsg(e)));
  }, [userId, convId]);
  useEffect(() => {
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', esc);
    return () => window.removeEventListener('keydown', esc);
  }, [onClose]);

  return (
    <div className="modal-back" onClick={onClose}>
      <div className="card modal modal-wide" role="dialog" aria-modal="true" aria-label={`${name}의 대화`} onClick={(e) => e.stopPropagation()}>
        <div className="row" style={{ marginBottom: 10 }}>
          <div style={{ minWidth: 0 }}>
            <h2 className="chat-view-title">{d?.title ? <MaskedText text={d.title} /> : '대화'}</h2>
            <p className="small muted">{name} · {d ? `${fmtTime(d.createdAt)} 시작 · 질문 ${d.messages.length}개` : '불러오는 중…'}</p>
          </div>
          <div className="spacer" />
          <button type="button" className="btn btn-sm btn-ghost" onClick={onClose} aria-label="닫기"><X size={16} /></button>
        </div>
        {error && <div className="alert alert-bad">{error}</div>}
        <div className="stack">
          {d?.messages.map((m) => (
            <div key={m.id} className="chat-view-turn">
              <div className="chat-view-q"><MaskedText text={m.maskedQuery} /></div>
              {m.result ? (
                <div className="chat-view-a">
                  {m.result.answer ? <AnswerMarkdown text={m.result.answer} /> : <p className="muted small">답변 없이 검색 결과만 있어요.</p>}
                  {m.result.docs.length > 0 && (
                    <p className="small muted chat-view-docs">
                      <FileText size={12} aria-hidden /> 근거 문서: {m.result.docs.map((x) => docLabel(x.docId)).join(', ')}
                    </p>
                  )}
                  <p className="small muted">{fmtTime(m.createdAt)}{m.result.agent ? ` · 분류 ${ROUTE_KO[m.result.agent.route] ?? m.result.agent.route}` : ''}</p>
                </div>
              ) : (
                <p className="small" style={{ color: 'var(--bad)' }}>저장된 답변을 읽지 못했어요.</p>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
