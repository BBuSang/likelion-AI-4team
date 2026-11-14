// 채팅: 대화 내역(두 번째 칸) + 질문·답변(가운데) + 근거 문서(오른쪽)
// 대화 내역은 이 브라우저 탭(sessionStorage)에만, 가린 글만 저장. 실명은 저장하지 않음. 서버 저장은 다음 단계
import { useEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { AlertTriangle, FileText, Files, MessageSquarePlus, Search, Send, ShieldCheck, Trash2 } from 'lucide-react';
import { useAuth } from '@/shared/contexts/AuthContext';
import { errMsg } from '@/shared/utils/error';
import * as chatApi from '../api/chat.api';
import type { AskResult } from '../api/chat.api';
import Restorable from '../components/Restorable';
import './chat.css';

interface Msg {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  result?: AskResult;
  error?: string;
}
interface Conv { id: string; title: string; at: number; msgs: Msg[] }

const ROLE_NAME = { STAFF: '직원', LEAD: '팀장', AUDITOR: '감사' } as const;
const SOURCE_KO: Record<string, string> = { internal: '사내 기록', law: '법령', product: '금융상품', noaction: '비조치의견서' };
const ROUTE_KO: Record<string, string> = { internal: '사내', public: '공개 자료', mixed: '사내+공개', out_of_scope: '범위 밖', clarify: '되묻기' };

/** 답변 밑 한 줄: 에이전트 분류 · 모델 · 시간 */
function footer(r: AskResult) {
  const t = r.timing;
  const parts: string[] = [];
  if (r.agent) parts.push(`분류: ${ROUTE_KO[r.agent.route] ?? r.agent.route}${r.agent.fallback ? '(기본값)' : ''}`);
  if (r.model) parts.push(r.model);
  if (t.classify) parts.push(`분류 ${t.classify.toFixed(1)}초`);
  if (t.search != null) parts.push(`검색 ${t.search.toFixed(1)}초`);
  if (t.llm) parts.push(`답변 ${t.llm.toFixed(1)}초`);
  return parts.join(' · ');
}
const uid = () => Math.random().toString(36).slice(2, 10);

function load(key: string): Conv[] {
  try {
    return JSON.parse(sessionStorage.getItem(key) ?? '[]') as Conv[];
  } catch {
    return [];
  }
}

function group(convs: Conv[]) {
  const today = new Date().toDateString();
  return [
    { name: '오늘', items: convs.filter((c) => new Date(c.at).toDateString() === today) },
    { name: '이전', items: convs.filter((c) => new Date(c.at).toDateString() !== today) },
  ].filter((g) => g.items.length);
}

export default function ChatPage() {
  const { user } = useAuth();
  const key = `chat:${user?.email}`;
  const [convs, setConvs] = useState<Conv[]>(() => load(key));
  const [currentId, setCurrentId] = useState<string | null>(convs[0]?.id ?? null);
  const [q, setQ] = useState('');
  const [filter, setFilter] = useState('');
  const [busy, setBusy] = useState(false);
  const [focusMsg, setFocusMsg] = useState<string | null>(null);
  const [focusDoc, setFocusDoc] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const sideRef = useRef<HTMLElement>(null);

  useEffect(() => {
    try { sessionStorage.setItem(key, JSON.stringify(convs.slice(0, 30))); } catch { /* 저장 못 해도 화면은 동작 */ }
  }, [convs, key]);

  const current = convs.find((c) => c.id === currentId) ?? null;
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [current?.msgs.length, busy]);

  // 오른쪽 근거 문서: 고른 답변, 없으면 근거 문서가 있는 마지막 답변 (범위 밖·준비 중 답변 뒤에도 앞 답변 근거가 보이게)
  const sourceMsg = useMemo(() => {
    const answers = (current?.msgs ?? []).filter((m) => m.result);
    const withDocs = answers.filter((m) => m.result!.docs.length);
    return answers.find((m) => m.id === focusMsg) ?? withDocs[withDocs.length - 1] ?? answers[answers.length - 1] ?? null;
  }, [current, focusMsg]);
  // 그 답변의 질문 (오른쪽 칸 머리에 어느 질문의 근거인지 표시)
  const sourceQuestion = useMemo(() => {
    const msgs = current?.msgs ?? [];
    const i = msgs.findIndex((m) => m.id === sourceMsg?.id);
    for (let j = i - 1; j >= 0; j--) if (msgs[j].role === 'user') return msgs[j].text;
    return null;
  }, [current, sourceMsg]);

  /** 답변의 '근거 문서' 버튼: 오른쪽 칸을 그 답변의 근거로 바꾸고 맨 위로 */
  const showSources = (id: string) => {
    setFocusMsg(id);
    setFocusDoc(null);
    sideRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const update = (id: string, fn: (c: Conv) => Conv) => setConvs((cs) => cs.map((c) => (c.id === id ? fn(c) : c)));

  const send = async (e: FormEvent) => {
    e.preventDefault();
    const query = q.trim();
    if (!query || busy) return;
    let convId = currentId;
    if (!current) {
      const c: Conv = { id: uid(), title: query.slice(0, 30), at: Date.now(), msgs: [] };
      setConvs((cs) => [c, ...cs]);
      convId = c.id;
      setCurrentId(c.id);
    }
    const userMsg: Msg = { id: uid(), role: 'user', text: query };
    update(convId!, (c) => ({ ...c, at: Date.now(), msgs: [...c.msgs, userMsg] }));
    setQ('');
    setBusy(true);
    try {
      const r = await chatApi.ask(query);
      const m: Msg = { id: uid(), role: 'assistant', text: r.answer ?? '', result: r };
      update(convId!, (c) => ({ ...c, msgs: [...c.msgs, m] }));
      setFocusMsg(m.id);
      setFocusDoc(null);
    } catch (err) {
      update(convId!, (c) => ({ ...c, msgs: [...c.msgs, { id: uid(), role: 'assistant', text: '', error: errMsg(err) }] }));
    } finally {
      setBusy(false);
    }
  };

  const newChat = () => { setCurrentId(null); setFocusMsg(null); setFocusDoc(null); };
  const remove = (id: string) => { setConvs((cs) => cs.filter((c) => c.id !== id)); if (id === currentId) newChat(); };
  const shown = convs.filter((c) => c.title.includes(filter.trim()));

  return (
    <>
      <aside className="panel side chat-list">
        <button className="btn btn-primary btn-block" onClick={newChat}><MessageSquarePlus size={16} aria-hidden /> 새 채팅</button>
        <label className="search">
          <Search size={14} aria-hidden />
          <input placeholder="대화 검색" value={filter} onChange={(e) => setFilter(e.target.value)} />
        </label>
        {group(shown).map((g) => (
          <div key={g.name}>
            <p className="list-head">{g.name}</p>
            {g.items.map((c) => (
              <div key={c.id} className={`conv${c.id === currentId ? ' active' : ''}`} onClick={() => { setCurrentId(c.id); setFocusMsg(null); setFocusDoc(null); }}>
                <span className="conv-title">{c.title}</span>
                <button className="icon-btn" aria-label="대화 지우기" onClick={(e) => { e.stopPropagation(); remove(c.id); }}><Trash2 size={13} /></button>
              </div>
            ))}
          </div>
        ))}
        {convs.length === 0 && <p className="small muted" style={{ marginTop: 14 }}>질문하면 여기에 쌓여요. 이 탭을 닫으면 지워져요.</p>}
      </aside>

      <section className="panel chat-main">
        <header className="chat-head">
          <h2>{current?.title ?? '새 채팅'}</h2>
          <div className="spacer" />
          {user && <span className="badge badge-brand">{ROLE_NAME[user.role]} · {user.team ?? '팀 없음'}</span>}
        </header>

        <div className="chat-body">
          {!current?.msgs.length && (
            <div className="empty">
              <ShieldCheck size={36} color="var(--brand)" aria-hidden />
              <h2>무엇을 찾아 드릴까요?</h2>
              <p className="muted">답변 속 개인정보는 가려진 채로 보여요. 권한이 있는 문서만 실명으로 볼 수 있어요.</p>
              <div className="examples">
                {['카드 분실 신고 처리 내역', '대출 상담 기록 찾아줘', '보이스피싱 피해 접수 건'].map((x) => (
                  <button key={x} className="btn btn-sm" onClick={() => { setQ(x); inputRef.current?.focus(); }}>{x}</button>
                ))}
              </div>
            </div>
          )}
          {current?.msgs.map((m) => m.role === 'user' ? (
            <div key={m.id} className="bubble-user">{m.text}</div>
          ) : (
            <div key={m.id} className={`answer${sourceMsg?.id === m.id ? ' focused' : ''}`} onClick={() => setFocusMsg(m.id)}>
              {m.error && <div className="alert alert-bad">{m.error}</div>}
              {m.result && (m.result.answer ? (
                <>
                  {m.result.askBack && <p className="small muted" style={{ marginBottom: 6 }}>같은 이름의 사람이 여럿이라 먼저 확인할게요.</p>}
                  <Restorable text={m.result.answer} docIds={m.result.docs.map((d) => d.docId)} onCite={(d) => { setFocusMsg(m.id); setFocusDoc(d); }} />
                  {!!m.result.agent?.pending.length && m.result.agent.sources.includes('internal') && (
                    <p className="small" style={{ color: 'var(--warn)', marginTop: 6 }}>
                      {m.result.agent.pending.map((s) => SOURCE_KO[s] ?? s).join(', ')} 자료는 아직 준비 중이라 사내 기록으로만 답했어요.
                    </p>
                  )}
                  {!m.result.askBack && !!m.result.ambiguousPeople?.length && (
                    <p className="small homonym-warn">
                      <AlertTriangle size={13} aria-hidden /> 동명이인 가능: 질문·답변의{' '}
                      {m.result.ambiguousPeople.map((t) => `이름·${t.slice(8, 12)}`).join(', ')}은(는) 같은 이름의 다른 사람이 있어요.
                      다른 사람 기록이 섞였을 수 있으니 근거 문서를 확인해 주세요.
                    </p>
                  )}
                  {!!m.result.check?.unverifiedTokens.length && (
                    <p className="small" style={{ color: 'var(--warn)', marginTop: 6 }}>확인 안 된 토큰 {m.result.check.unverifiedTokens.length}개를 가렸어요.</p>
                  )}
                </>
              ) : (
                <p className="muted">답변 생성이 꺼져 있어 검색 결과만 보여 드려요. 오른쪽 근거 문서를 확인해 주세요. ({m.result.docs.length}건)</p>
              ))}
              {m.result && (
                <div className="answer-foot">
                  <span className="small muted">{footer(m.result)}</span>
                  {m.result.docs.length > 0 && (
                    <button type="button" className={`btn btn-sm${sourceMsg?.id === m.id ? ' btn-outline' : ''}`}
                            onClick={(e) => { e.stopPropagation(); showSources(m.id); }}
                            aria-pressed={sourceMsg?.id === m.id}>
                      <Files size={14} aria-hidden /> {sourceMsg?.id === m.id ? '근거 문서 보는 중' : `근거 문서 ${m.result.docs.length}건`}
                    </button>
                  )}
                </div>
              )}
            </div>
          ))}
          {busy && <div className="answer"><span className="muted">문서를 찾는 중…</span></div>}
          <div ref={endRef} />
        </div>

        <form className="composer" onSubmit={send}>
          <input ref={inputRef} value={q} onChange={(e) => setQ(e.target.value)} placeholder="질문을 입력하세요" maxLength={1000} />
          <button className="btn btn-primary btn-sm" disabled={busy || !q.trim()} aria-label="보내기"><Send size={15} /></button>
        </form>
      </section>

      <aside className="panel side side-wide" ref={sideRef}>
        <h3 style={{ marginBottom: sourceQuestion ? 4 : 10 }}>근거 문서</h3>
        {sourceQuestion && <p className="small muted source-q" title={sourceQuestion}>질문: {sourceQuestion}</p>}
        {!sourceMsg && <p className="small muted">답변을 받으면 근거 문서가 여기에 보여요.</p>}
        {sourceMsg?.result && !sourceMsg.result.docs.length && <p className="small muted">이 답변에는 근거 문서가 없어요.</p>}
        {sourceMsg?.result?.docs.map((d, i) => (
          <div key={d.docId} className={`source${focusDoc === d.docId ? ' focused' : ''}`}>
            <div className="row">
              <FileText size={14} color="var(--brand)" aria-hidden />
              <span style={{ color: 'var(--brand)', fontWeight: 500 }}>[{i + 1}] {d.docId}</span>
              <div className="spacer" />
              <span className="badge badge-gray">{d.docType ?? '문서'}</span>
            </div>
            <div style={{ marginTop: 6 }}>
              <Restorable text={d.snippet} docIds={[d.docId]} compact />
            </div>
          </div>
        ))}
        {sourceMsg?.result && (
          <div className="alert alert-info small" style={{ marginTop: 10 }}>
            권한 밖 문서는 검색 단계에서 빠져요. 실명 보기는 모두 기록돼요.
          </div>
        )}
      </aside>
    </>
  );
}
