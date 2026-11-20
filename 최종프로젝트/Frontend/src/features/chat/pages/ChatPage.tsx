// 채팅: 대화 내역(두 번째 칸) + 질문·답변(가운데) + 근거 문서(오른쪽)
// 대화 내역은 계정별로 서버(DB)에 저장 (worklog 61장). 답은 토큰 상태, 친 질문 원문은 서버가 암호문으로만 저장해
// 본인에게만 풀어 줌 (64장). 실명 보기 결과는 저장하지 않음
import { useEffect, useMemo, useRef, useState } from 'react';
import type { FormEvent, ReactNode } from 'react';
import { AlertTriangle, ChevronDown, ChevronUp, FileText, Files, MessageSquarePlus, PanelLeft, Search, Send, Trash2 } from 'lucide-react';
import { LogoLarge } from '@/shared/components/Logo';
import { useAuth } from '@/shared/contexts/AuthContext';
import { errMsg } from '@/shared/utils/error';
import * as chatApi from '../api/chat.api';
import type { AskResult, ConversationDetail, ConversationSummary } from '../api/chat.api';
import AnswerMarkdown from '../components/AnswerMarkdown';
import MaskedText from '../components/MaskedText';
import Restorable from '../components/Restorable';
import StreamProgress from '../components/StreamProgress';
import type { Step } from '../components/StreamProgress';
import './chat.css';

interface Msg {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  result?: AskResult;
  error?: string;
  // 답변 스트리밍 중 (worklog 59장): text = 지금까지 받은 조각(서버 출력 검사를 거친 것), 끝나면 result 로 바꿈
  streaming?: boolean;
  steps?: Step[]; // 진행 단계 (worklog 60장)
  masked?: boolean; // 서버에서 불러온 질문 (가명처리된 글 → 토큰 칩으로)
}
/** id = 서버 대화 번호. loaded: 내용을 서버에서 불러왔는지 (목록만 받은 대화는 열 때 불러옴) */
interface Conv { id: number; title: string; at: number; msgs: Msg[]; loaded: boolean }

const ROLE_NAME = { STAFF: '직원', LEAD: '팀장', AUDITOR: '감사' } as const;
const SOURCE_KO: Record<string, string> = { internal: '사내 기록', law: '법령', product: '금융상품', noaction: '비조치의견서' };
const ROUTE_KO: Record<string, string> = { internal: '사내', public: '공개 자료', mixed: '사내+공개', out_of_scope: '범위 밖', clarify: '되묻기' };

// 이 글자 수나 줄 수보다 긴 근거 문서는 접어서 보여 줌 (상품 설명서는 짧아도 줄바꿈이 많음, worklog 57장)
const SOURCE_PREVIEW = 360;
const SOURCE_LINES = 6;

/** 근거 문서 본문: 길면 몇 줄만 보이게 접고 "전체 보기". 글은 자르지 않고 가리기만 (실명 보기가 전체 글에 그대로 동작) */
function SourceBody({ long, children }: { long: boolean; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="source-body-wrap">
      <div className={`source-body${long && !open ? ' collapsed' : ''}`}>{children}</div>
      {long && (
        <button type="button" className="source-more" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
          {open ? <><ChevronUp size={13} aria-hidden /> 접기</> : <><ChevronDown size={13} aria-hidden /> 전체 보기</>}
        </button>
      )}
    </div>
  );
}

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

const fromSummary = (s: ConversationSummary): Conv =>
  ({ id: s.id, title: s.title ?? '새 대화', at: Date.parse(s.updatedAt), msgs: [], loaded: false });

/** 서버에 저장된 질문·답변 → 화면 메시지 */
function toMsgs(d: ConversationDetail): Msg[] {
  return d.messages.flatMap((m): Msg[] => [
    // 본인 질문은 친 그대로(서버가 암호문을 풀어 줌). 못 풀면 가명처리된 질문을 칩으로
    m.query != null
      ? { id: `q${m.id}`, role: 'user', text: m.query }
      : { id: `q${m.id}`, role: 'user', text: m.maskedQuery, masked: true },
    m.result
      ? { id: `a${m.id}`, role: 'assistant', text: m.result.answer ?? '', result: m.result }
      : { id: `a${m.id}`, role: 'assistant', text: '', error: '저장된 답변을 읽지 못했어요.' },
  ]);
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
  const [convs, setConvs] = useState<Conv[]>([]);
  const [currentId, setCurrentId] = useState<number | null>(null);
  const [listState, setListState] = useState<'loading' | 'ok' | 'error'>('loading');
  const [openErr, setOpenErr] = useState('');
  const [q, setQ] = useState('');
  const [filter, setFilter] = useState('');
  const [busy, setBusy] = useState(false);
  const [focusMsg, setFocusMsg] = useState<string | null>(null);
  const [focusDoc, setFocusDoc] = useState<string | null>(null);
  // 대화 목록 접기·펼치기 (worklog 45장): 처음엔 넓은 창이면 펼침, 이후엔 마지막 선택을 기억
  const [listOpen, setListOpen] = useState<boolean>(() => {
    try {
      const v = localStorage.getItem('saebom.chatListOpen');
      if (v !== null) return v === '1';
    } catch { /* 저장소를 못 쓰면 기본값 */ }
    return window.innerWidth > 1100;
  });
  const toggleList = () => setListOpen((v) => {
    try { localStorage.setItem('saebom.chatListOpen', v ? '0' : '1'); } catch { /* 무시 */ }
    return !v;
  });
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const sideRef = useRef<HTMLElement>(null);

  // 대화 목록: 서버에서 (계정별). 예전 탭 저장분(sessionStorage)은 지움
  useEffect(() => {
    try { sessionStorage.removeItem(`chat:${user?.email}`); } catch { /* 무시 */ }
    chatApi.conversations()
      .then((list) => { setConvs(list.map(fromSummary)); setListState('ok'); })
      .catch(() => setListState('error'));
  }, [user?.email]);

  /** 대화 열기: 목록만 받은 대화는 내용을 서버에서 불러옴 */
  const open = async (id: number) => {
    setCurrentId(id);
    setFocusMsg(null);
    setFocusDoc(null);
    setOpenErr('');
    const c = convs.find((x) => x.id === id);
    if (!c || c.loaded) return;
    try {
      const d = await chatApi.conversation(id);
      update(id, (x) => ({ ...x, msgs: toMsgs(d), loaded: true }));
    } catch (err) {
      setOpenErr(errMsg(err));
    }
  };

  const current = convs.find((c) => c.id === currentId) ?? null;
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [current?.msgs.length, busy]);
  // 스트리밍으로 답이 길어지는 동안: 맨 아래를 보고 있을 때만 따라 내려감 (위로 올려 읽는 중이면 그대로)
  const streamingLen = current?.msgs.find((m) => m.streaming)?.text.length ?? 0;
  useEffect(() => {
    const box = endRef.current?.parentElement;
    if (streamingLen && box && box.scrollHeight - box.scrollTop - box.clientHeight < 160) {
      endRef.current?.scrollIntoView({ behavior: 'auto' });
    }
  }, [streamingLen]);

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
    for (let j = i - 1; j >= 0; j--) if (msgs[j].role === 'user') return msgs[j];
    return null;
  }, [current, sourceMsg]);

  /** 답변의 '근거 문서' 버튼: 오른쪽 칸을 그 답변의 근거로 바꾸고 맨 위로 */
  const showSources = (id: string) => {
    setFocusMsg(id);
    setFocusDoc(null);
    sideRef.current?.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const update = (id: number, fn: (c: Conv) => Conv) => setConvs((cs) => cs.map((c) => (c.id === id ? fn(c) : c)));

  const send = async (e: FormEvent) => {
    e.preventDefault();
    const query = q.trim();
    if (!query || busy) return;
    setBusy(true);
    setOpenErr('');
    let convId = currentId;
    if (!current) {
      // 새 대화는 서버에 먼저 만들고 (번호를 받아) 질문에 붙여 보냄 → 답이 끝나면 서버가 그 대화에 저장
      try {
        const s = await chatApi.createConversation();
        // 제목: 이번 화면에선 친 질문, 다시 열면 서버가 정한 가명처리된 첫 질문
        const c: Conv = { ...fromSummary(s), title: query.slice(0, 30), at: Date.now(), loaded: true };
        setConvs((cs) => [c, ...cs]);
        convId = c.id;
        setCurrentId(c.id);
      } catch (err) {
        setOpenErr(`대화를 만들지 못했어요. ${errMsg(err)}`);
        setBusy(false);
        return;
      }
    }
    const userMsg: Msg = { id: uid(), role: 'user', text: query };
    update(convId!, (c) => ({ ...c, at: Date.now(), msgs: [...c.msgs, userMsg] }));
    setQ('');
    // 답변 스트리밍 (worklog 59장): 자리를 먼저 만들고 단계·조각이 올 때마다 채움. 끝나면 결과(전체 글 기준 출력 검사)로 바꿈
    const id = uid();
    const patch = (fn: (m: Msg) => Msg) =>
      update(convId!, (c) => ({ ...c, msgs: c.msgs.map((m) => (m.id === id ? fn(m) : m)) }));
    update(convId!, (c) => ({ ...c, msgs: [...c.msgs, { id, role: 'assistant', text: '', streaming: true, steps: [] }] }));
    try {
      const r = await chatApi.askStream(query, {
        // 새 단계가 오면 앞 단계를 끝낸 것으로 (걸린 시간 = 다음 단계가 올 때까지)
        onStage: (stage, data) => patch((m) => {
          const now = performance.now();
          const steps = (m.steps ?? []).map((s, i, a) => (i === a.length - 1 && s.end == null ? { ...s, end: now } : s));
          return { ...m, steps: [...steps, { stage, start: now, data }] };
        }),
        onStepDone: (stage, data) => patch((m) => {
          const steps = [...(m.steps ?? [])];
          for (let i = steps.length - 1; i >= 0; i--) {
            if (steps[i].stage === stage) { steps[i] = { ...steps[i], result: data }; break; }
          }
          return { ...m, steps };
        }),
        onDelta: (t) => patch((m) => ({ ...m, text: m.text + t })),
      }, { conversationId: convId! });
      patch((m) => ({ id: m.id, role: 'assistant', text: r.answer ?? '', result: r }));
      setFocusMsg(id);
      setFocusDoc(null);
    } catch (err) {
      patch((m) => ({ id: m.id, role: 'assistant', text: '', error: errMsg(err) }));
    } finally {
      setBusy(false);
    }
  };

  const newChat = () => { setCurrentId(null); setFocusMsg(null); setFocusDoc(null); setOpenErr(''); };
  const remove = async (id: number) => {
    if (!window.confirm('이 대화를 지울까요? 계정에 저장된 내역에서도 지워지고 되돌릴 수 없어요.')) return;
    try {
      await chatApi.deleteConversation(id);
      setConvs((cs) => cs.filter((c) => c.id !== id));
      if (id === currentId) newChat();
    } catch (err) {
      setOpenErr(`대화를 지우지 못했어요. ${errMsg(err)}`);
    }
  };
  const shown = convs.filter((c) => c.title.includes(filter.trim()));

  return (
    <>
      <aside className={`panel side chat-list${listOpen ? '' : ' collapsed'}`} aria-hidden={!listOpen}>
        <button className="btn btn-primary btn-block" onClick={newChat}><MessageSquarePlus size={16} aria-hidden /> 새 채팅</button>
        <label className="search">
          <Search size={14} aria-hidden />
          <input placeholder="대화 검색" value={filter} onChange={(e) => setFilter(e.target.value)} />
        </label>
        {group(shown).map((g) => (
          <div key={g.name}>
            <p className="list-head">{g.name}</p>
            {g.items.map((c) => (
              <div key={c.id} className={`conv${c.id === currentId ? ' active' : ''}`} onClick={() => open(c.id)} title={c.title}>
                <span className="conv-title"><MaskedText text={c.title} /></span>
                <button className="icon-btn" aria-label="대화 지우기" onClick={(e) => { e.stopPropagation(); remove(c.id); }}><Trash2 size={13} /></button>
              </div>
            ))}
          </div>
        ))}
        {listState === 'loading' && <p className="small muted" style={{ marginTop: 14 }}>대화 목록을 불러오는 중…</p>}
        {listState === 'error' && <p className="small" style={{ marginTop: 14, color: 'var(--bad)' }}>대화 목록을 불러오지 못했어요.</p>}
        {listState === 'ok' && convs.length === 0 && (
          <p className="small muted" style={{ marginTop: 14 }}>대화가 없습니다.</p>
        )}
      </aside>

      <section className="panel chat-main">
        <header className="chat-head">
          <button type="button" className="btn btn-sm list-toggle" onClick={toggleList} aria-expanded={listOpen}
                  title={listOpen ? '대화 목록 접기' : '대화 목록 펼치기'}>
            <PanelLeft size={15} aria-hidden /> 
          </button>
          <h2 title={current?.title}>{current ? <MaskedText text={current.title} /> : '새 채팅'}</h2>
          <div className="spacer" />
          {user && <span className="badge badge-brand">{ROLE_NAME[user.role]} · {user.team ?? '팀 없음'}</span>}
        </header>

        <div className="chat-body">
          {openErr && <div className="alert alert-bad">{openErr}</div>}
          {current && !current.loaded && !openErr && <p className="small muted">대화를 불러오는 중…</p>}
          {(!current || (current.loaded && !current.msgs.length)) && (
            <div className="empty">
              <LogoLarge tone="blue" width={64} className="empty-logo" />
              <h2>무엇을 찾아 드릴까요?</h2>
              <p className="muted">답변 속 개인정보는 가려진 채로 보여요.<br></br>
                 권한이 있는 문서만 실명으로 볼 수 있어요.</p>
              
            </div>
          )}
          {current?.msgs.map((m) => m.role === 'user' ? (
            <div key={m.id} className="bubble-user">{m.masked ? <MaskedText text={m.text} /> : m.text}</div>
          ) : (
            <div key={m.id} className={`answer${sourceMsg?.id === m.id ? ' focused' : ''}`} onClick={() => setFocusMsg(m.id)}>
              {m.error && <div className="alert alert-bad">{m.error}</div>}
              {m.streaming && (
                <>
                  {m.text && <AnswerMarkdown text={m.text} />}
                  {m.steps?.length ? (
                    <StreamProgress steps={m.steps} writing={!!m.text} />
                  ) : (
                    <p className="small muted stream-stage" role="status"><span className="stream-dot" aria-hidden />보내는 중…</p>
                  )}
                </>
              )}
              {m.result && (m.result.answer ? (
                <>
                  {m.result.askBack && <p className="small muted" style={{ marginBottom: 6 }}>같은 이름의 사람이 여럿이라 먼저 확인할게요.</p>}
                  <Restorable markdown text={m.result.answer} docIds={m.result.docs.map((d) => d.docId)} onCite={(d) => { setFocusMsg(m.id); setFocusDoc(d); }} />
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
          <div ref={endRef} />
        </div>

        <form className="composer" onSubmit={send}>
          <input ref={inputRef} value={q} onChange={(e) => setQ(e.target.value)} placeholder="질문을 입력하세요" maxLength={1000} />
          <button className="btn btn-primary btn-sm" disabled={busy || !q.trim()} aria-label="보내기"><Send size={15} /></button>
        </form>
      </section>

      <aside className="panel side side-wide" ref={sideRef}>
        <h3 style={{ marginBottom: sourceQuestion ? 4 : 10 }}>근거 문서</h3>
        {sourceQuestion && (
          <p className="small muted source-q" title={sourceQuestion.text}>
            질문: {sourceQuestion.masked ? <MaskedText text={sourceQuestion.text} /> : sourceQuestion.text}
          </p>
        )}
        {!sourceMsg && <p className="small muted"></p>}
        {sourceMsg?.result && !sourceMsg.result.docs.length && <p className="small muted">이 답변에는 근거 문서가 없어요.</p>}
        {sourceMsg?.result?.docs.map((d, i) => (
          <div key={d.docId} className={`source${focusDoc === d.docId ? ' focused' : ''}`}>
            <div className="source-head">
              <FileText size={14} color="var(--brand)" aria-hidden />
              <span className="source-id">[{i + 1}] {d.docId}</span>
            </div>
            <p className="source-type">{d.docType ?? '문서'}</p>
            <SourceBody long={d.snippet.length > SOURCE_PREVIEW || d.snippet.split('\n').length > SOURCE_LINES}>
              <Restorable text={d.snippet} docIds={[d.docId]} compact />
            </SourceBody>
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
