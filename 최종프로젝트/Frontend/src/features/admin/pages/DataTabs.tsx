// 관리자: 저장된 문서(보기·지우기) + 문서 속 직원 목록 (worklog 15장)
// 문서 글은 가명처리된 토큰 상태만 보여 줌. 지우기는 사유 필수 + 문서 ID 다시 입력해서 확인
import { useEffect, useState } from 'react';
import { Search, Trash2 } from 'lucide-react';
import apiClient from '@/shared/api/client';
import { errMsg, fmtTime } from '@/shared/utils/error';
import MaskedText from '@/features/chat/components/MaskedText';
import * as api from '../api/admin.api';
import type { AdminUser, Employee, Team } from '../api/admin.api';

interface DocRow { doc_id: string; doc_type: string | null; owner_emp_id: string | null; team: string | null; visibility: string; source: string; created_at: string; chunks: number }
interface DocDetail extends Omit<DocRow, 'chunks' | 'source'> { basis?: Record<string, unknown>; chunks: { idx: number; masked_text: string; n_tokens: number }[]; people: number }
interface DelLog { id: number; doc_id: string; actor: string; reason: string; counts: Record<string, number>; created_at: string }

const VIS: Record<string, { n: string; cls: string }> = {
  public: { n: '공개', cls: 'badge-ok' }, team: { n: '팀', cls: 'badge-brand' }, restricted: { n: '제한(감사만)', cls: 'badge-warn' },
};
const ROLE: Record<string, string> = { STAFF: '직원', LEAD: '팀장', AUDITOR: '감사' };
const PAGE = 30;

export function DocsTab({ selected, onSelect, reloadKey }: { selected: string | null; onSelect: (id: string) => void; reloadKey: number }) {
  const [q, setQ] = useState('');
  const [vis, setVis] = useState('');
  const [source, setSource] = useState('');
  const [page, setPage] = useState(0);
  const [data, setData] = useState<{ total: number; items: DocRow[] }>({ total: 0, items: [] });
  const [log, setLog] = useState<DelLog[]>([]);
  const [error, setError] = useState('');

  useEffect(() => {
    const t = setTimeout(() => {
      apiClient.get('/api/admin/docs', { params: { q: q || undefined, visibility: vis || undefined, source: source || undefined, limit: PAGE, offset: page * PAGE } })
        .then((r) => { setData(r.data); setError(''); }).catch((e) => setError(errMsg(e)));
    }, 250);
    return () => clearTimeout(t);
  }, [q, vis, source, page, reloadKey]);
  useEffect(() => { apiClient.get<DelLog[]>('/api/admin/docs/deletion-log').then((r) => setLog(r.data)).catch(() => {}); }, [reloadKey]);

  const pages = Math.max(1, Math.ceil(data.total / PAGE));
  return (
    <>
      <div className="row" style={{ marginBottom: 10, flexWrap: 'wrap' }}>
        <label className="search" style={{ width: 260, margin: 0 }}>
          <Search size={14} aria-hidden /><input placeholder="문서 ID·종류·팀·담당 직원" value={q} onChange={(e) => { setQ(e.target.value); setPage(0); }} />
        </label>
        <select className="select" style={{ width: 150, height: 34 }} value={vis} onChange={(e) => { setVis(e.target.value); setPage(0); }}>
          <option value="">공개 범위 전체</option><option value="public">공개</option><option value="team">팀</option><option value="restricted">제한</option>
        </select>
        <select className="select" style={{ width: 150, height: 34 }} value={source} onChange={(e) => { setSource(e.target.value); setPage(0); }}>
          <option value="">들어온 경로 전체</option><option value="upload">화면 업로드</option><option value="batch">한 번에 적재</option>
        </select>
        <div className="spacer" />
        <span className="small muted">{data.total.toLocaleString()}건</span>
      </div>
      {error && <div className="alert alert-bad" style={{ marginBottom: 10 }}>{error}</div>}
      <table className="table">
        <thead><tr><th>문서 ID</th><th>종류</th><th>팀</th><th>담당 직원</th><th>공개 범위</th><th>경로</th><th>청크</th><th>저장</th></tr></thead>
        <tbody>
          {data.items.map((d) => (
            <tr key={d.doc_id} className={`clickable${selected === d.doc_id ? ' selected' : ''}`} onClick={() => onSelect(d.doc_id)}>
              <td className="nowrap" style={{ fontWeight: 500 }}>{d.doc_id}</td><td className="nowrap">{d.doc_type ?? '-'}</td><td className="nowrap">{d.team ?? '-'}</td><td>{d.owner_emp_id ?? '-'}</td>
              <td><span className={`badge ${VIS[d.visibility]?.cls ?? 'badge-gray'}`}>{VIS[d.visibility]?.n ?? d.visibility}</span></td>
              <td className="small nowrap">{d.source === 'upload' ? '화면 업로드' : '한 번에 적재'}</td><td>{d.chunks}</td>
              <td className="small muted nowrap">{fmtTime(d.created_at)}</td>
            </tr>
          ))}
          {data.items.length === 0 && <tr><td colSpan={8} className="muted" style={{ textAlign: 'center', padding: 30 }}>문서가 없어요.</td></tr>}
        </tbody>
      </table>
      <div className="row" style={{ justifyContent: 'center', marginTop: 10 }}>
        <button className="btn btn-sm" disabled={page === 0} onClick={() => setPage(page - 1)}>이전</button>
        <span className="small muted">{page + 1} / {pages}</span>
        <button className="btn btn-sm" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>다음</button>
      </div>
      <h3 style={{ margin: '20px 0 8px' }}>지운 기록</h3>
      <table className="table">
        <thead><tr><th>시각</th><th>문서</th><th>지운 사람</th><th>사유</th><th>지운 것</th></tr></thead>
        <tbody>
          {log.map((l) => (
            <tr key={l.id}>
              <td className="small muted nowrap">{fmtTime(l.created_at)}</td><td>{l.doc_id}</td><td className="small">{l.actor}</td><td className="small">{l.reason}</td>
              <td className="small muted nowrap">청크 {l.counts.chunks ?? 0} · 토큰 {l.counts.tokens ?? 0} · 사람 {l.counts.people ?? 0}</td>
            </tr>
          ))}
          {log.length === 0 && <tr><td colSpan={5} className="muted" style={{ textAlign: 'center', padding: 20 }}>지운 문서가 없어요.</td></tr>}
        </tbody>
      </table>
    </>
  );
}

export function DocDetailPanel({ docId, onDeleted, onChanged }: { docId: string; onDeleted: () => void; onChanged: () => void }) {
  const [d, setD] = useState<DocDetail | null>(null);
  const [error, setError] = useState('');
  const [confirm, setConfirm] = useState(false);
  const [reason, setReason] = useState('');
  const [typed, setTyped] = useState('');
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setD(null); setConfirm(false); setReason(''); setTyped(''); setError('');
    apiClient.get<DocDetail>(`/api/admin/docs/${encodeURIComponent(docId)}`).then((r) => setD(r.data)).catch((e) => setError(errMsg(e)));
  }, [docId]);

  const del = async () => {
    if (!reason.trim()) return setError('지우는 사유를 적어 주세요.');
    if (typed.trim() !== docId) return setError('문서 ID 를 똑같이 입력해 주세요.');
    setBusy(true);
    setError('');
    try {
      await apiClient.post(`/api/admin/docs/${encodeURIComponent(docId)}/delete`, { reason: reason.trim() });
      onDeleted();
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  if (!d) return error ? <div className="alert alert-bad">{error}</div> : <p className="small muted">불러오는 중…</p>;
  const reload = () => apiClient.get<DocDetail>(`/api/admin/docs/${encodeURIComponent(docId)}`).then((r) => setD(r.data));
  return (
    <div className="stack">
      <div>
        <h2>{d.doc_id}</h2>
        <p className="small muted">{d.doc_type ?? '문서'} · 저장 {fmtTime(d.created_at)}</p>
      </div>
      <div className="row" style={{ flexWrap: 'wrap' }}>
        <span className={`badge ${VIS[d.visibility]?.cls ?? 'badge-gray'}`}>{VIS[d.visibility]?.n ?? d.visibility}</span>
        <span className="badge badge-gray">팀 {d.team ?? '없음'}</span>
        <span className="badge badge-gray">담당 {d.owner_emp_id ?? '없음'}</span>
        <span className="badge badge-gray">사람 {d.people}명</span>
      </div>
      <MetaEditor key={d.doc_id} d={d} onSaved={() => { reload(); onChanged(); }} />
      <div>
        <h3 style={{ marginBottom: 6 }}>가명처리된 글</h3>
        <div style={{ maxHeight: 340, overflow: 'auto', border: '1px solid var(--line)', borderRadius: 8, padding: 10, fontSize: 12, lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>
          {d.chunks.map((c) => <div key={c.idx} style={{ marginBottom: 8 }}><MaskedText text={c.masked_text} /></div>)}
          {d.chunks.length === 0 && <span className="muted">청크가 없어요.</span>}
        </div>
      </div>
      {!confirm ? (
        <button className="btn btn-danger" onClick={() => setConfirm(true)}><Trash2 size={14} aria-hidden /> 문서 지우기</button>
      ) : (
        <div className="stack" style={{ border: '1px solid #f09595', borderRadius: 8, padding: 12, background: '#fffafa' }}>
          <p className="small" style={{ color: 'var(--bad)' }}>
            검색에서 바로 빠지고 되돌릴 수 없어요. 이 문서의 청크·사람 언급·검수 원문을 지우고, 다른 문서에 안 나오는 토큰·사람도 지워요. 지운 기록은 남아요.
          </p>
          <textarea className="textarea" placeholder="지우는 사유 (예: 잘못 올린 파일, 본인 삭제 요청)" value={reason} onChange={(e) => setReason(e.target.value)} maxLength={500} />
          <input className="input" placeholder={`확인: ${docId} 입력`} value={typed} onChange={(e) => setTyped(e.target.value)} />
          <div className="row">
            <button className="btn" onClick={() => setConfirm(false)}>취소</button>
            <button className="btn btn-danger" style={{ flex: 1 }} disabled={busy} onClick={del}>{busy ? '지우는 중…' : '영구 삭제'}</button>
          </div>
        </div>
      )}
      {error && <div className="alert alert-bad">{error}</div>}
    </div>
  );
}

export function EmployeesTab({ users }: { users: AdminUser[] }) {
  const [q, setQ] = useState('');
  const [list, setList] = useState<Employee[]>([]);
  const [error, setError] = useState('');
  useEffect(() => {
    const t = setTimeout(() => {
      api.employees(q.trim() || undefined).then((r) => { setList(r); setError(''); }).catch((e) => setError(errMsg(e)));
    }, 250);
    return () => clearTimeout(t);
  }, [q]);
  const account = (empId: string) => users.find((u) => u.empId === empId);
  return (
    <>
      <div className="row" style={{ marginBottom: 10 }}>
        <label className="search" style={{ width: 260, margin: 0 }}>
          <Search size={14} aria-hidden /><input placeholder="이름·팀·직원 번호" value={q} onChange={(e) => setQ(e.target.value)} />
        </label>
        <div className="spacer" />
        <span className="small muted">문서에서 추론한 직원 + 관리자가 만든 직원(U…) · {list.length}명</span>
      </div>
      {error && <div className="alert alert-bad" style={{ marginBottom: 10 }}>{error}</div>}
      <table className="table">
        <thead><tr><th>직원 번호</th><th>이름</th><th>팀</th><th>직급</th><th>추론 역할</th><th>문서</th><th>연결된 계정</th></tr></thead>
        <tbody>
          {list.map((e) => {
            const a = account(e.empId);
            return (
              <tr key={e.empId}>
                <td>{e.empId}</td><td style={{ fontWeight: 500 }}>{e.name}</td><td>{e.team ?? '-'}</td><td>{e.position ?? '-'}</td>
                <td>{ROLE[e.role] ?? e.role}</td><td>{e.nDocs}</td>
                <td className="small">{a ? <span className="badge badge-brand">{a.email}</span> : <span className="muted">-</span>}</td>
              </tr>
            );
          })}
          {list.length === 0 && <tr><td colSpan={7} className="muted" style={{ textAlign: 'center', padding: 30 }}>직원이 없어요.</td></tr>}
        </tbody>
      </table>
    </>
  );
}

// ---------------------------------------------------------------- 공개 범위·팀·담당자 바꾸기
const VIS_HELP: Record<string, string> = {
  public: '모든 직원이 봐요. 실명은 팀장·감사만 (담당자는 본인 문서).',
  team: '그 팀 팀장은 전부, 직원은 담당자가 있을 때 봐요. 담당자는 언제나 봐요.',
  restricted: '감사와 담당자만 봐요.',
};

function MetaEditor({ d, onSaved }: { d: DocDetail; onSaved: () => void }) {
  const [vis, setVis] = useState(d.visibility);
  const [team, setTeam] = useState(d.team ?? '');
  const [owner, setOwner] = useState(d.owner_emp_id ?? '');
  const [teams, setTeams] = useState<Team[]>([]);
  const [found, setFound] = useState<Employee[]>([]);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => { api.teams().then(setTeams).catch(() => {}); }, []);

  const changed = vis !== d.visibility || team !== (d.team ?? '') || owner !== (d.owner_emp_id ?? '');
  const search = async (q: string) => {
    setOwner(q);
    if (q.trim().length < 1) return setFound([]);
    try { setFound((await api.employees(q.trim())).slice(0, 6)); } catch { setFound([]); }
  };
  const save = async () => {
    if (vis === 'team' && !team) return setMsg({ ok: false, text: '팀 공개로 하려면 팀을 골라 주세요.' });
    setBusy(true);
    setMsg(null);
    try {
      await apiClient.post(`/api/admin/docs/${encodeURIComponent(d.doc_id)}/meta`, { visibility: vis, team: team || null, ownerEmpId: owner.trim() || null });
      setMsg({ ok: true, text: '바꿨어요. 검색에도 바로 반영돼요.' });
      onSaved();
    } catch (e) {
      setMsg({ ok: false, text: errMsg(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="stack" style={{ border: '1px solid var(--line)', borderRadius: 8, padding: 12, gap: 8 }}>
      <h3>누가 볼 수 있나</h3>
      <select className="select" value={vis} onChange={(e) => setVis(e.target.value)}>
        <option value="public">공개</option><option value="team">팀</option><option value="restricted">제한 (감사만)</option>
      </select>
      <p className="small muted">{VIS_HELP[vis]}</p>
      <label className="field" style={{ margin: 0 }}><span>팀</span>
        <select className="select" value={team} onChange={(e) => setTeam(e.target.value)}>
          <option value="">팀 없음</option>
          {teams.filter((t) => t.active || t.name === team).map((t) => <option key={t.id} value={t.name}>{t.name}</option>)}
        </select>
      </label>
      <label className="field" style={{ margin: 0, position: 'relative' }}><span>담당 직원</span>
        <input className="input" placeholder="이름·팀·직원 번호로 찾기 (비우면 없음)" value={owner} onChange={(e) => search(e.target.value)} />
        {found.length > 0 && (
          <div className="stack" style={{ gap: 0, border: '1px solid var(--line)', borderRadius: 8, marginTop: 4, background: '#fff' }}>
            {found.map((e) => (
              <button key={e.empId} type="button" className="btn btn-ghost btn-sm" style={{ justifyContent: 'flex-start' }}
                      onClick={() => { setOwner(e.empId); setFound([]); if (!team && e.team) setTeam(e.team); }}>
                {e.empId} · {e.name} <span className="muted">{e.team ?? ''}</span>
              </button>
            ))}
          </div>
        )}
      </label>
      <button className="btn btn-primary btn-sm" disabled={!changed || busy} onClick={save}>{busy ? '저장 중…' : '저장'}</button>
      {msg && <div className={`alert ${msg.ok ? 'alert-ok' : 'alert-bad'} small`}>{msg.text}</div>}
    </div>
  );
}
