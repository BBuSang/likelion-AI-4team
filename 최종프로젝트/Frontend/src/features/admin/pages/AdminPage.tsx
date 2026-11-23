// 관리자: 탭(승인 대기 / 사용자 / 팀 / 변경 기록) + 표 + 오른쪽 상세 (worklog 12장 API)
import { useCallback, useEffect, useState } from 'react';
import { RefreshCw, Search, UserPlus } from 'lucide-react';
import { errMsg, fmtTime } from '@/shared/utils/error';
import type { Role, UserStatus } from '@/shared/types/user.types';
import * as api from '../api/admin.api';
import type { AdminUser, AuditLog, Candidate, Employee, Position, Team } from '../api/admin.api';
import { DocDetailPanel, DocsTab, EmployeesTab } from './DataTabs';
import { PublicDocHint, PublicDocPanel, PublicDocsTab } from './PublicDocs';
import type { PublicSource } from './PublicDocs';
import UserChats from './UserChats';
import MaskedText from '@/features/chat/components/MaskedText';
import ConfirmDialog from '@/shared/components/ConfirmDialog';

const ROLES: { v: Role; n: string }[] = [{ v: 'STAFF', n: '직원' }, { v: 'LEAD', n: '팀장' }, { v: 'AUDITOR', n: '감사' }];
const roleName = (r: Role) => ROLES.find((x) => x.v === r)?.n ?? r;
const STATUS: Record<UserStatus, { n: string; cls: string }> = {
  PENDING: { n: '승인 대기', cls: 'badge-warn' }, ACTIVE: { n: '사용 중', cls: 'badge-brand' }, SUSPENDED: { n: '정지', cls: 'badge-bad' },
};
const ACTION: Record<string, string> = {
  approve: '승인', update: '정보 변경', delete_doc: '문서 삭제', update_doc: '문서 공개 범위', create_team: '팀 추가', update_team: '팀 변경', sync_teams: '팀 가져오기', bootstrap: '첫 관리자', view_chat: '채팅 열람',
};

type Tab = 'pending' | 'users' | 'docs' | 'emps' | 'teams' | 'log';
const DOC_SOURCES: ['pii' | PublicSource, string][] = [['pii', '개인정보 문서'], ['product', '금융상품'], ['law', '법령'], ['noaction', '금융당국 회신']];

export default function AdminPage() {
  const [tab, setTab] = useState<Tab>('pending');
  const [list, setList] = useState<AdminUser[]>([]);
  const [teams, setTeams] = useState<Team[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [selected, setSelected] = useState<AdminUser | null>(null);
  const [q, setQ] = useState('');
  const [error, setError] = useState('');
  const [docId, setDocId] = useState<string | null>(null);
  const [docsKey, setDocsKey] = useState(0);
  // 문서 탭 안의 출처: 개인정보 문서(사내) · 금융상품 · 법령 · 금융당국 회신(공개 자료, 읽기만)
  const [docSrc, setDocSrc] = useState<'pii' | PublicSource>('pii');
  const [pubId, setPubId] = useState<string | null>(null);

  const reload = useCallback(async () => {
    setError('');
    try {
      const [u, t] = await Promise.all([api.users(), api.teams()]);
      setList(u);
      setTeams(t);
      setSelected((s) => (s ? u.find((x) => x.id === s.id) ?? null : null));
    } catch (e) {
      setError(errMsg(e));
    }
  }, []);

  useEffect(() => { reload(); api.positions().then(setPositions).catch(() => {}); }, [reload]);

  const pending = list.filter((u) => u.status === 'PENDING');
  const rows = (tab === 'pending' ? pending : list).filter((u) => !q || `${u.name}${u.email}${u.team ?? ''}`.includes(q.trim()));

  return (
    <>
      <section className="panel page">
        <div className="page-head">
          <div>
            <h1>관리자</h1>
            <p>가입 승인 · 사용자 정보 · 저장된 문서 · 직원 목록 · 팀 · 변경 기록</p>
          </div>
          <div className="spacer" />
          <button className="btn btn-sm" onClick={reload}><RefreshCw size={14} aria-hidden /> 새로고침</button>
        </div>
        <div className="tabs">
          <button className={`tab${tab === 'pending' ? ' active' : ''}`} onClick={() => setTab('pending')}>승인 대기 {pending.length > 0 && <span className="badge badge-warn">{pending.length}</span>}</button>
          <button className={`tab${tab === 'users' ? ' active' : ''}`} onClick={() => setTab('users')}>사용자</button>
          <button className={`tab${tab === 'docs' ? ' active' : ''}`} onClick={() => setTab('docs')}>문서</button>
          <button className={`tab${tab === 'emps' ? ' active' : ''}`} onClick={() => setTab('emps')}>직원</button>
          <button className={`tab${tab === 'teams' ? ' active' : ''}`} onClick={() => setTab('teams')}>팀</button>
          <button className={`tab${tab === 'log' ? ' active' : ''}`} onClick={() => setTab('log')}>변경 기록</button>
        </div>
        {error && <div className="alert alert-bad" style={{ marginBottom: 12 }}>{error}</div>}

        {(tab === 'pending' || tab === 'users') && (
          <>
            <label className="search" style={{ maxWidth: 280, marginTop: 0 }}>
              <Search size={14} aria-hidden /><input placeholder="이름·이메일·팀" value={q} onChange={(e) => setQ(e.target.value)} />
            </label>
            <table className="table" style={{ marginTop: 10 }}>
              <thead><tr><th>이름</th><th>팀</th><th>직급</th><th>역할</th><th>직원 번호</th><th>상태</th><th>가입</th></tr></thead>
              <tbody>
                {rows.map((u) => (
                  <tr key={u.id} className={`clickable${selected?.id === u.id ? ' selected' : ''}`} onClick={() => setSelected(u)}>
                    <td><p style={{ fontWeight: 500 }}>{u.name}{u.admin && <span className="badge badge-gray" style={{ marginLeft: 6 }}>관리자</span>}</p><p className="small muted">{u.email}</p></td>
                    <td>{u.team ?? '-'}</td>
                    <td>{u.position ?? '-'}</td>
                    <td>{roleName(u.role)}</td>
                    <td>{u.empId ?? <span className="muted">-</span>}</td>
                    <td><span className={`badge ${STATUS[u.status].cls}`}>{STATUS[u.status].n}</span></td>
                    <td className="small muted nowrap">{fmtTime(u.createdAt)}</td>
                  </tr>
                ))}
                {rows.length === 0 && <tr><td colSpan={7} className="muted" style={{ textAlign: 'center', padding: 30 }}>{tab === 'pending' ? '승인을 기다리는 가입이 없어요.' : '사용자가 없어요.'}</td></tr>}
              </tbody>
            </table>
          </>
        )}
        {tab === 'docs' && (
          <>
            <div className="subtabs" role="tablist" aria-label="문서 출처">
              {DOC_SOURCES.map(([k, n]) => (
                <button key={k} role="tab" aria-selected={docSrc === k} className={`subtab${docSrc === k ? ' on' : ''}`}
                        onClick={() => { setDocSrc(k); setPubId(null); }}>{n}</button>
              ))}
            </div>
            {docSrc === 'pii' ? <DocsTab selected={docId} onSelect={setDocId} reloadKey={docsKey} />
                              : <PublicDocsTab source={docSrc} selected={pubId} onSelect={setPubId} />}
          </>
        )}
        {tab === 'emps' && <EmployeesTab users={list} />}
        {tab === 'teams' && <TeamsTab teams={teams} onChange={reload} />}
        {tab === 'log' && <LogTab users={list} />}
      </section>

      {tab === 'docs' && docSrc !== 'pii' && (
        <aside className="panel side side-wide">
          {pubId ? <PublicDocPanel key={`${docSrc}:${pubId}`} source={docSrc} id={pubId} /> : <PublicDocHint source={docSrc} />}
        </aside>
      )}
      {tab === 'docs' && docSrc === 'pii' && (
        <aside className="panel side side-wide">
          {docId ? <DocDetailPanel key={docId} docId={docId} onDeleted={() => { setDocId(null); setDocsKey((k) => k + 1); }} onChanged={() => setDocsKey((k) => k + 1)} />
                 : <p className="small muted">표에서 문서를 고르면 가명처리된 글과 지우기 버튼이 보여요.</p>}
        </aside>
      )}
      {(tab === 'pending' || tab === 'users') && (
        <aside className="panel side side-wide">
          {!selected && <p className="small muted">표에서 사람을 고르면 여기에 자세한 정보가 보여요.</p>}
          {selected?.status === 'PENDING' && <ApprovePanel key={selected.id} u={selected} teams={teams} positions={positions} onDone={reload} />}
          {selected && selected.status !== 'PENDING' && <EditPanel key={selected.id} u={selected} users={list} teams={teams} positions={positions} onDone={reload} />}
        </aside>
      )}
    </>
  );
}

// ---------------------------------------------------------------- 승인

function ApprovePanel({ u, teams, positions, onDone }: { u: AdminUser; teams: Team[]; positions: Position[]; onDone: () => void }) {
  const [cands, setCands] = useState<Candidate[] | null>(null);
  const [defRole, setDefRole] = useState<Role | null>(null);
  const [pick, setPick] = useState<string>('');           // empId 또는 'new'
  const [teamId, setTeamId] = useState(String(u.teamId ?? ''));
  const [positionId, setPositionId] = useState(String(u.positionId ?? ''));
  const [role, setRole] = useState<Role | ''>('');
  const [search, setSearch] = useState('');
  const [found, setFound] = useState<Employee[]>([]);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => {
    api.candidates(u.id).then((r) => {
      setCands(r.candidates);
      setDefRole(r.positionDefaultRole);
      const first = r.candidates.find((c) => !c.linkedTo);
      setPick(first ? first.employee.empId : 'new');
    }).catch((e) => setMsg({ ok: false, text: errMsg(e) }));
  }, [u.id]);

  const all = [...(cands ?? []).map((c) => ({ e: c.employee, linkedTo: c.linkedTo })),
               ...found.filter((f) => !(cands ?? []).some((c) => c.employee.empId === f.empId)).map((e) => ({ e, linkedTo: null }))];
  const picked = all.find((x) => x.e.empId === pick)?.e;
  const autoRole = picked?.role ?? defRole ?? 'STAFF';

  const doSearch = async () => {
    if (!search.trim()) return;
    try { setFound(await api.employees(search.trim())); } catch (e) { setMsg({ ok: false, text: errMsg(e) }); }
  };

  const approve = async () => {
    if (!pick) return setMsg({ ok: false, text: '연결할 직원을 고르거나 새 직원 번호를 선택해 주세요.' });
    setBusy(true);
    setMsg(null);
    try {
      await api.approve(u.id, {
        ...(pick === 'new' ? { createEmployee: true } : { empId: pick }),
        teamId: teamId ? Number(teamId) : undefined, positionId: positionId ? Number(positionId) : undefined,
        role: role || undefined,
      });
      onDone();
    } catch (e) {
      setMsg({ ok: false, text: errMsg(e) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="stack">
      <div>
        <h2>{u.name}</h2>
        <p className="small muted">{u.email} · 가입 {fmtTime(u.createdAt)}</p>
      </div>
      <div className="alert alert-warn small">가입 정보는 본인이 적은 값이에요. 문서 속 직원과 맞는지 확인하고 승인해 주세요.</div>
      <div className="row">
        <label className="field" style={{ flex: 1, margin: 0 }}><span>팀</span>
          <select className="select" value={teamId} onChange={(e) => setTeamId(e.target.value)}>
            {teams.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
          </select>
        </label>
        <label className="field" style={{ flex: 1, margin: 0 }}><span>직급</span>
          <select className="select" value={positionId} onChange={(e) => setPositionId(e.target.value)}>
            {positions.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
          </select>
        </label>
      </div>

      <div>
        <h3 style={{ marginBottom: 6 }}>문서 속 직원 연결</h3>
        {cands === null && <p className="small muted">후보를 찾는 중…</p>}
        {cands?.length === 0 && <p className="small muted">이름·팀이 같은 직원이 없어요. 아래에서 찾거나 새 직원 번호를 만들어 주세요.</p>}
        <div className="stack" style={{ gap: 6 }}>
          {all.map(({ e, linkedTo }) => (
            <label key={e.empId} className={`pick${pick === e.empId ? ' on' : ''}${linkedTo ? ' off' : ''}`}>
              <input type="radio" name="emp" disabled={!!linkedTo} checked={pick === e.empId} onChange={() => setPick(e.empId)} />
              <div style={{ flex: 1 }}>
                <p style={{ fontWeight: 500 }}>{e.name} <span className="muted small">{e.empId}</span></p>
                <p className="small muted">{e.team ?? '팀 모름'} · {e.position ?? '직급 모름'} · 추론 역할 {roleName(e.role)} · 문서 {e.nDocs}건</p>
                {linkedTo && <p className="small" style={{ color: 'var(--bad)' }}>이미 다른 계정(id {linkedTo})에 연결됨</p>}
              </div>
              {e.sameTeam === true && <span className="badge badge-ok">같은 팀</span>}
              {e.sameTeam === false && <span className="badge badge-warn">다른 팀</span>}
            </label>
          ))}
          <label className={`pick${pick === 'new' ? ' on' : ''}`}>
            <input type="radio" name="emp" checked={pick === 'new'} onChange={() => setPick('new')} />
            <div><p style={{ fontWeight: 500 }}><UserPlus size={14} aria-hidden /> 새 직원 번호 만들기</p><p className="small muted">문서에 나오지 않는 사람 (안 A)</p></div>
          </label>
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          <input className="input" style={{ height: 32 }} placeholder="이름 · 팀 · 직원 번호로 찾기" value={search}
                 onChange={(e) => setSearch(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && doSearch()} />
          <button className="btn btn-sm" onClick={doSearch}>찾기</button>
        </div>
      </div>

      <label className="field" style={{ margin: 0 }}><span>역할</span>
        <select className="select" value={role} onChange={(e) => setRole(e.target.value as Role | '')}>
          <option value="">기본값 ({roleName(autoRole)}{picked ? ' · 직원 추론' : ' · 직급 기본'})</option>
          {ROLES.map((r) => <option key={r.v} value={r.v}>{r.n}</option>)}
        </select>
      </label>
      {role === 'AUDITOR' && <div className="alert alert-warn small">감사 역할은 모든 문서를 보고 실명으로 볼 수 있어요.</div>}
      <button className="btn btn-primary" disabled={busy} onClick={approve}>{busy ? '승인 중…' : '승인'}</button>
      {msg && <div className={`alert ${msg.ok ? 'alert-ok' : 'alert-bad'}`}>{msg.text}</div>}
    </div>
  );
}

// ---------------------------------------------------------------- 사용자 정보 바꾸기

function EditPanel({ u, users, teams, positions, onDone }: { u: AdminUser; users: AdminUser[]; teams: Team[]; positions: Position[]; onDone: () => void }) {
  const who = (id: number | null) => (id === null ? '-' : id === 0 ? '서버' : users.find((x) => x.id === id)?.name ?? `id ${id}`);
  const [f, setF] = useState({ teamId: String(u.teamId ?? ''), positionId: String(u.positionId ?? ''), role: u.role, admin: u.admin });
  const [empId, setEmpId] = useState('');
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [log, setLog] = useState<AuditLog[]>([]);
  // 되돌리기 어려운 동작은 확인 대화상자를 거침 (정지·연결 풀기·연결 바꾸기)
  const [ask, setAsk] = useState<'suspend' | 'unlink' | 'relink' | null>(null);

  const loadLog = useCallback(() => { api.auditLog(u.id).then(setLog).catch(() => {}); }, [u.id]);
  useEffect(loadLog, [loadLog]);

  const run = async (body: api.UpdateBody, ok: string) => {
    setBusy(true);
    setMsg(null);
    try {
      await api.update(u.id, body);
      setMsg({ ok: true, text: ok });
      setEmpId('');
      onDone();
    } catch (e) {
      setMsg({ ok: false, text: errMsg(e) });
    } finally {
      setBusy(false);
    }
  };

  const save = () => run({
    teamId: f.teamId ? Number(f.teamId) : undefined, positionId: f.positionId ? Number(f.positionId) : undefined,
    role: f.role, admin: f.admin,
  }, '저장했어요.');

  return (
    <div className="stack">
      <div className="row">
        <div className="avatar">{u.name.slice(0, 1)}</div>
        <div><h2>{u.name}</h2><p className="small muted">{u.email}</p></div>
        <div className="spacer" />
        <span className={`badge ${STATUS[u.status].cls}`}>{STATUS[u.status].n}</span>
      </div>
      <p className="small muted">직원 번호 {u.empId ?? '없음'} · 승인 {fmtTime(u.approvedAt)}</p>
      <label className="field" style={{ margin: 0 }}><span>팀</span>
        <select className="select" value={f.teamId} onChange={(e) => setF({ ...f, teamId: e.target.value })}>
          <option value="">팀 없음</option>
          {teams.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
        </select>
      </label>
      <label className="field" style={{ margin: 0 }}><span>직급</span>
        <select className="select" value={f.positionId} onChange={(e) => setF({ ...f, positionId: e.target.value })}>
          <option value="">직급 없음</option>
          {positions.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
        </select>
      </label>
      <label className="field" style={{ margin: 0 }}><span>역할 (데이터 접근)</span>
        <select className="select" value={f.role} onChange={(e) => setF({ ...f, role: e.target.value as Role })}>
          {ROLES.map((r) => <option key={r.v} value={r.v}>{r.n}</option>)}
        </select>
      </label>
      <label className="row small"><input type="checkbox" checked={f.admin} onChange={(e) => setF({ ...f, admin: e.target.checked })} /> 관리자 페이지 권한</label>
      <button className="btn btn-primary" disabled={busy} onClick={save}>저장</button>

      <div style={{ borderTop: '1px solid var(--line)', paddingTop: 12 }} className="stack">
        <h3>직원 연결</h3>
        <div className="row">
          <input className="input" style={{ height: 32 }} placeholder="새 직원 번호 (예: E012)" value={empId} onChange={(e) => setEmpId(e.target.value)} />
          <button className="btn btn-sm" disabled={!empId.trim() || busy} onClick={() => setAsk('relink')}>바꾸기</button>
        </div>
        {u.empId && <button className="btn btn-sm btn-ghost" style={{ alignSelf: 'flex-start' }} disabled={busy} onClick={() => setAsk('unlink')}>연결 풀기</button>}
      </div>

      <div className="row" style={{ borderTop: '1px solid var(--line)', paddingTop: 12 }}>
        {u.status === 'ACTIVE'
          ? <button className="btn btn-danger btn-sm" disabled={busy} onClick={() => setAsk('suspend')}>계정 정지</button>
          : <button className="btn btn-sm" disabled={busy} onClick={() => run({ status: 'ACTIVE' }, '정지를 풀었어요.')}>정지 풀기</button>}
      </div>
      {msg && <div className={`alert ${msg.ok ? 'alert-ok' : 'alert-bad'}`}>{msg.text}</div>}

      {ask === 'suspend' && (
        <ConfirmDialog title="계정 정지" confirmLabel="정지" danger busy={busy}
                       message={<><b>{u.name}</b>({u.email}) 계정을 정지할까요? 로그인한 화면도 바로 막히고, 정지를 풀기 전까지 다시 로그인할 수 없어요.</>}
                       reason={{ label: '정지 사유 (변경 기록에 남아요)', placeholder: '예: 퇴사 처리, 비정상 접근 확인 중', max: 500 }}
                       onCancel={() => setAsk(null)}
                       onConfirm={(reason) => { setAsk(null); run({ status: 'SUSPENDED', reason }, '정지했어요. 이미 로그인한 화면도 바로 막혀요.'); }} />
      )}
      {ask === 'unlink' && (
        <ConfirmDialog title="직원 연결 풀기" confirmLabel="연결 풀기" danger busy={busy}
                       message={<><b>{u.name}</b> 계정과 문서 속 직원 <b>{u.empId}</b>의 연결을 풀까요? 이 계정은 더 이상 그 직원으로 인정되지 않아요. 다시 연결하려면 직원 번호를 넣고 바꾸기를 해야 해요.</>}
                       onCancel={() => setAsk(null)}
                       onConfirm={() => { setAsk(null); run({ unlinkEmployee: true }, '직원 연결을 풀었어요.'); }} />
      )}
      {ask === 'relink' && (
        <ConfirmDialog title="직원 연결 바꾸기" confirmLabel="바꾸기" busy={busy}
                       message={<><b>{u.name}</b> 계정의 직원 연결을 <b>{u.empId ?? '없음'}</b>에서 <b>{empId.trim()}</b>(으)로 바꿀까요? 이 계정은 앞으로 문서 속 직원 {empId.trim()}(으)로 인정돼요.</>}
                       onCancel={() => setAsk(null)}
                       onConfirm={() => { setAsk(null); run({ empId: empId.trim() }, '직원 연결을 바꿨어요.'); }} />
      )}

      <div style={{ borderTop: '1px solid var(--line)', paddingTop: 12 }}>
        <UserChats userId={u.id} name={u.name} onViewed={loadLog} />
      </div>

      <div>
        <h3 style={{ marginBottom: 6 }}>이 계정의 변경 기록</h3>
        {log.length === 0 && <p className="small muted">기록이 없어요.</p>}
        {log.slice(0, 8).map((l) => (
          <div key={l.id} className="small" style={{ padding: '6px 0', borderBottom: '1px solid #eef1f6' }}>
            <span className="badge badge-gray">{ACTION[l.action] ?? l.action}</span> <span className="muted">{fmtTime(l.createdAt)}</span>
            <p className="muted" style={{ wordBreak: 'break-all', marginTop: 2 }}><LogDetail l={l} who={who} /></p>
          </div>
        ))}
      </div>
    </div>
  );
}

/** 기록 한 줄의 내용. 채팅 열람은 "누가 누구의 어느 대화를 열람"으로 (worklog 61장) */
function LogDetail({ l, who }: { l: AuditLog; who: (id: number | null) => string }) {
  if (l.action === 'view_chat') {
    let id: unknown = null;
    let title: unknown = null;
    try {
      const c = JSON.parse(l.changes) as Record<string, [unknown, unknown]>;
      id = c.conversation?.[1];
      title = c.title?.[1];
    } catch { /* 옛 형식 */ }
    return (
      <>
        <b>{who(l.adminId)}</b> 님이 <b>{who(l.targetUserId)}</b> 님의{' '}
        {typeof title === 'string' && title ? <>「<MaskedText text={title} />」</> : id != null ? `#${id}` : ''} 대화를 열람했어요
      </>
    );
  }
  return <>{describe(l.action, l.changes)}</>;
}

// 변경 기록 한 줄을 읽기 쉬운 말로 (worklog: 변경 기록 가독성)
//   status ACTIVE→SUSPENDED → "계정 정지", SUSPENDED→ACTIVE → "정지 해제", 역할·팀·직급은 "역할: 직원 → 팀장" …
//   모르는 항목은 예전처럼 "항목: 전 → 후" 로
const VIS_KO: Record<string, string> = { public: '공개', team: '팀', restricted: '제한' };
const v = (x: unknown) => (x === null || x === undefined || x === '' ? '없음' : Array.isArray(x) ? x.join(', ') : String(x));
const statusName = (x: unknown) => STATUS[x as UserStatus]?.n ?? v(x);
const roleKo = (x: unknown) => (x ? roleName(x as Role) : '없음');

function describePart(action: string, k: string, a: unknown, b: unknown): string | null {
  switch (k) {
    case 'status':
      if (b === 'SUSPENDED') return '계정 정지';
      if (a === 'SUSPENDED' && b === 'ACTIVE') return '정지 해제';
      if (a === 'PENDING' && b === 'ACTIVE') return '가입 승인';
      return `계정 상태: ${statusName(a)} → ${statusName(b)}`;
    case 'reason': return `사유: ${v(b)}`;
    case 'role': return `역할: ${roleKo(a)} → ${roleKo(b)}`;
    case 'admin': return b ? '관리자 권한 부여' : '관리자 권한 해제';
    case 'empId':
      if (a == null) return `직원 연결: ${v(b)}`;
      if (b == null) return `직원 연결 해제 (${v(a)})`;
      return `직원 연결 변경: ${v(a)} → ${v(b)}`;
    case 'createdEmployee': return `새 직원 번호 만듦: ${v(b)}`;
    case 'position': return `직급: ${v(a)} → ${v(b)}`;
    case 'team':
      if (action === 'create_team') return `팀 추가: ${v(b)}`;
      if (action === 'update_team') return `${v(b)} 팀`;
      if (action === 'update_doc') return `문서 팀: ${v(a)} → ${v(b)}`;
      return `팀: ${v(a)} → ${v(b)}`;
    case 'active': return b ? '가입 선택칸에 다시 보이기' : '가입 선택칸에서 숨기기';
    case 'added': return `문서에서 가져온 팀: ${v(b)}`;
    case 'doc': return action === 'delete_doc' ? `문서 ${v(a)} 삭제` : `문서 ${v(a)}`;
    case 'chunks': return `청크 ${v(a)}개 지움`;
    case 'visibility': return `공개 범위: ${VIS_KO[String(a)] ?? v(a)} → ${VIS_KO[String(b)] ?? v(b)}`;
    case 'owner_emp_id': return `담당 직원: ${v(a)} → ${v(b)}`;
    default: return `${k}: ${v(a)} → ${v(b)}`;
  }
}

function describe(action: string, changes: string) {
  try {
    const c = JSON.parse(changes) as Record<string, [unknown, unknown]>;
    return Object.entries(c).map(([k, [a, b]]) => describePart(action, k, a, b)).filter(Boolean).join(' · ');
  } catch {
    return changes;
  }
}

// ---------------------------------------------------------------- 팀

function TeamsTab({ teams, onChange }: { teams: Team[]; onChange: () => void }) {
  const [name, setName] = useState('');
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [hide, setHide] = useState<Team | null>(null); // 숨기기 확인 대화상자 (보이기는 바로)
  const act = async (fn: () => Promise<unknown>, ok: string) => {
    setMsg(null);
    try { await fn(); setMsg({ ok: true, text: ok }); onChange(); } catch (e) { setMsg({ ok: false, text: errMsg(e) }); }
  };
  return (
    <div className="stack" style={{ maxWidth: 640 }}>
      <p className="small muted">팀 이름은 문서의 팀 이름과 글자가 같아야 같은 팀 문서를 볼 수 있어요. 문서에서 추론한 팀은 "문서에서 가져오기"로 넣어요.</p>
      <div className="row">
        <input className="input" placeholder="새 팀 이름" value={name} onChange={(e) => setName(e.target.value)} />
        <button className="btn" disabled={!name.trim()} onClick={() => act(() => api.createTeam(name), '팀을 추가했어요.').then(() => setName(''))}>추가</button>
        <button className="btn btn-outline" onClick={() => act(async () => { const r = await api.syncTeams(); setMsg({ ok: true, text: `${r.added}개 가져왔어요.` }); }, '가져왔어요.')}>문서에서 가져오기</button>
      </div>
      {msg && <div className={`alert ${msg.ok ? 'alert-ok' : 'alert-bad'}`}>{msg.text}</div>}
      <table className="table">
        <thead><tr><th>팀</th><th>출처</th><th>가입 선택칸</th><th /></tr></thead>
        <tbody>
          {teams.map((t) => (
            <tr key={t.id}>
              <td>{t.name}</td>
              <td><span className="badge badge-gray">{t.source === 'ai' ? '문서 추론' : '관리자 추가'}</span></td>
              <td>{t.active ? <span className="badge badge-brand">보임</span> : <span className="badge badge-gray">숨김</span>}</td>
              <td style={{ textAlign: 'right' }}>
                <button className="btn btn-sm" onClick={() => (t.active ? setHide(t) : act(() => api.setTeamActive(t.id, true), '다시 보이게 했어요.'))}>{t.active ? '숨기기' : '보이기'}</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {hide && (
        <ConfirmDialog title="팀 숨기기" confirmLabel="숨기기" danger
                       message={<><b>{hide.name}</b> 팀을 숨길까요? 가입 화면의 팀 선택칸에서 사라져서 새로 가입하는 사람은 이 팀을 고를 수 없어요. 이미 이 팀인 사용자와 문서는 그대로예요.</>}
                       onCancel={() => setHide(null)}
                       onConfirm={() => { const t = hide; setHide(null); act(() => api.setTeamActive(t.id, false), '숨겼어요.'); }} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------- 변경 기록

function LogTab({ users }: { users: AdminUser[] }) {
  const [log, setLog] = useState<AuditLog[]>([]);
  const [error, setError] = useState('');
  useEffect(() => { api.auditLog().then(setLog).catch((e) => setError(errMsg(e))); }, []);
  const who = (id: number | null) => (id === null ? '-' : id === 0 ? '서버' : users.find((u) => u.id === id)?.name ?? `id ${id}`);
  return (
    <>
      {error && <div className="alert alert-bad">{error}</div>}
      <table className="table">
        <thead><tr><th>시각</th><th>한 사람</th><th>작업</th><th>대상</th><th>내용</th></tr></thead>
        <tbody>
          {log.map((l) => (
            <tr key={l.id}>
              <td className="small muted nowrap">{fmtTime(l.createdAt)}</td>
              <td>{who(l.adminId)}</td>
              <td><span className="badge badge-gray">{ACTION[l.action] ?? l.action}</span></td>
              <td>{who(l.targetUserId)}</td>
              <td className="small" style={{ wordBreak: 'break-all' }}><LogDetail l={l} who={who} /></td>
            </tr>
          ))}
          {log.length === 0 && <tr><td colSpan={5} className="muted" style={{ textAlign: 'center', padding: 30 }}>기록이 없어요.</td></tr>}
        </tbody>
      </table>
    </>
  );
}
