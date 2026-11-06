// 마이페이지: 내 정보(DB 값), 비밀번호 바꾸기, 내 실명 보기 기록
import { useEffect, useState } from 'react';
import type { FormEvent } from 'react';
import apiClient from '@/shared/api/client';
import * as authApi from '@/features/auth/api/auth.api';
import type { UserResponse } from '@/features/auth/types/auth.types';
import { errMsg, fmtTime } from '@/shared/utils/error';

const ROLE: Record<string, { n: string; d: string }> = {
  STAFF: { n: '직원', d: '공개 문서, 같은 팀의 담당자 있는 문서, 내가 담당한 문서를 볼 수 있어요. 실명은 내가 담당한 문서만 볼 수 있어요.' },
  LEAD: { n: '팀장', d: '공개 문서와 우리 팀 문서 전부를 볼 수 있고, 그 문서들의 실명도 볼 수 있어요.' },
  AUDITOR: { n: '감사', d: '모든 문서를 볼 수 있어요. 실명을 볼 때는 사유를 남겨야 해요.' },
};
interface RestoreRow { id: number; doc_ids: string[]; restored: number; denied: number; purpose: string | null; created_at: string }

export default function MePage() {
  const [me, setMe] = useState<UserResponse | null>(null);
  const [log, setLog] = useState<RestoreRow[]>([]);
  const [pw, setPw] = useState({ cur: '', next: '', next2: '' });
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    authApi.getMe().then(setMe).catch((e) => setMsg({ ok: false, text: errMsg(e) }));
    apiClient.get<RestoreRow[]>('/api/me/restore-log').then((r) => setLog(r.data)).catch(() => {});
  }, []);

  const change = async (e: FormEvent) => {
    e.preventDefault();
    if (pw.next.length < 8) return setMsg({ ok: false, text: '새 비밀번호는 8자 이상이에요.' });
    if (pw.next !== pw.next2) return setMsg({ ok: false, text: '새 비밀번호가 서로 달라요.' });
    setBusy(true);
    try {
      await apiClient.put('/api/me/password', { currentPassword: pw.cur, newPassword: pw.next });
      setMsg({ ok: true, text: '비밀번호를 바꿨어요.' });
      setPw({ cur: '', next: '', next2: '' });
    } catch (err) {
      setMsg({ ok: false, text: errMsg(err) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="panel page">
      <div className="page-head"><div><h1>마이페이지</h1><p>내 정보와 권한, 비밀번호</p></div></div>
      <div className="grid2" style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) minmax(0,1fr)', gap: 12 }}>
        <div className="card stack">
          <div className="row">
            <div className="avatar" style={{ width: 44, height: 44, fontSize: 18 }}>{me?.name.slice(0, 1)}</div>
            <div><h2>{me?.name}</h2><p className="small muted">{me?.email}</p></div>
          </div>
          <table className="table">
            <tbody>
              <tr><td className="muted">팀</td><td>{me?.team ?? '-'}</td></tr>
              <tr><td className="muted">직급</td><td>{me?.position ?? '-'}</td></tr>
              <tr><td className="muted">역할</td><td>{me && <span className="badge badge-brand">{ROLE[me.role]?.n}</span>}{me?.admin && <span className="badge badge-gray" style={{ marginLeft: 6 }}>관리자</span>}</td></tr>
            </tbody>
          </table>
          {me && <div className="alert alert-info small">{ROLE[me.role]?.d}</div>}
          <p className="small muted">팀·직급·역할은 관리자만 바꿀 수 있어요.</p>
        </div>
        <form className="card" onSubmit={change}>
          <h2 style={{ marginBottom: 14 }}>비밀번호 바꾸기</h2>
          <label className="field"><span>지금 비밀번호</span><input className="input" type="password" autoComplete="current-password" value={pw.cur} onChange={(e) => setPw({ ...pw, cur: e.target.value })} /></label>
          <label className="field"><span>새 비밀번호</span><input className="input" type="password" autoComplete="new-password" placeholder="8자 이상" value={pw.next} onChange={(e) => setPw({ ...pw, next: e.target.value })} /></label>
          <label className="field"><span>새 비밀번호 확인</span><input className="input" type="password" autoComplete="new-password" value={pw.next2} onChange={(e) => setPw({ ...pw, next2: e.target.value })} /></label>
          <button className="btn btn-primary" disabled={busy || !pw.cur}>바꾸기</button>
          {msg && <div className={`alert ${msg.ok ? 'alert-ok' : 'alert-bad'}`} style={{ marginTop: 10 }}>{msg.text}</div>}
        </form>
      </div>
      <div className="card" style={{ marginTop: 12 }}>
        <h2 style={{ marginBottom: 10 }}>내 실명 보기 기록</h2>
        <table className="table">
          <thead><tr><th>시각</th><th>문서</th><th>복원</th><th>권한 없어 가림</th><th>사유</th></tr></thead>
          <tbody>
            {log.map((r) => <tr key={r.id}><td className="small muted nowrap">{fmtTime(r.created_at)}</td><td className="small">{r.doc_ids.join(', ') || '-'}</td><td>{r.restored}</td><td>{r.denied}</td><td className="small">{r.purpose ?? '-'}</td></tr>)}
            {log.length === 0 && <tr><td colSpan={5} className="muted" style={{ textAlign: 'center', padding: 24 }}>아직 실명 보기를 한 적이 없어요.</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
  );
}
