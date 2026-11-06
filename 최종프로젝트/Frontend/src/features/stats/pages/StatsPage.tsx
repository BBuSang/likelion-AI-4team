// 통계 (관리자): 문서·직원 수, 적재·검수 결과, 실명 보기 기록. 개인정보 값 없이 개수만
import { useEffect, useState } from 'react';
import apiClient from '@/shared/api/client';
import { errMsg } from '@/shared/utils/error';
import './stats.css';

interface Stats {
  documents: { total: number; by_visibility: Record<string, number>; chunks: number };
  employees: { total: number; by_role: Record<string, number> };
  ingest: { by_status: Record<string, number>; auto_approved: number; approved: number; rejected: number; added: number; removed: number; relabeled: number };
  restore: { role: string; count: number; tokens: number; denied: number }[];
  daily: { day: string; restores: number; reviews: number }[];
}
interface RestoreRow { id: number; actor: string; role: string; purpose: string | null; doc_ids: string[]; restored: number; denied: number; created_at: string }

const VIS: Record<string, string> = { public: '공개', team: '팀', restricted: '제한' };
const ROLE: Record<string, string> = { STAFF: '직원', LEAD: '팀장', AUDITOR: '감사' };

function Bars({ data }: { data: { name: string; n: number }[] }) {
  const max = Math.max(1, ...data.map((d) => d.n));
  return (
    <div className="hbars">
      {data.map((d) => (
        <div key={d.name} className="hbar">
          <span>{d.name}</span>
          <div><i style={{ width: `${(d.n / max) * 100}%` }} /></div>
          <b>{d.n.toLocaleString()}</b>
        </div>
      ))}
    </div>
  );
}

export default function StatsPage() {
  const [s, setS] = useState<Stats | null>(null);
  const [log, setLog] = useState<RestoreRow[]>([]);
  const [error, setError] = useState('');

  useEffect(() => {
    apiClient.get<Stats>('/api/admin/stats', { params: { days: 14 } }).then((r) => setS(r.data)).catch((e) => setError(errMsg(e)));
    apiClient.get<RestoreRow[]>('/api/admin/restore-log', { params: { limit: 20 } }).then((r) => setLog(r.data)).catch(() => {});
  }, []);

  if (!s) return <section className="panel page">{error ? <div className="alert alert-bad">{error}</div> : <p className="muted">불러오는 중…</p>}</section>;

  const restores = s.restore.reduce((a, r) => a + r.count, 0);
  const denied = s.restore.reduce((a, r) => a + r.denied, 0);
  const tokens = s.restore.reduce((a, r) => a + r.tokens, 0);
  const reviewed = s.ingest.approved + s.ingest.rejected;
  const maxDay = Math.max(1, ...s.daily.map((d) => d.restores + d.reviews));

  return (
    <section className="panel page">
      <div className="page-head"><div><h1>통계</h1><p>문서·검수·실명 보기 현황 (값 없이 개수만)</p></div></div>
      <div className="metrics">
        <div className="metric"><p>저장된 문서</p><b>{s.documents.total.toLocaleString()}</b><span>청크 {s.documents.chunks.toLocaleString()}</span></div>
        <div className="metric"><p>문서 속 직원</p><b>{s.employees.total}</b><span>팀장 {s.employees.by_role.LEAD ?? 0} · 감사 {s.employees.by_role.AUDITOR ?? 0}</span></div>
        <div className="metric"><p>검수한 문서</p><b>{reviewed}</b><span>자동 저장 {s.ingest.auto_approved} · 대기 {s.ingest.by_status.pending ?? 0}</span></div>
        <div className="metric"><p>실명 보기</p><b>{restores}</b><span>거부된 토큰 {denied} / {tokens + denied}</span></div>
      </div>

      <div className="grid2">
        <div className="card">
          <h2 style={{ marginBottom: 12 }}>최근 14일</h2>
          <div className="cols">
            {s.daily.map((d) => (
              <div key={d.day} className="col" title={`${d.day} · 실명 보기 ${d.restores} · 검수 ${d.reviews}`}>
                <div className="col-bar">
                  <i className="c-review" style={{ height: `${(d.reviews / maxDay) * 100}%` }} />
                  <i className="c-restore" style={{ height: `${(d.restores / maxDay) * 100}%` }} />
                </div>
                <span>{d.day.slice(3)}</span>
              </div>
            ))}
          </div>
          <div className="row small muted" style={{ marginTop: 8 }}>
            <span className="key c-restore" /> 실명 보기 <span className="key c-review" style={{ marginLeft: 10 }} /> 검수
          </div>
        </div>
        <div className="card">
          <h2 style={{ marginBottom: 12 }}>문서 공개 범위</h2>
          <Bars data={Object.entries(s.documents.by_visibility).map(([k, n]) => ({ name: VIS[k] ?? k, n }))} />
          <h2 style={{ margin: '18px 0 12px' }}>검수자가 고친 것</h2>
          <Bars data={[{ name: '놓친 것 추가', n: s.ingest.added }, { name: '잘못 잡은 것 삭제', n: s.ingest.removed }, { name: '라벨 변경', n: s.ingest.relabeled }]} />
        </div>
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <h2 style={{ marginBottom: 10 }}>역할별 실명 보기</h2>
        <table className="table">
          <thead><tr><th>역할</th><th>횟수</th><th>복원된 토큰</th><th>권한 없어 거부</th></tr></thead>
          <tbody>{s.restore.map((r) => <tr key={r.role}><td>{ROLE[r.role] ?? r.role}</td><td>{r.count}</td><td>{r.tokens}</td><td>{r.denied}</td></tr>)}</tbody>
        </table>
        <h2 style={{ margin: '18px 0 10px' }}>최근 실명 보기 기록</h2>
        <table className="table">
          <thead><tr><th>시각</th><th>계정</th><th>역할</th><th>문서</th><th>복원</th><th>거부</th><th>사유</th></tr></thead>
          <tbody>
            {log.map((r) => (
              <tr key={r.id}>
                <td className="small muted nowrap">{new Date(r.created_at).toLocaleString('ko-KR')}</td>
                <td className="small">{r.actor}</td><td>{ROLE[r.role] ?? r.role}</td>
                <td className="small">{r.doc_ids.join(', ') || '-'}</td><td>{r.restored}</td><td>{r.denied}</td>
                <td className="small">{r.purpose ?? '-'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
