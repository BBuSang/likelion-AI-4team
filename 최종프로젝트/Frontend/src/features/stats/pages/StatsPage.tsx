// 통계 (관리자): 문서·직원 수, 적재·검수 결과, 실명 보기 기록. 개인정보 값 없이 개수만
import { useEffect, useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import apiClient from '@/shared/api/client';
import { errMsg } from '@/shared/utils/error';
import MaskedText from '@/features/chat/components/MaskedText';
import './stats.css';

interface Stats {
  documents: { total: number; by_visibility: Record<string, number>; chunks: number };
  employees: { total: number; by_role: Record<string, number> };
  ingest: { by_status: Record<string, number>; auto_approved: number; approved: number; rejected: number; added: number; removed: number; relabeled: number };
  restore: { role: string; count: number; tokens: number; denied: number }[];
  daily: { day: string; restores: number; reviews: number }[];
}
// 이상 접근 탐지 (worklog 38장): 실명 보기 기록 기반 규칙 경고. 실명 없이 계정·토큰·개수만
interface Alert { id: string; rule: string; rule_ko: string; severity: 'high' | 'medium' | 'low'; actor: string; role: string | null; team: string | null; start: string; end: string; count: number; detail: string }
interface Anomalies { days: number; summary: { total: number; by_rule: Record<string, number>; by_severity: Record<string, number> }; alerts: Alert[] }
const SEV: Record<string, { n: string; cls: string }> = { high: { n: '높음', cls: 'badge-bad' }, medium: { n: '중간', cls: 'badge-warn' }, low: { n: '낮음', cls: 'badge-gray' } };
const SEV_RANK: Record<string, number> = { high: 0, medium: 1, low: 2 };
const RULE_ORDER = ['burst', 'probing', 'same_person', 'spike', 'breadth', 'purpose', 'off_hours'];
// 정렬: 시간순(최근 먼저, 기본) · 위험도순(높음 먼저, 같으면 최근) · 규칙순(규칙 묶음, 같으면 최근)
function sortAlerts(alerts: Alert[], by: 'time' | 'severity' | 'rule') {
  const t = (a: Alert) => -new Date(a.start).getTime();
  const key = (a: Alert): [number, number] =>
    by === 'severity' ? [SEV_RANK[a.severity] ?? 9, t(a)] : by === 'rule' ? [RULE_ORDER.indexOf(a.rule), t(a)] : [t(a), 0];
  return [...alerts].sort((x, y) => { const a = key(x), b = key(y); return a[0] - b[0] || a[1] - b[1]; });
}
// 최근 대화 (전체 사용자): 계정·제목(가명처리된 첫 질문)·질문 수만. 서버가 페이지 단위로 줌
interface RecentConv { id: number; userId: number; email: string | null; role: string | null; team: string | null; title: string; messages: number; updatedAt: string }
interface RecentPage { items: RecentConv[]; total: number; page: number; size: number }
const PAGE = 10; // 표 한 페이지 줄 수

/** 이전·다음 + 쪽 번호. pages 가 1 이하면 안 보임 */
function Pager({ page, pages, onChange }: { page: number; pages: number; onChange: (p: number) => void }) {
  if (pages <= 1) return null;
  // 현재 쪽 앞뒤 2개까지만 번호로 (많으면 처음·끝은 … 로)
  const nums = Array.from({ length: pages }, (_, i) => i).filter((i) => i === 0 || i === pages - 1 || Math.abs(i - page) <= 2);
  // 표 밑에 작게: 테두리 없는 글자 버튼, 지금 쪽만 옅은 배경
  return (
    <nav className="pager" aria-label="페이지">
      <button className="pager-btn" disabled={page === 0} onClick={() => onChange(page - 1)} aria-label="이전 페이지">
        <ChevronLeft size={14} aria-hidden />
      </button>
      {nums.map((i, k) => (
        <span key={i} className="pager-item">
          {k > 0 && i - nums[k - 1] > 1 && <span className="pager-gap">…</span>}
          <button className={`pager-btn${i === page ? ' on' : ''}`} aria-current={i === page ? 'page' : undefined}
                  onClick={() => onChange(i)}>{i + 1}</button>
        </span>
      ))}
      <button className="pager-btn" disabled={page >= pages - 1} onClick={() => onChange(page + 1)} aria-label="다음 페이지">
        <ChevronRight size={14} aria-hidden />
      </button>
    </nav>
  );
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
  const [anom, setAnom] = useState<Anomalies | null>(null);
  const [showAll, setShowAll] = useState(false);
  const [sortBy, setSortBy] = useState<'time' | 'severity' | 'rule'>('time');
  const [error, setError] = useState('');
  const [logPage, setLogPage] = useState(0);
  const [convs, setConvs] = useState<RecentPage | null>(null);
  const [convPage, setConvPage] = useState(0);
  const [convErr, setConvErr] = useState('');

  useEffect(() => {
    apiClient.get<Stats>('/api/admin/stats', { params: { days: 14 } }).then((r) => setS(r.data)).catch((e) => setError(errMsg(e)));
    // 실명 보기 기록은 최근 100건을 받아 화면에서 10건씩 나눠 보여 줌
    apiClient.get<RestoreRow[]>('/api/admin/restore-log', { params: { limit: 100 } }).then((r) => setLog(r.data)).catch(() => {});
    apiClient.get<Anomalies>('/api/admin/anomalies', { params: { days: 14 } }).then((r) => setAnom(r.data)).catch(() => {});
  }, []);

  // 최근 대화: 쪽을 바꿀 때마다 그 쪽만 서버에서
  useEffect(() => {
    apiClient.get<RecentPage>('/api/admin/recent-conversations', { params: { page: convPage, size: PAGE } })
      .then((r) => { setConvs(r.data); setConvErr(''); })
      .catch((e) => setConvErr(errMsg(e)));
  }, [convPage]);

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
        <div className="row" style={{ marginBottom: 10 }}>
          <h2>이상 접근 (최근 {anom?.days ?? 14}일)</h2>
          <div className="spacer" />
          {anom && <span className="small muted">높음 {anom.summary.by_severity.high ?? 0} · 중간 {anom.summary.by_severity.medium ?? 0} · 낮음 {anom.summary.by_severity.low ?? 0}</span>}
          {anom && anom.alerts.length > 0 && (
            <select className="select" style={{ height: 30, width: 110, marginLeft: 10 }} value={sortBy} aria-label="정렬"
                    onChange={(e) => setSortBy(e.target.value as 'time' | 'severity' | 'rule')}>
              <option value="time">시간순</option>
              <option value="severity">위험도순</option>
              <option value="rule">규칙순</option>
            </select>
          )}
        </div>
        {!anom && <p className="small muted">불러오는 중…</p>}
        {anom && anom.alerts.length === 0 && <p className="small muted">실명 보기 기록에서 이상한 접근이 보이지 않아요.</p>}
        {anom && anom.alerts.length > 0 && (
          <>
            <table className="table">
              <thead><tr><th>위험</th><th>규칙</th><th>계정</th><th>역할·팀</th><th>시각</th><th>내용</th></tr></thead>
              <tbody>
                {(showAll ? sortAlerts(anom.alerts, sortBy) : sortAlerts(anom.alerts, sortBy).slice(0, 10)).map((a) => (
                  <tr key={a.id}>
                    <td><span className={`badge ${SEV[a.severity].cls}`}>{SEV[a.severity].n}</span></td>
                    <td className="nowrap">{a.rule_ko}</td>
                    <td className="small">{a.actor}</td>
                    <td className="small">{ROLE[a.role ?? ''] ?? a.role ?? '-'}{a.team ? ` · ${a.team}` : ''}</td>
                    <td className="small muted nowrap">{new Date(a.start).toLocaleString('ko-KR', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' })}</td>
                    <td className="small">{a.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {anom.alerts.length > 10 && (
              <button className="btn btn-sm" style={{ marginTop: 8 }} onClick={() => setShowAll((v) => !v)}>
                {showAll ? '접기' : `전체 ${anom.alerts.length}건 보기`}
              </button>
            )}
          </>
        )}
        <p className="small muted" style={{ marginTop: 8 }}>
          규칙: 대량 복원(10분 10번) · 평소보다 급증(3배·20건, 감사 60건) · 업무 외 시간(22~7시·주말) · 넓은 범위(문서 30건, 감사 60건) · 형식적인 사유 · 권한 밖 시도 반복(1시간 5번) · 특정인 반복 조회(하루 5번·문서 3건)
        </p>
      </div>

      <div className="card" style={{ marginTop: 12 }}>
        <h2 style={{ marginBottom: 10 }}>역할별 실명 보기</h2>
        <table className="table">
          <thead><tr><th>역할</th><th>횟수</th><th>복원된 토큰</th><th>권한 없어 거부</th></tr></thead>
          <tbody>{s.restore.map((r) => <tr key={r.role}><td>{ROLE[r.role] ?? r.role}</td><td>{r.count}</td><td>{r.tokens}</td><td>{r.denied}</td></tr>)}</tbody>
        </table>
        <div className="row" style={{ margin: '18px 0 10px' }}>
          <h2>최근 대화 기록</h2>
          <div className="spacer" />
          {convs && <span className="small muted">전체 {convs.total.toLocaleString()}건</span>}
        </div>
        {convErr && <p className="small" style={{ color: 'var(--bad)' }}>대화 기록을 불러오지 못했어요. {convErr}</p>}
        {!convs && !convErr && <p className="small muted">불러오는 중…</p>}
        {convs && convs.items.length === 0 && <p className="small muted">아직 대화가 없어요.</p>}
        {convs && convs.items.length > 0 && (
          <div className="table-wrap">
            <table className="table">
              <thead><tr><th>시각</th><th>계정</th><th>역할·팀</th><th>대화 제목 (가명처리)</th><th>질문</th></tr></thead>
              <tbody>
                {convs.items.map((c) => (
                  <tr key={c.id}>
                    <td className="small muted nowrap">{new Date(c.updatedAt).toLocaleString('ko-KR')}</td>
                    <td className="small">{c.email ?? '-'}</td>
                    <td className="small">{ROLE[c.role ?? ''] ?? c.role ?? '-'}{c.team ? ` · ${c.team}` : ''}</td>
                    <td className="small conv-cell" title={c.title}><MaskedText text={c.title} /></td>
                    <td>{c.messages}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {convs && <Pager page={convPage} pages={Math.ceil(convs.total / PAGE)} onChange={setConvPage} />}

        <div className="row" style={{ margin: '18px 0 10px' }}>
          <h2>최근 실명 보기 기록</h2>
          <div className="spacer" />
          {log.length > 0 && <span className="small muted">최근 {log.length}건</span>}
        </div>
        <div className="table-wrap">
        <table className="table">
          <thead><tr><th>시각</th><th>계정</th><th>역할</th><th>문서</th><th>복원</th><th>거부</th><th>사유</th></tr></thead>
          <tbody>
            {log.slice(logPage * PAGE, (logPage + 1) * PAGE).map((r) => (
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
        <Pager page={logPage} pages={Math.ceil(log.length / PAGE)} onChange={setLogPage} />
      </div>
    </section>
  );
}
