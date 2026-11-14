// 데이터 적재: 파일 올리기(jsonl / 줄글) + 업로드 목록 + 고른 업로드의 문서 목록
import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileUp, RefreshCw } from 'lucide-react';
import { errMsg, fmtTime } from '@/shared/utils/error';
import * as api from '../api/ingest.api';
import type { Job, StagingRow } from '../api/ingest.api';
import { APPROVED_BY_KO, DOC_STATUS } from '../api/ingest.api';
import './review.css';

const JOB_STATUS: Record<string, { n: string; cls: string }> = {
  processing: { n: '탐지 중', cls: 'badge-brand' }, ready: { n: '탐지 끝', cls: 'badge-ok' }, failed: { n: '실패', cls: 'badge-bad' },
};

export default function IngestPage() {
  const nav = useNavigate();
  const [jobs, setJobs] = useState<Job[]>([]);
  const [sel, setSel] = useState<number | null>(null);
  const [docs, setDocs] = useState<StagingRow[]>([]);
  const [mode, setMode] = useState<'file' | 'text'>('file');
  const [file, setFile] = useState<File | null>(null);
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const reload = useCallback(async () => {
    try { setJobs(await api.jobs()); } catch (e) { setMsg({ ok: false, text: errMsg(e) }); }
  }, []);
  useEffect(() => { reload(); }, [reload]);

  // 탐지 중인 업로드가 있으면 3초마다 새로고침
  useEffect(() => {
    if (!jobs.some((j) => j.status === 'processing')) return;
    const t = setInterval(reload, 3000);
    return () => clearInterval(t);
  }, [jobs, reload]);

  useEffect(() => {
    if (sel === null) return;
    api.jobDocs(sel).then(setDocs).catch((e) => setMsg({ ok: false, text: errMsg(e) }));
  }, [sel, jobs]);

  const submit = async () => {
    setMsg(null);
    let body: Parameters<typeof api.upload>[0];
    if (mode === 'file') {
      if (!file) return setMsg({ ok: false, text: '파일을 골라 주세요.' });
      const content = await file.text();
      body = { source: file.name.endsWith('.jsonl') ? 'jsonl' : 'text', content, filename: file.name };
    } else {
      if (!text.trim()) return setMsg({ ok: false, text: '글을 붙여 넣어 주세요.' });
      body = { source: 'text', content: text };
    }
    setBusy(true);
    try {
      const r = await api.upload(body);
      setMsg({ ok: true, text: `${r.total}건을 올렸어요. 개인정보를 찾는 중이에요 (처음엔 1~2분).` });
      setFile(null);
      setText('');
      if (fileRef.current) fileRef.current.value = '';
      await reload();
      setSel(r.job_id);
    } catch (e) {
      setMsg({ ok: false, text: errMsg(e) });
    } finally {
      setBusy(false);
    }
  };

  const selJob = jobs.find((j) => j.id === sel);
  const firstPending = docs.find((d) => d.status === 'pending' || d.status === 'failed');

  return (
    <>
      <section className="panel page">
        <div className="page-head">
          <div><h1>데이터 적재</h1><p>올린 문서에서 개인정보를 찾아요. 애매한 곳이 없는 문서는 바로, 있는 문서는 사람이 확인한 뒤 가명처리해서 저장해요.</p></div>
          <div className="spacer" />
          <button className="btn btn-sm" onClick={reload}><RefreshCw size={14} aria-hidden /> 새로고침</button>
        </div>

        <div className="card" style={{ marginBottom: 18 }}>
          <div className="tabs" style={{ marginBottom: 12 }}>
            <button className={`tab${mode === 'file' ? ' active' : ''}`} onClick={() => setMode('file')}>파일 올리기</button>
            <button className={`tab${mode === 'text' ? ' active' : ''}`} onClick={() => setMode('text')}>글 붙여 넣기</button>
          </div>
          {mode === 'file' ? (
            <label className="drop">
              <FileUp size={26} color="var(--brand)" aria-hidden />
              <p>{file ? file.name : '.jsonl 또는 .txt 파일을 고르세요'}</p>
              <p className="small muted">jsonl: 한 줄에 {'{id, text, doc_type, entities?}'} · 정답 라벨(entities)이 있으면 탐지와 같을 때, 없으면 애매한 곳이 없을 때 바로 저장돼요</p>
              <input ref={fileRef} type="file" accept=".jsonl,.txt" hidden onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            </label>
          ) : (
            <textarea className="textarea" style={{ minHeight: 120 }} placeholder="문서 1건의 본문" value={text} onChange={(e) => setText(e.target.value)} />
          )}
          <div className="row" style={{ marginTop: 12 }}>
            <p className="small muted">원문은 검수가 끝날 때까지 별도 보관소(vault)에만 있어요.</p>
            <div className="spacer" />
            <button className="btn btn-primary" disabled={busy} onClick={submit}>{busy ? '올리는 중…' : '올리기'}</button>
          </div>
          {msg && <div className={`alert ${msg.ok ? 'alert-ok' : 'alert-bad'}`} style={{ marginTop: 10 }}>{msg.text}</div>}
        </div>

        <h2 style={{ marginBottom: 8 }}>업로드 목록</h2>
        <table className="table">
          <thead><tr><th>#</th><th>파일</th><th>올린 사람</th><th>건수</th><th>상태</th><th>검수 대기</th><th>저장</th><th>반려·실패</th><th>시각</th></tr></thead>
          <tbody>
            {jobs.map((j) => (
              <tr key={j.id} className={`clickable${sel === j.id ? ' selected' : ''}`} onClick={() => setSel(j.id)}>
                <td>{j.id}</td>
                <td>{j.filename ?? (j.source === 'text' ? '붙여 넣은 글' : '-')}</td>
                <td className="small">{j.uploader}</td>
                <td>{j.total}</td>
                <td><span className={`badge ${JOB_STATUS[j.status]?.cls ?? 'badge-gray'}`}>{JOB_STATUS[j.status]?.n ?? j.status}</span></td>
                <td>{j.pending ? <span className="badge badge-warn">{j.pending}</span> : 0}</td>
                <td>{j.stored}</td>
                <td>{j.rejected + j.failed}</td>
                <td className="small muted nowrap">{fmtTime(j.created_at)}</td>
              </tr>
            ))}
            {jobs.length === 0 && <tr><td colSpan={9} className="muted" style={{ textAlign: 'center', padding: 30 }}>아직 올린 파일이 없어요.</td></tr>}
          </tbody>
        </table>
      </section>

      <aside className="panel side side-wide">
        {!selJob && <p className="small muted">업로드를 고르면 문서 목록이 보여요.</p>}
        {selJob && (
          <div className="stack">
            <div>
              <h2>업로드 #{selJob.id}</h2>
              <p className="small muted">{selJob.filename ?? '붙여 넣은 글'} · {selJob.total}건
                {docs.length > 0 && ` · 자동 저장 ${docs.filter((d) => d.approved_by === 'auto' || d.approved_by === 'gold').length}건`}</p>
            </div>
            {selJob.status === 'processing' && <div className="alert alert-info small">개인정보를 찾는 중이에요…</div>}
            {firstPending && (
              <button className="btn btn-primary" onClick={() => nav(`/ingest/docs/${firstPending.id}`)}>검수 시작 (대기 {docs.filter((d) => d.status === 'pending').length}건)</button>
            )}
            <div>
              {docs.map((d) => (
                <div key={d.id} className="doc-row" onClick={() => nav(`/ingest/docs/${d.id}`)}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <p style={{ fontWeight: 500 }}>{d.ext_id ?? `#${d.seq}`} <span className="small muted">{d.doc_type ?? ''}</span></p>
                    <p className="small muted">
                      탐지 {d.n_detected}개
                      {d.approved_by && APPROVED_BY_KO[d.approved_by] ? ` · ${APPROVED_BY_KO[d.approved_by]}` : ''}
                      {d.status === 'pending' && d.n_flags ? ` · 확인할 곳 ${d.n_flags}개` : ''}
                      {d.status === 'pending' && d.n_hints ? ` · 놓쳤을 수 있는 곳 ${d.n_hints}개` : ''}
                    </p>
                  </div>
                  <span className={`badge ${DOC_STATUS[d.status]?.cls ?? 'badge-gray'}`}>{DOC_STATUS[d.status]?.n ?? d.status}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </aside>
    </>
  );
}
