// 검수: 원문 위에 탐지된 개인정보를 라벨 색 형광펜으로. 드래그 → 라벨 고르기(1~7), 형광펜 클릭 → 라벨 바꾸기·지우기
// 위치는 서버와 같은 코드포인트 단위 (브라우저 UTF-16 과 다를 수 있어 변환). 승인하면 가명처리 → 임베딩 → 저장
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { ArrowLeft, ChevronLeft, ChevronRight, Trash2 } from 'lucide-react';
import { errMsg } from '@/shared/utils/error';
import { LABELS, cpSlice, labelInfo, toCodePoint } from '@/shared/utils/pii';
import * as api from '../api/ingest.api';
import type { LinkGroup, Span, StagingDoc, StagingRow } from '../api/ingest.api';
import { DOC_STATUS } from '../api/ingest.api';
import './review.css';

const BASIS: Record<string, string> = {
  anchor_link: '전화·이메일 같은 값으로 기존 사람과 연결', aff_link: '소속이 같아 기존 사람과 연결',
  compat_link: '정보가 겹치지 않아 기존 사람과 연결', unique_link: '같은 이름이 한 명뿐이라 연결 (확인 권장)',
  new: '새 사람', ambiguous: '후보가 여럿 (골라 주세요)', unresolved: '판단 보류', chosen_link: '검수자가 고름',
};

type Pop = { kind: 'new'; start: number; end: number; x: number; y: number } | { kind: 'edit'; i: number; x: number; y: number };

function cpAt(node: Node, offset: number): number | null {
  if (node.nodeType === Node.TEXT_NODE) {
    const el = node.parentElement?.closest<HTMLElement>('[data-cp]');
    if (!el) return null;
    return Number(el.dataset.cp) + toCodePoint(node.textContent ?? '', offset);
  }
  const el = node as HTMLElement;
  const child = el.childNodes[offset] as HTMLElement | undefined;
  if (child?.dataset?.cp !== undefined) return Number(child.dataset.cp);
  if (el.dataset?.cp !== undefined) return Number(el.dataset.cp) + Array.from(el.textContent ?? '').length;
  return null;
}

export default function ReviewPage() {
  const { id } = useParams();
  const docId = Number(id);
  const nav = useNavigate();
  const [doc, setDoc] = useState<StagingDoc | null>(null);
  const [spans, setSpans] = useState<Span[]>([]);
  const [siblings, setSiblings] = useState<StagingRow[]>([]);
  const [links, setLinks] = useState<LinkGroup[]>([]);
  const [choices, setChoices] = useState<Record<string, string>>({});
  const [pop, setPop] = useState<Pop | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState('');
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setDoc(null); setPop(null); setError(''); setChoices({}); setRejecting(false); setReason('');
    api.doc(docId).then((d) => {
      setDoc(d);
      setSpans([...d.spans].sort((a, b) => a.start - b.start));
      setLinks(d.links);
      api.jobDocs(d.job_id).then(setSiblings).catch(() => {});
    }).catch((e) => setError(errMsg(e)));
  }, [docId]);

  const editable = doc?.status === 'pending' || doc?.status === 'failed';
  const chars = useMemo(() => Array.from(doc?.text ?? ''), [doc]);

  // 구간이 바뀌면 사람 잇기 미리보기 (0.5초 뒤)
  useEffect(() => {
    if (!doc || !editable) return;
    const t = setTimeout(() => {
      api.previewLinks(doc.id, spans, choices).then(setLinks).catch(() => {});
    }, 500);
    return () => clearTimeout(t);
  }, [spans, choices, doc, editable]);

  const segments = useMemo(() => {
    const out: { start: number; end: number; i: number | null }[] = [];
    let pos = 0;
    spans.forEach((s, i) => {
      if (s.start > pos) out.push({ start: pos, end: s.start, i: null });
      out.push({ start: s.start, end: s.end, i });
      pos = s.end;
    });
    if (pos < chars.length) out.push({ start: pos, end: chars.length, i: null });
    return out;
  }, [spans, chars]);

  const relPos = (rect: DOMRect) => {
    const box = boxRef.current!.getBoundingClientRect();
    const x = rect.left - box.left + rect.width / 2;
    // 창(최대 360px)이 글 상자 밖으로 안 나가게
    return { x: Math.min(Math.max(x, 185), Math.max(185, box.width - 185)), y: rect.bottom - box.top + boxRef.current!.scrollTop + 6 };
  };

  const onMouseUp = () => {
    if (!editable) return;
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) return;
    const r = sel.getRangeAt(0);
    if (!boxRef.current?.contains(r.commonAncestorContainer)) return;
    let a = cpAt(r.startContainer, r.startOffset);
    let b = cpAt(r.endContainer, r.endOffset);
    if (a === null || b === null) return;
    if (a > b) [a, b] = [b, a];
    while (a < b && /\s/.test(chars[a])) a++;
    while (b > a && /\s/.test(chars[b - 1])) b--;
    if (a >= b) return;
    if (spans.some((s) => s.start < b! && a! < s.end)) {
      setError('이미 표시한 구간과 겹쳐요. 먼저 기존 표시를 지워 주세요.');
      sel.removeAllRanges();
      return;
    }
    setError('');
    setPop({ kind: 'new', start: a, end: b, ...relPos(r.getBoundingClientRect()) });
  };

  const apply = useCallback((label: string) => {
    if (!pop) return;
    if (pop.kind === 'new') {
      setSpans((ss) => [...ss, { label, start: pop.start, end: pop.end }].sort((x, y) => x.start - y.start));
    } else {
      setSpans((ss) => ss.map((s, i) => (i === pop.i ? { ...s, label } : s)));
    }
    setPop(null);
    window.getSelection()?.removeAllRanges();
  }, [pop]);

  const removeAt = useCallback((i: number) => { setSpans((ss) => ss.filter((_, j) => j !== i)); setPop(null); }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!pop) return;
      const l = LABELS.find((x) => x.key === e.key);
      if (l) { e.preventDefault(); apply(l.label); }
      if (e.key === 'Escape') setPop(null);
      if ((e.key === 'Delete' || e.key === 'Backspace') && pop.kind === 'edit') { e.preventDefault(); removeAt(pop.i); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [pop, apply, removeAt]);

  // 탐지 대비 바뀐 수
  const changes = useMemo(() => {
    if (!doc) return { added: 0, removed: 0, relabeled: 0 };
    const key = (s: Span) => `${s.start}-${s.end}`;
    const det = new Map(doc.detected.map((s) => [key(s), s.label]));
    const fin = new Map(spans.map((s) => [key(s), s.label]));
    let added = 0, removed = 0, relabeled = 0;
    fin.forEach((l, k) => { if (!det.has(k)) added++; else if (det.get(k) !== l) relabeled++; });
    det.forEach((_, k) => { if (!fin.has(k)) removed++; });
    return { added, removed, relabeled };
  }, [doc, spans]);

  const idx = siblings.findIndex((s) => s.id === docId);
  const go = (d?: StagingRow) => d && nav(`/ingest/docs/${d.id}`);
  const nextPending = () => siblings.find((s, i) => i > idx && (s.status === 'pending' || s.status === 'failed'))
    ?? siblings.find((s) => s.id !== docId && (s.status === 'pending' || s.status === 'failed'));

  const act = async (action: 'approve' | 'reject') => {
    if (!doc) return;
    if (action === 'reject' && !reason.trim()) return setError('반려 사유를 적어 주세요.');
    setBusy(true);
    setError('');
    try {
      await api.review(doc.id, action === 'approve'
        ? { action, spans: spans.map((s) => ({ ...s, text: cpSlice(doc.text, s.start, s.end) })), choices }
        : { action, reason: reason.trim() });
      const n = nextPending();
      if (n) go(n); else nav('/ingest');
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  if (!doc) return <section className="panel page">{error ? <div className="alert alert-bad">{error}</div> : <p className="muted">불러오는 중…</p>}</section>;

  const counts = LABELS.map((l) => ({ ...l, n: spans.filter((s) => s.label === l.label).length }));
  const persons = links.filter((g) => g.mentions.length);

  return (
    <>
      <section className="panel page" style={{ display: 'flex', flexDirection: 'column' }}>
        <div className="page-head">
          <button className="btn btn-sm btn-ghost" onClick={() => nav('/ingest')}><ArrowLeft size={15} aria-hidden /> 목록</button>
          <div>
            <h1>{doc.doc_id}</h1>
            <p>{doc.doc_type ?? '문서'} · 업로드 #{doc.job_id}</p>
          </div>
          <span className={`badge ${DOC_STATUS[doc.status]?.cls ?? 'badge-gray'}`}>{DOC_STATUS[doc.status]?.n ?? doc.status}</span>
          <div className="spacer" />
          <button className="btn btn-sm" disabled={idx <= 0} onClick={() => go(siblings[idx - 1])} aria-label="이전 문서"><ChevronLeft size={15} /></button>
          <span className="small muted">{idx + 1} / {siblings.length || 1}</span>
          <button className="btn btn-sm" disabled={idx < 0 || idx >= siblings.length - 1} onClick={() => go(siblings[idx + 1])} aria-label="다음 문서"><ChevronRight size={15} /></button>
        </div>
        <div className="legend">
          {LABELS.map((l) => <span key={l.label} className={`pii pii-${l.cls}`}>{l.key} {l.name}</span>)}
          <span className="small muted">{editable ? '글자를 드래그하고 숫자 키나 버튼으로 라벨을 고르세요. 형광펜을 누르면 바꾸거나 지울 수 있어요.' : '처리된 문서라 읽기만 할 수 있어요.'}</span>
        </div>
        {error && <div className="alert alert-bad" style={{ marginBottom: 10 }}>{error}</div>}
        <div className="doc-text" ref={boxRef} onMouseUp={onMouseUp}>
          {segments.map((g) => {
            const t = chars.slice(g.start, g.end).join('');
            if (g.i === null) return <span key={`t${g.start}`} data-cp={g.start}>{t}</span>;
            const s = spans[g.i];
            const info = labelInfo(s.label);
            return (
              <mark key={`m${g.start}`} data-cp={g.start} data-name={info?.name ?? s.label}
                    className={`hl hl-${info?.cls ?? 'other'}`}
                    onMouseUp={(e) => e.stopPropagation()}
                    onClick={(e) => editable && setPop({ kind: 'edit', i: g.i!, ...relPos((e.target as HTMLElement).getBoundingClientRect()) })}>
                {t}
              </mark>
            );
          })}
          {pop && (
            <div className="label-pop" style={{ left: pop.x, top: pop.y }} onMouseUp={(e) => e.stopPropagation()}>
              {LABELS.map((l) => (
                <button key={l.label} className={`pii pii-${l.cls}`} onClick={() => apply(l.label)}>{l.key} {l.name}</button>
              ))}
              {pop.kind === 'edit' && <button className="btn btn-sm btn-danger" onClick={() => removeAt(pop.i)} aria-label="지우기"><Trash2 size={13} /></button>}
            </div>
          )}
        </div>
      </section>

      <aside className="panel side side-wide stack">
        <div>
          <h3 style={{ marginBottom: 8 }}>찾은 개인정보 {spans.length}개</h3>
          <div className="count-grid">
            {counts.map((c) => <div key={c.label} className="count"><span className={`dot hl-${c.cls}`} />{c.name}<b>{c.n}</b></div>)}
          </div>
          <p className="small muted" style={{ marginTop: 8 }}>탐지 대비 추가 {changes.added} · 삭제 {changes.removed} · 라벨 변경 {changes.relabeled}</p>
        </div>

        <div>
          <h3 style={{ marginBottom: 6 }}>같은 사람 잇기</h3>
          {persons.length === 0 && <p className="small muted">이름 표시가 없어요.</p>}
          {persons.map((g) => {
            const name = cpSlice(doc.text, g.mentions[0][0], g.mentions[0][1]);
            const ask = editable && (g.needs_check || g.candidates.length > 1);
            return (
              <div key={g.id} className="link-row">
                <div className="row">
                  <span className="pii pii-person">{name}</span>
                  <span className="small muted">{g.mentions.length}번 나옴</span>
                  <div className="spacer" />
                  {g.needs_check && <span className="badge badge-warn">확인</span>}
                </div>
                <p className="small muted" style={{ marginTop: 4 }}>{BASIS[g.basis] ?? g.basis}{g.evidence.length ? ` · 근거 ${g.evidence.join(', ')}` : ''}</p>
                {ask && (
                  <select className="select" style={{ height: 32, marginTop: 6 }} value={choices[g.id] ?? ''}
                          onChange={(e) => setChoices((c) => { const n = { ...c }; if (e.target.value) n[g.id] = e.target.value; else delete n[g.id]; return n; })}>
                    <option value="">자동 판단 그대로</option>
                    {g.candidates.map((p) => <option key={p} value={p}>기존 사람 {p.slice(0, 10)}</option>)}
                    <option value="new">새 사람으로</option>
                    <option value="unresolved">판단 보류</option>
                  </select>
                )}
              </div>
            );
          })}
        </div>

        {editable && (
          <div className="stack" style={{ marginTop: 'auto', borderTop: '1px solid var(--line)', paddingTop: 12 }}>
            {!rejecting ? (
              <>
                <button className="btn btn-primary" disabled={busy} onClick={() => act('approve')}>{busy ? '저장 중…' : '승인하고 저장'}</button>
                <button className="btn" disabled={busy} onClick={() => setRejecting(true)}>반려</button>
                <p className="small muted">승인하면 가명처리 → 임베딩 → 저장하고 다음 문서로 넘어가요.</p>
              </>
            ) : (
              <>
                <textarea className="textarea" placeholder="반려 사유 (예: 개인정보가 아닌 내부 문서)" value={reason} onChange={(e) => setReason(e.target.value)} maxLength={500} />
                <div className="row">
                  <button className="btn" onClick={() => setRejecting(false)}>취소</button>
                  <button className="btn btn-danger" style={{ flex: 1 }} disabled={busy} onClick={() => act('reject')}>반려하기</button>
                </div>
              </>
            )}
          </div>
        )}
      </aside>
    </>
  );
}
