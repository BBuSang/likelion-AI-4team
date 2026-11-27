// 검수: 원문 위에 탐지된 개인정보를 라벨 색 형광펜으로. 드래그 → 라벨 고르기(1~7), 형광펜 클릭 → 라벨 바꾸기·지우기
// 위치는 서버와 같은 코드포인트 단위 (브라우저 UTF-16 과 다를 수 있어 변환). 승인하면 가명처리 → 임베딩 → 저장
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { AlertTriangle, ArrowLeft, ChevronLeft, ChevronRight, Pencil, Plus, Trash2, Users } from 'lucide-react';
import ConfirmDialog from '@/shared/components/ConfirmDialog';
import { errMsg } from '@/shared/utils/error';
import { LABELS, cpSlice, labelInfo, toCodePoint } from '@/shared/utils/pii';
import DocTypeSelect from '../components/DocTypeSelect';
import * as api from '../api/ingest.api';
import type { LinkGroup, ReviewFlag, ReviewHint, Span, StagingDoc, StagingRow } from '../api/ingest.api';
import { DOC_STATUS, FLAG_KO } from '../api/ingest.api';
import './review.css';

// 사람 잇기 판단 (AI linking.py). '이미 저장된 사람' = 앞서 저장한 문서에 나온 사람
const BASIS: Record<string, string> = {
  anchor_link: '이미 저장된 사람과 같은 사람으로 이었어요. 전화번호·이메일 같은 값이 같아요.',
  aff_link: '이미 저장된 사람과 같은 사람으로 이었어요. 이름과 소속이 같아요.',
  compat_link: '이미 저장된 사람과 같은 사람으로 이었어요. 이름이 같고 다른 정보도 서로 어긋나지 않아요.',
  unique_link: '이미 저장된 같은 이름의 사람과 같은 사람으로 이었어요. 그 이름이 한 명뿐이라 이은 것이고, 이름 말고는 근거가 없어요.',
  new: '처음 나온 사람이라 새 사람으로 저장해요.',
  ambiguous: '같은 이름의 사람이 여럿이라 누구인지 정하지 못했어요.',
  unresolved: '누구인지 정하지 않고 이름으로만 저장해요.',
  chosen_link: '검수자가 고른 사람과 이어요.',
};
// 확인이 필요한 판단에서 검수자가 할 일
const CHECK_HOW: Record<string, string> = {
  unique_link: '동명이인일 수 있어요. 문맥상 같은 사람이면 그대로 두고, 다른 사람이면 "새 사람으로", 모르겠으면 "판단 보류"를 고르세요.',
  ambiguous: '아래에서 어느 사람인지 고르세요. 모르겠으면 "판단 보류"를 고르세요.',
};
// 근거 코드 → 말 ('name' 은 위 문장에 이미 들어 있어 따로 안 보임)
const EVIDENCE_KO: Record<string, string> = {
  phone_number: '전화번호', email: '이메일', account_number: '계좌번호', card_number: '카드번호',
  resident_registration_number: '주민번호', address: '주소', affiliation: '소속', compatible: '어긋나는 정보 없음',
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
  const [docType, setDocType] = useState(''); // 승인할 때 저장할 문서 종류 (처음엔 지금 값)
  const [editing, setEditing] = useState(false); // 본문 고치기 중
  const [draft, setDraft] = useState('');
  const [names, setNames] = useState<Record<string, string>>({}); // 화자 이름 바꾸기 {지금 이름: 새 이름}
  const [note, setNote] = useState(''); // 본문을 고친 뒤 안내
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setDoc(null); setPop(null); setError(''); setChoices({}); setRejecting(false); setReason(''); setDocType('');
    setEditing(false); setDraft(''); setNames({}); setNote('');
    api.doc(docId).then((d) => {
      setDoc(d);
      setDocType(d.doc_type ?? '');
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

  // 검수로 온 이유: 위치가 그대로인 구간에만 표시, 힌트는 아직 아무 구간과도 안 겹친 것만
  const flagAt = useMemo(() => new Map((doc?.review?.flags ?? []).map((f) => [`${f.start}-${f.end}`, f])), [doc]);
  const hints = useMemo(() => (doc?.review?.hints ?? []).filter((h) => !spans.some((s) => s.start < h.end && h.start < s.end)),
    [doc, spans]);

  const segments = useMemo(() => {
    const out: { start: number; end: number; i: number | null; hint?: ReviewHint }[] = [];
    const plain = (a: number, b: number) => {  // 표시 안 된 글 — 힌트 자리는 따로 떼어 밑줄
      let p = a;
      hints.filter((h) => h.start >= a && h.end <= b).forEach((h) => {
        if (h.start > p) out.push({ start: p, end: h.start, i: null });
        out.push({ start: h.start, end: h.end, i: null, hint: h });
        p = h.end;
      });
      if (p < b) out.push({ start: p, end: b, i: null });
    };
    let pos = 0;
    spans.forEach((s, i) => {
      if (s.start > pos) plain(pos, s.start);
      out.push({ start: s.start, end: s.end, i });
      pos = s.end;
    });
    if (pos < chars.length) plain(pos, chars.length);
    return out;
  }, [spans, chars, hints]);

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

  // 저장하지 않은 수정 수: 처음 불러온 doc.spans 와 다른 곳(더함·뺌·라벨 바꿈) + 사람 잇기 선택(choices)
  const unsaved = useMemo(() => {
    if (!doc) return 0;
    const key = (s: Span) => `${s.start}-${s.end}`;
    const init = new Map(doc.spans.map((s) => [key(s), s.label]));
    const cur = new Map(spans.map((s) => [key(s), s.label]));
    let n = 0;
    cur.forEach((l, k) => { if (init.get(k) !== l) n++; });
    init.forEach((_, k) => { if (!cur.has(k)) n++; });
    return n + Object.keys(choices).length + (docType.trim() !== (doc.doc_type ?? '') ? 1 : 0)
      + (editing && draft !== doc.text ? 1 : 0) + Object.entries(names).filter(([k, v]) => v.trim() && v.trim() !== k).length;
  }, [doc, spans, choices, docType, editing, draft, names]);

  // 수정이 있으면 새로고침·탭 닫기 전에 브라우저가 물어봄 (문구는 브라우저가 정함)
  useEffect(() => {
    if (!unsaved) return;
    const onUnload = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = ''; };
    window.addEventListener('beforeunload', onUnload);
    return () => window.removeEventListener('beforeunload', onUnload);
  }, [unsaved]);

  // 이전·다음·목록: 수정이 있으면 확인 대화상자를 거쳐 이동 (승인·반려 뒤 자동 이동은 묻지 않음)
  const [leaveTo, setLeaveTo] = useState<string | null>(null);
  const leave = (path: string) => (unsaved ? setLeaveTo(path) : nav(path));

  const idx = siblings.findIndex((s) => s.id === docId);
  const go = (d?: StagingRow) => d && nav(`/ingest/docs/${d.id}`);
  const goAsk = (d?: StagingRow) => d && leave(`/ingest/docs/${d.id}`);
  const nextPending = () => siblings.find((s, i) => i > idx && (s.status === 'pending' || s.status === 'failed'))
    ?? siblings.find((s) => s.id !== docId && (s.status === 'pending' || s.status === 'failed'));

  // 본문 저장 (직접 고치기·화자 이름 바꾸기 둘 다) → 서버가 구간을 옮기고 바뀐 곳을 다시 탐지 → 다시 불러옴
  const saveText = async (text: string) => {
    if (!doc) return;
    if (!text.trim()) return setError('본문이 비었어요.');
    setBusy(true);
    setError('');
    try {
      const r = await api.editText(doc.id, { text, spans: spans.map((s) => ({ ...s, text: cpSlice(doc.text, s.start, s.end) })) });
      const d = await api.doc(doc.id);
      setDoc(d);
      setSpans([...d.spans].sort((a, b) => a.start - b.start));
      setLinks(d.links);
      setEditing(false);
      setNames({});
      setPop(null);
      setNote(r.changed
        ? `본문을 고쳤어요. 개인정보 표시는 그대로 ${r.kept}곳, 고친 곳에서 다시 찾은 곳 ${r.added}곳${r.dropped ? `, 고친 곳에 걸려 지운 곳 ${r.dropped}곳` : ''}. 고친 부분을 한 번 더 봐 주세요.`
        : '바뀐 곳이 없어요.');
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  const act = async (action: 'approve' | 'reject') => {
    if (!doc) return;
    if (action === 'reject' && !reason.trim()) return setError('반려 사유를 적어 주세요.');
    setBusy(true);
    setError('');
    try {
      await api.review(doc.id, action === 'approve'
        ? { action, spans: spans.map((s) => ({ ...s, text: cpSlice(doc.text, s.start, s.end) })), choices,
            docType: docType.trim() && docType.trim() !== (doc.doc_type ?? '') ? docType.trim() : undefined }
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

  const autoType = doc.meta?.doc_type_auto as { doc_type: string; score: number; second: string | null; second_score: number | null } | undefined;
  const speakers = api.speakersIn(doc.text);
  const counts = LABELS.map((l) => ({ ...l, n: spans.filter((s) => s.label === l.label).length }));
  const persons = links.filter((g) => g.mentions.length);

  return (
    <>
      <section className="panel page" style={{ display: 'flex', flexDirection: 'column' }}>
        <div className="page-head">
          <button className="btn btn-sm btn-ghost" onClick={() => leave('/ingest')}><ArrowLeft size={15} aria-hidden /> 목록</button>
          <div>
            <h1>{doc.doc_id}</h1>
            <p>{doc.doc_type ?? '문서'} · 업로드 #{doc.job_id}</p>
          </div>
          <span className={`badge ${DOC_STATUS[doc.status]?.cls ?? 'badge-gray'}`}>{DOC_STATUS[doc.status]?.n ?? doc.status}</span>
          <div className="spacer" />
          <button className="btn btn-sm" disabled={idx <= 0} onClick={() => goAsk(siblings[idx - 1])} aria-label="이전 문서"><ChevronLeft size={15} /></button>
          <span className="small muted">{idx + 1} / {siblings.length || 1}</span>
          <button className="btn btn-sm" disabled={idx < 0 || idx >= siblings.length - 1} onClick={() => goAsk(siblings[idx + 1])} aria-label="다음 문서"><ChevronRight size={15} /></button>
        </div>
        <div className="legend">
          {LABELS.map((l) => <span key={l.label} className={`pii pii-${l.cls}`}>{l.key} {l.name}</span>)}
          <span className="small muted">{!editable ? '처리된 문서라 읽기만 할 수 있어요.'
            : editing ? '본문을 고친 뒤 저장하세요. 안 고친 곳의 개인정보 표시는 그대로 두고, 고친 곳은 다시 찾아요.'
            : '글자를 드래그하고 숫자 키나 버튼으로 라벨을 고르세요. 형광펜을 누르면 바꾸거나 지울 수 있어요.'}</span>
          {editable && !editing && (
            <button className="btn btn-sm" style={{ marginLeft: 'auto' }} disabled={busy}
                    onClick={() => { setDraft(doc.text); setEditing(true); setPop(null); setNote(''); }}>
              <Pencil size={13} aria-hidden /> 본문 고치기
            </button>
          )}
        </div>
        {note && !editing && <div className="alert alert-ok" style={{ marginBottom: 10 }}>{note}</div>}
        {error && <div className="alert alert-bad" style={{ marginBottom: 10 }}>{error}</div>}
        {editing ? (
          <div className="stack" style={{ flex: 1, minHeight: 0 }}>
            <textarea className="textarea doc-edit" value={draft} onChange={(e) => setDraft(e.target.value)} aria-label="본문 고치기" autoFocus />
            <div className="row" style={{ gap: 8 }}>
              <span className="small muted">{Array.from(draft).length.toLocaleString()}자</span>
              <div className="spacer" />
              <button className="btn btn-sm" disabled={busy} onClick={() => { setEditing(false); setDraft(''); }}>취소</button>
              <button className="btn btn-sm btn-primary" disabled={busy || draft === doc.text} onClick={() => saveText(draft)}>
                {busy ? '저장하는 중…' : '본문 저장'}
              </button>
            </div>
          </div>
        ) : (
        <div className="doc-text" ref={boxRef} onMouseUp={onMouseUp}>
          {segments.map((g) => {
            const t = chars.slice(g.start, g.end).join('');
            if (g.i === null && g.hint) {
              return <span key={`h${g.start}`} data-cp={g.start} className="hint-miss"
                           title={`놓쳤을 수 있는 곳: ${labelInfo(g.hint.label)?.name ?? g.hint.label} (모델 ${Math.round(g.hint.p * 100)}%)`}>{t}</span>;
            }
            if (g.i === null) return <span key={`t${g.start}`} data-cp={g.start}>{t}</span>;
            const s = spans[g.i];
            const info = labelInfo(s.label);
            const flag: ReviewFlag | undefined = flagAt.get(`${s.start}-${s.end}`);
            return (
              <mark key={`m${g.start}`} data-cp={g.start} data-name={info?.name ?? s.label}
                    className={`hl hl-${info?.cls ?? 'other'}${flag && editable ? ' hl-flag' : ''}`}
                    title={flag ? FLAG_KO[flag.flag] : undefined}
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
        )}
      </section>

      <aside className="panel side side-wide stack">
        <div>
          <h3 style={{ marginBottom: 6 }}>문서 종류</h3>
          {editable
            ? <DocTypeSelect value={docType} onChange={setDocType} allowAuto={false} disabled={busy} />
            : <p className="small">{doc.doc_type ?? '-'}</p>}
          {autoType && (
            <p className="small muted" style={{ marginTop: 6 }}>
              자동 판단: {autoType.doc_type} ({Math.round(autoType.score * 100)}%)
              {autoType.second && <> · 다음 후보 {autoType.second} ({Math.round((autoType.second_score ?? 0) * 100)}%)</>}
            </p>
          )}
        </div>

        {speakers.length > 0 && (
          <div>
            <h3 style={{ marginBottom: 6 }}><Users size={15} aria-hidden /> 화자 {speakers.length}명</h3>
            {editable && !editing ? <>
              {speakers.map((sp) => (
                <div key={sp.name} className="row small" style={{ gap: 6, marginTop: 4 }}>
                  <span style={{ minWidth: 56 }}>{sp.name}</span>
                  <span className="muted" style={{ minWidth: 34 }}>{sp.lines}줄</span>
                  <input className="input" style={{ height: 28, flex: 1 }} maxLength={30} placeholder="새 이름"
                         list="speaker-names" value={names[sp.name] ?? ''} disabled={busy} aria-label={`${sp.name} 새 이름`}
                         onChange={(e) => setNames((n) => ({ ...n, [sp.name]: e.target.value.replace(/[:\n]/g, '') }))} />
                </div>
              ))}
              <datalist id="speaker-names">{speakers.map((sp) => <option key={sp.name} value={sp.name} />)}</datalist>
              <div className="row" style={{ marginTop: 6 }}>
                <span className="small muted">다른 화자 이름을 넣으면 한 사람으로 합쳐져요</span>
                <div className="spacer" />
                <button className="btn btn-sm" disabled={busy || !Object.entries(names).some(([k, v]) => v.trim() && v.trim() !== k)}
                        onClick={() => saveText(api.renameSpeakers(doc.text, names))}>이름 바꾸기</button>
              </div>
            </> : <p className="small muted">{speakers.map((sp) => `${sp.name} ${sp.lines}줄`).join(' · ')}</p>}
          </div>
        )}

        <div>
          <h3 style={{ marginBottom: 8 }}>찾은 개인정보 {spans.length}개</h3>
          <div className="count-grid">
            {counts.map((c) => <div key={c.label} className="count"><span className={`dot hl-${c.cls}`} />{c.name}<b>{c.n}</b></div>)}
          </div>
          <p className="small muted" style={{ marginTop: 8 }}>탐지 대비 추가 {changes.added} · 삭제 {changes.removed} · 라벨 변경 {changes.relabeled}</p>
        </div>

        {editable && doc.review?.needs_review && (
          <div className="review-why">
            <h3 style={{ marginBottom: 6 }}><AlertTriangle size={15} aria-hidden /> 확인이 필요한 곳</h3>
            {(doc.review.flags ?? []).map((f) => (
              <p key={`f${f.start}`} className="small">
                <span className="pii pii-flag">{cpSlice(doc.text, f.start, f.end)}</span> {labelInfo(f.label)?.name ?? f.label} — {FLAG_KO[f.flag]}
                {f.p !== null && <span className="muted"> (모델 {Math.round(f.p * 100)}%)</span>}
              </p>
            ))}
            {(doc.review.link_check ?? []).length > 0 && <p className="small">동명이인 확인 필요</p>}
            {doc.review.doc_type_check && <p className="small">문서 종류가 애매해요 ({doc.review.doc_type_check.pick}{doc.review.doc_type_check.second ? ` / ${doc.review.doc_type_check.second}` : ''}). 위에서 확인해 주세요.</p>}
            {hints.length > 0 && <>
              <p className="small muted" style={{ marginTop: 8 }}>놓쳤을 수 있는 곳 (물결 밑줄) — 개인정보면 추가하세요</p>
              {hints.map((h) => (
                <div key={`h${h.start}`} className="row small" style={{ marginTop: 4 }}>
                  <span className="hint-miss">{cpSlice(doc.text, h.start, h.end)}</span>
                  <span className="muted">{labelInfo(h.label)?.name ?? h.label}? {Math.round(h.p * 100)}%</span>
                  <div className="spacer" />
                  <button className="btn btn-sm" onClick={() => setSpans((ss) => [...ss, { label: h.label, start: h.start, end: h.end }].sort((x, y) => x.start - y.start))}>
                    <Plus size={13} aria-hidden /> 추가
                  </button>
                </div>
              ))}
            </>}
            <p className="small muted" style={{ marginTop: 8 }}>점선 테두리 구간을 확인하고, 틀렸으면 눌러서 고치거나 지우세요.</p>
          </div>
        )}

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
                <p className="small muted" style={{ marginTop: 4 }}>{BASIS[g.basis] ?? g.basis}</p>
                {g.evidence.some((x) => EVIDENCE_KO[x]) && (
                  <p className="small muted">같은 값: {g.evidence.map((x) => EVIDENCE_KO[x]).filter(Boolean).join(', ')}</p>
                )}
                {g.needs_check && CHECK_HOW[g.basis] && (
                  <p className="small" style={{ marginTop: 4, color: 'var(--warn)' }}>{CHECK_HOW[g.basis]}</p>
                )}
                {ask && (
                  <select className="select" style={{ height: 32, marginTop: 6 }} value={choices[g.id] ?? ''}
                          onChange={(e) => setChoices((c) => { const n = { ...c }; if (e.target.value) n[g.id] = e.target.value; else delete n[g.id]; return n; })}>
                    <option value="">자동 판단 그대로{g.basis === 'unique_link' ? ' (같은 사람)' : ''}</option>
                    {g.candidates.map((p) => <option key={p} value={p}>이미 저장된 사람 {p.slice(0, 10)}</option>)}
                    <option value="new">새 사람으로 (동명이인)</option>
                    <option value="unresolved">판단 보류 (이름으로만 저장)</option>
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

      {leaveTo && (
        <ConfirmDialog title="저장하지 않은 수정" confirmLabel="이동" danger
                       message={`저장하지 않은 수정 ${unsaved}건이 사라져요. 이동할까요?`}
                       onCancel={() => setLeaveTo(null)}
                       onConfirm={() => { const to = leaveTo; setLeaveTo(null); nav(to); }} />
      )}
    </>
  );
}
