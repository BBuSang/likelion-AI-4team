// 관리자 문서 탭: 공개 자료 보기 (금융상품 설명서·약관, 법령, 금융당국 회신). 읽기만 — 고치기·지우기 없음
// 출처마다 AI 서버가 다른 곳에서 읽음 (상품 = 파일, 법령·회신 = DB). 한 출처가 안 돼도 다른 출처는 보임
import { useEffect, useState } from 'react';
import { ExternalLink, Search } from 'lucide-react';
import apiClient from '@/shared/api/client';
import Pager from '@/shared/components/Pager';
import { errMsg } from '@/shared/utils/error';

export type PublicSource = 'product' | 'law' | 'noaction';

interface Row { id: string; title: string; kind: string | null; date?: string | null; products?: number; sections?: number; articles?: number; serial_no?: string | null; category?: string | null }
interface Detail {
  id: string; title: string; kind: string | null; date?: string | null; file?: string | null; serial_no?: string | null; category?: string | null;
  reg_date?: string | null; status?: string | null; answer_type?: string | null; url?: string | null;
  products?: { product_id: string; name: string; role: string }[];
  sections: { label: string; page?: string | number | null; text: string }[];
  sections_available?: boolean;
}

const PAGE = 30;
const ROLE_KO: Record<string, string> = { terms: '약관', explainer: '설명서' };
const INFO: Record<PublicSource, { search: string; cols: string[]; empty: string; pick: string }> = {
  product: { search: '문서 ID · 문서 이름 · 상품 이름', cols: ['문서 ID', '문서 이름', '종류', '연결 상품', '섹션'], empty: '문서가 없어요.', pick: '표에서 문서를 고르면 연결된 상품과 섹션 내용이 보여요.' },
  law: { search: '법령 이름', cols: ['법령 이름', '종류', '시행일', '조문'], empty: '법령이 없어요.', pick: '표에서 법령을 고르면 조문 목록이 보여요.' },
  noaction: { search: '제목·일련번호', cols: ['구분', '제목', '분야', '회신일'], empty: '회신이 없어요.', pick: '표에서 회신을 고르면 질의 요지 · 회답 · 이유가 보여요.' },
};

export function PublicDocsTab({ source, selected, onSelect }: { source: PublicSource; selected: string | null; onSelect: (id: string) => void }) {
  const [q, setQ] = useState('');
  const [page, setPage] = useState(0);
  const [data, setData] = useState<{ total: number; items: Row[] } | null>(null);
  const [error, setError] = useState('');

  // 출처를 바꾸면 검색·쪽을 처음으로
  useEffect(() => { setQ(''); setPage(0); }, [source]);
  useEffect(() => {
    const t = setTimeout(() => {
      apiClient.get(`/api/admin/public-docs/${source}`, { params: { q: q || undefined, limit: PAGE, offset: page * PAGE } })
        .then((r) => { setData(r.data); setError(''); })
        .catch((e) => { setData(null); setError(errMsg(e)); });
    }, 250);
    return () => clearTimeout(t);
  }, [source, q, page]);

  const info = INFO[source];
  const pages = Math.max(1, Math.ceil((data?.total ?? 0) / PAGE));
  return (
    <>
      <div className="row" style={{ marginBottom: 10, flexWrap: 'wrap' }}>
        <label className="search" style={{ width: 260, margin: 0 }}>
          <Search size={14} aria-hidden /><input placeholder={info.search} value={q} onChange={(e) => { setQ(e.target.value); setPage(0); }} />
        </label>
        <div className="spacer" />
        {data && <span className="small muted">{data.total.toLocaleString()}건</span>}
      </div>
      {error && <div className="alert alert-bad" style={{ marginBottom: 10 }}>자료를 불러오지 못했어요. {error}</div>}
      {!data && !error && <p className="small muted">불러오는 중…</p>}
      {data && (
        <div className="table-scroll">
          <table className="table">
            <thead><tr>{info.cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
            <tbody>
              {data.items.map((d) => (
                <tr key={d.id} className={`clickable${selected === d.id ? ' selected' : ''}`} onClick={() => onSelect(d.id)}>
                  {source === 'product' && <>
                    <td className="nowrap" style={{ fontWeight: 500 }}>{d.id}</td><td>{d.title}</td><td className="nowrap">{d.kind ?? '-'}</td>
                    <td>{d.products}</td><td>{d.sections || '-'}</td>
                  </>}
                  {source === 'law' && <>
                    <td style={{ fontWeight: 500 }}>{d.title}</td><td className="nowrap">{d.kind ?? '-'}</td>
                    <td className="small muted nowrap">{d.date ?? '-'}</td><td>{d.articles?.toLocaleString()}</td>
                  </>}
                  {source === 'noaction' && <>
                    <td className="nowrap"><span className="badge badge-gray">{d.kind ?? '-'}</span></td><td>{d.title}</td>
                    <td className="small nowrap">{d.category ?? '-'}</td><td className="small muted nowrap">{d.date ?? '-'}</td>
                  </>}
                </tr>
              ))}
              {data.items.length === 0 && <tr><td colSpan={info.cols.length} className="muted" style={{ textAlign: 'center', padding: 30 }}>{info.empty}</td></tr>}
            </tbody>
          </table>
        </div>
      )}
      {data && <Pager page={page} pages={pages} onChange={setPage} />}
    </>
  );
}

export function PublicDocHint({ source }: { source: PublicSource }) {
  return <p className="small muted">{INFO[source].pick}</p>;
}

/** 오른쪽 칸: 고른 자료의 정보 + 섹션(상품)·조문(법령)·질의/회답/이유(회신). 조문이 많으면 안에서 찾기 */
export function PublicDocPanel({ source, id }: { source: PublicSource; id: string }) {
  const [d, setD] = useState<Detail | null>(null);
  const [error, setError] = useState('');
  const [find, setFind] = useState('');

  useEffect(() => {
    setD(null); setError(''); setFind('');
    apiClient.get<Detail>(`/api/admin/public-docs/${source}/${encodeURIComponent(id)}`).then((r) => setD(r.data)).catch((e) => setError(errMsg(e)));
  }, [source, id]);

  if (!d) return error ? <div className="alert alert-bad">{error}</div> : <p className="small muted">불러오는 중…</p>;
  const secs = find.trim() ? d.sections.filter((s) => s.label.includes(find.trim()) || s.text.includes(find.trim())) : d.sections;
  const secName = source === 'law' ? '조문' : source === 'noaction' ? '내용' : '섹션';
  return (
    <div className="stack">
      <div>
        <h3 style={{ marginBottom: 4 }}>{d.title}</h3>
        <p className="small muted">
          {[d.kind, d.serial_no && `일련번호 ${d.serial_no}`, d.category, d.date && (source === 'law' ? `시행 ${d.date}` : `회신 ${d.date}`), d.file]
            .filter(Boolean).join(' · ')}
        </p>
        {d.url && <a className="small" href={d.url} target="_blank" rel="noreferrer"><ExternalLink size={12} aria-hidden /> 원문 보기</a>}
      </div>

      {source === 'product' && (
        <div>
          <p className="small" style={{ fontWeight: 500, marginBottom: 4 }}>연결된 상품 {d.products?.length ?? 0}개</p>
          <div className="pub-products">
            {d.products?.map((p) => <span key={p.product_id} className="badge badge-brand" title={p.product_id}>{p.name} · {ROLE_KO[p.role] ?? p.role}</span>)}
          </div>
        </div>
      )}

      <div>
        <div className="row" style={{ marginBottom: 6 }}>
          <p className="small" style={{ fontWeight: 500 }}>{secName} {d.sections.length.toLocaleString()}개</p>
          <div className="spacer" />
          {d.sections.length > 10 && (
            <label className="search" style={{ width: 150, height: 28, margin: 0 }}>
              <Search size={12} aria-hidden /><input placeholder={`${secName} 찾기`} value={find} onChange={(e) => setFind(e.target.value)} />
            </label>
          )}
        </div>
        {source === 'product' && d.sections_available === false && <p className="small muted">이 서버에는 문서 원문(섹션) 파일이 없어요.</p>}
        {secs.map((s, i) => (
          <div key={i} className="pub-section">
            <p className="pub-section-label">{s.label || '(제목 없음)'}{s.page ? <span className="muted"> · {s.page}쪽</span> : null}</p>
            <p className="pub-section-text">{s.text}</p>
          </div>
        ))}
        {find.trim() && secs.length === 0 && <p className="small muted">찾는 {secName}이(가) 없어요.</p>}
      </div>
    </div>
  );
}
