// 문서 종류 고르기: (자동 판단) + 저장된 문서 종류(많은 순) + 직접 입력
// value '' = 자동 (올리기 화면에서만). 목록은 화면에 처음 뜰 때 한 번 불러옴
import { useEffect, useState } from 'react';
import * as api from '../api/ingest.api';

const AUTO = '';
const CUSTOM = '__custom__';

export default function DocTypeSelect({ value, onChange, allowAuto = true, disabled = false, id }: {
  value: string; onChange: (v: string) => void; allowAuto?: boolean; disabled?: boolean; id?: string;
}) {
  const [types, setTypes] = useState<api.DocTypeCount[] | null>(null);
  const [custom, setCustom] = useState(false);

  useEffect(() => { api.docTypes().then(setTypes).catch(() => setTypes([])); }, []);

  const known = !!types?.some((t) => t.doc_type === value);
  // 목록에 없는 값(직접 입력했거나 처음 받은 종류)은 직접 입력 칸으로 보여 줌
  const showCustom = custom || (value !== AUTO && types !== null && !known);
  const selected = showCustom ? CUSTOM : value;

  return (
    <span className="row" style={{ gap: 6, flexWrap: 'wrap' }}>
      <select id={id} className="select" style={{ height: 32, width: 'auto', minWidth: 160 }} disabled={disabled}
        value={selected}
        onChange={(e) => {
          const v = e.target.value;
          if (v === CUSTOM) { setCustom(true); onChange(known ? '' : value); return; }
          setCustom(false);
          onChange(v);
        }}>
        {allowAuto && <option value={AUTO}>자동 판단</option>}
        {!allowAuto && value === AUTO && !showCustom && <option value={AUTO} disabled>종류 고르기</option>}
        {types?.map((t) => <option key={t.doc_type} value={t.doc_type}>{t.doc_type} ({t.docs.toLocaleString()})</option>)}
        <option value={CUSTOM}>직접 입력…</option>
      </select>
      {showCustom && (
        <input className="input" style={{ height: 32, width: 150 }} maxLength={50} placeholder="새 종류 이름" autoFocus={custom}
          disabled={disabled} value={value} onChange={(e) => onChange(e.target.value)} />
      )}
    </span>
  );
}
