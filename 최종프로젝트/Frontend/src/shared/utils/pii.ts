// 개인정보 라벨: 탐지기 라벨(person 등) ↔ 토큰 접두어(PERSON 등) ↔ 화면 이름·색 클래스
// AI app/pii/pii_pipeline.py 의 PREFIX, app/ingest/service.py 의 LABELS 와 같음

export const LABELS = [
  { label: 'person', prefix: 'PERSON', name: '이름', cls: 'person', key: '1' },
  { label: 'phone_number', prefix: 'PHONE', name: '전화', cls: 'phone', key: '2' },
  { label: 'email', prefix: 'EMAIL', name: '이메일', cls: 'email', key: '3' },
  { label: 'address', prefix: 'ADDRESS', name: '주소', cls: 'address', key: '4' },
  { label: 'resident_registration_number', prefix: 'RRN', name: '주민번호', cls: 'rrn', key: '5' },
  { label: 'account_number', prefix: 'ACCOUNT', name: '계좌', cls: 'account', key: '6' },
  { label: 'card_number', prefix: 'CARD', name: '카드', cls: 'card', key: '7' },
] as const;

export type PiiLabel = (typeof LABELS)[number]['label'];

export const labelInfo = (label: string) => LABELS.find((l) => l.label === label);

const byPrefix = (prefix: string) => LABELS.find((l) => l.prefix === prefix);

/** 답변 속 토큰 `<PERSON_ab12…>` */
export const TOKEN_RE = /<([A-Z_]+)_([0-9A-Za-z]+)>/g;

export type Piece =
  | { kind: 'text'; text: string }
  | { kind: 'token'; token: string; name: string; cls: string; short: string }
  | { kind: 'cite'; docId: string };

/** 글 → 일반 글 / 토큰 / [문서 ID] 조각 */
export function splitAnswer(text: string): Piece[] {
  const out: Piece[] = [];
  const re = /<([A-Z_]+)_([0-9A-Za-z]+)>|\[([A-Z]\d{3,}[A-Za-z0-9-]*)\]/g;
  let last = 0;
  for (const m of text.matchAll(re)) {
    const i = m.index ?? 0;
    if (i > last) out.push({ kind: 'text', text: text.slice(last, i) });
    if (m[1]) {
      const info = byPrefix(m[1]);
      out.push({ kind: 'token', token: m[0], name: info?.name ?? m[1], cls: info?.cls ?? 'other', short: m[2].slice(0, 4) });
    } else {
      out.push({ kind: 'cite', docId: m[3] });
    }
    last = i + m[0].length;
  }
  if (last < text.length) out.push({ kind: 'text', text: text.slice(last) });
  return out;
}

// ---------- 검수 화면: 서버 위치는 파이썬 글자(코드포인트) 단위, 브라우저는 UTF-16 ----------

/** UTF-16 위치 → 코드포인트 위치 */
export function toCodePoint(text: string, utf16: number): number {
  return Array.from(text.slice(0, utf16)).length;
}

/** 코드포인트 위치로 자르기 */
export function cpSlice(text: string, start: number, end: number): string {
  return Array.from(text).slice(start, end).join('');
}
