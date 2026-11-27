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
  // [문서 ID]: 사내 H0071 · 공개 RAG LAW-1/PRD-1/CASE-1 · 비조치의견서 L3563/O45/F7
  const re = /<([A-Z_]+)_([0-9A-Za-z]+)>|\[((?:LAW|PRD|CASE)-\d+|[A-Z]\d+[A-Za-z0-9-]*)\]/g;
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

// ---------- 질문에 쓴 이름 중 데이터에 없는 사람: 가리지 않고 친 말 그대로 ----------

const escapeRe = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** 가명처리된 질문(masked)과 친 질문 원문(raw)을 맞대어, tokens 자리에 원문에서 쓴 말을 찾음 → {토큰: 친 말}.
 *  서버가 '예린사원' → '예린 사원'처럼 띄어쓰기를 바꾸므로 공백은 느슨하게 봄. 못 맞추면 {} (전처럼 가린 채로) */
export function revealTokens(raw: string, masked: string, tokens: string[]): Record<string, string> {
  if (!tokens.length || !raw || !masked) return {};
  const parts = masked.split(/(<[A-Z_]+(?:_[0-9A-Za-z]+)?>)/); // 홀수 칸 = 토큰 (<MASKED> 포함)
  const order: string[] = [];
  const src = parts.map((p, i) => {
    if (i % 2 === 1) { order.push(p); return '(.+?)'; }
    return p.split(/\s+/).map(escapeRe).join('\\s*');
  }).join('');
  const m = new RegExp(`^\\s*${src}\\s*$`, 's').exec(raw);
  if (!m) return {};
  const out: Record<string, string> = {};
  order.forEach((t, i) => {
    const v = m[i + 1].trim();
    if (tokens.includes(t) && v && !(t in out)) out[t] = v;
  });
  return out;
}

/** 글 속 토큰을 reveal 의 말로 바꿈 (화면 표시용) */
export function applyReveal(text: string, reveal?: Record<string, string>): string {
  if (!reveal || !Object.keys(reveal).length) return text;
  return text.replace(/<[A-Z_]+_[0-9A-Za-z]+>/g, (t) => reveal[t] ?? t);
}

// 회신 사례 번호 화면 이름: 안쪽 번호(L·O·F + 원본 사이트 번호)는 그대로 두고 보이는 글만 바꿈
// L = 법령해석(lawreqIdx), O = 비조치의견서(opinionIdx), F = 현장건의(checkplaceNo)
const CASE_KIND: Record<string, string> = { L: '법령해석', O: '비조치', F: '현장건의' };

/** 문서 ID → 화면에 보일 이름 (L5453 → "법령해석 5453"). 그 밖의 ID(H0071·LAW-1 …)는 그대로 */
export function docLabel(docId: string): string {
  const m = /^([LOF])(\d+)$/.exec(docId);
  return m ? `${CASE_KIND[m[1]]} ${m[2]}` : docId;
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
