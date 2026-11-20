// 답변 마크다운 → 화면 (worklog 56장). LLM 답변이 ### 제목·**굵게**·목록·표로 와서 그대로 보이던 것을 꾸며 보여 줌.
// 라이브러리를 안 쓰는 이유: 토큰 `<PERSON_…>` 이 HTML 태그처럼 보여 마크다운 라이브러리가 지워 버림.
// 안전: React 요소로만 만든다 (HTML 그대로 넣기·링크 만들기 없음). 글 조각 안의 토큰·[근거] 는 MaskedText 가 칩·버튼으로.
import type { ReactNode } from 'react';
import MaskedText from './MaskedText';

interface Props {
  text: string;
  onCite?: (docId: string) => void;
  denied?: string[];
}

type Block =
  | { kind: 'h'; level: number; text: string }
  | { kind: 'p'; lines: string[] }
  | { kind: 'list'; items: ListItem[]; ordered: boolean; start: number }
  | { kind: 'quote'; lines: string[] }
  | { kind: 'code'; text: string }
  | { kind: 'table'; head: string[]; rows: string[][] }
  | { kind: 'hr' };

interface ListItem { text: string; children: ListItem[]; ordered: boolean }

const LIST_RE = /^(\s*)([-*•+]|\d{1,3}[.)])\s+(.*)$/;
const isRow = (l: string) => /^\s*\|.*\|\s*$/.test(l);
const isSep = (l: string) => /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$/.test(l);
const cells = (l: string) => l.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());

/** 줄 → 블록 */
function parseBlocks(src: string): Block[] {
  const lines = src.replace(/\r\n?/g, '\n').split('\n');
  const out: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (/^\s*```/.test(line)) {
      const buf: string[] = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) buf.push(lines[i++]);
      i++;
      out.push({ kind: 'code', text: buf.join('\n') });
      continue;
    }
    const h = /^\s*(#{1,6})\s+(.*?)\s*#*\s*$/.exec(line);
    if (h) { out.push({ kind: 'h', level: h[1].length, text: h[2] }); i++; continue; }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { out.push({ kind: 'hr' }); i++; continue; }
    if (isRow(line) && i + 1 < lines.length && isSep(lines[i + 1])) {
      const head = cells(line);
      const rows: string[][] = [];
      i += 2;
      while (i < lines.length && isRow(lines[i])) rows.push(cells(lines[i++]));
      out.push({ kind: 'table', head, rows });
      continue;
    }
    if (/^\s*>/.test(line)) {
      const buf: string[] = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) buf.push(lines[i++].replace(/^\s*>\s?/, ''));
      out.push({ kind: 'quote', lines: buf });
      continue;
    }
    if (LIST_RE.test(line)) {
      const first = LIST_RE.exec(line)!;
      const ordered = /\d/.test(first[2]);
      const root: ListItem[] = [];
      const stack: { indent: number; items: ListItem[] }[] = [{ indent: first[1].length, items: root }];
      while (i < lines.length) {
        const m = LIST_RE.exec(lines[i]);
        if (!m) {
          // 목록 항목의 이어지는 줄 (들여쓴 일반 줄) → 앞 항목에 붙임
          if (lines[i].trim() && /^\s{2,}/.test(lines[i]) && stack[stack.length - 1].items.length) {
            const items = stack[stack.length - 1].items;
            items[items.length - 1].text += '\n' + lines[i].trim();
            i++;
            continue;
          }
          break;
        }
        const indent = m[1].length;
        const item: ListItem = { text: m[3], children: [], ordered: /\d/.test(m[2]) };
        while (stack.length > 1 && indent < stack[stack.length - 1].indent) stack.pop();
        const top = stack[stack.length - 1];
        if (indent > top.indent + 1 && top.items.length) {
          const parent = top.items[top.items.length - 1];
          parent.children.push(item);
          stack.push({ indent, items: parent.children });
        } else {
          top.items.push(item);
        }
        i++;
      }
      out.push({ kind: 'list', items: root, ordered, start: ordered ? parseInt(first[2], 10) || 1 : 1 });
      continue;
    }
    const buf: string[] = [];
    while (i < lines.length && lines[i].trim() && !LIST_RE.test(lines[i]) && !/^\s*(#{1,6}\s|>|```)/.test(lines[i])
           && !(isRow(lines[i]) && i + 1 < lines.length && isSep(lines[i + 1]))) {
      buf.push(lines[i++].trimEnd());
    }
    out.push({ kind: 'p', lines: buf });
  }
  return out;
}

const INLINE_RE = /`([^`]+)`|\*\*(.+?)\*\*|__(.+?)__|\*(?![\s*])(.+?)\*/g;

export default function AnswerMarkdown({ text, onCite, denied }: Props) {
  let key = 0;
  const k = () => key++;

  /** 글 한 줄 → 굵게·기울임·코드 + 토큰·근거 */
  const inline = (s: string): ReactNode[] => {
    const out: ReactNode[] = [];
    let last = 0;
    for (const m of s.matchAll(INLINE_RE)) {
      const at = m.index ?? 0;
      if (at > last) out.push(<MaskedText key={k()} text={s.slice(last, at)} onCite={onCite} denied={denied} />);
      if (m[1] !== undefined) out.push(<code key={k()}>{m[1]}</code>);
      else if (m[2] !== undefined || m[3] !== undefined) out.push(<strong key={k()}>{inline(m[2] ?? m[3])}</strong>);
      else out.push(<em key={k()}>{inline(m[4])}</em>);
      last = at + m[0].length;
    }
    if (last < s.length) out.push(<MaskedText key={k()} text={s.slice(last)} onCite={onCite} denied={denied} />);
    return out;
  };

  const withBreaks = (ls: string[]) => ls.flatMap((l, j) => (j ? [<br key={k()} />, ...inline(l)] : inline(l)));

  const list = (items: ListItem[], ordered: boolean, start = 1): ReactNode => {
    const body = items.map((it) => (
      <li key={k()}>
        {withBreaks(it.text.split('\n'))}
        {it.children.length > 0 && list(it.children, it.children[0].ordered)}
      </li>
    ));
    return ordered ? <ol key={k()} start={start}>{body}</ol> : <ul key={k()}>{body}</ul>;
  };

  return (
    <div className="md">
      {parseBlocks(text).map((b) => {
        switch (b.kind) {
          case 'h': {
            const Tag = (`h${Math.min(Math.max(b.level, 3), 5)}`) as 'h3' | 'h4' | 'h5'; // 답변 안 제목은 h3~h5 (화면 제목보다 작게)
            return <Tag key={k()}>{inline(b.text)}</Tag>;
          }
          case 'p': return <p key={k()}>{withBreaks(b.lines)}</p>;
          case 'list': return list(b.items, b.ordered, b.start);
          case 'quote': return <blockquote key={k()}>{withBreaks(b.lines)}</blockquote>;
          case 'code': return <pre key={k()}><code>{b.text}</code></pre>;
          case 'hr': return <hr key={k()} />;
          case 'table':
            return (
              <div key={k()} className="md-table">
                <table>
                  <thead><tr>{b.head.map((c) => <th key={k()}>{inline(c)}</th>)}</tr></thead>
                  <tbody>{b.rows.map((r) => <tr key={k()}>{b.head.map((_, j) => <td key={k()}>{inline(r[j] ?? '')}</td>)}</tr>)}</tbody>
                </table>
              </div>
            );
        }
      })}
    </div>
  );
}
