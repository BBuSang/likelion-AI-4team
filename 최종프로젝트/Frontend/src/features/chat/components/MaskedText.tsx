// 토큰 `<PERSON_…>` 은 라벨 색 칩, `[H0006]` 은 근거 문서 링크로 보여 줌
import type { ReactNode } from 'react';
import { Lock } from 'lucide-react';
import { docLabel, splitAnswer } from '@/shared/utils/pii';

interface Props {
  text: string;
  onCite?: (docId: string) => void;
  denied?: string[]; // 실명 보기 뒤에도 권한이 없어 남은 토큰
  labels?: boolean; // 근거 문서: 줄 머리 소제목(질의요지:·회답:·이유: …)을 굵게
}

// 회신 사례 근거 문서의 소제목 (noaction search.case_block·exmnt_block, agents/agent._single_evidence)
const LABEL_RE = /(^|\n)(제목|질의요지|건의내용|회답|이유|당국 검토):/g;

function withLabels(text: string, key: number, lineStart: boolean) {
  const parts: ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(LABEL_RE)) {
    if (!m[1] && !lineStart) continue; // 토큰·인용 바로 뒤 조각의 맨 앞은 줄 머리가 아님
    const start = (m.index ?? 0) + m[1].length;
    if (start > last) parts.push(text.slice(last, start));
    parts.push(<b key={start}>{m[2]}:</b>);
    last = start + m[2].length + 1;
  }
  if (last < text.length) parts.push(text.slice(last));
  return <span key={key}>{parts}</span>;
}

export default function MaskedText({ text, onCite, denied, labels }: Props) {
  return (
    <>
      {splitAnswer(text).map((p, i) => {
        if (p.kind === 'text') return labels ? withLabels(p.text, i, i === 0) : <span key={i}>{p.text}</span>;
        if (p.kind === 'cite') {
          return (
            <button key={i} className="cite" onClick={() => onCite?.(p.docId)} title={`근거 문서 보기 (${p.docId})`}>
              {docLabel(p.docId)}
            </button>
          );
        }
        const isDenied = denied?.includes(p.token);
        return (
          <span key={i} className={`pii pii-${p.cls}`} title={isDenied ? '권한이 없어 가린 채로 둬요' : '가명처리된 개인정보'}>
            {p.name} · {p.short}{isDenied && <Lock size={11} style={{ marginLeft: 3, verticalAlign: -1 }} aria-label="권한 없음" />}
          </span>
        );
      })}
    </>
  );
}
