// 토큰 `<PERSON_…>` 은 라벨 색 칩, `[H0006]` 은 근거 문서 링크로 보여 줌
import { Lock } from 'lucide-react';
import { splitAnswer } from '@/shared/utils/pii';

interface Props {
  text: string;
  onCite?: (docId: string) => void;
  denied?: string[]; // 실명 보기 뒤에도 권한이 없어 남은 토큰
}

export default function MaskedText({ text, onCite, denied }: Props) {
  return (
    <>
      {splitAnswer(text).map((p, i) => {
        if (p.kind === 'text') return <span key={i}>{p.text}</span>;
        if (p.kind === 'cite') {
          return (
            <button key={i} className="cite" onClick={() => onCite?.(p.docId)} title="근거 문서 보기">
              {p.docId}
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
