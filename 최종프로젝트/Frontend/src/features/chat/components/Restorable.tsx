// 가린 글 + "실명 보기" 버튼. 실명은 이 컴포넌트 상태에만 두고 저장하지 않음 (다시 가리면 버림)
import { useState } from 'react';
import { Eye, EyeOff } from 'lucide-react';
import { useAuth } from '@/shared/contexts/AuthContext';
import { errMsg } from '@/shared/utils/error';
import * as chatApi from '../api/chat.api';
import type { RestoreResult } from '../api/chat.api';
import MaskedText from './MaskedText';
import RestoreModal from './RestoreModal';

interface Props {
  text: string;
  docIds: string[];
  onCite?: (docId: string) => void;
  compact?: boolean;
}

export default function Restorable({ text, docIds, onCite, compact }: Props) {
  const { user } = useAuth();
  const [result, setResult] = useState<RestoreResult | null>(null);
  const [asking, setAsking] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const hasToken = /<[A-Z_]+_[0-9A-Za-z]+>/.test(text);

  const run = async (purpose?: string) => {
    setAsking(false);
    setBusy(true);
    setError('');
    try {
      setResult(await chatApi.restore(text, docIds, purpose));
    } catch (e) {
      setError(errMsg(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div style={{ whiteSpace: 'pre-wrap', fontSize: compact ? 12 : 14, lineHeight: 1.8 }}>
        <MaskedText text={result ? result.text : text} onCite={onCite} denied={result?.denied} />
      </div>
      {hasToken && (
        <div className="row" style={{ marginTop: 8, justifyContent: 'flex-end', gap: 8 }}>
          {result && (
            <span className="small muted">
              {result.restored}개 복원{result.denied.length ? ` · ${result.denied.length}개는 권한이 없어 가린 채로` : ''}
            </span>
          )}
          {result ? (
            <button className="btn btn-sm" onClick={() => setResult(null)}><EyeOff size={14} aria-hidden /> 다시 가리기</button>
          ) : (
            <button className="btn btn-sm btn-outline" disabled={busy}
                    onClick={() => (user?.role === 'AUDITOR' || user?.role === 'LEAD' ? setAsking(true) : run())}>
              <Eye size={14} aria-hidden /> {busy ? '확인 중…' : '실명 보기'}
            </button>
          )}
        </div>
      )}
      {error && <div className="alert alert-bad" style={{ marginTop: 8 }}>{error}</div>}
      {asking && <RestoreModal onCancel={() => setAsking(false)} onConfirm={run} />}
    </div>
  );
}
