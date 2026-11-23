// 되돌리기 어려운 동작 전에 묻는 확인 대화상자 (모양은 RestoreModal 과 같음)
// - Escape·바깥 클릭 = 취소, 처음 포커스는 취소 버튼 (Enter 를 눌러도 실수로 실행되지 않게)
// - reason 을 주면 사유 입력칸이 생기고, 비어 있으면 확인 버튼이 막힘
import { useEffect, useId, useRef, useState } from 'react';
import type { ReactNode } from 'react';

interface Props {
  title: string;
  message: ReactNode;          // 누구에게 어떤 일이 생기는지
  confirmLabel: string;        // 예: '정지', '지우기'
  danger?: boolean;            // 확인 버튼을 빨간 테두리로
  reason?: { label: string; placeholder?: string; max?: number };
  busy?: boolean;
  onCancel: () => void;
  onConfirm: (reason: string) => void;
}

export default function ConfirmDialog({ title, message, confirmLabel, danger, reason, busy, onCancel, onConfirm }: Props) {
  const [text, setText] = useState('');
  const cancelRef = useRef<HTMLButtonElement>(null);
  const titleId = useId();
  const max = reason?.max ?? 500;
  const blocked = busy || (!!reason && !text.trim());

  useEffect(() => { cancelRef.current?.focus(); }, []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') { e.stopPropagation(); onCancel(); } };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onCancel]);

  return (
    <div className="modal-back" onClick={onCancel}>
      <div className="card modal" role="alertdialog" aria-modal="true" aria-labelledby={titleId} onClick={(e) => e.stopPropagation()}>
        <h2 id={titleId}>{title}</h2>
        <div className="muted small" style={{ margin: '4px 0 14px', lineHeight: 1.6 }}>{message}</div>
        {reason && (
          <label className="field" style={{ margin: 0 }}>
            <span>{reason.label}</span>
            <textarea className="textarea" value={text} placeholder={reason.placeholder} maxLength={max}
                      onChange={(e) => setText(e.target.value)} />
            <span className="small muted" style={{ textAlign: 'right' }}>{text.length}/{max}</span>
          </label>
        )}
        <div className="row" style={{ marginTop: 14, justifyContent: 'flex-end' }}>
          <button ref={cancelRef} className="btn" onClick={onCancel}>취소</button>
          <button className={`btn ${danger ? 'btn-danger' : 'btn-primary'}`} disabled={blocked} onClick={() => onConfirm(text.trim())}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
