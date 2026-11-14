// 팀장·감사 역할의 실명 보기: 사유 입력 (백엔드·AI 도 사유 없으면 거절, worklog 25-9)
import { useState } from 'react';

interface Props {
  onCancel: () => void;
  onConfirm: (purpose: string) => void;
}

export default function RestoreModal({ onCancel, onConfirm }: Props) {
  const [purpose, setPurpose] = useState('');
  const [error, setError] = useState('');
  return (
    <div className="modal-back" onClick={onCancel}>
      <div className="card modal" onClick={(e) => e.stopPropagation()}>
        <h2>실명 보기 사유</h2>
        <p className="muted small" style={{ margin: '4px 0 14px' }}>팀장·감사 역할은 사유를 남겨야 실명을 볼 수 있어요. 사유는 복원 기록에 저장돼요.</p>
        <textarea className="textarea" value={purpose} placeholder="민원 2026-118 처리 적정성 감사"
                  onChange={(e) => { setPurpose(e.target.value); setError(''); }} maxLength={500} />
        {error && <p className="small" style={{ color: 'var(--bad)', marginTop: 6 }}>{error}</p>}
        <div className="row" style={{ marginTop: 14, justifyContent: 'flex-end' }}>
          <button className="btn" onClick={onCancel}>취소</button>
          <button className="btn btn-primary" onClick={() => (purpose.trim() ? onConfirm(purpose.trim()) : setError('사유를 입력해 주세요.'))}>
            실명 보기
          </button>
        </div>
      </div>
    </div>
  );
}
