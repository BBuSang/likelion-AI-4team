// 이메일 인증 칸 (가입·비밀번호 재설정 공통)
// 이메일 + [코드 받기] → 코드 6자리 + [확인] → 안내 한 줄 + [다시 받기] → "인증을 마쳤어요"
import { useEffect, useState } from 'react';
import { MailCheck } from 'lucide-react';
import { errMsg } from '@/shared/utils/error';

type Props = {
  email: string;
  onEmailChange: (v: string) => void;
  verified: boolean;
  onVerified: (v: boolean) => void;
  send: (email: string) => Promise<unknown>;
  verify: (email: string, code: string) => Promise<unknown>;
  onError: (msg: string) => void;
  /** 코드를 보낸 뒤 보여 줄 한 줄 안내 */
  sentText?: string;
  autoFocus?: boolean;
};

const CODE_SEC = 600;
const RESEND_SEC = 60;

export default function EmailCodeField({
  email, onEmailChange, verified, onVerified, send, verify, onError,
  sentText = '코드를 보냈어요. 스팸함도 확인해 주세요.', autoFocus,
}: Props) {
  const [code, setCode] = useState('');
  const [sentAt, setSentAt] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    if (!sentAt || verified) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [sentAt, verified]);

  const elapsed = sentAt ? Math.floor((now - sentAt) / 1000) : 0;
  const resendIn = sentAt ? Math.max(0, RESEND_SEC - elapsed) : 0;
  const expiresIn = sentAt ? Math.max(0, CODE_SEC - elapsed) : 0;
  const mmss = (n: number) => `${Math.floor(n / 60)}:${String(n % 60).padStart(2, '0')}`;
  const emailOk = /^\S+@\S+\.\S+$/.test(email);
  const locked = verified || sentAt !== null;

  const doSend = async () => {
    if (!emailOk) return onError('이메일 형식을 확인해 주세요.');
    setBusy(true); onError('');
    try {
      await send(email);
      setSentAt(Date.now()); setNow(Date.now()); setCode('');
    } catch (err) {
      onError(errMsg(err));
    } finally {
      setBusy(false);
    }
  };

  const doVerify = async () => {
    if (!/^\d{6}$/.test(code)) return onError('메일로 받은 6자리 숫자를 입력해 주세요.');
    setBusy(true); onError('');
    try {
      await verify(email, code);
      onVerified(true);
    } catch (err) {
      onError(errMsg(err));
    } finally {
      setBusy(false);
    }
  };

  const change = () => { onVerified(false); setSentAt(null); setCode(''); onError(''); };

  return (
    <>
      <div className="field">
        <label htmlFor="ecf-email" className="field-label">이메일</label>
        <div className="row">
          <input id="ecf-email" className="input" style={{ flex: 1, minWidth: 0 }} type="email" autoComplete="username"
                 value={email} onChange={(e) => { onEmailChange(e.target.value); onError(''); }}
                 placeholder="이메일을 입력하세요" readOnly={locked} autoFocus={autoFocus}
                 onKeyDown={(e) => { if (e.key === 'Enter' && !locked) { e.preventDefault(); doSend(); } }} />
          {locked
            ? <button type="button" className="btn code-btn" onClick={change} disabled={busy}>변경</button>
            : <button type="button" className="btn btn-outline code-btn" onClick={doSend} disabled={busy || !emailOk}>코드 받기</button>}
        </div>
        {verified && <span className="code-done"><MailCheck size={14} aria-hidden /> 인증을 마쳤어요</span>}
      </div>

      {sentAt !== null && !verified && (
        <div className="field">
          <label htmlFor="ecf-code" className="field-label">
            인증 코드 <span className="muted">{expiresIn > 0 ? `· ${mmss(expiresIn)} 남음` : '· 만료됨, 다시 받아 주세요'}</span>
          </label>
          <div className="row">
            <input id="ecf-code" className="input code-input" inputMode="numeric" autoComplete="one-time-code" maxLength={6}
                   value={code} placeholder="6자리 숫자" autoFocus
                   onChange={(e) => { setCode(e.target.value.replace(/\D/g, '')); onError(''); }}
                   onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); doVerify(); } }} />
            <button type="button" className="btn btn-primary code-btn" onClick={doVerify}
                    disabled={busy || code.length !== 6 || expiresIn === 0}>확인</button>
          </div>
          <div className="row code-sent">
            <span className="code-sent-text" title={sentText}>{sentText}</span>
            <button type="button" className="btn btn-outline code-btn" onClick={doSend} disabled={busy || resendIn > 0}>
              {resendIn > 0 ? `${resendIn}초` : '다시 받기'}
            </button>
          </div>
        </div>
      )}
    </>
  );
}
