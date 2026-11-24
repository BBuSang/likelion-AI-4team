// 비밀번호 재설정: 이메일 인증 → 새 비밀번호 → 완료
import { useState } from 'react';
import type { FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { CheckCircle2 } from 'lucide-react';
import * as authApi from '../api/auth.api';
import EmailCodeField from '../components/EmailCodeField';
import AuthHero from '../components/AuthHero';
import { errMsg } from '@/shared/utils/error';
import '@/shared/components/layout.css';

export default function ResetPasswordPage() {
  const [email, setEmail] = useState('');
  const [verified, setVerified] = useState(false);
  const [pw, setPw] = useState({ a: '', b: '' });
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [done, setDone] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!verified) return setError('이메일 인증을 먼저 마쳐 주세요.');
    if (pw.a.length < 8) return setError('비밀번호는 8자 이상이에요.');
    if (pw.a !== pw.b) return setError('비밀번호가 서로 달라요.');
    setBusy(true);
    try {
      await authApi.resetPassword(email, pw.a);
      setDone(true);
    } catch (err) {
      setError(errMsg(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="auth">
      <div className="auth-split">
        <AuthHero />
        <div className="panel auth-card">
        {done ? (
          <div style={{ textAlign: 'center', padding: '10px 0' }}>
            <CheckCircle2 size={40} color="var(--brand)" aria-hidden />
            <h1 style={{ marginTop: 10 }}>비밀번호를 바꿨어요</h1>
            <p className="muted" style={{ margin: '8px 0 20px' }}>새 비밀번호로 로그인해 주세요.</p>
            <Link className="btn btn-primary btn-block" to="/login">로그인 화면으로</Link>
          </div>
        ) : (
          <form onSubmit={submit}>
            <h1>비밀번호 재설정</h1>
            <p className="muted small" style={{ margin: '4px 0 18px' }}>가입한 이메일로 받은 코드를 확인한 뒤 새 비밀번호를 정해요.</p>
            <EmailCodeField email={email} onEmailChange={setEmail} verified={verified} onVerified={setVerified}
                            onError={setError} send={authApi.sendResetCode} verify={authApi.verifyResetCode}
                            sentText="가입된 이메일이면 코드를 보냈어요." autoFocus />
            {verified && (
              <>
                <label className="field"><span>새 비밀번호</span>
                  <input className="input" type="password" autoComplete="new-password" placeholder="8자 이상" autoFocus
                         value={pw.a} onChange={(e) => { setPw({ ...pw, a: e.target.value }); setError(''); }} />
                </label>
                <label className="field"><span>새 비밀번호 확인</span>
                  <input className="input" type="password" autoComplete="new-password"
                         value={pw.b} onChange={(e) => { setPw({ ...pw, b: e.target.value }); setError(''); }} />
                </label>
                <button className="btn btn-primary btn-block" disabled={busy}>{busy ? '바꾸는 중…' : '비밀번호 바꾸기'}</button>
              </>
            )}
          </form>
        )}
        {error && <div className="alert alert-bad" style={{ marginTop: 12 }}>{error}</div>}
        {!done && <p className="small muted" style={{ marginTop: 16, textAlign: 'center' }}><Link to="/login">로그인으로 돌아가기</Link></p>}
        </div>
      </div>
    </div>
  );
}
