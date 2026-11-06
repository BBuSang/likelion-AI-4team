import { useState } from 'react';
import type { FormEvent } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { ShieldCheck } from 'lucide-react';
import axios from 'axios';
import { useAuth } from '@/shared/contexts/AuthContext';
import { errMsg } from '@/shared/utils/error';
import '@/shared/components/layout.css';

export default function LoginPage() {
  const { login } = useAuth();
  const nav = useNavigate();
  const from = (useLocation().state as { from?: { pathname: string } } | null)?.from?.pathname ?? '/chat';
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [rememberMe, setRememberMe] = useState(false);
  const [error, setError] = useState<{ kind: 'warn' | 'bad'; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!email.trim() || !password) {
      setError({ kind: 'bad', text: '이메일과 비밀번호를 입력해 주세요.' });
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await login({ email, password, rememberMe });
      nav(from, { replace: true });
    } catch (err) {
      // 403 = 승인 대기·정지 (비밀번호는 맞음), 401 = 이메일·비밀번호 틀림
      const pending = axios.isAxiosError(err) && err.response?.status === 403;
      setError({ kind: pending ? 'warn' : 'bad', text: errMsg(err) });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="auth">
      <div className="auth-split">
        <div className="auth-hero">
          <div className="row" style={{ fontWeight: 600 }}>
            <span className="brand-mark" style={{ background: '#fff', color: 'var(--brand)' }}>S</span> 새봄
          </div>
          <div className="spacer" />
          <h1>개인정보는 가리고<br />업무 답은 정확하게</h1>
          <p style={{ color: '#cfdcff', marginTop: 10 }}>사내 문서 검색 도우미. 답변 속 개인정보는 가려진 채로 보이고, 권한이 있는 문서만 실명으로 볼 수 있어요.</p>
          <div className="row small" style={{ marginTop: 22, color: '#cfdcff' }}>
            <ShieldCheck size={16} aria-hidden /> 모든 실명 보기는 기록돼요
          </div>
        </div>
        <form className="panel auth-card" onSubmit={submit}>
          <h1 style={{ marginBottom: 20 }}>로그인</h1>
          <label className="field"><span>이메일</span>
            <input className="input" type="email" autoComplete="username" placeholder="name@company.com"
                   value={email} onChange={(e) => { setEmail(e.target.value); setError(null); }} />
          </label>
          <label className="field"><span>비밀번호</span>
            <input className="input" type="password" autoComplete="current-password" placeholder="8자 이상"
                   value={password} onChange={(e) => { setPassword(e.target.value); setError(null); }} />
          </label>
          <label className="row small muted" style={{ marginBottom: 16 }}>
            <input type="checkbox" checked={rememberMe} onChange={(e) => setRememberMe(e.target.checked)} /> 로그인 유지 (24시간)
          </label>
          <button className="btn btn-primary btn-block" disabled={busy}>{busy ? '확인 중…' : '로그인'}</button>
          {error && <div className={`alert alert-${error.kind}`} style={{ marginTop: 12 }}>{error.text}</div>}
          <p className="small muted" style={{ marginTop: 18, textAlign: 'center' }}>
            계정이 없나요? <Link to="/signup">가입 신청</Link>
          </p>
          <p className="small muted" style={{ marginTop: 6, textAlign: 'center' }}>
            비밀번호를 잊었다면 관리자에게 문의해 주세요.
          </p>
        </form>
      </div>
    </div>
  );
}
