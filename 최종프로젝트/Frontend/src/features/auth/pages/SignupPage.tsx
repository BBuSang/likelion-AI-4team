// 가입 신청 3단계: ① 이메일·비밀번호 → ② 이름·팀·직급(선택칸) → ③ 승인 대기 안내
import { useEffect, useState } from 'react';
import type { FormEvent } from 'react';
import { Link } from 'react-router-dom';
import { CheckCircle2 } from 'lucide-react';
import * as authApi from '../api/auth.api';
import type { OptionsResponse } from '../types/auth.types';
import { errMsg } from '@/shared/utils/error';
import '@/shared/components/layout.css';

export default function SignupPage() {
  const [step, setStep] = useState(1);
  const [opts, setOpts] = useState<OptionsResponse>({ teams: [], positions: [] });
  const [f, setF] = useState({ email: '', password: '', password2: '', name: '', teamId: '', positionId: '' });
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const set = (k: keyof typeof f) => (e: { target: { value: string } }) => { setF({ ...f, [k]: e.target.value }); setError(''); };

  useEffect(() => {
    authApi.getOptions().then(setOpts).catch((e) => setError(errMsg(e, '팀·직급 목록을 불러오지 못했어요.')));
  }, []);

  const next = async (e: FormEvent) => {
    e.preventDefault();
    if (!/^\S+@\S+\.\S+$/.test(f.email)) return setError('이메일 형식을 확인해 주세요.');
    if (f.password.length < 8) return setError('비밀번호는 8자 이상이에요.');
    if (f.password !== f.password2) return setError('비밀번호가 서로 달라요.');
    setBusy(true);
    try {
      if (!(await authApi.checkEmail(f.email))) return setError('이미 가입된 이메일이에요.');
      setStep(2);
    } catch (err) {
      setError(errMsg(err));
    } finally {
      setBusy(false);
    }
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!f.name.trim()) return setError('이름을 입력해 주세요.');
    if (!f.teamId || !f.positionId) return setError('팀과 직급을 골라 주세요.');
    setBusy(true);
    try {
      await authApi.signup({ email: f.email, password: f.password, name: f.name,
                             teamId: Number(f.teamId), positionId: Number(f.positionId) });
      setStep(3);
    } catch (err) {
      setError(errMsg(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="auth">
      <div className="panel auth-single">
        <div className="steps">{[1, 2, 3].map((i) => <i key={i} className={i <= step ? 'on' : ''} />)}</div>
        {step === 1 && (
          <form onSubmit={next}>
            <h1>가입 신청</h1>
            <p className="muted small" style={{ margin: '4px 0 18px' }}>로그인에 쓸 이메일과 비밀번호</p>
            <label className="field"><span>이메일</span><input className="input" type="email" value={f.email} onChange={set('email')} placeholder="name@company.com" /></label>
            <label className="field"><span>비밀번호</span><input className="input" type="password" value={f.password} onChange={set('password')} placeholder="8자 이상" autoComplete="new-password" /></label>
            <label className="field"><span>비밀번호 확인</span><input className="input" type="password" value={f.password2} onChange={set('password2')} autoComplete="new-password" /></label>
            <button className="btn btn-primary btn-block" disabled={busy}>다음</button>
          </form>
        )}
        {step === 2 && (
          <form onSubmit={submit}>
            <h1>소속 정보</h1>
            <p className="muted small" style={{ margin: '4px 0 18px' }}>관리자가 확인한 뒤 사용할 수 있어요. 문서 속 직원과 이어 주는 데 써요.</p>
            <label className="field"><span>이름</span><input className="input" value={f.name} onChange={set('name')} placeholder="홍길동" /></label>
            <label className="field"><span>팀</span>
              <select className="select" value={f.teamId} onChange={set('teamId')}>
                <option value="">팀 고르기</option>
                {opts.teams.map((t) => <option key={t.id} value={t.id}>{t.name}</option>)}
              </select>
            </label>
            <label className="field"><span>직급</span>
              <select className="select" value={f.positionId} onChange={set('positionId')}>
                <option value="">직급 고르기</option>
                {opts.positions.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
            </label>
            {opts.teams.length === 0 && <p className="small muted" style={{ marginBottom: 10 }}>팀 목록이 비어 있어요. 관리자에게 문의해 주세요.</p>}
            <div className="row">
              <button type="button" className="btn" onClick={() => setStep(1)}>이전</button>
              <button className="btn btn-primary" style={{ flex: 1 }} disabled={busy}>{busy ? '보내는 중…' : '가입 신청'}</button>
            </div>
          </form>
        )}
        {step === 3 && (
          <div style={{ textAlign: 'center', padding: '10px 0' }}>
            <CheckCircle2 size={40} color="var(--brand)" aria-hidden />
            <h1 style={{ marginTop: 10 }}>신청했어요</h1>
            <p className="muted" style={{ margin: '8px 0 20px' }}>관리자가 승인하면 로그인할 수 있어요.</p>
            <Link className="btn btn-primary btn-block" to="/login">로그인 화면으로</Link>
          </div>
        )}
        {error && <div className="alert alert-bad" style={{ marginTop: 12 }}>{error}</div>}
        {step < 3 && <p className="small muted" style={{ marginTop: 16, textAlign: 'center' }}>이미 계정이 있나요? <Link to="/login">로그인</Link></p>}
      </div>
    </div>
  );
}
