// 로그인·가입·비밀번호 재설정 화면 왼쪽 파란 카드 (worklog 43·49장)
// 큰 흰 로고(403, 0 이 자물쇠)를 위에 크게, 이름·설명은 아래
import { ShieldCheck } from 'lucide-react';
import { LogoLarge } from '@/shared/components/Logo';

export default function AuthHero() {
  return (
    <div className="auth-hero">
      <LogoLarge tone="white" width={190} className="hero-logo" />
      <div className="spacer" />
      <h1>개인정보는 가리고<br />업무 답은 정확하게</h1>
      <p style={{ color: '#cfdcff', marginTop: 10 }}>사내 문서 검색 도우미. 답변 속 개인정보는 가려진 채로 보이고, 권한이 있는 문서만 실명으로 볼 수 있어요.</p>
      <div className="row small" style={{ marginTop: 22, color: '#cfdcff' }}>
        <ShieldCheck size={16} aria-hidden /> 모든 실명 보기는 기록돼요
      </div>
    </div>
  );
}
