// 로그인 뒤 공통 틀: 왼쪽 메뉴 + 가운데 화면
// 메뉴는 권한별로 보임: 데이터 적재·통계·관리자는 관리자만 (실제 차단은 백엔드)
import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { BarChart3, LogOut, MessageCircle, Upload, UserRound, Users } from 'lucide-react';
import { useAuth } from '@/shared/contexts/AuthContext';
import { LogoMark, LogoRow } from './Logo';
import './layout.css';

const ROLE_NAME = { STAFF: '직원', LEAD: '팀장', AUDITOR: '감사' } as const;

export default function AppLayout() {
  const { user, logout } = useAuth();
  const nav = useNavigate();
  const items = [
    { to: '/chat', icon: MessageCircle, label: '채팅', show: true },
    { to: '/ingest', icon: Upload, label: '데이터 적재', show: !!user?.admin },
    { to: '/stats', icon: BarChart3, label: '통계', show: !!user?.admin },
    { to: '/admin', icon: Users, label: '관리자', show: !!user?.admin },
    { to: '/me', icon: UserRound, label: '마이페이지', show: true },
  ];
  return (
    <div className="app">
      <aside className="nav panel">
        <div className="brand">
          {/* 넓은 메뉴: 가로 로고 / 아이콘만 남는 좁은 메뉴: 자물쇠 마크 */}
          <span className="nav-label" title="사공삼(403)"><LogoRow height={26} /></span>
          <span className="nav-mark" title="사공삼(403)"><LogoMark /></span>
        </div>
        <nav className="nav-items">
          {items.filter((i) => i.show).map(({ to, icon: Icon, label }) => (
            <NavLink key={to} to={to} title={label} className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
              <Icon size={17} aria-hidden /> <span className="nav-label">{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="spacer" />
        {user && (
          <div className="me-box" title={`${user.name}${user.admin ? ' (관리자)' : ''} · ${user.team ?? '팀 없음'} · ${ROLE_NAME[user.role]}`}>
            <div className="avatar">{user.name.slice(0, 1)}</div>
            <div className="nav-label" style={{ minWidth: 0 }}>
              {/* 이름(+관리자) / 팀 / 역할을 한 줄씩 — 길어도 다음 줄로 밀리지 않고 말줄임 */}
              <p className="me-name"><span className="me-line">{user.name}</span>{user.admin && <span className="me-admin">관리자</span>}</p>
              <p className="small muted me-line">{user.team ?? '팀 없음'}</p>
              <p className="small muted me-line">{ROLE_NAME[user.role]}</p>
            </div>
          </div>
        )}
        {/* <Link to="/me" className="nav-item" title="설정"><Settings2 size={17} aria-hidden /> <span className="nav-label">설정</span></Link> */}
        <button className="nav-item" title="로그아웃" onClick={() => { logout(); nav('/login'); }}>
          <LogOut size={17} aria-hidden /> <span className="nav-label">로그아웃</span>
        </button>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
