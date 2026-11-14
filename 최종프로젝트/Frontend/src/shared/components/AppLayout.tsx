// 로그인 뒤 공통 틀: 왼쪽 메뉴 + 가운데 화면 (worklog 14장)
// 메뉴는 권한별로 보임: 데이터 적재·통계·관리자는 관리자만 (실제 차단은 백엔드)
import { Link, NavLink, Outlet, useNavigate } from 'react-router-dom';
import { BarChart3, LogOut, MessageCircle, Settings2, Upload, UserRound, Users } from 'lucide-react';
import { useAuth } from '@/shared/contexts/AuthContext';
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
          <span className="brand-mark">S</span>
          <span>새봄</span>
        </div>
        <nav className="nav-items">
          {items.filter((i) => i.show).map(({ to, icon: Icon, label }) => (
            <NavLink key={to} to={to} className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
              <Icon size={17} aria-hidden /> {label}
            </NavLink>
          ))}
        </nav>
        <div className="spacer" />
        {user && (
          <div className="me-box">
            <div className="avatar">{user.name.slice(0, 1)}</div>
            <div style={{ minWidth: 0 }}>
              <p className="me-name">{user.name}</p>
              <p className="small muted">{user.team ?? '팀 없음'} · {ROLE_NAME[user.role]}{user.admin ? ' · 관리자' : ''}</p>
            </div>
          </div>
        )}
        <Link to="/me" className="nav-item"><Settings2 size={17} aria-hidden /> 설정</Link>
        <button className="nav-item" onClick={() => { logout(); nav('/login'); }}>
          <LogOut size={17} aria-hidden /> 로그아웃
        </button>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
