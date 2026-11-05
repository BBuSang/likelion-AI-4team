// 로그인 안 했으면 /auth 로. role 을 주면 그 권한이 있어야 들어감 (관리자 페이지 = role="ADMIN")
// 화면에서 막는 것은 편의일 뿐이고, 실제 차단은 백엔드가 함
import type { ReactNode } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/shared/contexts/AuthContext';
import type { Role } from '@/shared/types/user.types';

interface Props {
  children: ReactNode;
  role?: Role;
}

const ProtectedRoute = ({ children, role }: Props) => {
  const { user, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) return null;
  if (!user) return <Navigate to="/auth" state={{ from: location }} replace />;
  if (role && user.role !== role) return <Navigate to="/" replace />;
  return <>{children}</>;
};

export default ProtectedRoute;
