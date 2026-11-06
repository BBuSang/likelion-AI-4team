// 로그인 안 했으면 /auth 로. admin 을 주면 관리자만, roles 를 주면 그 역할만 들어감
// 화면에서 막는 것은 편의일 뿐이고, 실제 차단은 백엔드가 함
import type { ReactNode } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '@/shared/contexts/AuthContext';
import type { Role } from '@/shared/types/user.types';

interface Props {
  children: ReactNode;
  admin?: boolean;
  roles?: Role[];
}

const ProtectedRoute = ({ children, admin, roles }: Props) => {
  const { user, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) return null;
  if (!user) return <Navigate to="/auth" state={{ from: location }} replace />;
  if (admin && !user.admin) return <Navigate to="/" replace />;
  if (roles && !roles.includes(user.role)) return <Navigate to="/" replace />;
  return <>{children}</>;
};

export default ProtectedRoute;
