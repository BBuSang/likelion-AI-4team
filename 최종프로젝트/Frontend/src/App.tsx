// 라우팅: 로그인 쪽(메뉴 없음) / 로그인 뒤(AppLayout: 왼쪽 메뉴 + 화면). 관리자 화면은 <ProtectedRoute admin>
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { AuthProvider } from '@/shared/contexts/AuthContext';
import ProtectedRoute from '@/shared/components/ProtectedRoute';
import AppLayout from '@/shared/components/AppLayout';
import LoginPage from '@/features/auth/pages/LoginPage';
import SignupPage from '@/features/auth/pages/SignupPage';
import ChatPage from '@/features/chat/pages/ChatPage';
import AdminPage from '@/features/admin/pages/AdminPage';
import IngestPage from '@/features/review/pages/IngestPage';
import ReviewPage from '@/features/review/pages/ReviewPage';
import StatsPage from '@/features/stats/pages/StatsPage';
import MePage from '@/features/profile/pages/MePage';

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<SignupPage />} />
          <Route element={<ProtectedRoute><AppLayout /></ProtectedRoute>}>
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/me" element={<MePage />} />
            <Route path="/admin" element={<ProtectedRoute admin><AdminPage /></ProtectedRoute>} />
            <Route path="/ingest" element={<ProtectedRoute admin><IngestPage /></ProtectedRoute>} />
            <Route path="/ingest/docs/:id" element={<ProtectedRoute admin><ReviewPage /></ProtectedRoute>} />
            <Route path="/stats" element={<ProtectedRoute admin><StatsPage /></ProtectedRoute>} />
          </Route>
          <Route path="*" element={<Navigate to="/chat" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
