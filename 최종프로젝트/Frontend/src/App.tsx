// ============================================
// App - 라우팅만 잡아둔 상태. 화면(페이지)은 디자인이 나오면 features/*/pages 에 추가
// ============================================

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { AuthProvider } from '@/shared/contexts/AuthContext';

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* TODO(디자인 후): /auth, /chat, /review, /admin, /profile
              보호 페이지는 <ProtectedRoute>, 관리자 페이지는 <ProtectedRoute role="ADMIN"> 로 감쌈 */}
          <Route path="*" element={null} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
