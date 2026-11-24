import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.tsx'
// 글꼴: Pretendard 가변 글꼴을 프로젝트에 포함 (화면에 쓰는 글자 묶음만 받아 옴)
import 'pretendard/dist/web/variable/pretendardvariable-dynamic-subset.css'
import './shared/styles/theme.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
