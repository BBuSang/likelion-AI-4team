import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    host: true, // 같은 와이파이의 다른 기기에서도 접속 (http://<이 PC IP>:5173)
    // 화면이 /api 를 부르면 이 PC 의 스프링으로 넘김 → 다른 기기에서도 백엔드 주소를 몰라도 되고 CORS 도 필요 없음
    proxy: {
      '/api': { target: 'http://localhost:8080', changeOrigin: false },
    },
  },
})
