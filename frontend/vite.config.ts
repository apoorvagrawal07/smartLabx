import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const proxy = Object.fromEntries(
  ['/chat', '/memories', '/generate', '/openapi.json'].map((route) => [route, {
    target: process.env.SMARTLAB_API_URL || 'http://127.0.0.1:8000',
    changeOrigin: true,
    timeout: 240_000,
    proxyTimeout: 240_000,
  }]),
);

export default defineConfig({
  plugins: [react()],
  server: { proxy },
  preview: { proxy },
});