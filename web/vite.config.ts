import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://127.0.0.1:8765' } },
  build: { outDir: 'dist', emptyOutDir: true, chunkSizeWarningLimit: 900 },
  test: { environment: 'jsdom', setupFiles: './tests/setup.ts', include: ['tests/**/*.test.{ts,tsx}'] },
});
