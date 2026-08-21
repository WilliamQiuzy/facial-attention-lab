import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { verifyApprovedAssetFiles } from './build/approvedAssets'

const approvedAssetPaths = verifyApprovedAssetFiles()

const verifyApprovedAssetsPlugin = {
  name: 'verify-approved-synthetic-assets',
  enforce: 'pre' as const,
  buildStart() {
    verifyApprovedAssetFiles()
  },
}

export default defineConfig({
  plugins: [verifyApprovedAssetsPlugin, react()],
  server: {
    fs: {
      allow: [
        path.resolve(process.cwd()),
        ...new Set(
          approvedAssetPaths.map(
            (filePath) => path.dirname(filePath),
          ),
        ),
      ],
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test/setup.ts',
    css: true,
  },
})
