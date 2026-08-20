import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'node:path'
import { verifyApprovedAssetFiles } from './build/approvedAssets'
import { verifyPresentationAssetFiles } from './build/presentationAssets'

const approvedAssetPaths = verifyApprovedAssetFiles()
const presentationAssetPaths = verifyPresentationAssetFiles()

const verifyApprovedAssetsPlugin = {
  name: 'verify-approved-synthetic-assets',
  enforce: 'pre' as const,
  buildStart() {
    verifyApprovedAssetFiles()
    verifyPresentationAssetFiles()
  },
}

export default defineConfig({
  plugins: [verifyApprovedAssetsPlugin, react()],
  server: {
    fs: {
      allow: [
        path.resolve(process.cwd()),
        ...new Set(
          [...approvedAssetPaths, ...presentationAssetPaths].map(
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
