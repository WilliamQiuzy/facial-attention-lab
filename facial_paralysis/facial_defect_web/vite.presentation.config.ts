import react from '@vitejs/plugin-react'
import path from 'node:path'
import { defineConfig } from 'vite'
import { viteSingleFile } from 'vite-plugin-singlefile'
import { verifyPresentationAssetFiles } from './build/presentationAssets'

const presentationAssetPaths = verifyPresentationAssetFiles()

export default defineConfig({
  publicDir: false,
  plugins: [
    {
      name: 'verify-presentation-synthetic-assets',
      enforce: 'pre',
      buildStart() {
        verifyPresentationAssetFiles()
      },
    },
    react(),
    viteSingleFile({ removeViteModuleLoader: true }),
  ],
  server: {
    fs: {
      allow: [
        path.resolve(process.cwd()),
        ...presentationAssetPaths.map((assetPath) => path.dirname(assetPath)),
      ],
    },
  },
  build: {
    assetsInlineLimit: Number.MAX_SAFE_INTEGER,
    emptyOutDir: true,
    modulePreload: false,
    outDir: 'presentation-dist',
    sourcemap: false,
    rollupOptions: {
      input: path.resolve(process.cwd(), 'presentation.html'),
    },
  },
})
