/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_ENABLE_CONNECTED_MODE?: 'true' | 'false'
  readonly VITE_ATTENTION_API_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
