/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_FACIAL_PARALYSIS_API_URL?: string
  readonly VITE_ENABLE_DEMONSTRATION?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
