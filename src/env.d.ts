/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_APP_WS_URL: string
  // 添加其他环境变量...
}

interface ImportMeta {
  readonly env: ImportMetaEnv
} 