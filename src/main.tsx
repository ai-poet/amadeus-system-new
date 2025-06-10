import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import router from './routes';
import './index.css'
import { ThemeProvider } from './components/themeProvider'
import { RouterProvider } from 'react-router-dom'
import { StoreProvider } from './store/storeProvider';
import './i18n';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <StoreProvider>
      <ThemeProvider defaultTheme="dark" storageKey="vite-ui-theme">
        <RouterProvider router={router} />
      </ThemeProvider>
    </StoreProvider>
  </StrictMode>,
)
