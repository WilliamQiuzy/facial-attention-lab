import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { EnvironmentStrip } from './components/EnvironmentStrip'
import { PresentationDemoPage } from './pages/PresentationDemoPage'
import './styles/tokens.css'
import './styles/global.css'
import './styles/workbench.css'
import './styles/presentation.css'

function PresentationStandaloneApp() {
  return (
    <div className="app-root">
      <a className="skip-link" href="#main-content">
        Skip to main content
      </a>
      <EnvironmentStrip />
      <header className="site-header presentation-standalone-header">
        <div className="site-header__inner">
          <span className="brand">
            <span className="brand__mark" aria-hidden="true">
              FR
            </span>
            <span className="brand__name">Facial Reconstruction Imaging</span>
          </span>
          <span className="presentation-standalone-header__label">
            Offline demo
          </span>
        </div>
      </header>
      <main id="main-content">
        <PresentationDemoPage />
      </main>
      <footer className="site-footer presentation-standalone-footer">
        <div className="site-footer__grid">
          <p className="site-footer__brand">
            Facial Reconstruction Imaging · Research prototype
          </p>
          <p className="site-footer__boundary">
            Synthetic presentation only · Clinical use blocked
          </p>
        </div>
      </footer>
    </div>
  )
}

const root = document.getElementById('root')
if (!root) throw new Error('Missing presentation root element.')

createRoot(root).render(
  <StrictMode>
    <PresentationStandaloneApp />
  </StrictMode>,
)
