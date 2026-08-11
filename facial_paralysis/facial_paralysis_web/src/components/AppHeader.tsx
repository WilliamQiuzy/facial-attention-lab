import { CircleHelp, FlaskConical } from 'lucide-react'

export function AppHeader() {
  return (
    <>
      <div className="research-strip">
        <div className="header-inner">
          <span><FlaskConical aria-hidden="true" size={14} /> Research use only</span>
          <span>FACES protocol · Source script v0.01</span>
        </div>
      </div>
      <header className="app-header">
        <div className="header-inner masthead">
          <a className="product-mark" href="#top" aria-label="FACES Research Capture home">
            <span className="motion-mark" aria-hidden="true"><i /><i /><i /></span>
            <span><strong>FACES</strong><small>Research Capture</small></span>
          </a>
          <nav aria-label="Primary navigation">
            <a href="#capture">Capture</a>
            <a href="#protocol">Protocol</a>
            <a href="#analysis">Analysis</a>
          </nav>
          <a className="help-link" href="#research-boundary"><CircleHelp aria-hidden="true" size={18} /> About this prototype</a>
        </div>
      </header>
    </>
  )
}
