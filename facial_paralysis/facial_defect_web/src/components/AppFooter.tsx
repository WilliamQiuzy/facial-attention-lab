import { Link } from 'react-router-dom'

export function AppFooter() {
  return (
    <footer className="site-footer workspace-footer">
      <div className="site-footer__grid">
        <p className="site-footer__brand">
          FaceAI · Research prototype
        </p>
        <nav aria-label="Resource navigation">
          <Link to="/about">Help</Link>
        </nav>
        <p className="site-footer__boundary">
          Sample data only · Session resets on refresh
        </p>
      </div>
    </footer>
  )
}
