import { Link } from 'react-router-dom'

export function AppFooter() {
  return (
    <footer className="site-footer">
      <div className="site-footer__grid">
        <div>
          <p className="site-footer__brand">Facial Attention Lab</p>
          <p>
            An independent interface study for facial-scar attention research. No
            affiliation or endorsement is implied.
          </p>
        </div>
        <div>
          <p className="site-footer__heading">Research boundaries</p>
          <Link to="/methods">Methods & limitations</Link>
          <Link to="/model">Data provenance</Link>
        </div>
        <div>
          <p className="site-footer__heading">Current capability</p>
          <p>Synthetic images · Simulated maps · No patient data</p>
        </div>
      </div>
      <div className="site-footer__bottom">
        <span>Prototype build 0.1</span>
        <span>Research use only</span>
      </div>
    </footer>
  )
}
