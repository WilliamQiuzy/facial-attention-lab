import { Link } from 'react-router-dom'

export function OverviewPage() {
  return (
    <div className="workspace-page help-page">
      <header className="workspace-page__header page-shell">
        <div>
          <h1>Help & research information</h1>
          <p>
            Use Patients for the session-only photo workflow. The links below are
            for research setup, verification, and troubleshooting.
          </p>
        </div>
      </header>

      <div className="help-page__content page-shell">
        <section aria-labelledby="daily-workflow-title">
          <h2 id="daily-workflow-title">Daily workflow</h2>
          <ol>
            <li>
              Open <Link to="/patients">Patients</Link> and create or select a
              synthetic/test record.
            </li>
            <li>
              Add a photo visit, confirm image quality, then run the simulated
              analysis.
            </li>
            <li>
              Review the original image, overlay, density field, and facial-area
              summary on one scrolling page.
            </li>
            <li>
              Use <Link to="/reviews">Reviews</Link> for patient-workflow results
              awaiting review.
            </li>
          </ol>
        </section>

        <section className="help-page__advanced">
          <details>
            <summary>Advanced research tools</summary>
            <div className="help-page__advanced-content">
              <p>
                Technical pages for research setup, verification, and troubleshooting.
                These are not part of the clinician workflow.
              </p>
              <nav className="help-page__links" aria-label="Research tools">
                <Link to="/cases">Synthetic cases</Link>
                <Link to="/research/reviews">Research reviews</Link>
                <Link to="/runs">Runs</Link>
                <Link to="/jobs">Jobs</Link>
                <Link to="/models">Models</Link>
                <Link to="/methods">Methods</Link>
                <Link to="/integration">Integration</Link>
              </nav>
            </div>
          </details>
        </section>

        <p className="help-page__boundary">
          Research prototype for synthetic/test records only. Session data resets on
          refresh. It is not clinical guidance and clinical use is blocked.
        </p>
      </div>
    </div>
  )
}
