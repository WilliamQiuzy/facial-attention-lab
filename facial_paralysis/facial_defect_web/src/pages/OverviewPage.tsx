import { ArrowRight, Database, Eye, ShieldCheck, Stethoscope } from 'lucide-react'
import { Link } from 'react-router-dom'

export function OverviewPage() {
  return (
    <>
      <section className="hero">
        <div className="hero__content">
          <p className="eyebrow">Facial scar attention research</p>
          <h1>Make visual attention easier to discuss.</h1>
          <p className="hero__lede">
            A clinician-first workspace for exploring how attention maps could support
            careful conversations around facial reconstruction—without turning attention
            into a judgment about appearance.
          </p>
          <div className="button-row">
            <Link className="button button--primary" to="/analysis">
              Explore the synthetic demo <ArrowRight aria-hidden="true" size={18} />
            </Link>
            <Link className="button button--secondary" to="/methods">
              Review methods & limits
            </Link>
          </div>
          <p className="hero__boundary">
            Uses two AI-generated, unpaired faces and simulated UI metrics. No human gaze,
            model prediction, or patient record is shown.
          </p>
        </div>

        <div className="hero__visual" aria-label="Simulated attention-map interface preview">
          <div className="hero__orb hero__orb--one" />
          <div className="hero__orb hero__orb--two" />
          <div className="hero__preview-card">
            <span className="status-pill status-pill--dark">SIMULATED UI</span>
            <Eye aria-hidden="true" size={48} strokeWidth={1.2} />
            <p>Attention is a measurement target.</p>
            <strong>It is not a value judgment.</strong>
          </div>
        </div>
      </section>

      <section className="trust-row" aria-label="Current prototype safeguards">
        <div>
          <ShieldCheck aria-hidden="true" />
          <span>Zero patient data</span>
        </div>
        <div>
          <Database aria-hidden="true" />
          <span>Source shown on every result</span>
        </div>
        <div>
          <Stethoscope aria-hidden="true" />
          <span>Clinical use remains blocked</span>
        </div>
      </section>

      <section className="section page-shell" aria-labelledby="pathway-title">
        <div className="section-heading">
          <div>
            <p className="eyebrow">One workspace, two conversations</p>
            <h2 id="pathway-title">Designed for a careful handoff.</h2>
          </div>
          <p>
            The clinician view keeps provenance and quality gates close. The patient view
            translates the same demonstration into plain, non-diagnostic language.
          </p>
        </div>

        <div className="path-grid">
          <article className="path-card path-card--blue">
            <span className="path-card__index">01</span>
            <p className="eyebrow">Clinician & researcher</p>
            <h3>Inspect the evidence boundary.</h3>
            <p>
              Review synthetic assets, simulated maps, metric definitions, versioning, and
              every gate required before a future model can connect.
            </p>
            <Link to="/analysis">
              Open attention demo <ArrowRight aria-hidden="true" />
            </Link>
          </article>

          <article className="path-card path-card--light">
            <span className="path-card__index">02</span>
            <p className="eyebrow">Patient & family</p>
            <h3>Keep the meaning human.</h3>
            <p>
              See a plain-language explanation that separates where people look from how
              they feel, what they think, or whether a procedure succeeded.
            </p>
            <Link to="/patient-report">
              Preview patient explanation <ArrowRight aria-hidden="true" />
            </Link>
          </article>
        </div>
      </section>

      <section className="readiness-band">
        <div className="page-shell readiness-band__inner">
          <div>
            <p className="eyebrow">Built for what comes next</p>
            <h2>A stable carrier for your model—not a premature clinical claim.</h2>
          </div>
          <div className="readiness-list">
            <div>
              <span>Interface prototype</span>
              <strong className="status-text status-text--ready">Ready</strong>
            </div>
            <div>
              <span>Model connector</span>
              <strong className="status-text status-text--gated">Gated</strong>
            </div>
            <div>
              <span>Clinical decision use</span>
              <strong className="status-text status-text--blocked">Blocked</strong>
            </div>
          </div>
        </div>
      </section>
    </>
  )
}
