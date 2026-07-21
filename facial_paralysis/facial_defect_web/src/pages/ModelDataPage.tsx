import { ArrowDown, Ban, Braces, Check, Database, LockKeyhole, ServerCog } from 'lucide-react'

const sources = [
  {
    name: 'mock_simulation',
    state: 'Active now',
    detail: 'Deterministic interface fixtures. Not human gaze and not model output.',
  },
  {
    name: 'observed_gaze',
    state: 'Future protocol',
    detail: 'Aggregated participant gaze after consent, calibration, QC, and cohort freeze.',
  },
  {
    name: 'model_prediction',
    state: 'Future validation',
    detail: 'Versioned salience output evaluated against held-out observed gaze.',
  },
]

export function ModelDataPage() {
  return (
    <div className="model-page">
      <header className="model-hero">
        <div className="page-shell model-hero__grid">
          <div>
            <p className="eyebrow">Integration boundary</p>
            <h1>Model & data readiness</h1>
            <p className="lede">
              A typed, fail-closed seam for connecting research evidence later—without
              allowing the interface to silently promote simulated data into a model or
              clinical claim.
            </p>
          </div>
          <div className="connection-status">
            <span className="connection-status__light" aria-hidden="true" />
            <div>
              <p>Runtime state</p>
              <strong>Disconnected by design</strong>
              <span>Default build makes zero network requests.</span>
            </div>
          </div>
        </div>
      </header>

      <section className="page-shell model-section" aria-labelledby="source-title">
        <div className="section-heading section-heading--compact">
          <div>
            <p className="eyebrow">One required field</p>
            <h2 id="source-title">Every result declares its origin.</h2>
          </div>
          <p>
            The application treats simulation, empirical gaze, and model inference as three
            different evidence classes. The UI cannot substitute one for another.
          </p>
        </div>
        <div className="source-grid">
          {sources.map((source, index) => (
            <article key={source.name}>
              <span className="source-grid__index">0{index + 1}</span>
              <code>{source.name}</code>
              <strong>{source.state}</strong>
              <p>{source.detail}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="integration-band">
        <div className="page-shell">
          <div className="integration-heading">
            <div>
              <p className="eyebrow">Future runtime</p>
              <h2>From approved inputs to reviewable evidence</h2>
            </div>
            <div className="model-version-card">
              <p>Model version</p>
              <strong>Not connected</strong>
              <span>Required before any prediction request</span>
            </div>
          </div>

          <div className="integration-flow" aria-label="Future model integration flow">
            <article>
              <Database aria-hidden="true" /><span>01</span><h3>Approved input</h3>
              <p>Two approved image assets, hashes, pairing state, and study version.</p>
            </article>
            <ArrowDown className="integration-flow__arrow" aria-hidden="true" />
            <article>
              <ServerCog aria-hidden="true" /><span>02</span><h3>Versioned service</h3>
              <p>Separate observed-gaze and salience-prediction operations.</p>
            </article>
            <ArrowDown className="integration-flow__arrow" aria-hidden="true" />
            <article>
              <Braces aria-hidden="true" /><span>03</span><h3>Validated envelope</h3>
              <p>Origin, capability, uncertainty, quality, and provenance are mandatory.</p>
            </article>
            <ArrowDown className="integration-flow__arrow" aria-hidden="true" />
            <article>
              <LockKeyhole aria-hidden="true" /><span>04</span><h3>Human review</h3>
              <p>QC failure returns no metric. The interface never fills gaps with guesses.</p>
            </article>
          </div>
        </div>
      </section>

      <section className="page-shell endpoint-section" aria-labelledby="endpoints-title">
        <div>
          <p className="eyebrow">Typed endpoints</p>
          <h2 id="endpoints-title">Observed and predicted stay separate.</h2>
        </div>
        <div className="endpoint-list">
          <article><span>Observed aggregate</span><code>/api/v1/attention-analyses</code><p>Expected origin: observed_gaze</p></article>
          <article><span>Model inference</span><code>/api/v1/salience-predictions</code><p>Expected origin: model_prediction</p></article>
        </div>
      </section>

      <section className="requirements-section">
        <div className="page-shell requirements-grid">
          <div>
            <p className="eyebrow">Promotion checklist</p>
            <h2>What must be true before connection</h2>
          </div>
          <div className="requirements-list">
            <article><Check aria-hidden="true" /><div><h3>Input contract</h3><p>Two approved image assets, immutable hashes, and explicit pairing state.</p></div></article>
            <article><Ban aria-hidden="true" /><div><h3>ROI review</h3><p>Unreviewed or low-confidence regions fail closed and return no metric.</p></div></article>
            <article><Ban aria-hidden="true" /><div><h3>Minimum eligible sample</h3><p>Observed-gaze aggregates require the protocol-defined QC threshold.</p></div></article>
            <article><Ban aria-hidden="true" /><div><h3>Validation & governance</h3><p>Held-out validation, subgroup review, privacy, and institutional approval.</p></div></article>
          </div>
        </div>
      </section>
    </div>
  )
}
