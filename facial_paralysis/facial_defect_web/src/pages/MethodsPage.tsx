import { AlertTriangle, Eye, Fingerprint, LockKeyhole, Scale, ShieldCheck } from 'lucide-react'

const measures = [
  { title: 'Fixation duration', text: 'Total time that qualified fixations remain inside the reviewed scar region.' },
  { title: 'Fixation count', text: 'Number of qualified fixations that enter the reviewed scar region.' },
  { title: 'Time to first fixation', text: 'Elapsed time from image onset until the first qualified fixation enters the region.' },
  { title: 'Scar-region gaze share', text: 'Proportion of valid gaze time assigned to the reviewed region of interest.' },
]

export function MethodsPage() {
  return (
    <article className="methods-page">
      <header className="methods-hero">
        <div className="page-shell methods-hero__inner">
          <p className="eyebrow">Research reference</p>
          <h1>Methods, provenance & safeguards</h1>
          <p className="lede">
            What this interface is designed to represent, what the current prototype actually
            contains, and which claims remain out of bounds.
          </p>
        </div>
      </header>

      <div className="page-shell methods-layout">
        <nav className="methods-nav" aria-label="Methods sections">
          <a href="#measurement">Measurement</a>
          <a href="#provenance">Provenance</a>
          <a href="#limits">Interpretation limits</a>
          <a href="#privacy">Privacy & governance</a>
        </nav>

        <div className="methods-content">
          <section id="measurement" aria-labelledby="measurement-title">
            <div className="methods-section-heading">
              <Eye aria-hidden="true" />
              <div><p className="eyebrow">Proposal-aligned measurement</p><h2 id="measurement-title">Four eye-tracking measures</h2></div>
            </div>
            <p>
              The proposed research would standardize facial images, map qualified gaze to a
              reviewed scar region, and compute observer-level measures before aggregation.
              The current website contains simulated values only.
            </p>
            <div className="definition-list">
              {measures.map((measure, index) => (
                <article key={measure.title}><span>0{index + 1}</span><div><h3>{measure.title}</h3><p>{measure.text}</p></div></article>
              ))}
            </div>
          </section>

          <section id="provenance" aria-labelledby="provenance-title">
            <div className="methods-section-heading">
              <Fingerprint aria-hidden="true" />
              <div><p className="eyebrow">Evidence identity</p><h2 id="provenance-title">Provenance travels with the result</h2></div>
            </div>
            <p>
              A visible source field distinguishes <code>mock_simulation</code>, <code>observed_gaze</code>,
              and <code>model_prediction</code>. Asset hashes, region version, analysis version,
              pairing state, QC, and creation time belong in the same result envelope.
            </p>
            <div className="provenance-rule">
              <ShieldCheck aria-hidden="true" />
              <div><strong>Current artifact</strong><p>Two hash-pinned, AI-generated faces · unpaired_demo · simulated_ui_only · no human gaze · no model</p></div>
            </div>
          </section>

          <section id="limits" aria-labelledby="limits-title">
            <div className="methods-section-heading">
              <Scale aria-hidden="true" />
              <div><p className="eyebrow">Interpretation boundary</p><h2 id="limits-title">Attention is not emotion, judgment, stigma, or outcome.</h2></div>
            </div>
            <p>
              Gaze can describe when and where visual attention landed under a defined study
              protocol. It cannot, by itself, reveal why a person looked, whether the person
              approved or disapproved, how the image affected quality of life, or whether a
              reconstructive procedure succeeded.
            </p>
            <aside className="methods-warning">
              <AlertTriangle aria-hidden="true" />
              <p>This prototype must never diagnose scar severity, predict social evaluation, recommend a procedure, or replace patient-reported outcomes.</p>
            </aside>
          </section>

          <section id="privacy" aria-labelledby="privacy-title">
            <div className="methods-section-heading">
              <LockKeyhole aria-hidden="true" />
              <div><p className="eyebrow">Privacy & governance</p><h2 id="privacy-title">No sensitive-data shortcut</h2></div>
            </div>
            <p>
              In the default build, no photos, identifiers, or analysis payloads are persisted
              in browser storage, and no network request is made. Upload remains unavailable.
            </p>
            <p>
              A future human study still creates participant data even when the stimulus image
              is synthetic. IRB and protocol decisions remain institutional gates, alongside
              consent, calibration thresholds, access control, retention, deletion, and cohort definitions.
            </p>
          </section>
        </div>
      </div>
    </article>
  )
}
