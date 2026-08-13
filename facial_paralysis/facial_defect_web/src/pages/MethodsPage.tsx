import { AlertTriangle, Ban, Eye, Fingerprint, LockKeyhole, Scale, ShieldCheck } from 'lucide-react'
import { listWorkbenchAssets } from '../workbench/catalog'
import { useWorkspace } from '../workbench/WorkspaceProvider'

export function MethodsPage() {
  const { gatewayMode } = useWorkspace()
  const catalogCount = listWorkbenchAssets().length
  const connected = gatewayMode === 'connected'

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
              <div><p className="eyebrow">Intended spatial contract</p><h2 id="measurement-title">What this workbench can represent</h2></div>
            </div>
            <p>
              No trained facial-defect attention model exists in this repository yet. The
              default output is a deterministic synthetic field used to rehearse the
              clinician interface; it is not observed gaze or a patient result.
            </p>
            <p>
              A future compatible service is expected to return population-level predicted
              observer-attention spatial density with an explicit spatial field. That intended
              density contract is separate from the current mock AOI calculation.
            </p>
            <p>
              Current mock values are fixed-template simulated point-weight shares. Each
              simulated point&apos;s intensity is assigned by its center. Radius is ignored, so
              this is not a raster or density-kernel integral.
            </p>
            <div className="provenance-rule" role="note" aria-label="Clinical AOI meaning">
              <Eye aria-hidden="true" />
              <div>
                <strong>Mock AOIs are interface-rehearsal summaries</strong>
                <p>
                  Mock AOIs are automatic post-inference summaries and do not modify the
                  simulation. The subsite and hemiface rows are separate complete partitions;
                  the central facial triangle is an overlapping, non-additive reference.
                </p>
                <p>
                  In a frontal image, patient left appears on the viewer&apos;s right. A future
                  surgical-site mask would be a separate, versioned contextual annotation; it
                  would not stand in for attention, severity, or the immutable image bound.
                </p>
              </div>
            </div>
            <div className="provenance-rule" role="note" aria-label="Current model boundary">
              <Ban aria-hidden="true" />
              <div>
                <strong>Facial-paralysis evidence remains a separate system</strong>
                <p>
                  The current facial_paralysis scoring path returns palsy probability and
                  eyes/mouth ordinal outputs. Separate analysis code derives landmark-derived
                  left-right asymmetry, eye-closure dynamics, and a Mayo FACES label-free
                  research measurement summary. None of these outputs is a pixel attention map.
                </p>
                <p>
                  No checked-in checkpoint includes an HB task; the architecture can support
                  one, but Mayo HB calibration has not started. The FACES-action-derived
                  regional research measurement summary is not a validated eFACE, Sunnybrook,
                  or HB composite or grade.
                </p>
                <p>
                  A severity or ordinal payload without spatial points fails closed. The browser
                  never converts palsy scores, task logits, temporal pooling weights, or
                  occlusion analyses into a heatmap.
                </p>
                <p>
                  The v2 attention checkpoint uses temporal frame pooling, not spatial
                  facial attention. The v4 checkpoint includes a coarse3 head, but the
                  current prediction script does not export it.
                </p>
              </div>
            </div>
          </section>

          <section id="provenance" aria-labelledby="provenance-title">
            <div className="methods-section-heading">
              <Fingerprint aria-hidden="true" />
              <div><p className="eyebrow">Evidence identity</p><h2 id="provenance-title">Provenance travels with the result</h2></div>
            </div>
            <p>
              The current result envelope accepts <code>mock_simulation</code> from the local
              deterministic implementation. It reserves <code>model_prediction</code> only for
              a future, separately defined spatial-attention API with an explicit heatmap.
              <code>observed_gaze</code> remains a future study data class outside the current
              inference gateway.
            </p>
            <p>
              Connected version 1 is a synthetic spatial contract rehearsal. It requires
              <code>registration_geometry_unavailable_v1</code> and does not claim that
              landmarks were supplied. It does not carry landmarks or polygons, source
              dimensions, orientation or mirror metadata, or registration quality control.
            </p>
            <p>
              Connected AOI reporting remains unavailable and fails closed until the contract
              is extended.
            </p>
            <p>
              No observer-attention checkpoint or output is implemented. The current
              <code>HeatmapPoint[]</code> display-points representation and a future
              patient-media reference are provisional. Model and backend owners must jointly
              freeze the media reference, coordinate frame, spatial representation,
              normalization, display scale, and production request/response schema before
              integration.
            </p>
            <p>
              A connected result states whether observed gaze is included in that result
              payload separately from training-data provenance. This rehearsal includes no
              observed-gaze payload and reports training-data provenance as not disclosed.
            </p>
            <div
              className="provenance-rule"
              role="status"
              aria-label="Methods runtime boundary"
            >
              <ShieldCheck aria-hidden="true" />
              <div>
                <strong>
                  {connected
                    ? 'Synthetic spatial contract rehearsal enabled'
                    : 'Local mock gateway active'}
                </strong>
                <p>
                  {connected
                    ? 'Explicit opt-in. The single WorkbenchGateway port requests model_prediction output only when a run executes and never falls back to the mock engine.'
                    : 'Default mode performs no inference network request. The single WorkbenchGateway port resolves deterministic mock_simulation output in memory.'}
                </p>
                {connected ? (
                  <p>
                    The current facial_paralysis functional-assessment system remains separate
                    and is not connected; enabling this seam does not make it a
                    spatial-attention model.
                  </p>
                ) : null}
                {connected ? <p>Accepted capability: <code>research_unvalidated</code>.</p> : null}
              </div>
            </div>
            <div className="provenance-rule">
              <Fingerprint aria-hidden="true" />
              <div>
                <strong>Current canonical catalog</strong>
                <p>
                  {catalogCount} hash-pinned, standalone AI-generated synthetic cases ·
                  unpaired_demo · no patient data · no human gaze
                </p>
              </div>
            </div>
          </section>

          <section id="limits" aria-labelledby="limits-title">
            <div className="methods-section-heading">
              <Scale aria-hidden="true" />
              <div><p className="eyebrow">Interpretation boundary</p><h2 id="limits-title">Attention is not emotion, judgment, stigma, or outcome.</h2></div>
            </div>
            <p>
              Observed gaze can describe when and where visual attention landed under a defined
              study protocol. A predicted density field is not observed gaze. Neither can, by
              itself, reveal why a person looked, whether the person approved or disapproved,
              how the image affected quality of life, or whether a reconstructive procedure
              succeeded.
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
              In default mock mode, no photos, identifiers, or analysis payloads are persisted
              in browser storage and no inference network request is made. Connected mode is
              an explicit configuration opt-in; it does not enable upload, storage, or fallback.
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
