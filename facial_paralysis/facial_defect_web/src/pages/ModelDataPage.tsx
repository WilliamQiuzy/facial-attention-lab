import {
  ArrowDown,
  Ban,
  Braces,
  Check,
  Database,
  LockKeyhole,
  ServerCog,
} from 'lucide-react'
import { listWorkbenchAssets } from '../workbench/catalog'
import { useWorkspace } from '../workbench/WorkspaceProvider'

export function ModelDataPage() {
  const { gatewayMode } = useWorkspace()
  const connected = gatewayMode === 'connected'
  const catalogCount = listWorkbenchAssets().length

  const sources = [
    {
      name: 'mock_simulation',
      state: connected ? 'Default-mode implementation' : 'Active in this session',
      detail:
        'Deterministic local output from MockWorkbenchGateway. No network, human gaze, or patient data.',
    },
    {
      name: 'observed_gaze',
      state: 'Not implemented',
      detail:
        'A future consented study data class. It is outside the current WorkbenchGateway inference response contract.',
    },
    {
      name: 'model_prediction',
      state: connected
        ? 'Synthetic spatial contract rehearsal enabled'
        : 'Synthetic spatial contract rehearsal available',
      detail: (
        <>
          This seam rehearses a future population-level predicted observer-attention
          spatial density contract; no attention checkpoint or output is implemented.
          HttpWorkbenchGateway accepts this origin only with spatial display points, strict
          response model identity, declared semantics,{' '}
          <code>registration_geometry_unavailable_v1</code>, capability{' '}
          <code>research_unvalidated</code>, and the permanent clinical-use block. It does
          not carry landmarks or polygons, source dimensions, orientation or mirror
          metadata, or registration quality control. Connected AOI reporting is unavailable
          and fails closed until the contract is extended.
        </>
      ),
    },
  ] as const

  return (
    <div className="model-page">
      <header className="model-hero">
        <div className="page-shell model-hero__grid">
          <div>
            <p className="eyebrow">Implemented integration boundary</p>
            <h1>Model & data readiness</h1>
            <p className="lede">
              One typed gateway port serves deterministic local simulation by default and an
              explicitly enabled research HTTP implementation without changing the operational
              pages or weakening fail-closed validation.
            </p>
          </div>
          <div
            className="connection-status"
            role="status"
            aria-label="Current gateway mode"
          >
            <span className="connection-status__light" aria-hidden="true" />
            <div>
              <p>Runtime state</p>
              <strong>
                {connected
                  ? 'Research HTTP seam enabled'
                  : 'Local mock gateway active'}
              </strong>
              <span>
                {connected
                  ? 'Explicit opt-in · network requests occur only when a run executes'
                  : 'Default mode · no inference network requests'}
              </span>
            </div>
          </div>
        </div>
      </header>

      <section
        className="page-shell model-section"
        role="note"
        aria-label="Current model compatibility"
      >
        <div className="section-heading section-heading--compact">
          <div>
            <p className="eyebrow">Current project boundary</p>
            <h2>The research model and this spatial demo are not interchangeable.</h2>
          </div>
          <p>
            The available functional-assessment research outputs are non-spatial severity or
            regional summaries. They are not connected to this web workbench.
          </p>
        </div>
        <div className="provenance-rule">
          <Ban aria-hidden="true" />
          <div>
            <strong>
              {connected
                ? 'HTTP transport enabled; compatible model absent'
                : 'No compatible model backend connected'}
            </strong>
            <p>
              {connected
                ? 'The research HTTP seam is enabled, but the current functional-assessment research system is still not connected. No trained facial-defect attention model is configured.'
                : 'The browser currently runs a deterministic synthetic spatial mock only. No trained facial-defect attention model exists in this repository yet.'}
            </p>
            <p>
              This functional-assessment research system does not emit a spatial heatmap, and
              a non-spatial severity or ordinal response cannot be converted into one. A
              connected response without valid spatial points fails closed.
            </p>
            <p>
              A future extended contract may add post-inference AOI summaries without changing
              the prediction. A surgical-site mask is not part of the current request; any
              future mask must be a separate, versioned contextual annotation and cannot stand
              in for attention or severity.
            </p>
          </div>
        </div>
      </section>

      <section className="page-shell model-section" aria-labelledby="source-title">
        <div className="section-heading section-heading--compact">
          <div>
            <p className="eyebrow">Origin is a safety field</p>
            <h2 id="source-title">Every result declares its evidence class.</h2>
          </div>
          <p>
            Simulation and connected prediction are accepted by different gateway modes.
            Observed gaze remains outside this inference contract and is never synthesized as a
            substitute.
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
              <p className="eyebrow">Current runtime contract</p>
              <h2>One gateway from approved input to reviewable evidence</h2>
            </div>
            <div className="model-version-card">
              <p>Gateway implementation</p>
              <strong>
                {connected ? 'HttpWorkbenchGateway' : 'MockWorkbenchGateway'}
              </strong>
              <span>Implements <code>WorkbenchGateway</code></span>
            </div>
          </div>

          <div className="integration-flow" aria-label="Current inference gateway flow">
            <article>
              <Database aria-hidden="true" /><span>01</span><h3>Canonical case</h3>
              <p>
                {catalogCount} hash-pinned standalone synthetic cases, each independent and
                unpaired.
              </p>
            </article>
            <ArrowDown className="integration-flow__arrow" aria-hidden="true" />
            <article>
              <ServerCog aria-hidden="true" /><span>02</span><h3>Exact request</h3>
              <p>
                The connected wire sends the request-contract version, run and attempt IDs,
                case and asset IDs, source SHA-256, and full-image source-binding
                ID/version/geometry. Internal mock model and configuration fields are not sent.
              </p>
            </article>
            <ArrowDown className="integration-flow__arrow" aria-hidden="true" />
            <article>
              <Braces aria-hidden="true" /><span>03</span><h3>Exact response</h3>
              <p>
                The response must echo that request identity, report the connected model,
                artifact, preprocessing, calibration, and display scale, and pass origin,
                capability, spatial-semantics, point-count, registration-unavailable,
                quality, and provenance checks.
              </p>
            </article>
            <ArrowDown className="integration-flow__arrow" aria-hidden="true" />
            <article>
              <LockKeyhole aria-hidden="true" /><span>04</span><h3>Research review</h3>
              <p>
                Failure returns no result. Clinical use stays blocked in both gateway modes.
              </p>
            </article>
          </div>
        </div>
      </section>

      <section className="page-shell endpoint-section" aria-labelledby="endpoints-title">
        <div>
          <p className="eyebrow">Single HTTP operation</p>
          <h2 id="endpoints-title">One gateway operation, exact binding.</h2>
          <p>
            Connected mode never falls back to mock after timeout, abort, HTTP error,
            malformed response, or binding mismatch.
          </p>
        </div>
        <div className="endpoint-list">
          <article>
            <span>Connected inference</span>
            <code>/api/v1/workbench/inference</code>
            <p>POST one minimal connected identity; accept only an exactly matching model_prediction envelope.</p>
          </article>
          <article>
            <span>Explicit opt-in</span>
            <code>VITE_ENABLE_CONNECTED_MODE=true</code>
            <p>Requires an absolute approved base URL in <code>VITE_ATTENTION_API_URL</code>.</p>
          </article>
        </div>
      </section>

      <section className="requirements-section">
        <div className="page-shell requirements-grid">
          <div>
            <p className="eyebrow">Fail-closed checklist</p>
            <h2>What every inference must preserve</h2>
          </div>
          <div className="requirements-list">
            <article>
              <Check aria-hidden="true" />
              <div><h3>Catalog boundary</h3><p>Exactly {catalogCount} approved standalone synthetic assets with immutable SHA-256 values.</p></div>
            </article>
            <article>
              <Ban aria-hidden="true" />
              <div><h3>Internal image bound</h3><p>The immutable full-image bound identifies the exact input. It is not a surgical-site mask and does not modify the spatial field.</p></div>
            </article>
            <article>
              <Ban aria-hidden="true" />
              <div><h3>Response identity</h3><p>Wrong origin, capability, watermark, request echo, model identity, digest, or provenance fails closed.</p></div>
            </article>
            <article>
              <Ban aria-hidden="true" />
              <div><h3>No fallback or promotion</h3><p>Connected failure never becomes mock output; neither mode is clinically validated.</p></div>
            </article>
          </div>
        </div>
      </section>

      <section className="page-shell model-section" aria-label="Contract freeze boundary">
        <div className="section-heading section-heading--compact">
          <div>
            <p className="eyebrow">Not production-frozen</p>
            <h2>Display points and patient-media reference remain provisional.</h2>
          </div>
          <p>
            The current <code>HeatmapPoint[]</code> representation is capped at 4,096 display
            points. It is not a frozen raster or radial-basis scientific field. Patient-media
            reference and authorization are unimplemented. Model and backend owners must
            jointly freeze those contracts, coordinate frame, normalization, and display
            scale before integration.
          </p>
        </div>
      </section>
    </div>
  )
}
