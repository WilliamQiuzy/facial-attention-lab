import {
  ArrowLeft,
  CheckCircle2,
  Image as ImageIcon,
  RotateCcw,
  ShieldCheck,
} from 'lucide-react'
import { Link, useParams } from 'react-router-dom'
import { FailClosedState } from '../components/FailClosedState'
import { StatusBadge } from '../components/StatusBadge'
import { getWorkbenchAsset } from '../workbench/catalog'
import { getCaseRoi } from '../workbench/reducer'
import { isVerifiedFullImageSourceBinding } from '../workbench/sourceBinding'
import { useWorkspace } from '../workbench/WorkspaceProvider'

export function RoiReviewPage() {
  const { caseId } = useParams()
  const { state, actions, persistence } = useWorkspace()
  const asset = caseId ? getWorkbenchAsset(caseId) : undefined

  if (!caseId || !asset) {
    return (
      <FailClosedState
        eyebrow="Unknown case"
        title="Case unavailable"
        requestedId={caseId}
        description="The requested ID is not in the canonical ten-case synthetic catalog. No fallback case was substituted."
        backTo="/cases"
        backLabel="Back to cases"
      />
    )
  }

  const sourceBinding = getCaseRoi(state, asset.id)
  const verified = isVerifiedFullImageSourceBinding(asset, sourceBinding)

  return (
    <section className="workspace-page source-binding-page">
      <header className="workspace-page__header page-shell">
        <div>
          <Link className="workspace-back-link" to="/cases">
            <ArrowLeft aria-hidden="true" /> Back to synthetic cases
          </Link>
          <p className="workspace-kicker">Internal input identity</p>
          <h1>Source image binding</h1>
          <p>
            <code>{asset.id}</code> · {asset.label}
          </p>
        </div>
        <div
          className="source-binding-page__status"
          role="status"
          aria-label="Source binding status"
        >
          <StatusBadge tone={verified ? 'success' : 'warning'}>
            {verified ? 'Verified' : 'Needs restoration'}
          </StatusBadge>
          <span>
            {sourceBinding ? `Version ${sourceBinding.version}` : 'Binding missing'}
          </span>
        </div>
      </header>

      <div className="source-binding-layout page-shell">
        <section
          className="workspace-panel source-binding-image"
          aria-label="Canonical source image"
        >
          <div className="workspace-panel__heading">
            <div>
              <p className="workspace-kicker">Canonical synthetic asset</p>
              <h2>Full source image</h2>
            </div>
            <ImageIcon aria-hidden="true" />
          </div>
          <figure>
            <img
              src={asset.url}
              alt={`${asset.label}: AI-generated synthetic face`}
              width="1024"
              height="1024"
              loading="eager"
              decoding="async"
              fetchPriority="high"
            />
            <figcaption>
              AI-generated synthetic · independent identity · unpaired
            </figcaption>
          </figure>
        </section>

        <aside className="workspace-panel source-binding-explanation">
          <ShieldCheck aria-hidden="true" />
          <p className="workspace-kicker">What this controls</p>
          <h2>{verified ? 'Ready for a new run' : 'Restore before running'}</h2>
          <p className="source-binding-explanation__definition">
            Internal full-image source binding — not an anatomical AOI and not a
            surgical-site mask.
          </p>
          <p>
            This binding identifies the complete input image used by the workbench.
            Anatomical AOIs summarize a completed attention field after inference; they
            never crop or alter this input.
          </p>

          {!verified ? (
            <div className="source-binding-explanation__recovery">
              <p>
                Restoring writes the canonical full-image identity for this case.
                Incompatible current results become stale, while their complete history
                remains available.
              </p>
              <button
                className="workspace-button workspace-button--primary"
                type="button"
                onClick={() => actions.restoreFullImageSourceBinding(asset.id)}
              >
                <RotateCcw aria-hidden="true" /> Restore full-image binding
              </button>
              <small>This action does not run inference.</small>
            </div>
          ) : (
            <div className="source-binding-explanation__ready">
              <CheckCircle2 aria-hidden="true" />
              <p>
                The complete canonical image is bound. Starting inference remains a
                separate action.
              </p>
              <Link
                className="workspace-button workspace-button--primary"
                to={`/analysis?case=${asset.id}`}
              >
                Continue to Run
              </Link>
            </div>
          )}

          <details className="source-binding-technical">
            <summary>Technical details</summary>
            <dl>
              <div>
                <dt>Asset SHA-256</dt>
                <dd><code>{asset.sha256}</code></dd>
              </div>
              <div>
                <dt>Binding version</dt>
                <dd>{sourceBinding?.version ?? 'Unavailable'}</dd>
              </div>
              <div>
                <dt>Persistence</dt>
                <dd>{persistence.replace('_', ' ')}</dd>
              </div>
            </dl>
          </details>
        </aside>
      </div>
    </section>
  )
}
