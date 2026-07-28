import {
  ArrowLeft,
  Download,
} from 'lucide-react'
import { Link, useLocation } from 'react-router-dom'
import { AttentionResultView } from '../components/AttentionResultView'
import { FailClosedState } from '../components/FailClosedState'
import {
  createPatientExportManifest,
  downloadPatientExport,
} from '../workbench/reviewExport'
import { evaluatePatientReportEligibility } from '../workbench/reviewPolicy'
import { useWorkspace } from '../workbench/WorkspaceProvider'

const PROTOTYPE_LIKE_REVIEW_IDS = new Set([
  ...Object.getOwnPropertyNames(Object.prototype).map((property) =>
    property.toLowerCase(),
  ),
  'prototype',
])

function UnavailablePatientPreview({
  reason,
  description,
}: {
  readonly reason: string
  readonly description: string
}) {
  return (
    <FailClosedState
      eyebrow="Patient handoff gate"
      title="Patient preview unavailable"
      requestedId={reason}
      description={description}
      backTo="/research/reviews"
      backLabel="Back to research reviews"
    />
  )
}

export function PatientReportPage() {
  const { search } = useLocation()
  const { state } = useWorkspace()
  const queryParameters = new URLSearchParams(search)
  const reviewParameters = queryParameters.getAll('review')
  const reviewId = reviewParameters.length === 1 ? reviewParameters[0] : ''

  if ([...queryParameters.keys()].some((key) => key !== 'review')) {
    return (
      <UnavailablePatientPreview
        reason="Unexpected query parameters were supplied"
        description="Patient explanation eligibility is blocked because this route accepts only one exact review parameter."
      />
    )
  }

  if (
    reviewParameters.length === 0 ||
    (reviewParameters.length === 1 && !reviewId.trim())
  ) {
    return (
      <UnavailablePatientPreview
        reason="No exact review ID was supplied"
        description="Patient explanation eligibility is blocked until one exact in-session research review is supplied. No fixture was substituted."
      />
    )
  }
  if (reviewParameters.length !== 1) {
    return (
      <UnavailablePatientPreview
        reason="Duplicate review parameters were supplied"
        description="Patient explanation eligibility is blocked because the route does not identify one authoritative review."
      />
    )
  }
  if (reviewId !== reviewId.trim()) {
    return (
      <UnavailablePatientPreview
        reason="The review ID must match exactly"
        description="Patient explanation eligibility is blocked because leading or trailing whitespace changes the requested review identity."
      />
    )
  }
  if (PROTOTYPE_LIKE_REVIEW_IDS.has(reviewId.toLowerCase())) {
    return (
      <UnavailablePatientPreview
        reason={reviewId}
        description="Patient explanation eligibility is blocked because the supplied review identifier is reserved and cannot identify an authoritative review."
      />
    )
  }

  const eligibility = evaluatePatientReportEligibility(state, reviewId)
  if (!eligibility.eligible) {
    const details = eligibility.blockers.map((entry) => entry.message).join(' ')
    return (
      <UnavailablePatientPreview
        reason={reviewId}
        description={`Patient explanation eligibility is blocked. ${details}`}
      />
    )
  }

  const downloadManifest = () => {
    const result = createPatientExportManifest(state, reviewId)
    if (result.eligible) downloadPatientExport(result.manifest)
  }

  return (
    <article className="workspace-page task6-page task6-patient-page">
      <div className="task6-patient-toolbar page-shell">
        <Link to={`/research/reviews/${encodeURIComponent(reviewId)}`}>
          <ArrowLeft aria-hidden="true" /> Return to research review
        </Link>
      </div>

      <header className="page-shell patient-explanation-header">
        <p className="workspace-kicker">Patient explanation · synthetic demo</p>
        <h1>Simulated attention explanation</h1>
        <p>
          This is a research-interface demonstration. It is not an individual
          prediction, a clinical measurement, or evidence about a patient.
        </p>
      </header>

      <section
        className="page-shell task6-patient-preview"
        aria-label="Approved simulated patient explanation"
      >
        <div className="patient-result-heading">
          <h2>Result</h2>
          <p>
            Begin with the AOI summary. Open the density field or overlay only
            when useful.
          </p>
        </div>

        <div className="patient-result">
          <AttentionResultView
            asset={eligibility.asset}
            output={eligibility.output}
            roi={eligibility.roi}
            layout="patient-compact"
          />
        </div>

        <details className="patient-result-disclosure">
          <summary>How to discuss this result</summary>
          <div>
            <p>
              The display compares relative simulated signal across the displayed
              image field.
            </p>
            <p>
              It cannot explain what a person thought or felt, diagnose anything, assess
              healing, or establish whether a procedure succeeded.
            </p>
          </div>
        </details>

        <details className="patient-result-disclosure">
          <summary>Technical options</summary>
          <div>
            <p>
              Export the restricted research manifest for an authorized technical
              review. It contains no patient identifiers.
            </p>
            <button
              className="workspace-button workspace-button--secondary"
              type="button"
              aria-label="Download safe JSON manifest"
              onClick={downloadManifest}
            >
              <Download aria-hidden="true" /> Download safe JSON manifest
            </button>
          </div>
        </details>
      </section>
    </article>
  )
}
