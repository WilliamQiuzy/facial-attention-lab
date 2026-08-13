import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from 'react'
import { Link, useParams } from 'react-router-dom'
import { approvedAssets } from '../data/approvedAssetManifest'
import { CapturePanel } from '../components/CapturePanel'
import { CaptureQualityChecklist } from '../components/CaptureQualityChecklist'
import { PatientAoiSummary } from '../components/PatientAoiSummary'
import { PatientAttentionImages } from '../components/PatientAttentionImages'
import { PatientIdentityHeader } from '../components/PatientIdentityHeader'
import { PatientJobProgress } from '../components/PatientJobProgress'
import {
  PatientWorkflowProviderError,
  usePatientWorkflow,
} from '../patientWorkflow/PatientWorkflowProvider'
import {
  getOwnRecordValue,
  selectCurrentCapture,
  selectCurrentResult,
  selectCurrentRun,
  selectVisitNextAction,
} from '../patientWorkflow/selectors'
import type {
  CaptureQualityChecks,
  PatientReviewDecision,
  PatientVisit,
} from '../patientWorkflow/types'

const RESULT_DISCLAIMER =
  'Simulated estimate of where observers may attend. Not eye-tracking, diagnosis, severity, treatment guidance, or evidence of surgical outcome.'

const SYNTHETIC_LONGITUDINAL_BOUNDARY =
  'A standalone catalog demo is already used by another visit in this record. Upload a separate test image; catalog demos cannot establish longitudinal identity.'

const CURRENT_SYNTHETIC_BOUNDARY =
  'This visit already uses the standalone catalog demo. Take or upload a different test image to replace it.'

const TIMEPOINT_LABELS: Readonly<
  Record<PatientVisit['timepoint'], string>
> = {
  preoperative: 'Preoperative',
  postoperative: 'Postoperative',
  follow_up: 'Follow-up',
}

function formatDate(date: string): string {
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC',
  }).format(new Date(`${date}T00:00:00.000Z`))
}

function safeWorkflowError(error: unknown): string {
  if (error instanceof PatientWorkflowProviderError) {
    return error.message
  }
  return 'The requested workflow action could not be completed safely.'
}

export function PatientVisitPage() {
  const { patientId = '', visitId = '' } = useParams()
  const { state, actions } = usePatientWorkflow()
  const patient = getOwnRecordValue(state.patientsById, patientId)
  const visit = getOwnRecordValue(state.visitsById, visitId)
  const [actionError, setActionError] = useState<string>()
  const [reviewDecision, setReviewDecision] =
    useState<PatientReviewDecision | ''>('')
  const [reviewNote, setReviewNote] = useState('')
  const [reviewError, setReviewError] = useState<string>()
  const reviewedDecisionRef = useRef<HTMLInputElement>(null)
  const reviewNoteRef = useRef<HTMLTextAreaElement>(null)
  const stateHeadingRef = useRef<HTMLHeadingElement>(null)
  const visitAvailable =
    Boolean(patient) &&
    Boolean(visit) &&
    visit?.patientId === patient?.id
  const nextAction = visitAvailable
    ? selectVisitNextAction(state, visitId)
    : undefined

  useEffect(() => {
    if (
      nextAction === 'retry_analysis' ||
      nextAction === 'review_result' ||
      nextAction === 'visit_complete'
    ) {
      const heading = stateHeadingRef.current
      heading?.focus({ preventScroll: true })
      heading?.scrollIntoView?.({
        behavior: 'auto',
        block: 'start',
      })
    }
  }, [nextAction])

  if (!patient || !visit || visit.patientId !== patient.id) {
    return (
      <section className="patient-workflow-page patient-page-content page-shell">
        <h1>Visit unavailable</h1>
        <p>
          This patient visit is not available in the current session.
        </p>
        <Link className="patient-link-action" to="/patients">
          Back to patients
        </Link>
      </section>
    )
  }

  const capture = selectCurrentCapture(state, visit.id)
  const run = selectCurrentRun(state, visit.id)
  const result = selectCurrentResult(state, visit.id)
  const previewUrl = actions.getCapturePreviewUrl(visit.id)
  const standaloneAsset = approvedAssets[0]
  const syntheticDemoUsedByAnotherVisit = state.captureOrder.some(
    (captureId) => {
      const candidate = getOwnRecordValue(
        state.capturesById,
        captureId,
      )
      return (
        candidate?.patientId === patient.id &&
        candidate.visitId !== visit.id &&
        candidate.source === 'synthetic_demo'
      )
    },
  )
  const syntheticUnavailableReason =
    syntheticDemoUsedByAnotherVisit
      ? SYNTHETIC_LONGITUDINAL_BOUNDARY
      : capture?.source === 'synthetic_demo'
        ? CURRENT_SYNTHETIC_BOUNDARY
        : undefined

  const attachFile = (
    file: File,
    source: 'camera' | 'upload',
  ) => actions.attachSessionCapture(visit.id, file, source)

  const attachSynthetic = () => {
    if (!standaloneAsset) {
      return Promise.reject(
        new Error('No approved standalone synthetic image is available.'),
      )
    }
    return actions.attachSyntheticCapture(
      visit.id,
      standaloneAsset.id,
    )
  }

  const setQuality = (
    check: keyof CaptureQualityChecks,
    passed: boolean,
  ) => {
    setActionError(undefined)
    try {
      actions.setQualityCheck(visit.id, check, passed)
    } catch (error) {
      setActionError(safeWorkflowError(error))
    }
  }

  const runAnalysis = () => {
    setActionError(undefined)
    try {
      actions.submitAnalysis(visit.id)
    } catch (error) {
      setActionError(safeWorkflowError(error))
    }
  }

  const retryAnalysis = () => {
    setActionError(undefined)
    try {
      actions.retryAnalysis(visit.id)
    } catch (error) {
      setActionError(safeWorkflowError(error))
    }
  }

  const completeReview = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setReviewError(undefined)

    if (!reviewDecision) {
      setReviewError('Choose Reviewed or Repeat photo.')
      reviewedDecisionRef.current?.focus()
      return
    }
    const note = reviewNote.trim()
    if (reviewDecision === 'repeat_photo' && !note) {
      setReviewError('Enter a reason for the repeat photo.')
      reviewNoteRef.current?.focus()
      return
    }

    try {
      actions.completeReview(
        visit.id,
        reviewDecision,
        note || undefined,
      )
      if (reviewDecision === 'repeat_photo') {
        actions.requestRetake(visit.id)
      }
    } catch (error) {
      setReviewError(safeWorkflowError(error))
    }
  }

  let workflowContent

  if (nextAction === 'capture_photo') {
    workflowContent = (
      <CapturePanel
        title="Add frontal photograph"
        onSelectFile={attachFile}
        onUseSynthetic={attachSynthetic}
        syntheticUnavailableReason={syntheticUnavailableReason}
      />
    )
  } else if (
    (nextAction === 'confirm_quality' ||
      nextAction === 'run_analysis') &&
    capture
  ) {
    workflowContent = (
      <div className="patient-capture-quality-step">
        <CapturePanel
          title="Current photograph"
          previewUrl={previewUrl}
          previewWidth={capture.width}
          previewHeight={capture.height}
          onSelectFile={attachFile}
          onUseSynthetic={attachSynthetic}
          syntheticUnavailableReason={syntheticUnavailableReason}
        />
        <CaptureQualityChecklist
          checks={capture.qualityChecks}
          onChange={setQuality}
          onRun={runAnalysis}
        />
      </div>
    )
  } else if (
    nextAction === 'processing' &&
    run &&
    (run.status === 'queued' ||
      run.status === 'running' ||
      run.status === 'succeeded')
  ) {
    workflowContent = (
      <div className="patient-processing-step">
        <PatientJobProgress
          status={run.status}
          previewUrl={previewUrl}
          width={capture?.width}
          height={capture?.height}
        />
        <Link
          className="patient-link-action"
          to={`/patients/${patient.id}`}
        >
          Return to patient record
        </Link>
      </div>
    )
  } else if (nextAction === 'retry_analysis' && capture) {
    const faceRegistrationFailed =
      run?.status === 'failed' &&
      run.failure?.field?.startsWith('faceRegistration') === true
    workflowContent = (
      <section
        className="patient-retry-step"
        aria-labelledby="patient-retry-title"
      >
        <h2
          ref={stateHeadingRef}
          id="patient-retry-title"
          tabIndex={-1}
        >
          {faceRegistrationFailed
            ? 'Face alignment needs attention'
            : 'Analysis needs attention'}
        </h2>
        {faceRegistrationFailed ? (
          <p>
            We could not match one clear face to this photograph.
            Retake or upload a centered frontal image with only one
            face visible.
          </p>
        ) : (
          <>
            <p>
              The current exact-bound simulated run did not complete.
            </p>
            <p>
              Retry uses this same photograph and confirmed quality
              record.
            </p>
            <button
              className="patient-primary-action"
              type="button"
              onClick={retryAnalysis}
            >
              Retry exact photo analysis
            </button>
          </>
        )}
        <CapturePanel
          title={
            faceRegistrationFailed
              ? 'Replace photograph'
              : 'Replace photograph instead'
          }
          previewUrl={previewUrl}
          previewWidth={capture.width}
          previewHeight={capture.height}
          onSelectFile={attachFile}
          onUseSynthetic={attachSynthetic}
          syntheticUnavailableReason={syntheticUnavailableReason}
        />
      </section>
    )
  } else if (
    nextAction === 'review_result' &&
    capture &&
    result &&
    previewUrl
  ) {
    workflowContent = (
      <section
        className="patient-result-review"
        aria-labelledby="patient-result-title"
      >
        <header className="patient-result-review__header">
          <h2
            ref={stateHeadingRef}
            id="patient-result-title"
            tabIndex={-1}
          >
            Review result
          </h2>
          <p className="patient-result-review__disclaimer">
            {RESULT_DISCLAIMER}
          </p>
        </header>

        <PatientAttentionImages
          previewUrl={previewUrl}
          width={capture.width}
          height={capture.height}
          points={result.output.points}
          faceRegistration={result.faceRegistration}
        />
        <PatientAoiSummary
          points={result.output.points}
          faceRegistration={result.faceRegistration}
        />

        <details className="patient-result-review__technical">
          <summary>Technical details</summary>
          <p className="patient-result-review__engine-boundary">
            Demo engine only: this illustrative field is seeded by the
            capture hash and positioned using on-device face landmarks.
            It does not detect this person’s scar and was not produced by
            the checked-in facial-paralysis model.
          </p>
          <dl>
            <div>
              <dt>Capture version</dt>
              <dd>{capture.version}</dd>
            </div>
            <div>
              <dt>Capture protocol</dt>
              <dd>Frontal, relaxed, non-mirrored</dd>
            </div>
            <div>
              <dt>Result source</dt>
              <dd>Local deterministic workflow simulation</dd>
            </div>
          </dl>
        </details>

        <form
          className="patient-result-review__form"
          noValidate
          onSubmit={completeReview}
        >
          <fieldset>
            <legend>Review decision</legend>
            <label>
              <input
                ref={reviewedDecisionRef}
                type="radio"
                name="reviewDecision"
                value="reviewed"
                checked={reviewDecision === 'reviewed'}
                aria-invalid={
                  reviewError === 'Choose Reviewed or Repeat photo.'
                }
                aria-describedby={
                  reviewError === 'Choose Reviewed or Repeat photo.'
                    ? 'patient-review-error'
                    : undefined
                }
                onChange={() => {
                  setReviewDecision('reviewed')
                  setReviewError(undefined)
                }}
              />
              <span>Reviewed</span>
            </label>
            <label>
              <input
                type="radio"
                name="reviewDecision"
                value="repeat_photo"
                checked={reviewDecision === 'repeat_photo'}
                aria-invalid={
                  reviewError === 'Choose Reviewed or Repeat photo.'
                }
                aria-describedby={
                  reviewError === 'Choose Reviewed or Repeat photo.'
                    ? 'patient-review-error'
                    : undefined
                }
                onChange={() => {
                  setReviewDecision('repeat_photo')
                  setReviewError(undefined)
                }}
              />
              <span>Repeat photo</span>
            </label>
          </fieldset>

          <label className="patient-result-review__note">
            <span>
              {reviewDecision === 'repeat_photo'
                ? 'Reason for repeat photo'
                : 'Review note (optional)'}
            </span>
            <textarea
              ref={reviewNoteRef}
              name="reviewNote"
              autoComplete="off"
              value={reviewNote}
              required={reviewDecision === 'repeat_photo'}
              aria-invalid={
                reviewError === 'Enter a reason for the repeat photo.'
              }
              aria-describedby={
                reviewError === 'Enter a reason for the repeat photo.'
                  ? 'patient-review-error'
                  : undefined
              }
              onChange={(event) => {
                setReviewNote(event.currentTarget.value)
                if (
                  reviewError ===
                  'Enter a reason for the repeat photo.'
                ) {
                  setReviewError(undefined)
                }
              }}
            />
          </label>

          {reviewError ? (
            <p
              className="patient-result-review__error"
              id="patient-review-error"
              role="alert"
            >
              {reviewError}
            </p>
          ) : null}

          <button className="patient-primary-action" type="submit">
            Complete review
          </button>
        </form>
      </section>
    )
  } else if (nextAction === 'visit_complete') {
    workflowContent = (
      <section className="patient-visit-complete">
        <h2 ref={stateHeadingRef} tabIndex={-1}>
          Review complete
        </h2>
        <p>
          This photo visit is complete in the current session.
        </p>
        <div className="patient-visit-complete__actions">
          <Link
            className="patient-primary-action"
            to={`/patients/${patient.id}`}
          >
            Return to patient record
          </Link>
          <Link
            className="patient-secondary-action"
            to="/reviews"
          >
            Back to reviews
          </Link>
        </div>
      </section>
    )
  } else if (nextAction === 'retake') {
    workflowContent = (
      <CapturePanel
        title="Repeat photo"
        previewUrl={previewUrl}
        previewWidth={capture?.width}
        previewHeight={capture?.height}
        onSelectFile={attachFile}
        onUseSynthetic={attachSynthetic}
        syntheticUnavailableReason={syntheticUnavailableReason}
      />
    )
  } else {
    workflowContent = (
      <section className="patient-visit-unavailable" role="alert">
        <h2>Workflow unavailable</h2>
        <p>
          The current visit state could not be verified safely.
        </p>
      </section>
    )
  }

  return (
    <div className="patient-workflow-page patient-visit-page">
      <header className="patient-page-header page-shell">
        <Link
          className="patient-back-link"
          to={`/patients/${patient.id}`}
        >
          Back to patient record
        </Link>
        <PatientIdentityHeader
          patient={patient}
          headingLevel={1}
          compact
        />
        <p className="patient-visit-page__context">
          {TIMEPOINT_LABELS[visit.timepoint]} photo visit ·{' '}
          {formatDate(visit.visitDate)}
        </p>
      </header>

      <section className="patient-page-content page-shell">
        {actionError ? (
          <p className="patient-visit-page__error" role="alert">
            {actionError}
          </p>
        ) : null}
        {workflowContent}
      </section>
    </div>
  )
}
