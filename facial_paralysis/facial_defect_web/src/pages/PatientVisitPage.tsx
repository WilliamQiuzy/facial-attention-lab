import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from 'react'
import { Link, useParams } from 'react-router-dom'
import { CapturePanel } from '../components/CapturePanel'
import { CaptureQualityChecklist } from '../components/CaptureQualityChecklist'
import { PatientAoiSummary } from '../components/PatientAoiSummary'
import {
  PatientAttentionDensity,
  PatientAttentionImages,
  PatientAttentionPrimaryImages,
} from '../components/PatientAttentionImages'
import { PatientIdentityHeader } from '../components/PatientIdentityHeader'
import { PatientJobProgress } from '../components/PatientJobProgress'
import {
  findPatientSamplePhotoAsset,
  patientSamplePhotoAssetForPatientTimepoint,
} from '../data/patientSamplePhotoPair'
import {
  PatientWorkflowProviderError,
  usePatientWorkflow,
} from '../patientWorkflow/PatientWorkflowProvider'
import {
  getOwnRecordValue,
  selectCurrentCapture,
  selectCurrentReview,
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
  'Illustrative estimate of where observers may attend. Not measured eye-tracking, diagnosis, treatment guidance, or evidence of surgical outcome.'

const INCOMPATIBLE_SAMPLE_BOUNDARY =
  'This record already uses a different sample photo. Use Camera or Upload photo to keep the same patient across visits.'

const CURRENT_SYNTHETIC_BOUNDARY =
  'This visit already uses a sample photo. Use Camera or Upload photo to replace it.'

const FOLLOW_UP_SAMPLE_BOUNDARY =
  'Sample photos are available for preoperative and postoperative visits. Use Camera or Upload photo for follow-up.'

const REVIEW_DECISION_REQUIRED =
  'Choose Accept photo for comparison or Request a new photo.'

const NEW_PHOTO_REASON_REQUIRED =
  'Enter a reason for requesting a new photo.'

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

function formatDateTime(date: string): string {
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    timeZone: 'UTC',
    timeZoneName: 'short',
  }).format(new Date(date))
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
  const [decisionJustSaved, setDecisionJustSaved] =
    useState(false)
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
  const currentReview = selectCurrentReview(state, visit.id)
  const previewUrl = actions.getCapturePreviewUrl(visit.id)
  const sampleAsset = patientSamplePhotoAssetForPatientTimepoint(
    patient.id,
    visit.timepoint,
  )
  const incompatibleSampleUsedByAnotherVisit = state.captureOrder.some(
    (captureId) => {
      const candidate = getOwnRecordValue(
        state.capturesById,
        captureId,
      )
      if (
        candidate?.patientId !== patient.id ||
        candidate.visitId === visit.id ||
        candidate.source !== 'synthetic_demo'
      ) {
        return false
      }
      const otherSample = candidate.syntheticSourceAssetId
        ? findPatientSamplePhotoAsset(
            candidate.syntheticSourceAssetId,
          )
        : undefined
      return (
        !sampleAsset ||
        !otherSample ||
        otherSample.pairId !== sampleAsset.pairId ||
        otherSample.timepoint === sampleAsset.timepoint
      )
    },
  )
  const syntheticUnavailableReason =
    capture?.source === 'synthetic_demo'
      ? CURRENT_SYNTHETIC_BOUNDARY
      : !sampleAsset
        ? FOLLOW_UP_SAMPLE_BOUNDARY
        : incompatibleSampleUsedByAnotherVisit
          ? INCOMPATIBLE_SAMPLE_BOUNDARY
          : undefined
  const previewDisclosure =
    capture?.source === 'synthetic_demo'
      ? sampleAsset?.disclosure
      : undefined

  const attachFile = (
    file: File,
    source: 'camera' | 'upload',
  ) => actions.attachSessionCapture(visit.id, file, source)

  const attachSynthetic = () => {
    if (!sampleAsset) {
      return Promise.reject(
        new Error('No approved sample photo is available.'),
      )
    }
    return actions.attachSyntheticCapture(
      visit.id,
      sampleAsset.id,
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
      setReviewError(REVIEW_DECISION_REQUIRED)
      reviewedDecisionRef.current?.focus()
      return
    }
    const note = reviewNote.trim()
    if (reviewDecision === 'repeat_photo' && !note) {
      setReviewError(NEW_PHOTO_REASON_REQUIRED)
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
      } else {
        setDecisionJustSaved(true)
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
          previewDisclosure={previewDisclosure}
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
              The current photo analysis did not complete.
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
          previewDisclosure={previewDisclosure}
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
                  reviewError === REVIEW_DECISION_REQUIRED
                }
                aria-describedby={
                  reviewError === REVIEW_DECISION_REQUIRED
                    ? 'patient-review-error'
                    : undefined
                }
                onChange={() => {
                  setReviewDecision('reviewed')
                  setReviewError(undefined)
                }}
              />
              <span>Accept photo for comparison</span>
            </label>
            <label>
              <input
                type="radio"
                name="reviewDecision"
                value="repeat_photo"
                aria-label="Request a new photo"
                checked={reviewDecision === 'repeat_photo'}
                aria-invalid={
                  reviewError === REVIEW_DECISION_REQUIRED
                }
                aria-describedby={
                  reviewError === REVIEW_DECISION_REQUIRED
                    ? 'new-photo-help patient-review-error'
                    : 'new-photo-help'
                }
                onChange={() => {
                  setReviewDecision('repeat_photo')
                  setReviewError(undefined)
                }}
              />
              <span className="patient-result-review__decision-copy">
                <strong>Request a new photo</strong>
                <small id="new-photo-help">
                  Use when this image should not be used for comparison.
                </small>
              </span>
            </label>
          </fieldset>

          <label className="patient-result-review__note">
            <span>
              {reviewDecision === 'repeat_photo'
                ? 'Reason for requesting a new photo'
                : 'Review note (optional)'}
            </span>
            <textarea
              ref={reviewNoteRef}
              name="reviewNote"
              autoComplete="off"
              value={reviewNote}
              required={reviewDecision === 'repeat_photo'}
              aria-invalid={
                reviewError === NEW_PHOTO_REASON_REQUIRED
              }
              aria-describedby={
                reviewError === NEW_PHOTO_REASON_REQUIRED
                  ? 'patient-review-error'
                  : undefined
              }
              onChange={(event) => {
                setReviewNote(event.currentTarget.value)
                if (
                  reviewError === NEW_PHOTO_REASON_REQUIRED
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
            Save decision
          </button>
        </form>

        <PatientAoiSummary
          points={result.output.points}
          faceRegistration={result.faceRegistration}
        />

        <details className="patient-result-review__technical">
          <summary>Technical details</summary>
          <p className="patient-result-review__engine-boundary">
            Illustrative engine only: this field is seeded by the
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
              <dd>Local illustrative sample engine</dd>
            </div>
          </dl>
        </details>
      </section>
    )
  } else if (
    nextAction === 'visit_complete' &&
    capture &&
    result &&
    currentReview
  ) {
    workflowContent = (
      <section
        className="patient-visit-record"
        aria-labelledby="patient-visit-record-title"
      >
        <header className="patient-visit-record__header">
          <div>
            <p className="patient-visit-record__eyebrow">
              Single visit
            </p>
            <h2
              ref={stateHeadingRef}
              id="patient-visit-record-title"
              tabIndex={-1}
            >
              {TIMEPOINT_LABELS[visit.timepoint]} visit record
            </h2>
          </div>
          <span className="patient-visit-record__status">
            Complete · Accepted for comparison
          </span>
        </header>

        {decisionJustSaved ? (
          <p className="patient-visit-record__saved" role="status">
            <strong>Decision saved.</strong> This visit is now available
            from the patient record.
          </p>
        ) : null}

        {previewUrl ? (
          <>
            <section
              className="patient-visit-record__result"
              aria-labelledby="patient-visit-record-result-title"
            >
              <div className="patient-visit-record__section-heading">
                <h3 id="patient-visit-record-result-title">
                  Visit photo and result
                </h3>
                <p>{RESULT_DISCLAIMER}</p>
              </div>
              {previewDisclosure ? (
                <p className="patient-visit-record__provenance">
                  {previewDisclosure}
                </p>
              ) : null}
              <div className="patient-attention-images">
                <PatientAttentionPrimaryImages
                  previewUrl={previewUrl}
                  width={capture.width}
                  height={capture.height}
                  points={result.output.points}
                />
              </div>
            </section>

            <section
              className="patient-visit-record__review"
              aria-labelledby="patient-visit-record-review-title"
            >
              <h3 id="patient-visit-record-review-title">
                Saved review
              </h3>
              <dl>
                <div>
                  <dt>Decision</dt>
                  <dd>Accepted for comparison</dd>
                </div>
                <div>
                  <dt>Completed</dt>
                  <dd>{formatDateTime(currentReview.completedAt)}</dd>
                </div>
                {currentReview.note ? (
                  <div>
                    <dt>Review note</dt>
                    <dd>{currentReview.note}</dd>
                  </div>
                ) : null}
              </dl>
            </section>

            <details className="patient-visit-record__additional">
              <summary>Additional details</summary>
              <div className="patient-visit-record__additional-content">
                <PatientAttentionDensity
                  width={capture.width}
                  height={capture.height}
                  points={result.output.points}
                  faceRegistration={result.faceRegistration}
                />
                <PatientAoiSummary
                  points={result.output.points}
                  faceRegistration={result.faceRegistration}
                />
                <section className="patient-visit-record__technical">
                  <h3>Technical details</h3>
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
                      <dd>Local illustrative sample engine</dd>
                    </div>
                    <div>
                      <dt>Result created</dt>
                      <dd>{formatDateTime(result.createdAt)}</dd>
                    </div>
                  </dl>
                </section>
              </div>
            </details>
          </>
        ) : (
          <div className="patient-visit-record__unavailable" role="status">
            <h3>Historical photo unavailable</h3>
            <p>
              This session no longer contains the exact photograph
              bound to this stored result. No other visit image has
              been substituted.
            </p>
          </div>
        )}

        <div className="patient-visit-record__actions">
          <Link
            className="patient-primary-action"
            to={`/patients/${patient.id}`}
          >
            Return to patient record
          </Link>
        </div>
      </section>
    )
  } else if (nextAction === 'retake') {
    workflowContent = (
      <CapturePanel
        title="Add replacement photo"
        previewUrl={previewUrl}
        previewWidth={capture?.width}
        previewHeight={capture?.height}
        previewDisclosure={previewDisclosure}
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
