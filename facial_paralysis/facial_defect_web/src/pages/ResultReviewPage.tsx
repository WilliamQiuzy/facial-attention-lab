import {
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  ClipboardPenLine,
  LockKeyhole,
  RotateCcw,
  ShieldX,
} from 'lucide-react'
import { useId, useRef, useState, type FormEvent, type ReactNode } from 'react'
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom'
import { FailClosedState } from '../components/FailClosedState'
import { StatusBadge } from '../components/StatusBadge'
import { getWorkbenchAsset } from '../workbench/catalog'
import {
  evaluatePatientReportEligibility,
  reviewRuntimeStateBlockers,
  selectExactResultTarget,
  type ExactResultTarget,
  type ReviewPolicyBlocker,
} from '../workbench/reviewPolicy'
import { useWorkspace } from '../workbench/WorkspaceProvider'
import type {
  InferenceOutput,
  ResearchReviewNote,
  ResultReview,
  ResultReviewEvent,
  ReviewStatus,
} from '../workbench/types'
import { CONNECTED_ATTENTION_REQUEST_PROFILE_VERSION } from '../workbench/types'

const REVIEW_STATUSES: readonly ReviewStatus[] = [
  'awaiting_review',
  'approved_for_research',
  'changes_requested',
  'revoked',
]

const REVIEW_KEYS = [
  'id',
  'runId',
  'attemptId',
  'resultDigest',
  'inputFingerprint',
  'authorId',
  'reviewerId',
  'status',
  'decision',
  'events',
] as const

const REVIEW_EVENT_KEYS = ['sequence', 'decision', 'actorId', 'note'] as const
const REVIEW_NOTE_KEYS = ['rationale', 'limitations'] as const

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return false
  }
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

function hasExactOwnKeys(
  value: Record<string, unknown>,
  expected: readonly string[],
): boolean {
  const ownKeys = Reflect.ownKeys(value)
  return (
    ownKeys.length === expected.length &&
    ownKeys.every(
      (key) => typeof key === 'string' && expected.includes(key),
    )
  )
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

function isReviewStatus(value: unknown): value is ReviewStatus {
  return REVIEW_STATUSES.some((status) => status === value)
}

function normalizeReviewEvent(
  value: unknown,
  index: number,
): ResultReviewEvent | undefined {
  if (!isPlainRecord(value) || !hasExactOwnKeys(value, REVIEW_EVENT_KEYS)) {
    return undefined
  }
  const note = value.note
  if (!isPlainRecord(note) || !hasExactOwnKeys(note, REVIEW_NOTE_KEYS)) {
    return undefined
  }
  if (
    value.sequence !== index + 1 ||
    !isReviewStatus(value.decision) ||
    (value.actorId !== 'demo_author' && value.actorId !== 'demo_reviewer') ||
    !isNonEmptyString(note.rationale) ||
    !isNonEmptyString(note.limitations)
  ) {
    return undefined
  }
  return {
    sequence: value.sequence,
    decision: value.decision,
    actorId: value.actorId,
    note: {
      rationale: note.rationale,
      limitations: note.limitations,
    },
  }
}

function hasCoherentReviewLifecycle(
  events: readonly ResultReviewEvent[],
  status: ReviewStatus,
): boolean {
  const first = events[0]
  if (
    !first ||
    first.decision !== 'awaiting_review' ||
    first.actorId !== 'demo_author'
  ) {
    return false
  }
  for (let index = 1; index < events.length; index += 1) {
    const previous = events[index - 1]
    const current = events[index]
    if (!previous || !current) return false
    const legalTransition =
      (previous.decision === 'awaiting_review' &&
        (current.decision === 'approved_for_research' ||
          current.decision === 'changes_requested')) ||
      (previous.decision === 'changes_requested' &&
        current.decision === 'awaiting_review') ||
      (previous.decision === 'approved_for_research' &&
        current.decision === 'revoked')
    const legalActor =
      current.decision === 'awaiting_review'
        ? current.actorId === 'demo_author'
        : current.actorId === 'demo_reviewer'
    if (!legalTransition || !legalActor) return false
  }
  return events[events.length - 1]?.decision === status
}

function normalizeResultReview(
  value: unknown,
  requestedId: string,
): ResultReview | undefined {
  try {
    if (!isPlainRecord(value) || !hasExactOwnKeys(value, REVIEW_KEYS)) {
      return undefined
    }
    if (
      value.id !== requestedId ||
      !isNonEmptyString(value.runId) ||
      !isNonEmptyString(value.attemptId) ||
      !isNonEmptyString(value.resultDigest) ||
      !isNonEmptyString(value.inputFingerprint) ||
      value.authorId !== 'demo_author' ||
      value.reviewerId !== 'demo_reviewer' ||
      !isReviewStatus(value.status) ||
      !isReviewStatus(value.decision) ||
      value.decision !== value.status ||
      !Array.isArray(value.events) ||
      Object.getPrototypeOf(value.events) !== Array.prototype ||
      value.events.length === 0
    ) {
      return undefined
    }

    const rawEvents = value.events
    const eventKeys = Reflect.ownKeys(rawEvents)
    if (
      eventKeys.length !== rawEvents.length + 1 ||
      !Object.hasOwn(rawEvents, 'length')
    ) {
      return undefined
    }

    const events: ResultReviewEvent[] = []
    for (let index = 0; index < rawEvents.length; index += 1) {
      if (!Object.hasOwn(rawEvents, index)) return undefined
      const event = normalizeReviewEvent(rawEvents[index], index)
      if (!event) return undefined
      events.push(event)
    }
    if (!hasCoherentReviewLifecycle(events, value.status)) return undefined

    return {
      id: value.id,
      runId: value.runId,
      attemptId: value.attemptId,
      resultDigest: value.resultDigest,
      inputFingerprint: value.inputFingerprint,
      authorId: value.authorId,
      reviewerId: value.reviewerId,
      status: value.status,
      decision: value.decision,
      events,
    }
  } catch {
    return undefined
  }
}

function humanize(value: string): string {
  return value.replaceAll('_', ' ')
}

function statusTone(status: ReviewStatus) {
  if (status === 'approved_for_research') return 'success' as const
  if (status === 'revoked') return 'blocked' as const
  if (status === 'changes_requested') return 'warning' as const
  return 'info' as const
}

type ReviewTechnicalEvidence = {
  readonly reviewId?: string
  readonly runId: string
  readonly attemptId: string
  readonly resultDigest: string
  readonly inputFingerprint?: string
  readonly assetSha256?: string
  readonly roiId?: string
  readonly roiVersion?: number
  readonly modelVersion?: string
  readonly modelMode?: string
  readonly configurationHash?: string
  readonly connectedModelId?: string
  readonly connectedModelVersion?: string
  readonly artifactSha256?: string
  readonly preprocessingVersion?: string
  readonly calibrationVersion?: string
  readonly displayScaleId?: string
  readonly origin?: string
  readonly engineVersion?: string
  readonly capabilityStatus?: string
  readonly blockers?: readonly ReviewPolicyBlocker[]
}

function TechnicalValue({
  label,
  value,
}: {
  readonly label: string
  readonly value: string
}) {
  return <div><dt>{label}</dt><dd><code>{value}</code></dd></div>
}

function ReviewTechnicalDetails({ evidence }: { readonly evidence: ReviewTechnicalEvidence }) {
  const connected = evidence.origin === 'model_prediction'

  return (
    <details className="task6-technical-details task6-review-technical">
      <summary>Technical details</summary>
      <dl>
        {evidence.reviewId ? <TechnicalValue label="Review ID" value={evidence.reviewId} /> : null}
        <TechnicalValue label="Run ID" value={evidence.runId} />
        <TechnicalValue label="Attempt ID" value={evidence.attemptId} />
        <TechnicalValue label="Result digest" value={evidence.resultDigest} />
        {evidence.inputFingerprint ? <TechnicalValue label="Input fingerprint" value={evidence.inputFingerprint} /> : null}
        {evidence.assetSha256 ? <TechnicalValue label="Asset SHA-256" value={evidence.assetSha256} /> : null}
        {evidence.roiId && evidence.roiVersion !== undefined ? (
          <TechnicalValue
            label="Full-image source binding"
            value={`${evidence.roiId} · version ${evidence.roiVersion}`}
          />
        ) : null}
        {connected ? (
          <TechnicalValue
            label="Connected request contract"
            value={CONNECTED_ATTENTION_REQUEST_PROFILE_VERSION}
          />
        ) : evidence.modelVersion ? (
          <TechnicalValue
            label="Simulation profile"
            value={evidence.modelVersion}
          />
        ) : null}
        {!connected && evidence.modelMode ? (
          <TechnicalValue label="Simulation mode" value={evidence.modelMode} />
        ) : null}
        {!connected && evidence.configurationHash ? <TechnicalValue label="Configuration hash" value={evidence.configurationHash} /> : null}
        {evidence.origin ? <TechnicalValue label="Output origin" value={evidence.origin} /> : null}
        {evidence.engineVersion ? (
          <TechnicalValue
            label={connected ? 'Connected engine version' : 'Simulation engine version'}
            value={evidence.engineVersion}
          />
        ) : null}
        {evidence.capabilityStatus ? <TechnicalValue label="Capability" value={evidence.capabilityStatus} /> : null}
        {connected && evidence.connectedModelId ? <TechnicalValue label="Connected model ID" value={evidence.connectedModelId} /> : null}
        {connected && evidence.connectedModelVersion ? <TechnicalValue label="Connected model version" value={evidence.connectedModelVersion} /> : null}
        {connected && evidence.artifactSha256 ? <TechnicalValue label="Artifact SHA-256" value={evidence.artifactSha256} /> : null}
        {connected && evidence.preprocessingVersion ? <TechnicalValue label="Preprocessing version" value={evidence.preprocessingVersion} /> : null}
        {connected && evidence.calibrationVersion ? <TechnicalValue label="Calibration version" value={evidence.calibrationVersion} /> : null}
        {connected && evidence.displayScaleId ? <TechnicalValue label="Display scale ID" value={evidence.displayScaleId} /> : null}
        {evidence.blockers && evidence.blockers.length > 0 ? (
          <div>
            <dt>Safety checks</dt>
            <dd>
              <ul>
                {evidence.blockers.map((entry) => (
                  <li key={entry.code}><code>{entry.code}</code>: {entry.message}</li>
                ))}
              </ul>
            </dd>
          </div>
        ) : null}
      </dl>
      {connected ? (
        <p className="task6-review-technical__identity-note">
          This synthetic spatial contract rehearsal uses the response-reported
          connected model identity as research provenance, not clinical
          certification.
        </p>
      ) : null}
    </details>
  )
}

function CaseLead({
  caseId,
  status,
  tone,
  nextAction,
}: {
  readonly caseId: string
  readonly status: string
  readonly tone: 'success' | 'info' | 'warning' | 'blocked'
  readonly nextAction: string
}) {
  const asset = getWorkbenchAsset(caseId)
  return (
    <div className="task6-case-lead">
      <div className="task6-case-lead__identity">
        <span>Case</span>
        <strong>{asset?.label ?? 'Case unavailable'}</strong>
        <code>{caseId}</code>
      </div>
      <div className="task6-case-lead__status">
        <span>Result status</span>
        <StatusBadge tone={tone}>{status}</StatusBadge>
        <p>{nextAction}</p>
      </div>
    </div>
  )
}

function ReviewBlockers({ blockers }: { readonly blockers: readonly ReviewPolicyBlocker[] }) {
  if (blockers.length === 0) return null
  return (
    <div className="task6-review-blockers" role="alert">
      <LockKeyhole aria-hidden="true" />
      <div>
        <strong>Exact target is blocked</strong>
        <ul>
          {blockers.map((entry) => (
            <li key={entry.code}>
              <span>{entry.message}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

type StructuredNoteFormProps = {
  readonly heading: string
  readonly description: string
  readonly actions: readonly {
    readonly label: string
    readonly icon?: ReactNode
    readonly tone: 'primary' | 'secondary'
    readonly disabled?: boolean
    readonly onSubmit: (note: ResearchReviewNote) => boolean | void
  }[]
}

function StructuredNoteForm({
  heading,
  description,
  actions,
}: StructuredNoteFormProps) {
  const [rationale, setRationale] = useState('')
  const [limitations, setLimitations] = useState('')
  const [error, setError] = useState<string>()
  const [invalidFields, setInvalidFields] = useState({
    rationale: false,
    limitations: false,
  })
  const rationaleRef = useRef<HTMLTextAreaElement>(null)
  const limitationsRef = useRef<HTMLTextAreaElement>(null)
  const validationMessageId = useId()

  const submit = (action: StructuredNoteFormProps['actions'][number]) => {
    const normalized = {
      rationale: rationale.trim(),
      limitations: limitations.trim(),
    }
    const missing = {
      rationale: normalized.rationale.length === 0,
      limitations: normalized.limitations.length === 0,
    }
    setInvalidFields(missing)
    if (missing.rationale || missing.limitations) {
      setError('Rationale and limitations are both required.')
      if (missing.rationale) {
        rationaleRef.current?.focus()
      } else {
        limitationsRef.current?.focus()
      }
      return
    }
    setError(undefined)
    if (action.onSubmit(normalized) !== false) {
      setInvalidFields({ rationale: false, limitations: false })
      setRationale('')
      setLimitations('')
    }
  }

  return (
    <form
      className="task6-note-form"
      aria-label="Structured research review note"
      onSubmit={(event: FormEvent) => event.preventDefault()}
    >
      <div className="task6-note-form__heading">
        <ClipboardPenLine aria-hidden="true" />
        <div><h2>{heading}</h2><p>{description}</p></div>
      </div>
      <label>
        <span>Rationale</span>
        <textarea
          ref={rationaleRef}
          name="rationale"
          autoComplete="off"
          aria-invalid={invalidFields.rationale || undefined}
          aria-describedby={invalidFields.rationale ? validationMessageId : undefined}
          required
          rows={4}
          value={rationale}
          onChange={(event) => {
            const value = event.currentTarget.value
            setRationale(value)
            if (invalidFields.rationale && value.trim().length > 0) {
              setInvalidFields((current) => ({ ...current, rationale: false }))
            }
          }}
          placeholder="State why this exact synthetic result should advance or change."
        />
      </label>
      <label>
        <span>Limitations</span>
        <textarea
          ref={limitationsRef}
          name="limitations"
          autoComplete="off"
          aria-invalid={invalidFields.limitations || undefined}
          aria-describedby={invalidFields.limitations ? validationMessageId : undefined}
          required
          rows={4}
          value={limitations}
          onChange={(event) => {
            const value = event.currentTarget.value
            setLimitations(value)
            if (invalidFields.limitations && value.trim().length > 0) {
              setInvalidFields((current) => ({ ...current, limitations: false }))
            }
          }}
          placeholder="State the evidence limits and the clinical-use boundary."
        />
      </label>
      {error ? (
        <p
          className="task6-note-form__error"
          id={validationMessageId}
          role="alert"
        >
          {error}
        </p>
      ) : null}
      <div className="task6-note-form__actions">
        {actions.map((action) => (
          <button
            className={`workspace-button workspace-button--${action.tone}`}
            type="button"
            disabled={action.disabled}
            key={action.label}
            onClick={() => submit(action)}
          >
            {action.icon}{action.label}
          </button>
        ))}
      </div>
    </form>
  )
}

function CreateReviewSurface({
  runId,
  attemptId,
  target,
}: {
  readonly runId: string
  readonly attemptId: string
  readonly target: ExactResultTarget
}) {
  const { actions } = useWorkspace()
  const navigate = useNavigate()
  const [failure, setFailure] = useState<string>()

  const create = (note: ResearchReviewNote): boolean => {
    try {
      const { reviewId } = actions.createReview({
        runId,
        attemptId,
        note,
      })
      navigate(`/research/reviews/${encodeURIComponent(reviewId)}`, {
        replace: true,
      })
      return true
    } catch (error) {
      setFailure(
        error instanceof Error
          ? error.message
          : 'The exact result could not enter research review.',
      )
      return false
    }
  }

  return (
    <section className="workspace-page task6-page task6-review-page">
      <header className="workspace-page__header page-shell task6-page__header">
        <div>
          <Link className="workspace-back-link" to="/research/reviews">
            <ArrowLeft aria-hidden="true" /> Back to review results
          </Link>
          <p className="workspace-kicker">Results</p>
          <h1>Review result</h1>
          <p>
            Add a clear rationale and the evidence limits for this result.
          </p>
          <CaseLead
            caseId={target.run.caseId}
            status="Ready for author note"
            tone="info"
            nextAction="Next: add the rationale and limitations."
          />
        </div>
        <div className="task6-boundary-note" role="note">
          <LockKeyhole aria-hidden="true" />
          <div>
            <strong>Research-only review</strong>
            <span>This note does not approve clinical or patient use.</span>
          </div>
        </div>
      </header>

      <div className="page-shell">
        <ReviewTechnicalDetails
          evidence={{
            runId,
            attemptId,
            resultDigest: target.output.resultDigest,
            inputFingerprint: target.attempt.binding.inputFingerprint,
            assetSha256: target.attempt.binding.assetSha256,
            roiId: target.attempt.binding.roiId,
            roiVersion: target.attempt.binding.roiVersion,
            modelVersion: target.attempt.binding.modelVersion,
            modelMode: target.attempt.binding.modelMode,
            configurationHash: target.attempt.binding.configurationHash,
            origin: target.output.origin,
            engineVersion: target.output.provenance.engineVersion,
            capabilityStatus: target.output.capabilityStatus,
            ...(target.output.origin === 'model_prediction'
              ? {
                  connectedModelId: target.output.modelIdentity.modelId,
                  connectedModelVersion: target.output.modelIdentity.modelVersion,
                  artifactSha256: target.output.modelIdentity.artifactSha256,
                  preprocessingVersion:
                    target.output.modelIdentity.preprocessingVersion,
                  calibrationVersion:
                    target.output.modelIdentity.calibrationVersion,
                  displayScaleId: target.output.modelIdentity.displayScaleId,
                }
              : {}),
          }}
        />
      </div>

      <div className="page-shell task6-review-layout">
        <section className="workspace-panel task6-result-binding" aria-label="Exact result binding">
          <p className="workspace-kicker">Before you continue</p>
          <h2>Record what the result supports</h2>
          <p className="task6-result-binding__copy">
            Explain why this result is suitable for research discussion and name what the
            evidence cannot establish. Your note stays attached to this exact result.
          </p>
          <div className="task6-clinical-block">
            <ShieldX aria-hidden="true" /> Clinical use and patient interpretation remain blocked.
          </div>
        </section>

        <section className="workspace-panel task6-review-command">
          <StructuredNoteForm
            heading="Submit structured author note"
            description="Both fields become an append-only review event. Free-form notes never enter a patient export."
            actions={[
              {
                label: 'Create review',
                icon: <ArrowRight aria-hidden="true" />,
                tone: 'primary',
                onSubmit: create,
              },
            ]}
          />
          {failure ? <p className="task6-note-form__error" role="alert">{failure}</p> : null}
        </section>
      </div>
    </section>
  )
}

function ReviewHistory({ review }: { readonly review: ResultReview }) {
  return (
    <section
      className="workspace-panel task6-review-history"
      role="region"
      aria-label="Review event history"
    >
      <div className="task6-section-heading">
        <RotateCcw aria-hidden="true" />
        <div><p className="workspace-kicker">Append-only</p><h2>Review event history</h2></div>
      </div>
      <ol>
        {review.events.map((event) => (
          <li key={event.sequence}>
            <div className="task6-review-history__sequence">
              <span>Event {String(event.sequence).padStart(2, '0')}</span>
              <strong>Decision: {humanize(event.decision)}</strong>
            </div>
            <dl>
              <div><dt>Role</dt><dd>{event.actorId === 'demo_author' ? 'Author' : 'Independent reviewer'}</dd></div>
              <div><dt>Rationale</dt><dd>{event.note.rationale}</dd></div>
              <div><dt>Limitations</dt><dd>{event.note.limitations}</dd></div>
            </dl>
          </li>
        ))}
      </ol>
    </section>
  )
}

function ExistingReviewSurface({ review }: { readonly review: ResultReview }) {
  const { state, actions } = useWorkspace()
  const selection = selectExactResultTarget(state, review)
  const patient = evaluatePatientReportEligibility(state, review.id)
  const run = Object.hasOwn(state.runsById, review.runId)
    ? state.runsById[review.runId]
    : undefined
  const attempt = Object.hasOwn(state.attemptsById, review.attemptId)
    ? state.attemptsById[review.attemptId]
    : undefined
  const binding = attempt?.binding
  const caseId = selection.ok
    ? selection.target.run.caseId
    : run?.caseId ?? 'Case unavailable'
  const nextAction = review.status === 'awaiting_review'
    ? 'Next: record an independent decision.'
    : review.status === 'changes_requested'
      ? 'Next: update the rationale and resubmit.'
      : review.status === 'approved_for_research'
        ? patient.eligible
          ? 'Next: open the gated patient explanation or revoke approval.'
          : 'Next: resolve the safety checks or revoke approval.'
        : 'No further action. Patient handoff is blocked.'
  const technicalBlockers = [
    ...(selection.ok ? [] : selection.blockers),
    ...(review.status === 'approved_for_research' && !patient.eligible
      ? patient.blockers
      : []),
  ]

  return (
    <section className="workspace-page task6-page task6-review-page">
      <p
        className="sr-only"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        Review status: {humanize(review.status)}. {review.events.length} recorded event{review.events.length === 1 ? '' : 's'}. {nextAction}
      </p>
      <header className="workspace-page__header page-shell task6-page__header">
        <div>
          <Link className="workspace-back-link" to="/research/reviews">
            <ArrowLeft aria-hidden="true" /> Back to review results
          </Link>
          <p className="workspace-kicker">Results</p>
          <h1>Review result</h1>
          <p>
            Check the recorded evidence and choose the next safe action.
          </p>
          <CaseLead
            caseId={caseId}
            status={humanize(review.status)}
            tone={statusTone(review.status)}
            nextAction={nextAction}
          />
        </div>
        <div className="task6-boundary-note" role="note">
          <LockKeyhole aria-hidden="true" />
          <div>
            <strong>Research-only decision</strong>
            <span>{review.events.length} recorded event{review.events.length === 1 ? '' : 's'} · clinical use remains blocked</span>
          </div>
        </div>
      </header>

      <div className="page-shell">
        <ReviewTechnicalDetails
          evidence={{
            reviewId: review.id,
            runId: review.runId,
            attemptId: review.attemptId,
            resultDigest: review.resultDigest,
            inputFingerprint: review.inputFingerprint,
            assetSha256: binding?.assetSha256,
            roiId: binding?.roiId,
            roiVersion: binding?.roiVersion,
            modelVersion: binding?.modelVersion,
            modelMode: binding?.modelMode,
            configurationHash: binding?.configurationHash,
            origin: attempt?.result?.output.origin,
            engineVersion: attempt?.result?.output.provenance.engineVersion,
            capabilityStatus: attempt?.result?.output.capabilityStatus,
            ...(attempt?.result?.output.origin === 'model_prediction'
              ? {
                  connectedModelId:
                    attempt.result.output.modelIdentity.modelId,
                  connectedModelVersion:
                    attempt.result.output.modelIdentity.modelVersion,
                  artifactSha256:
                    attempt.result.output.modelIdentity.artifactSha256,
                  preprocessingVersion:
                    attempt.result.output.modelIdentity.preprocessingVersion,
                  calibrationVersion:
                    attempt.result.output.modelIdentity.calibrationVersion,
                  displayScaleId:
                    attempt.result.output.modelIdentity.displayScaleId,
                }
              : {}),
            blockers: technicalBlockers,
          }}
        />
      </div>

      <div className="page-shell task6-review-detail-grid">
        <ReviewHistory review={review} />
        <aside className="task6-review-rail">
          {!selection.ok ? <ReviewBlockers blockers={selection.blockers} /> : null}

          {review.status !== 'revoked' ? (
            <section className="workspace-panel task6-review-command">
              <StructuredNoteForm
                heading={
                  review.status === 'changes_requested'
                    ? 'Resubmit structured response'
                    : review.status === 'approved_for_research'
                      ? 'Withdraw research approval'
                      : 'Record independent decision'
                }
                description="Rationale and limitations are required for every lifecycle event."
                actions={
                  review.status === 'awaiting_review'
                    ? [
                        {
                          label: 'Approve for research demo',
                          icon: <CheckCircle2 aria-hidden="true" />,
                          tone: 'primary',
                          disabled: !selection.ok,
                          onSubmit: (note) => actions.approveReview(review.id, note),
                        },
                        {
                          label: 'Request changes',
                          tone: 'secondary',
                          disabled: !selection.ok,
                          onSubmit: (note) => actions.requestReviewChanges(review.id, note),
                        },
                      ]
                    : review.status === 'changes_requested'
                      ? [
                          {
                            label: 'Resubmit for review',
                            icon: <RotateCcw aria-hidden="true" />,
                            tone: 'primary',
                            disabled: !selection.ok,
                            onSubmit: (note) => actions.resubmitReview(review.id, note),
                          },
                        ]
                      : [
                          {
                            label: 'Revoke research approval',
                            icon: <ShieldX aria-hidden="true" />,
                            tone: 'secondary',
                            onSubmit: (note) => actions.revokeReview(review.id, note),
                          },
                        ]
                }
              />
            </section>
          ) : (
            <div className="task6-revoked-state">
              <ShieldX aria-hidden="true" />
              <strong>Research approval revoked</strong>
              <p>Patient preview and export remain unavailable for this review.</p>
            </div>
          )}

          {patient.eligible ? (
            <Link
              className="workspace-button workspace-button--primary task6-patient-handoff"
              to={`/patient-report?review=${encodeURIComponent(review.id)}`}
            >
              Open gated patient explanation <ArrowRight aria-hidden="true" />
            </Link>
          ) : review.status === 'approved_for_research' ? (
            <ReviewBlockers blockers={patient.blockers} />
          ) : null}
        </aside>
      </div>
    </section>
  )
}

function creationBinding(search: string):
  | { readonly ok: true; readonly runId: string; readonly attemptId: string }
  | { readonly ok: false; readonly reason: string } {
  const parameters = new URLSearchParams(search)
  const runs = parameters.getAll('run')
  const attempts = parameters.getAll('attempt')
  const onlyKnownKeys = [...parameters.keys()].every(
    (key) => key === 'run' || key === 'attempt',
  )
  if (runs.length > 1 || attempts.length > 1) {
    return { ok: false, reason: 'Duplicate run or attempt parameters were supplied' }
  }
  const runId = runs[0] ?? ''
  const attemptId = attempts[0] ?? ''
  if (
    !onlyKnownKeys ||
    runs.length !== 1 ||
    attempts.length !== 1 ||
    !runId ||
    !attemptId ||
    runId !== runId.trim() ||
    attemptId !== attemptId.trim()
  ) {
    return { ok: false, reason: 'No exact run and attempt were supplied' }
  }
  return { ok: true, runId, attemptId }
}

export function ResultReviewPage() {
  const { reviewId } = useParams()
  const { search } = useLocation()
  const { state } = useWorkspace()
  const runtimeBlockers = reviewRuntimeStateBlockers(state)

  if (runtimeBlockers.length > 0) {
    return (
      <FailClosedState
        eyebrow="Exact review binding required"
        title="Review target unavailable"
        requestedId={runtimeBlockers.map((entry) => entry.message).join(' ')}
        description="The malformed run and attempt session state was not interpreted or replaced. No result evidence or review action is available."
        backTo="/research/reviews"
        backLabel="Back to reviews"
      />
    )
  }

  if (reviewId === 'new') {
    const binding = creationBinding(search)
    if (!binding.ok) {
      return (
        <FailClosedState
          eyebrow="Exact review binding required"
          title="Review target unavailable"
          requestedId={binding.reason}
          description="Review creation requires exactly one run and one attempt from the current in-memory session. No target was substituted."
          backTo="/research/reviews"
          backLabel="Back to reviews"
        />
      )
    }
    const run = Object.hasOwn(state.runsById, binding.runId)
      ? state.runsById[binding.runId]
      : undefined
    const attempt = Object.hasOwn(state.attemptsById, binding.attemptId)
      ? state.attemptsById[binding.attemptId]
      : undefined
    const output = attempt?.result?.output
    const fingerprint = attempt?.binding?.inputFingerprint
    const selected = output && fingerprint
      ? selectExactResultTarget(state, {
          runId: binding.runId,
          attemptId: binding.attemptId,
          resultDigest: output.resultDigest,
          inputFingerprint: fingerprint,
        })
      : undefined
    if (!run || !selected?.ok) {
      const reason = selected && !selected.ok
        ? selected.blockers.map((entry) => entry.message).join(' ')
        : `${binding.runId} · ${binding.attemptId}`
      return (
        <FailClosedState
          eyebrow="Exact review binding required"
          title="Review target unavailable"
          requestedId={reason}
          description="Only one exact, active, succeeded, current result may enter research review. No fixture or alternate attempt was substituted."
          backTo="/research/reviews"
          backLabel="Back to reviews"
        />
      )
    }
    return (
      <CreateReviewSurface
        runId={binding.runId}
        attemptId={binding.attemptId}
        target={selected.target}
      />
    )
  }

  const rawReview: unknown =
    reviewId && Object.hasOwn(state.reviewsById, reviewId)
      ? state.reviewsById[reviewId]
      : undefined
  const exactReview = reviewId
    ? normalizeResultReview(rawReview, reviewId)
    : undefined
  if (!reviewId || !exactReview) {
    return (
      <FailClosedState
        eyebrow="Session state unavailable"
        title="Review unavailable in this session"
        requestedId={reviewId}
        description="The exact route ID does not resolve to a well-formed in-memory review. Refresh-lost, malformed, digest-only, and prototype-polluted records are never replaced with a fixture."
        backTo="/research/reviews"
        backLabel="Back to reviews"
      />
    )
  }

  return <ExistingReviewSurface review={exactReview} />
}
