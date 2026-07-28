import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { AttentionResultView } from '../components/AttentionResultView'
import { FailClosedState } from '../components/FailClosedState'
import { StatusBadge } from '../components/StatusBadge'
import { getWorkbenchAsset } from '../workbench/catalog'
import { selectExactResultTarget } from '../workbench/reviewPolicy'
import { useWorkspace } from '../workbench/WorkspaceProvider'
import { CONNECTED_ATTENTION_REQUEST_PROFILE_VERSION } from '../workbench/types'

function formatState(value: string): string {
  return value.replaceAll('_', ' ')
}

function conciseLabel(label: string): string {
  return label.replace(/^Standalone synthetic case — /, '')
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

function hasOwn(record: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(record, key)
}

function displayString(value: unknown, fallback: string): string {
  return isNonEmptyString(value) ? value : fallback
}

function storedFreshness(value: unknown): 'current' | 'stale' | 'revoked' | undefined {
  return value === 'current' || value === 'stale' || value === 'revoked'
    ? value
    : undefined
}

function yesNo(value: unknown): string {
  if (value === true) return 'Yes'
  if (value === false) return 'No'
  return 'Unavailable'
}

function statusTone(status: string) {
  if (status === 'succeeded') return 'success' as const
  if (status === 'failed' || status === 'blocked') return 'blocked' as const
  if (status === 'cancelled') return 'warning' as const
  return 'info' as const
}

export function RunDetailPage() {
  const { runId } = useParams()
  const { state, actions } = useWorkspace()
  const [actionFailure, setActionFailure] = useState<string>()
  const run =
    runId && Object.prototype.hasOwnProperty.call(state.runsById, runId)
      ? state.runsById[runId]
      : undefined

  if (!runId || !run) {
    return (
      <FailClosedState
        eyebrow="Session state unavailable"
        title="Run unavailable in this session"
        requestedId={runId}
        description="The exact route ID does not match an in-memory run. This can happen after refresh or session expiry; no fixture run was substituted."
        backTo="/runs"
        backLabel="Back to session runs"
      />
    )
  }

  const asset = getWorkbenchAsset(run.caseId)
  if (!asset) {
    return (
      <FailClosedState
        eyebrow="Case binding unavailable"
        title="Simulation unavailable"
        requestedId={run.caseId}
        description="The selected simulation is not bound to a canonical case, so no alternate case was substituted."
        backTo="/runs"
        backLabel="Back to recent simulations"
      />
    )
  }

  const attempts = (Array.isArray(run.attemptIds) ? run.attemptIds : []).flatMap(
    (attemptId) => {
      if (
        !isNonEmptyString(attemptId) ||
        !Object.prototype.hasOwnProperty.call(state.attemptsById, attemptId)
      ) return []
      const attempt: unknown = state.attemptsById[attemptId]
      return isRecord(attempt) && attempt.clientRunId === run.clientRunId
        ? [{ attemptId, record: attempt }]
        : []
    },
  )
  const activeAttemptId = isNonEmptyString(run.activeAttemptId)
    ? run.activeAttemptId
    : undefined
  const activeAttempt = activeAttemptId
    ? attempts.find((attempt) => attempt.attemptId === activeAttemptId)
    : undefined
  const exactActiveAttempt =
    activeAttempt?.record.status === run.status ? activeAttempt : undefined
  const exactBinding = isRecord(exactActiveAttempt?.record.binding)
    ? exactActiveAttempt.record.binding
    : undefined
  const binding = exactBinding ?? attempts
    .map((attempt) => attempt.record.binding)
    .find(isRecord)
  const exactResult = isRecord(exactActiveAttempt?.record.result)
    ? exactActiveAttempt.record.result
    : undefined
  const evidenceResult = exactResult ?? [...attempts]
    .reverse()
    .map((attempt) => attempt.record.result)
    .find(isRecord)
  const exactOutput = isRecord(exactResult?.output) ? exactResult.output : undefined
  const rawEvidenceOutput = isRecord(evidenceResult?.output)
    ? evidenceResult.output
    : undefined
  const exactResultDigest = isNonEmptyString(exactOutput?.resultDigest)
    ? exactOutput.resultDigest
    : undefined
  const exactInputFingerprint = isNonEmptyString(exactBinding?.inputFingerprint)
    ? exactBinding.inputFingerprint
    : undefined
  const reviewSelection =
    exactActiveAttempt && exactResultDigest && exactInputFingerprint
    ? selectExactResultTarget(state, {
        runId: run.clientRunId,
        attemptId: exactActiveAttempt.attemptId,
        resultDigest: exactResultDigest,
        inputFingerprint: exactInputFingerprint,
      })
    : undefined
  const reviewEligible = reviewSelection?.ok === true
  const exactVisual = reviewSelection?.ok
    ? (() => {
        const output = reviewSelection.target.output
        const resultAsset = getWorkbenchAsset(output.binding.assetId)
        const resultRoi = hasOwn(state.roisByCase, output.binding.caseId)
          ? state.roisByCase[output.binding.caseId]
          : undefined
        return resultAsset && resultRoi
          ? { asset: resultAsset, output, roi: resultRoi }
          : undefined
      })()
    : undefined
  const evidenceOutput: Record<string, unknown> | undefined = reviewSelection?.ok
    ? reviewSelection.target.output as unknown as Record<string, unknown>
    : rawEvidenceOutput
  const connectedEvidence = evidenceOutput?.origin === 'model_prediction'
  const evidenceFreshness = storedFreshness(evidenceResult?.freshness)
  const resultState = reviewEligible
    ? 'Current'
    : evidenceResult
      ? evidenceFreshness === 'stale' || evidenceFreshness === 'revoked'
        ? evidenceFreshness
        : 'Integrity unavailable'
      : 'Not available yet'
  const existingReview = reviewEligible && exactActiveAttempt && exactResultDigest && exactInputFingerprint
    ? state.reviewOrder
        .filter((reviewId) =>
          Object.prototype.hasOwnProperty.call(state.reviewsById, reviewId),
        )
        .map((reviewId) => state.reviewsById[reviewId])
        .find(
          (review) =>
            review?.runId === run.clientRunId &&
            review.attemptId === exactActiveAttempt.attemptId &&
            review.resultDigest === exactResultDigest &&
            review.inputFingerprint === exactInputFingerprint,
        )
    : undefined
  const activeStatus = exactActiveAttempt?.record.status
  const canCancel = activeStatus === 'queued' || activeStatus === 'running'
  const canRetry = exactActiveAttempt
    ? ['blocked', 'failed', 'cancelled'].includes(String(activeStatus))
    : false
  const analysisHref = `/analysis?case=${encodeURIComponent(run.caseId)}&run=${encodeURIComponent(run.clientRunId)}`
  const reviewHref = reviewEligible && exactActiveAttempt
    ? existingReview
      ? `/research/reviews/${encodeURIComponent(existingReview.id)}`
      : `/research/reviews/new?run=${encodeURIComponent(run.clientRunId)}&attempt=${encodeURIComponent(exactActiveAttempt.attemptId)}`
    : undefined

  let nextStep = 'Return to Analysis to continue this simulation.'
  if (run.status === 'queued') {
    nextStep = 'The simulation is waiting to start. You can cancel this request.'
  } else if (run.status === 'running') {
    nextStep = 'The simulation is running. You can cancel this request.'
  } else if (run.status === 'failed') {
    nextStep = 'The simulation did not complete. Retry the exact same input.'
  } else if (run.status === 'cancelled') {
    nextStep = 'The request was cancelled. Retry the exact same input when ready.'
  } else if (run.status === 'blocked') {
    nextStep = 'The simulation is blocked. Retry the exact same input when ready.'
  } else if (run.status === 'succeeded' && reviewEligible) {
    nextStep = 'The current result is ready for research review.'
  } else if (run.status === 'succeeded' && evidenceResult) {
    nextStep = evidenceFreshness === 'stale' || evidenceFreshness === 'revoked'
      ? `This result is ${evidenceFreshness} and cannot be reviewed.`
      : 'Result integrity is unavailable, so this result cannot be reviewed.'
  }

  const bindingAssetSha = displayString(binding?.assetSha256, 'Unavailable')
  const bindingModel = connectedEvidence
    ? CONNECTED_ATTENTION_REQUEST_PROFILE_VERSION
    : displayString(binding?.modelVersion, 'Not configured')
  const bindingFingerprint = displayString(binding?.inputFingerprint, 'Unavailable')
  const bindingRoi = isNonEmptyString(binding?.roiId) &&
    typeof binding?.roiVersion === 'number' &&
    Number.isFinite(binding.roiVersion)
    ? `${binding.roiId} · version ${binding.roiVersion}`
    : 'Unavailable'
  const evidenceDigest = displayString(evidenceOutput?.resultDigest, 'Unavailable')
  const evidenceProvenance = isRecord(evidenceOutput?.provenance)
    ? evidenceOutput.provenance
    : undefined
  const evidenceModelIdentity = isRecord(evidenceOutput?.modelIdentity)
    ? evidenceOutput.modelIdentity
    : undefined
  const connectedIdentityAvailable =
    !connectedEvidence ||
    (evidenceModelIdentity !== undefined &&
      isNonEmptyString(evidenceModelIdentity.modelId) &&
      isNonEmptyString(evidenceModelIdentity.modelVersion) &&
      isNonEmptyString(evidenceModelIdentity.artifactSha256) &&
      isNonEmptyString(evidenceModelIdentity.preprocessingVersion) &&
      isNonEmptyString(evidenceModelIdentity.calibrationVersion) &&
      isNonEmptyString(evidenceModelIdentity.displayScaleId))
  const provenanceAvailable =
    evidenceProvenance !== undefined &&
    isNonEmptyString(evidenceProvenance.engine) &&
    isNonEmptyString(evidenceProvenance.engineVersion) &&
    isNonEmptyString(evidenceOutput?.origin) &&
    typeof evidenceProvenance.networkAccessed === 'boolean' &&
    typeof evidenceProvenance.storageAccessed === 'boolean' &&
    connectedIdentityAvailable &&
    (connectedEvidence
      ? evidenceProvenance.observedGazePayloadIncluded === false &&
        evidenceProvenance.trainingDataProvenance === 'not_disclosed'
      : typeof evidenceProvenance.humanGazeData === 'boolean')
  const reviewIntegrity = reviewSelection?.ok
    ? 'Exact current result'
    : reviewSelection
      ? reviewSelection.blockers.map((entry) => entry.message).join(' ')
      : evidenceResult
        ? 'Stored result integrity is unavailable.'
        : 'No succeeded active result is available.'

  const cancel = () => {
    setActionFailure(undefined)
    actions.cancelRun(run.clientRunId)
  }

  const retry = () => {
    setActionFailure(undefined)
    try {
      actions.retryRun(run.clientRunId)
    } catch (error) {
      setActionFailure(
        error instanceof Error
          ? error.message
          : 'The exact input could not be retried.',
      )
    }
  }

  return (
    <div className="workspace-page run-detail-page">
      <header className="workspace-page__header page-shell">
        <div>
          <Link className="workspace-back-link" to="/runs">
            {connectedEvidence ? '← Recent runs' : '← Recent simulations'}
          </Link>
          <h1>{conciseLabel(asset.label)}</h1>
          <p>{asset.id}</p>
        </div>
      </header>

      <section
        className={`run-primary page-shell${exactVisual ? ' run-primary--result' : ''}`}
        aria-label="Selected run"
      >
        {exactVisual ? (
          <div className="run-primary__result-story">
            <AttentionResultView
              asset={exactVisual.asset}
              output={exactVisual.output}
              roi={exactVisual.roi}
              layout="clinician-stack"
            />
          </div>
        ) : (
          <figure className="run-primary__preview">
            <img
              src={asset.url}
              alt={`${conciseLabel(asset.label)} synthetic preview`}
              width="1024"
              height="1024"
              loading="eager"
              fetchPriority="high"
              decoding="async"
            />
          </figure>
        )}

        <div className="run-primary__next-step">
          <div
            className="run-primary__progress"
            role="status"
            aria-label="Run progress"
            aria-live="polite"
            aria-atomic="true"
          >
            <span aria-label={`Run status ${formatState(run.status)}`}>
              <StatusBadge tone={statusTone(run.status)}>{formatState(run.status)}</StatusBadge>
            </span>
            <span className="run-primary__result-state">Result: {resultState}</span>
          </div>
          <h2>Next step</h2>
          <p>{nextStep}</p>
          <div className="run-primary__actions">
            {canCancel ? (
              <button
                className="workspace-button workspace-button--primary"
                type="button"
                onClick={cancel}
              >
                Cancel request
              </button>
            ) : null}
            {canRetry ? (
              <button
                className="workspace-button workspace-button--primary"
                type="button"
                onClick={retry}
              >
                Retry exact input
              </button>
            ) : null}
            {reviewEligible && reviewHref ? (
              <Link className="workspace-button workspace-button--primary" to={reviewHref}>
                Review result
              </Link>
            ) : null}
            <Link className="workspace-button workspace-button--secondary" to={analysisHref}>
              Return to Analysis
            </Link>
          </div>
          {actionFailure ? <p className="run-primary__failure" role="alert">{actionFailure}</p> : null}
        </div>
      </section>

      <details className="run-technical page-shell">
        <summary>Technical details</summary>
        <div className="run-technical__content">
          <section className="workspace-panel run-summary" aria-label="Run binding">
            <h2>Binding</h2>
            <dl>
              <div><dt>Internal run ID</dt><dd><code>{run.clientRunId}</code></dd></div>
              <div><dt>Case binding</dt><dd><code>{run.caseId}</code></dd></div>
              <div><dt>Asset SHA-256</dt><dd><code>{bindingAssetSha}</code></dd></div>
              <div>
                <dt>
                  {connectedEvidence
                    ? 'Connected request contract'
                    : 'Simulation profile'}
                </dt>
                <dd>{bindingModel}</dd>
              </div>
              <div><dt>Full-image source binding</dt><dd>{bindingRoi}</dd></div>
              <div>
                <dt>Input fingerprint</dt>
                <dd><code>{bindingFingerprint}</code></dd>
              </div>
            </dl>
          </section>

          <section className="workspace-panel attempt-panel" aria-label="Attempt timeline">
            <div className="workspace-panel__heading">
              <div><h2>Attempt history</h2></div>
              <strong>{attempts.length}</strong>
            </div>
            <ol className="attempt-timeline">
              {attempts.map(({ attemptId, record }, index) => {
                const parentAttemptId = displayString(record.parentAttemptId, '')
                const attemptStatus = displayString(record.status, 'unavailable')
                const failure = isRecord(record.failure) ? record.failure : undefined
                const failureReason = displayString(failure?.reason, '')
                const failureMessage = displayString(failure?.message, '')
                return (
                  <li data-testid="attempt-row" key={attemptId}>
                    <span className="attempt-timeline__index">{index + 1}</span>
                    <div>
                      <code>{attemptId}</code>
                      {parentAttemptId ? (
                        <span>Retry of {parentAttemptId}</span>
                      ) : (
                        <span>Initial attempt</span>
                      )}
                      {failureReason && failureMessage ? (
                        <small>{failureReason}: {failureMessage}</small>
                      ) : null}
                    </div>
                    <StatusBadge tone={statusTone(attemptStatus)}>
                      {formatState(attemptStatus)}
                    </StatusBadge>
                  </li>
                )
              })}
            </ol>
          </section>

          <section
            className="workspace-panel run-result"
            aria-label="Result availability"
            data-testid="result-availability"
          >
            <h2>{reviewEligible ? 'Result available' : 'Result unavailable'}</h2>
            <dl>
              <div><dt>Digest</dt><dd><code>{evidenceDigest}</code></dd></div>
              <div><dt>Freshness</dt><dd>{evidenceFreshness ?? 'Unavailable'}</dd></div>
              <div>
                <dt>Review integrity</dt>
                <dd>{reviewIntegrity}</dd>
              </div>
            </dl>
          </section>

          <section className="workspace-panel run-provenance" aria-label="Run provenance">
            <h2>Provenance</h2>
            {provenanceAvailable && evidenceProvenance ? (
              <dl>
                <div><dt>Engine</dt><dd>{displayString(evidenceProvenance.engine, 'Unavailable')}</dd></div>
                <div><dt>Origin</dt><dd>{displayString(evidenceOutput?.origin, 'Unavailable')}</dd></div>
                <div>
                  <dt>
                    {connectedEvidence
                      ? 'Connected engine version'
                      : 'Simulation engine version'}
                  </dt>
                  <dd>
                    {displayString(evidenceProvenance.engineVersion, 'Unavailable')}
                  </dd>
                </div>
                <div>
                  <dt>Result digest</dt>
                  <dd><code>{evidenceDigest}</code></dd>
                </div>
                {connectedEvidence && evidenceModelIdentity ? (
                  <>
                    <div><dt>Connected model ID</dt><dd>{displayString(evidenceModelIdentity.modelId, 'Unavailable')}</dd></div>
                    <div><dt>Connected model version</dt><dd>{displayString(evidenceModelIdentity.modelVersion, 'Unavailable')}</dd></div>
                    <div><dt>Artifact SHA-256</dt><dd><code>{displayString(evidenceModelIdentity.artifactSha256, 'Unavailable')}</code></dd></div>
                    <div><dt>Preprocessing version</dt><dd>{displayString(evidenceModelIdentity.preprocessingVersion, 'Unavailable')}</dd></div>
                    <div><dt>Calibration version</dt><dd>{displayString(evidenceModelIdentity.calibrationVersion, 'Unavailable')}</dd></div>
                    <div><dt>Display scale ID</dt><dd>{displayString(evidenceModelIdentity.displayScaleId, 'Unavailable')}</dd></div>
                  </>
                ) : null}
                <div data-testid="provenance-network">
                  <dt>Network accessed</dt><dd>{yesNo(evidenceProvenance.networkAccessed)}</dd>
                </div>
                <div data-testid="provenance-storage">
                  <dt>Persistent storage accessed</dt><dd>{yesNo(evidenceProvenance.storageAccessed)}</dd>
                </div>
                {connectedEvidence ? (
                  <>
                    <div data-testid="provenance-observed-gaze-payload">
                      <dt>Observed gaze in result payload</dt>
                      <dd>{yesNo(evidenceProvenance.observedGazePayloadIncluded)}</dd>
                    </div>
                    <div data-testid="provenance-training-data">
                      <dt>Training-data provenance</dt>
                      <dd>Not disclosed</dd>
                    </div>
                  </>
                ) : (
                  <div data-testid="provenance-human-gaze">
                    <dt>Human gaze used by simulation</dt>
                    <dd>{yesNo(evidenceProvenance.humanGazeData)}</dd>
                  </div>
                )}
              </dl>
            ) : (
              <p>Provenance is unavailable because the stored evidence is incomplete.</p>
            )}
            {connectedEvidence && provenanceAvailable ? (
              <p className="run-provenance__note">
                This synthetic spatial contract rehearsal uses the
                response-reported connected model identity as research provenance,
                not clinical certification.
              </p>
            ) : null}
          </section>
        </div>
      </details>
    </div>
  )
}
