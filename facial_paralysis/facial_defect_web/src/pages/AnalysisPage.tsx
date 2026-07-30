import {
  ArrowLeft,
  ArrowRight,
  Ban,
  FlaskConical,
  LoaderCircle,
  RotateCcw,
  Square,
} from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { AttentionResultView } from '../components/AttentionResultView'
import { FailClosedState } from '../components/FailClosedState'
import { PreflightGateList } from '../components/PreflightGateList'
import { RunTimeline } from '../components/RunTimeline'
import { getWorkbenchAsset } from '../workbench/catalog'
import { deriveClinicalAoiPresentation } from '../workbench/clinicalAoiPresentation'
import {
  CONNECTED_INFERENCE_WATERMARK,
  validateInferenceOutputEnvelope,
} from '../workbench/inferenceEnvelope'
import { createCanonicalInferenceBindingSnapshot } from '../workbench/mockEngine'
import { getCaseRoi } from '../workbench/reducer'
import { selectExactResultTarget } from '../workbench/reviewPolicy'
import { isVerifiedFullImageSourceBinding } from '../workbench/sourceBinding'
import { useWorkspace } from '../workbench/WorkspaceProvider'
import {
  CONNECTED_ATTENTION_REQUEST_PROFILE_VERSION,
  WorkbenchError,
  type InferenceBinding,
  type InferenceConfiguration,
  type MockModelVersion,
  type NormalizedRoi,
} from '../workbench/types'

const DEFAULT_MODEL_VERSION: MockModelVersion = 'mock-salience-v0.3'
const DEFAULT_CONFIG: InferenceConfiguration = {
  threshold: 0.42,
  smoothing: 0.27,
}

function formatShare(value: number): string {
  return `${Math.round(value * 100)}%`
}

function sameGeometry(first: NormalizedRoi, second: NormalizedRoi): boolean {
  return (
    first.x === second.x &&
    first.y === second.y &&
    first.width === second.width &&
    first.height === second.height
  )
}

function isCanonicalBinding(binding: InferenceBinding): boolean {
  try {
    return (
      JSON.stringify(createCanonicalInferenceBindingSnapshot(binding)) ===
      JSON.stringify(binding)
    )
  } catch {
    return false
  }
}

function sameStoredBinding(
  first: InferenceBinding,
  second: InferenceBinding,
): boolean {
  return JSON.stringify(first) === JSON.stringify(second)
}

function InvalidAnalysisCase({ reason }: { readonly reason: string }) {
  return (
    <FailClosedState
      eyebrow="Authoritative case routing"
      title="Case unavailable"
      requestedId={reason}
      description="Inference opens only for one exact ID in the canonical ten-case synthetic catalog. No fixture or alternate case was substituted."
      backTo="/cases"
      backLabel="Back to cases"
    />
  )
}

function MissingSourceBinding({
  caseId,
  label,
}: {
  readonly caseId: string
  readonly label: string
}) {
  return (
    <section className="workspace-page fail-closed-page">
      <div className="fail-closed-state page-shell">
        <Ban aria-hidden="true" />
        <p className="workspace-kicker">Run preparation</p>
        <h1>Source image binding required</h1>
        <code>{caseId}</code>
        <p>
          {label} is a recognized synthetic case, but its required full-image source
          binding is missing. No simulation was started.
        </p>
        <p className="run-command-panel__failure" role="alert">
          Restore the internal full-image binding before running this case.
        </p>
        <div className="fail-closed-state__actions">
          <Link
            className="workspace-button workspace-button--primary"
            to={`/cases/${encodeURIComponent(caseId)}/roi`}
          >
            Restore source binding
          </Link>
          <Link className="workspace-button workspace-button--quiet" to="/cases">
            Back to cases
          </Link>
        </div>
      </div>
    </section>
  )
}

function InvalidAnalysisRun({
  caseId,
  reason,
  integrityRunId,
}: {
  readonly caseId: string
  readonly reason: string
  readonly integrityRunId?: string
}) {
  if (integrityRunId) {
    return (
      <section className="workspace-page fail-closed-page">
        <div className="fail-closed-state page-shell">
          <Ban aria-hidden="true" />
          <p className="workspace-kicker">Exact session run routing</p>
          <h1>Run unavailable</h1>
          <code>{reason}</code>
          <p className="run-command-panel__failure" role="alert">
            Result integrity validation failed. The selected succeeded record is
            internally inconsistent, so its map, digest, and review are unavailable.
          </p>
          <div className="fail-closed-state__actions">
            <Link
              className="workspace-button workspace-button--secondary"
              to={`/runs/${encodeURIComponent(integrityRunId)}`}
            >
              Open exact run detail
            </Link>
            <Link
              className="workspace-button workspace-button--quiet"
              to={`/analysis?case=${encodeURIComponent(caseId)}`}
            >
              Clear run selection
            </Link>
          </div>
        </div>
      </section>
    )
  }

  return (
    <FailClosedState
      eyebrow="Exact session run routing"
      title="Run unavailable"
      requestedId={reason}
      description="The run query must identify one in-memory run bound to this exact case. No active, recent, or fixture run was substituted."
      backTo={`/analysis?case=${encodeURIComponent(caseId)}`}
      backLabel="Clear run selection"
    />
  )
}

export function AnalysisPage() {
  const { search } = useLocation()
  const navigate = useNavigate()
  const { state, actions, gatewayMode } = useWorkspace()
  const searchParameters = new URLSearchParams(search)
  const caseParameters = searchParameters.getAll('case')
  const runParameters = searchParameters.getAll('run')
  const requestedCaseId = caseParameters.length === 1 ? caseParameters[0].trim() : ''
  const requestedRunId = runParameters.length === 1 ? runParameters[0] : ''
  const asset = requestedCaseId ? getWorkbenchAsset(requestedCaseId) : undefined
  const roi = asset ? getCaseRoi(state, asset.id) : undefined
  const hasValidOptionalRunQuery =
    runParameters.length === 0 ||
    (runParameters.length === 1 &&
      requestedRunId.length > 0 &&
      requestedRunId === requestedRunId.trim())
  const selectedRun =
    hasValidOptionalRunQuery &&
    requestedRunId &&
    Object.prototype.hasOwnProperty.call(state.runsById, requestedRunId)
      ? state.runsById[requestedRunId]
      : undefined
  const selectedAttempt =
    selectedRun?.activeAttemptId &&
    Object.prototype.hasOwnProperty.call(state.attemptsById, selectedRun.activeAttemptId)
      ? state.attemptsById[selectedRun.activeAttemptId]
      : undefined
  const selectedBinding = selectedAttempt?.binding
  const exactSelectedRunAttemptBindingLinkage = Boolean(
    selectedRun &&
      selectedAttempt &&
      selectedBinding &&
      selectedRun.status === selectedAttempt.status &&
      selectedRun.activeAttemptId === selectedAttempt.id &&
      selectedRun.attemptIds.includes(selectedAttempt.id) &&
      selectedAttempt.clientRunId === selectedRun.clientRunId &&
      selectedAttempt.attemptToken === selectedBinding.attemptToken &&
      selectedBinding.clientRunId === selectedRun.clientRunId &&
      selectedBinding.caseId === selectedRun.caseId &&
      selectedBinding.assetId === selectedRun.assetId &&
      selectedBinding.roiStatus === 'approved',
  )
  const runMatchesRoute = Boolean(
    selectedRun &&
      asset &&
      selectedRun.clientRunId === requestedRunId &&
      selectedRun.caseId === asset.id &&
      selectedRun.assetId === asset.id,
  )
  const run = runMatchesRoute ? selectedRun : undefined
  const attempt = run ? selectedAttempt : undefined
  const attemptBusy =
    attempt?.status === 'queued' || attempt?.status === 'running'
  const binding = attempt?.binding
  const [failure, setFailure] = useState<string>()
  const resultRef = useRef<HTMLDivElement>(null)
  const resultHeadingRef = useRef<HTMLHeadingElement>(null)
  const previousSelectionRef = useRef<
    | {
        readonly attemptId: string
        readonly status: string
      }
    | undefined
  >(undefined)
  const focusedAttemptRef = useRef<string | undefined>(undefined)
  const exactRunAttemptBindingLinkage = Boolean(
    exactSelectedRunAttemptBindingLinkage &&
      asset &&
      run &&
      attempt &&
      binding &&
      binding.caseId === asset.id &&
      binding.assetId === asset.id &&
      binding.roiStatus === 'approved',
  )
  const currentCanonicalBindingMatches = Boolean(
    exactRunAttemptBindingLinkage &&
      asset &&
      binding &&
      binding.assetSha256 === asset.sha256 &&
      isCanonicalBinding(binding),
  )
  const bindingMatchesCurrentRoi = Boolean(
    currentCanonicalBindingMatches &&
      asset &&
      binding &&
      isVerifiedFullImageSourceBinding(asset, roi) &&
      roi &&
      binding.roiId === roi.id &&
      binding.roiVersion === roi.version &&
      sameGeometry(binding.roiGeometry, roi.geometry),
  )
  const storedResult = attempt?.result
  const historicalOutputMatchesBinding = Boolean(
    exactRunAttemptBindingLinkage &&
      binding &&
      storedResult &&
      sameStoredBinding(storedResult.output.binding, binding),
  )
  const validatedOutput =
    currentCanonicalBindingMatches && binding && storedResult
      ? validateInferenceOutputEnvelope(storedResult.output, binding, gatewayMode)
      : undefined
  const exactResultSelection =
    run && attempt?.status === 'succeeded' && binding && storedResult
      ? selectExactResultTarget(state, {
          runId: run.clientRunId,
          attemptId: attempt.id,
          resultDigest: storedResult.output.resultDigest,
          inputFingerprint: binding.inputFingerprint,
        })
      : undefined
  const output =
    attempt?.status === 'succeeded' &&
    bindingMatchesCurrentRoi &&
    storedResult?.freshness === 'current' &&
    validatedOutput?.valid === true &&
    exactResultSelection?.ok === true
      ? exactResultSelection.target.output
      : undefined
  const staleResultRequiresRerun = Boolean(
    attempt?.status === 'succeeded' &&
      ((storedResult?.freshness === 'stale' && historicalOutputMatchesBinding) ||
        (storedResult?.freshness === 'current' &&
          validatedOutput?.valid === true &&
          !bindingMatchesCurrentRoi)),
  )
  const resultIntegrityFailed = Boolean(
    attempt?.status === 'succeeded' &&
      storedResult?.freshness === 'current' &&
      !output &&
      !staleResultRequiresRerun,
  )
  const selectedRunIntegrityFailed = Boolean(
    selectedRun &&
      selectedAttempt?.status === 'succeeded' &&
      selectedAttempt.result?.freshness === 'current' &&
      (!exactSelectedRunAttemptBindingLinkage ||
        !selectedBinding ||
        !isCanonicalBinding(selectedBinding)),
  )
  useEffect(() => {
    const currentSelection = attempt
      ? { attemptId: attempt.id, status: attempt.status }
      : undefined
    const previousSelection = previousSelectionRef.current
    previousSelectionRef.current = currentSelection
    const newlySucceeded = Boolean(
      currentSelection?.status === 'succeeded' &&
        previousSelection?.attemptId === currentSelection.attemptId &&
        previousSelection.status !== 'succeeded',
    )

    if (
      !newlySucceeded ||
      !output ||
      !attempt ||
      focusedAttemptRef.current === attempt.id ||
      !resultHeadingRef.current
    ) {
      return
    }

    focusedAttemptRef.current = attempt.id
    resultHeadingRef.current.focus({ preventScroll: true })
    const reducedMotion =
      typeof window.matchMedia === 'function' &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches
    if (typeof resultHeadingRef.current.scrollIntoView === 'function') {
      resultHeadingRef.current.scrollIntoView({
        behavior: reducedMotion ? 'auto' : 'smooth',
        block: 'start',
      })
    }
  }, [attempt, output])

  if (caseParameters.length === 0 || (caseParameters.length === 1 && !requestedCaseId)) {
    return <InvalidAnalysisCase reason="No case ID was supplied" />
  }
  if (caseParameters.length !== 1) {
    return <InvalidAnalysisCase reason="More than one case ID was supplied" />
  }
  if (!asset) {
    return <InvalidAnalysisCase reason={requestedCaseId} />
  }
  if (!roi) {
    return <MissingSourceBinding caseId={asset.id} label={asset.label} />
  }
  if (!hasValidOptionalRunQuery) {
    const reason = runParameters.length > 1
      ? 'More than one run ID was supplied'
      : requestedRunId.length === 0
        ? 'The run ID was empty'
        : 'The run ID must match exactly without surrounding whitespace'
    return <InvalidAnalysisRun caseId={asset.id} reason={reason} />
  }
  if (requestedRunId && (!selectedRun || !runMatchesRoute)) {
    return (
      <InvalidAnalysisRun
        caseId={asset.id}
        reason={requestedRunId}
        integrityRunId={
          selectedRunIntegrityFailed ? selectedRun?.clientRunId : undefined
        }
      />
    )
  }

  const ready = isVerifiedFullImageSourceBinding(asset, roi)
  const canCancel = attempt?.status === 'queued' || attempt?.status === 'running'
  const canRetry =
    attempt?.status === 'cancelled' ||
    attempt?.status === 'failed' ||
    attempt?.status === 'blocked'
  const executionBoundary =
    gatewayMode === 'mock'
      ? 'MOCK ONLY · UNVALIDATED · NOT A PATIENT RESULT'
      : CONNECTED_INFERENCE_WATERMARK
  const connectedMode = gatewayMode === 'connected'
  const connectedResult = output?.origin === 'model_prediction'
  const aoiEvidence = output && !connectedResult
    ? deriveClinicalAoiPresentation(output.heatmap, output.binding.roiGeometry)
    : undefined
  const existingReview =
    run && attempt && output
      ? state.reviewOrder
          .filter((reviewId) =>
            Object.prototype.hasOwnProperty.call(state.reviewsById, reviewId),
          )
          .map((reviewId) => state.reviewsById[reviewId])
          .find(
            (review) =>
              review?.runId === run.clientRunId &&
              review.attemptId === attempt.id &&
              review.resultDigest === output.resultDigest &&
              review.inputFingerprint === output.binding.inputFingerprint,
          )
      : undefined

  const startRun = () => {
    setFailure(undefined)
    try {
      const identifiers = actions.startRun({
        caseId: asset.id,
        modelVersion: DEFAULT_MODEL_VERSION,
        config: DEFAULT_CONFIG,
      })
      const nextSearch = new URLSearchParams({
        case: asset.id,
        run: identifiers.runId,
      })
      navigate(`/analysis?${nextSearch.toString()}`, { replace: true })
    } catch (error) {
      setFailure(
        error instanceof WorkbenchError
          ? `${error.reason}: ${error.message}`
          : connectedMode
            ? 'The research observer-attention prediction could not be started.'
            : 'The simulation could not be started.',
      )
    }
  }

  const retryRun = () => {
    if (!run) return
    setFailure(undefined)
    try {
      actions.retryRun(run.clientRunId)
    } catch (error) {
      setFailure(
        error instanceof WorkbenchError
          ? `${error.reason}: ${error.message}`
          : 'The exact input could not be retried.',
      )
    }
  }

  return (
    <section className="workspace-page inference-page">
      <header className="workspace-page__header page-shell inference-page__header">
        <div>
          <Link className="workspace-back-link" to="/cases">
            <ArrowLeft aria-hidden="true" /> Back to cases
          </Link>
          <h1>
            {connectedMode
              ? 'Research observer-attention prediction'
              : 'Simulated observer-attention density'}
          </h1>
          <p>
            {asset.label.replace(/^Standalone synthetic case — /, '')}
          </p>
        </div>
        {connectedMode ? (
          <div
            className="inference-page__boundary"
            role="region"
            aria-label="Execution boundary"
          >
          <strong>{executionBoundary}</strong>
          </div>
        ) : null}
      </header>

      <div
        className={`inference-layout page-shell${output ? ' inference-layout--result' : ''}`}
      >
        <div className="inference-visual-column">
          <section className="workspace-panel inference-visual" aria-label="Selected synthetic image">
            <div className="workspace-panel__heading">
              <h2
                ref={output ? resultHeadingRef : undefined}
                tabIndex={output ? -1 : undefined}
              >
                {output ? 'Result' : 'Source image'}
              </h2>
            </div>

            {output ? (
              <div
                ref={resultRef}
                className="inference-result"
                role="region"
                tabIndex={-1}
                aria-label={
                  connectedResult
                    ? 'Research observer-attention prediction result'
                    : 'Simulation result'
                }
              >
                <AttentionResultView
                  asset={asset}
                  output={output}
                  roi={roi}
                  layout="clinician-stack"
                />
              </div>
            ) : (
              <>
                {resultIntegrityFailed ? (
                  <div className="inference-result__integrity-failure" role="alert">
                    <strong>Result integrity validation failed.</strong>
                    <span>
                      The selected current result cannot be displayed or reviewed. Open
                      the exact run detail to audit the stored record.
                    </span>
                  </div>
                ) : staleResultRequiresRerun ? (
                  <div className="inference-result__unavailable" role="status">
                    <strong>Current result unavailable</strong>
                    <span>
                      The source image binding changed. The previous result is available
                      in run details. Run again to create a result for the current image.
                    </span>
                  </div>
                ) : null}
                <figure className="inference-source-preview">
                  <img
                    src={asset.url}
                    alt={`${asset.label}: AI-generated synthetic face`}
                    width="1024"
                    height="1024"
                    loading="eager"
                    decoding="async"
                    fetchPriority="high"
                  />
                  <figcaption>AI-generated synthetic source image</figcaption>
                </figure>
              </>
            )}
          </section>
        </div>

        <aside className="inference-command-column">
          <output
            className="sr-only"
            role="status"
            aria-label="Analysis status announcement"
            aria-live="polite"
            aria-atomic="true"
          >
            {attempt?.status ?? 'not started'}
          </output>
          <section
            className="workspace-panel run-command-panel"
            aria-label="Run command"
            aria-busy={attemptBusy}
          >
            <div className="workspace-panel__heading">
              <h2>
                {output
                  ? 'Next step'
                  : connectedMode
                    ? 'Run research prediction'
                    : 'Run simulation'}
              </h2>
            </div>
            <div className="run-command-panel__state">
              <span>Active attempt</span>
              <output
                aria-label="Active attempt status"
              >
                {attempt?.status ?? 'not started'}
              </output>
            </div>
            {attemptBusy ? (
              <div className="workspace-loading-state">
                <LoaderCircle
                  className="workspace-loading-icon"
                  aria-hidden="true"
                />
                <span>
                  <strong>
                    {attempt.status === 'queued'
                      ? 'Starting analysis…'
                      : 'Preparing result…'}
                  </strong>
                  <small>
                    Keep this page open. You can cancel this research
                    request if needed.
                  </small>
                </span>
              </div>
            ) : null}
            {attempt?.failure ? (
              <p className="run-command-panel__failure" role="alert">
                {attempt.failure.reason}: {attempt.failure.message}
              </p>
            ) : null}
            {failure ? <p className="run-command-panel__failure" role="alert">{failure}</p> : null}
            {!ready ? (
              <p className="run-command-panel__failure" role="alert">
                <strong>Full-image source binding unavailable.</strong>{' '}
                <Link to={`/cases/${encodeURIComponent(asset.id)}/roi`}>
                  Restore source binding
                </Link>
              </p>
            ) : null}
            <div className="run-command-panel__actions">
              {!attempt ? (
                <button
                  className="workspace-button workspace-button--primary"
                  type="button"
                  disabled={!ready}
                  onClick={startRun}
                >
                  <FlaskConical aria-hidden="true" />{' '}
                  {connectedMode ? 'Run research prediction' : 'Run simulation'}
                </button>
              ) : null}
              {canCancel && run ? (
                <button
                  className="workspace-button workspace-button--primary"
                  type="button"
                  onClick={() => actions.cancelRun(run.clientRunId)}
                >
                  <Square aria-hidden="true" /> Cancel run
                </button>
              ) : null}
              {canRetry ? (
                <button
                  className="workspace-button workspace-button--primary"
                  type="button"
                  onClick={retryRun}
                >
                  <RotateCcw aria-hidden="true" /> Retry exact input
                </button>
              ) : null}
              {attempt?.status === 'succeeded' && staleResultRequiresRerun ? (
                <button
                  className="workspace-button workspace-button--primary"
                  type="button"
                  disabled={!ready}
                  onClick={startRun}
                >
                  <FlaskConical aria-hidden="true" /> Run with current image
                </button>
              ) : null}
              {run && attempt?.status === 'succeeded' && output ? (
                <Link
                  className="workspace-button workspace-button--primary run-command-panel__review"
                  to={
                    existingReview
                      ? `/research/reviews/${encodeURIComponent(existingReview.id)}`
                      : `/research/reviews/new?run=${encodeURIComponent(run.clientRunId)}&attempt=${encodeURIComponent(attempt.id)}`
                  }
                >
                  Review this result <ArrowRight aria-hidden="true" />
                </Link>
              ) : null}
              {attempt?.status === 'succeeded' &&
              !output &&
              !staleResultRequiresRerun ? (
                <button
                  className="workspace-button workspace-button--primary"
                  type="button"
                  disabled={!ready}
                  onClick={startRun}
                >
                  <FlaskConical aria-hidden="true" /> Start a new run
                </button>
              ) : null}
            </div>
          </section>
        </aside>
      </div>

      <details className="inference-advanced page-shell">
        <summary>Technical details</summary>
        <div className="inference-advanced__content">
          <section aria-label="Selected case binding">
            <h2>Source lineage</h2>
            <div className="inference-case-strip">
              <div>
                <span>Case / asset</span>
                <code>{asset.id}</code>
              </div>
              <div>
                <span>Canonical SHA-256</span>
                <code>{asset.sha256}</code>
              </div>
              <div>
                <span>Full-image source bound</span>
                <strong>
                  {ready
                    ? `Verified full image · v${roi.version}`
                    : `Unavailable · v${roi.version}`}
                </strong>
              </div>
              <div>
                <span>Relationship</span>
                <strong>independent · unpaired</strong>
              </div>
            </div>
          </section>

          <section className="workspace-panel inference-config" aria-label="Execution provenance">
            <div className="workspace-panel__heading">
              <h2>Execution boundary</h2>
            </div>
            <div className="inference-config__summary">
              <p>
                {connectedMode
                  ? 'The connected gateway may return a research-unvalidated observer-attention prediction. It is not observed gaze, validated, or deployable for clinical use.'
                  : 'New runs use one fixed deterministic simulation configuration. It is not a trained, validated, or deployable clinical model.'}
              </p>
            </div>
          </section>

          <PreflightGateList asset={asset} roi={roi} gatewayMode={gatewayMode} />

          {output && (connectedResult || aoiEvidence?.ok) ? (
            <section
              className="workspace-panel inference-evidence"
              aria-label="Clinical AOI evidence"
            >
              <div className="workspace-panel__heading">
                <h2>
                  {connectedResult ? 'AOI summary unavailable' : 'Result evidence'}
                </h2>
              </div>
              {connectedResult ? (
                <div className="inference-evidence__unavailable">
                  <p>
                    Registration geometry was not supplied with this connected
                    result.
                  </p>
                  <p>
                    AOI evidence requires registered landmarks or polygons,
                    explicit orientation metadata, and registration quality
                    control.
                  </p>
                </div>
              ) : aoiEvidence?.ok ? (
                <>
                  <div
                    className="inference-metrics"
                    role="region"
                    aria-label="Simulated clinical AOI shares"
                  >
                    <div>
                      <span>Central triangle share</span>
                      <strong>{formatShare(aoiEvidence.centralTriangleShare)}</strong>
                    </div>
                    <div>
                      <span>Patient-left share</span>
                      <strong>{formatShare(aoiEvidence.hemifaces.patientLeftShare)}</strong>
                    </div>
                    <div>
                      <span>Patient-right share</span>
                      <strong>{formatShare(aoiEvidence.hemifaces.patientRightShare)}</strong>
                    </div>
                    <div>
                      <span>Dominant anatomical AOI</span>
                      <strong>{aoiEvidence.dominantSubsite?.label ?? 'None available'}</strong>
                    </div>
                  </div>
                  <p className="inference-evidence__note">
                    Shares of simulated point weights assigned by the fixed
                    anatomical template; not gaze duration or clinical
                    measurements.
                  </p>
                </>
              ) : null}
              <dl className="inference-provenance" aria-label="Result provenance">
                <div><dt>Result digest</dt><dd><output aria-label="Result digest">{output.resultDigest}</output></dd></div>
                <div><dt>Origin</dt><dd>{output.origin}</dd></div>
                <div><dt>Capability</dt><dd>{output.capabilityStatus}</dd></div>
                <div><dt>Engine</dt><dd>{output.provenance.engine}</dd></div>
                <div>
                  <dt>
                    {connectedResult
                      ? 'Connected engine version'
                      : 'Simulation engine version'}
                  </dt>
                  <dd>{output.provenance.engineVersion}</dd>
                </div>
                <div>
                  <dt>
                    {connectedResult
                      ? 'Connected request contract'
                      : 'Simulation profile'}
                  </dt>
                  <dd>
                    <span>
                      {connectedResult
                        ? CONNECTED_ATTENTION_REQUEST_PROFILE_VERSION
                        : output.binding.modelVersion}
                    </span>
                    {connectedResult ? null : ` · ${output.binding.modelMode}`}
                  </dd>
                </div>
                {output.origin === 'model_prediction' ? (
                  <>
                    <div><dt>Connected model ID</dt><dd>{output.modelIdentity.modelId}</dd></div>
                    <div><dt>Connected model version</dt><dd>{output.modelIdentity.modelVersion}</dd></div>
                    <div><dt>Artifact SHA-256</dt><dd>{output.modelIdentity.artifactSha256}</dd></div>
                    <div><dt>Preprocessing version</dt><dd>{output.modelIdentity.preprocessingVersion}</dd></div>
                    <div><dt>Calibration version</dt><dd>{output.modelIdentity.calibrationVersion}</dd></div>
                    <div><dt>Display scale ID</dt><dd>{output.modelIdentity.displayScaleId}</dd></div>
                  </>
                ) : (
                  <div>
                    <dt>Simulation configuration hash</dt>
                    <dd>{output.binding.configurationHash}</dd>
                  </div>
                )}
                <div><dt>Network accessed</dt><dd>{output.provenance.networkAccessed ? 'Yes' : 'No'}</dd></div>
                <div><dt>Persistent storage</dt><dd>{output.provenance.storageAccessed ? 'Yes' : 'No'}</dd></div>
                {output.origin === 'model_prediction' ? (
                  <>
                    <div>
                      <dt>Observed gaze in result payload</dt>
                      <dd>{output.provenance.observedGazePayloadIncluded ? 'Yes' : 'No'}</dd>
                    </div>
                    <div>
                      <dt>Training-data provenance</dt>
                      <dd>Not disclosed</dd>
                    </div>
                  </>
                ) : (
                  <div>
                    <dt>Human gaze used by simulation</dt>
                    <dd>{output.provenance.humanGazeData ? 'Yes' : 'No'}</dd>
                  </div>
                )}
              </dl>
              {connectedResult ? (
                <p className="inference-provenance__note">
                  This is a synthetic spatial contract rehearsal. The
                  response-reported connected model identity is research provenance,
                  not clinical certification.
                </p>
              ) : null}
              <div className="quality-gate-row" aria-label="Output quality gates">
                <span>Binding integrity passed</span>
                <span>Internal image binding passed</span>
                <span>Finite normalized values passed</span>
                <strong>Clinical-use eligibility: blocked</strong>
              </div>
            </section>
          ) : null}

          {run ? (
            <section className="inference-advanced__run" aria-label="Run history and provenance">
              <Link className="run-command-panel__detail" to={`/runs/${run.clientRunId}`}>
                Open exact run detail
              </Link>
              <RunTimeline run={run} attemptsById={state.attemptsById} />
            </section>
          ) : null}
        </div>
      </details>
    </section>
  )
}
