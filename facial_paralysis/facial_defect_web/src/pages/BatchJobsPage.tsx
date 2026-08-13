import {
  Ban,
  CheckCircle2,
  FlaskConical,
  ListChecks,
  LoaderCircle,
  RotateCcw,
  XCircle,
} from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  formatJobProgressSummary,
  JobProgressMatrix,
  type JobProgressRow,
} from '../components/JobProgressMatrix'
import { StatusBadge } from '../components/StatusBadge'
import { listWorkbenchAssets } from '../workbench/catalog'
import {
  auditBatchManifest,
  createBatchManifest,
  type BatchManifest,
} from '../workbench/batchManifest'
import { manifestMatchesBatchDraft } from '../workbench/batchSession'
import { useWorkspace } from '../workbench/WorkspaceProvider'
import '../styles/task5.css'

function plural(count: number, singular: string, pluralForm = `${singular}s`) {
  return count === 1 ? singular : pluralForm
}

export function BatchJobsPage() {
  const catalog = listWorkbenchAssets()
  const {
    state: workspaceState,
    actions,
    batchState: session,
    batchActions,
    gatewayMode,
  } = useWorkspace()
  const manifestAudit = session.manifest
    ? auditBatchManifest(session.manifest, workspaceState)
    : undefined
  const draftMatches = manifestMatchesBatchDraft(session)
  const reviewedManifestItems = session.manifest
    ? [...session.manifest.items, ...session.manifest.excludedItems]
    : []
  const readyCount = session.manifest?.items.filter((item) => item.preflight === 'ready').length ?? 0
  const blockedCount = reviewedManifestItems.filter(
    (item) => item.preflight === 'blocked',
  ).length
  const [exclusionsConfirmed, setExclusionsConfirmed] = useState(false)
  const [submissionError, setSubmissionError] = useState(false)
  const progressHeadingRef = useRef<HTMLHeadingElement>(null)
  const previousJobIdRef = useRef(session.job?.id)
  const draftConfirmationKey = [
    session.selectedCaseIds.join('|'),
    session.modelVersion,
    session.config.threshold,
    session.config.smoothing,
    session.manifest?.hash ?? '',
  ].join('::')

  useEffect(() => {
    setExclusionsConfirmed(false)
    setSubmissionError(false)
  }, [draftConfirmationKey])

  useEffect(() => {
    const jobId = session.job?.id
    if (!previousJobIdRef.current && jobId) {
      const heading = progressHeadingRef.current
      heading?.focus({ preventScroll: true })
      heading?.scrollIntoView?.({
        behavior: 'auto',
        block: 'start',
      })
    }
    previousJobIdRef.current = jobId
  }, [session.job?.id])

  const currentCaseReview = useMemo(
    () => createBatchManifest({
      workspaceState,
      selectedCaseIds: catalog.map((asset) => asset.id),
      modelVersion: session.modelVersion,
      config: session.config,
    }),
    [catalog, session.config, session.modelVersion, workspaceState],
  )
  const readinessByCase = new Map(
    currentCaseReview.items.map((item) => [item.caseId, item.preflight] as const),
  )
  const readyCaseIds = currentCaseReview.items
    .filter((item) => item.preflight === 'ready')
    .map((item) => item.caseId)

  const progressRows = useMemo<readonly JobProgressRow[]>(() => {
    if (!session.manifest) return []
    return [...session.manifest.items, ...session.manifest.excludedItems].map((item) => {
      const runId = session.job?.runIdsByCase[item.caseId]
      const run = runId ? workspaceState.runsById[runId] : undefined
      const activeAttempt = run?.activeAttemptId
        ? workspaceState.attemptsById[run.activeAttemptId]
        : undefined
      const stale = manifestAudit?.staleCaseIds.includes(item.caseId) === true
      const status = stale
        ? 'blocked'
        : item.preflight === 'blocked'
          ? 'blocked'
          : run?.status ?? 'ready'
      const detail = stale
        ? 'Review these cases again before starting.'
        : item.preflight === 'blocked'
          ? 'Restore this case’s full-image source binding before running.'
          : status === 'running'
            ? 'Simulation is running.'
            : status === 'queued' || status === 'draft' || status === 'validating'
              ? 'Waiting to start.'
              : status === 'succeeded'
                ? 'Result is ready.'
                : status === 'failed'
                  ? `${activeAttempt?.failure?.message ?? 'Simulation failed.'} Retry this case.`
                  : status === 'cancelled'
                    ? 'Simulation was cancelled. Retry when ready.'
                    : 'Ready to start.'

      return {
        caseId: item.caseId,
        label: item.label,
        status,
        detail,
        ...(runId ? { runId } : {}),
        attemptCount: run?.attemptIds.length ?? 0,
        ...(activeAttempt?.parentAttemptId
          ? { parentAttemptId: activeAttempt.parentAttemptId }
          : {}),
        retryEligible: status === 'failed' || status === 'cancelled',
      }
    })
  }, [manifestAudit?.staleCaseIds, session.job, session.manifest, workspaceState])

  const activeRows = progressRows.filter((row) =>
    ['draft', 'validating', 'queued', 'running'].includes(row.status),
  )
  const retryRows = progressRows.filter((row) => row.retryEligible && row.runId)
  const submittedCount = Object.keys(session.job?.runIdsByCase ?? {}).length
  const succeededCount = progressRows.filter(
    (row) => row.status === 'succeeded' && row.runId,
  ).length
  const allSubmittedSucceeded =
    submittedCount > 0 && succeededCount === submittedCount
  const progressAnnouncement = formatJobProgressSummary(progressRows)

  const selectReady = () => {
    setExclusionsConfirmed(false)
    batchActions.selectAllCases(readyCaseIds)
  }

  const selectAll = () => {
    setExclusionsConfirmed(false)
    batchActions.selectAllCases(catalog.map((asset) => asset.id))
  }

  const clearSelection = () => {
    setExclusionsConfirmed(false)
    batchActions.clearSelection()
  }

  const preflight = () => {
    const selectedCaseIds = session.selectedCaseIds
    const selectedEveryReadyCase =
      selectedCaseIds.length === readyCaseIds.length &&
      readyCaseIds.every((caseId) => selectedCaseIds.includes(caseId))
    const excludedCaseIds = selectedEveryReadyCase
      ? currentCaseReview.items
          .filter((item) => item.preflight === 'blocked')
          .map((item) => item.caseId)
      : []
    const manifest = createBatchManifest({
      workspaceState,
      selectedCaseIds,
      excludedCaseIds,
      modelVersion: session.modelVersion,
      config: session.config,
    })
    setExclusionsConfirmed(false)
    batchActions.setManifest(manifest)
  }

  const submit = () => {
    const manifest = session.manifest
    if (
      !manifest ||
      !draftMatches ||
      !manifestAudit?.valid ||
      session.job ||
      (blockedCount > 0 && !exclusionsConfirmed)
    ) return
    setSubmissionError(batchActions.startBatch() === undefined)
  }

  const cancelBatch = () => {
    for (const row of activeRows) {
      if (row.runId) actions.cancelRun(row.runId)
    }
  }

  const retryOne = (runId: string) => actions.retryRun(runId)
  const retryEligible = () => {
    for (const row of retryRows) {
      if (row.runId) actions.retryRun(row.runId)
    }
  }

  const submitLabel = `Start ${readyCount} ${plural(readyCount, 'simulation')}`
  const retryLabel = `Retry ${retryRows.length} eligible ${plural(
    retryRows.length,
    'attempt',
  )}`

  return (
    <div className="workspace-page task5-page task5-jobs-page">
      <header className="workspace-page__header page-shell">
        <div>
          <p className="workspace-kicker">Simulations</p>
          <h1>Run several cases</h1>
          <p>
            Choose the synthetic cases you want to run, review the selection, then start
            the approved simulations together.
          </p>
        </div>
        <div className="task5-boundary-chip">
          <FlaskConical aria-hidden="true" />
          <span>
            <strong>Research simulation</strong>
            {gatewayMode === 'mock'
              ? 'Memory only · no upload, network, or storage'
              : 'Network required for simulations · no browser storage'}
          </span>
        </div>
      </header>

      <section className="task5-layout page-shell">
        <aside
          className="task5-control-rail"
          aria-label={
            session.job
              ? 'Batch submission summary'
              : 'Batch configuration'
          }
        >
          {session.job ? (
            <div className="task5-batch-summary">
              <p className="workspace-kicker">Selection complete</p>
              <h2>
                {submittedCount}{' '}
                {plural(submittedCount, 'simulation')}{' '}
                {allSubmittedSucceeded ? 'completed' : 'started'}
              </h2>
              <p>
                {allSubmittedSucceeded
                  ? submittedCount === 1
                    ? 'The result is ready.'
                    : `All ${submittedCount} results are ready.`
                  : blockedCount > 0
                  ? `${blockedCount} ${plural(blockedCount, 'case')} need source binding and were not submitted.`
                  : 'All reviewed cases were submitted.'}
              </p>
            </div>
          ) : (
            <>
              <div className="task5-section-heading">
            <div><p className="workspace-kicker">Step 1</p><h2>Select cases</h2></div>
            <output aria-label="Selected cases">{session.selectedCaseIds.length}</output>
          </div>
          <div className="task5-control-row">
            <button
              className="workspace-button workspace-button--secondary"
              type="button"
              disabled={Boolean(session.job)}
              onClick={selectReady}
            >
              Select ready cases
            </button>
            <button
              className="workspace-button workspace-button--quiet"
              type="button"
              disabled={Boolean(session.job)}
              onClick={selectAll}
              aria-label={`Select all ${catalog.length} cases`}
            >
              Select all {catalog.length}
            </button>
            <button
              className="workspace-button workspace-button--quiet"
              type="button"
              disabled={Boolean(session.job)}
              onClick={clearSelection}
            >
              Clear
            </button>
          </div>
          <div className="task5-case-checklist">
            {catalog.map((asset) => {
              const ready = readinessByCase.get(asset.id) === 'ready'
              return (
                <div className="task5-case-checklist__row" key={asset.id}>
                  <label>
                    <input
                      type="checkbox"
                      name="caseIds"
                      disabled={Boolean(session.job)}
                      checked={session.selectedCaseIds.includes(asset.id)}
                      onChange={() => {
                        setExclusionsConfirmed(false)
                        batchActions.toggleCase(asset.id)
                      }}
                    />
                    <span className="task5-case-checklist__identity">
                      <code>{asset.id}</code>
                      {asset.label}
                    </span>
                    <StatusBadge tone={ready ? 'success' : 'warning'}>
                      {ready ? 'Ready' : 'Needs source binding'}
                    </StatusBadge>
                  </label>
                  {!ready ? (
                    <Link to={`/cases/${encodeURIComponent(asset.id)}/roi`}>
                      Restore source binding
                    </Link>
                  ) : null}
                </div>
              )
            })}
          </div>

          <details className="task5-config-block">
            <summary>Advanced settings</summary>
            <div className="task5-config-block__content">
              <label>
                <span>Mock model</span>
                <select
                  name="modelVersion"
                  disabled={Boolean(session.job)}
                  value={session.modelVersion}
                  onChange={(event) => {
                    setExclusionsConfirmed(false)
                    batchActions.updateModel(
                      event.currentTarget.value as typeof session.modelVersion,
                    )
                  }}
                >
                  <option value="mock-salience-v0.3">mock-salience-v0.3</option>
                  <option value="mock-salience-v0.4">mock-salience-v0.4</option>
                </select>
              </label>
              {(['threshold', 'smoothing'] as const).map((field) => {
                const percent = Math.round(session.config[field] * 100)
                return (
                  <label key={field}>
                    <span>{field[0].toUpperCase() + field.slice(1)} <output>{percent}%</output></span>
                    <input
                      name={field}
                      type="range"
                      min="0"
                      max="100"
                      disabled={Boolean(session.job)}
                      value={percent}
                      aria-label={`${field[0].toUpperCase() + field.slice(1)} ${percent} percent`}
                      onChange={(event) => {
                        setExclusionsConfirmed(false)
                        batchActions.updateConfig(
                          field,
                          Number(event.currentTarget.value) / 100,
                        )
                      }}
                    />
                  </label>
                )
              })}
            </div>
          </details>

          <button
            className="workspace-button workspace-button--primary task5-primary-action"
            type="button"
            disabled={session.selectedCaseIds.length === 0 || Boolean(session.job)}
            onClick={preflight}
          >
            <ListChecks aria-hidden="true" /> Review selected cases
          </button>
            </>
          )}
        </aside>

        <div className="task5-main-column">
          {session.manifest && !session.job ? (
            <ManifestPreview
              manifest={session.manifest}
              exclusionsConfirmed={exclusionsConfirmed}
              onConfirmExclusions={setExclusionsConfirmed}
              controlsDisabled={Boolean(session.job)}
            />
          ) : !session.manifest ? (
            <section className="task5-empty-panel">
              <ListChecks aria-hidden="true" />
              <h2>Choose cases to begin</h2>
              <p>Select ready cases or make your own selection, then review it before starting.</p>
            </section>
          ) : null}

          {session.manifest && !session.job && (!draftMatches || manifestAudit?.valid === false) ? (
            <div className="task5-alert" role="alert">
              <XCircle aria-hidden="true" />
              <span>
                <strong>This review is out of date.</strong> Review selected cases again
                before starting. The earlier selection cannot be reused.
              </span>
            </div>
          ) : null}

          {submissionError ? (
            <div className="task5-alert" role="alert">
              <XCircle aria-hidden="true" />
              <span>
                <strong>The simulations did not start.</strong> Nothing was submitted.
                Review the cases again and retry.
              </span>
            </div>
          ) : null}

          {session.manifest && !session.job ? (
            <div className="task5-command-bar">
              <div>
                <span>Step 3</span>
                <strong>{readyCount} will run · {blockedCount} need source binding</strong>
              </div>
              <button
                className="workspace-button workspace-button--primary"
                type="button"
                disabled={
                  !draftMatches ||
                  manifestAudit?.valid !== true ||
                  readyCount === 0 ||
                  (blockedCount > 0 && !exclusionsConfirmed) ||
                  Boolean(session.job)
                }
                onClick={submit}
                aria-label={submitLabel}
              >
                <CheckCircle2 aria-hidden="true" /> {submitLabel}
              </button>
            </div>
          ) : null}

          {session.job ? (
            <>
              <p
                className="sr-only"
                role="status"
                aria-label="Batch progress announcement"
                aria-live="polite"
                aria-atomic="true"
              >
                {progressAnnouncement}
              </p>
              <section
                className="task5-progress"
                aria-label="Batch progress"
                aria-busy={activeRows.length > 0}
              >
              <div className="task5-section-heading task5-progress__heading">
                <div>
                  <p className="workspace-kicker">03 · Execute</p>
                  <h2 ref={progressHeadingRef} tabIndex={-1}>
                    Batch progress
                  </h2>
                  <span>{submittedCount} submitted · {blockedCount} blocked</span>
                </div>
                {activeRows.length > 0 || retryRows.length > 0 ? (
                  <div className="task5-control-row">
                    {activeRows.length > 0 ? (
                      <button
                        className="workspace-button workspace-button--quiet"
                        type="button"
                        onClick={cancelBatch}
                        aria-label="Cancel batch"
                      >
                        <Ban aria-hidden="true" /> Cancel batch
                      </button>
                    ) : null}
                    {retryRows.length > 0 ? (
                      <button
                        className="workspace-button workspace-button--secondary"
                        type="button"
                        onClick={retryEligible}
                        aria-label={retryLabel}
                      >
                        <RotateCcw aria-hidden="true" /> {retryLabel}
                      </button>
                    ) : null}
                  </div>
                ) : null}
              </div>
              {activeRows.length > 0 ? (
                <div className="workspace-loading-state task5-progress__active">
                  <LoaderCircle
                    className="workspace-loading-icon"
                    aria-hidden="true"
                  />
                  <span>
                    <strong>
                      Analyzing {activeRows.length}{' '}
                      {plural(activeRows.length, 'case')}…
                    </strong>
                    <small>
                      Completed cases will update below as results are
                      prepared.
                    </small>
                  </span>
                </div>
              ) : null}
              <JobProgressMatrix rows={progressRows} onRetry={retryOne} />
              <details className="task5-technical-details task5-progress-technical">
                <summary>Technical details</summary>
                <div className="task5-technical-details__content">
                  <dl>
                    <div><dt>Batch job ID</dt><dd><code>{session.job.id}</code></dd></div>
                    <div><dt>Manifest hash</dt><dd><code aria-label="Manifest hash">{session.job.manifestHash}</code></dd></div>
                  </dl>
                  <div className="task5-technical-case-list">
                    {progressRows.map((row) => (
                      <div key={row.caseId}>
                        <code>{row.caseId}</code>
                        <span>Run ID: {row.runId ?? 'Not submitted'}</span>
                        <span>
                          Attempts: {row.attemptCount}
                          {row.parentAttemptId ? ` · Parent ${row.parentAttemptId}` : ''}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </details>
              </section>
            </>
          ) : null}
        </div>
      </section>
    </div>
  )
}

function ManifestPreview({
  manifest,
  exclusionsConfirmed,
  onConfirmExclusions,
  controlsDisabled,
}: {
  readonly manifest: BatchManifest
  readonly exclusionsConfirmed: boolean
  readonly onConfirmExclusions: (confirmed: boolean) => void
  readonly controlsDisabled: boolean
}) {
  const reviewedItems = [...manifest.items, ...manifest.excludedItems]
  const ready = manifest.items.filter((item) => item.preflight === 'ready').length
  const blockedItems = reviewedItems.filter((item) => item.preflight === 'blocked')
  const blocked = blockedItems.length
  return (
    <section className="task5-manifest" aria-label="Selected case review">
      <div className="task5-section-heading task5-manifest__heading">
        <div>
          <p className="workspace-kicker">Step 2</p>
          <h2>Review selected cases</h2>
        </div>
      </div>
      <div className="task5-manifest__counts">
        <span><strong aria-label="Ready cases">{ready}</strong> Ready</span>
        <span><strong aria-label="Blocked cases">{blocked}</strong> Need source binding</span>
      </div>
      {blocked > 0 ? (
        <p className="task5-manifest__warning">
          {blocked} {plural(blocked, 'case')} need source binding and will not run.
        </p>
      ) : (
        <p className="task5-manifest__ready">All selected cases are ready to run.</p>
      )}
      <div className="task5-preflight-list">
        {reviewedItems.map((item) => (
          <article data-testid="preflight-item" key={item.caseId}>
            <div><code>{item.caseId}</code><span>{item.label}</span></div>
            <div>
              <StatusBadge tone={item.preflight === 'ready' ? 'success' : 'warning'}>
                {item.preflight === 'ready' ? 'Ready' : 'Needs source binding'}
              </StatusBadge>
            </div>
          </article>
        ))}
      </div>

      {blockedItems.length > 0 ? (
        <section className="task5-exclusions" aria-label="Cases excluded from this run">
          <div>
            <strong>These cases will not run</strong>
            <ul>
              {blockedItems.map((item) => (
                <li key={item.caseId}>
                  <code>{item.caseId}</code>
                  <span>{item.label}</span>
                  <Link to={`/cases/${encodeURIComponent(item.caseId)}/roi`}>
                    Restore source binding
                  </Link>
                </li>
              ))}
            </ul>
          </div>
          <label>
            <input
              type="checkbox"
              name="confirmExclusions"
              checked={exclusionsConfirmed}
              disabled={controlsDisabled}
              onChange={(event) => onConfirmExclusions(event.currentTarget.checked)}
            />
            <span>I understand that {blocked} {plural(blocked, 'case')} need source binding and will not run</span>
          </label>
        </section>
      ) : null}

      <details className="task5-technical-details">
        <summary>Technical details</summary>
        <div className="task5-technical-details__content">
          <dl>
            <div><dt>Manifest hash</dt><dd><code aria-label="Manifest hash">{manifest.hash}</code></dd></div>
            <div><dt>Model version</dt><dd><code>{manifest.modelVersion}</code></dd></div>
            <div><dt>Model mode</dt><dd><code>{manifest.modelMode}</code></dd></div>
            <div><dt>Threshold</dt><dd>{Math.round(manifest.config.threshold * 100)}%</dd></div>
            <div><dt>Smoothing</dt><dd>{Math.round(manifest.config.smoothing * 100)}%</dd></div>
            <div><dt>Persistence</dt><dd>{manifest.persistence}</dd></div>
          </dl>
          <p>
            This immutable manifest preserves exact case, asset, full-image source binding,
            model, and configuration lineage. Blocked items remain in the manifest and are never
            submitted.
          </p>
          <div className="task5-technical-case-list">
            {reviewedItems.map((item) => (
              <div key={item.caseId}>
                <code>{item.caseId}</code>
                <span>Asset SHA-256: {item.assetSha256}</span>
                <span>Source binding: {item.roiId ?? 'unavailable'} · version {item.roiVersion ?? 'unavailable'} · {item.roiStatus}</span>
              </div>
            ))}
          </div>
        </div>
      </details>
    </section>
  )
}
