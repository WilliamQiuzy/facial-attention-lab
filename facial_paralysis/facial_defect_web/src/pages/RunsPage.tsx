import { ArrowRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import { StatusBadge } from '../components/StatusBadge'
import { getWorkbenchAsset } from '../workbench/catalog'
import { selectExactResultTarget } from '../workbench/reviewPolicy'
import type { InferenceRun, WorkspaceState } from '../workbench/types'
import { useWorkspace } from '../workbench/WorkspaceProvider'

type RunDisplayState =
  | InferenceRun['status']
  | 'result ready'
  | 'stale'
  | 'revoked'
  | 'integrity unavailable'
  | 'result unavailable'

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

function statusTone(status: RunDisplayState) {
  if (status === 'result ready') return 'success' as const
  if (
    status === 'failed' ||
    status === 'blocked' ||
    status === 'revoked' ||
    status === 'integrity unavailable'
  ) return 'blocked' as const
  if (
    status === 'cancelled' ||
    status === 'stale' ||
    status === 'result unavailable'
  ) return 'warning' as const
  return 'info' as const
}

function displayStateForRun(
  state: WorkspaceState,
  run: InferenceRun,
): RunDisplayState {
  if (run.status !== 'succeeded') return run.status
  const attemptId = run.activeAttemptId
  if (
    !attemptId ||
    !run.attemptIds.includes(attemptId) ||
    !Object.prototype.hasOwnProperty.call(state.attemptsById, attemptId)
  ) return 'result unavailable'

  const attempt: unknown = state.attemptsById[attemptId]
  if (
    !isRecord(attempt) ||
    attempt.clientRunId !== run.clientRunId ||
    attempt.status !== 'succeeded' ||
    !isRecord(attempt.binding) ||
    !isRecord(attempt.result)
  ) return 'result unavailable'
  const result = attempt.result
  if (!isRecord(result.output)) return 'integrity unavailable'
  if (result.freshness === 'stale' || result.freshness === 'revoked') {
    return result.freshness
  }
  if (result.freshness !== 'current') return 'integrity unavailable'
  if (
    !isNonEmptyString(result.output.resultDigest) ||
    !isNonEmptyString(attempt.binding.inputFingerprint)
  ) return 'integrity unavailable'

  const selected = selectExactResultTarget(state, {
    runId: run.clientRunId,
    attemptId,
    resultDigest: result.output.resultDigest,
    inputFingerprint: attempt.binding.inputFingerprint,
  })
  return selected.ok ? 'result ready' : 'integrity unavailable'
}

export function RunsPage() {
  const { state } = useWorkspace()
  const runs = [...state.runOrder]
    .reverse()
    .filter((runId) => Object.prototype.hasOwnProperty.call(state.runsById, runId))
    .map((runId) => state.runsById[runId])
    .filter((run) => run !== undefined)

  return (
    <div className="workspace-page runs-page">
      <header className="workspace-page__header page-shell">
        <div>
          <h1>Recent simulations</h1>
          <p>Available until refresh.</p>
        </div>
        <Link className="workspace-button workspace-button--primary" to="/cases">
          Start simulation
          <ArrowRight aria-hidden="true" />
        </Link>
      </header>

      <section className="run-list-shell page-shell" aria-label="Session run list">
        {runs.length > 0 ? (
          <div className="run-list">
            {runs.map((run) => {
              const asset = getWorkbenchAsset(run.caseId)
              const displayState = displayStateForRun(state, run)
              return (
                <article className="run-row" data-testid="run-row" key={run.clientRunId}>
                  <div className="run-row__case">
                    <h2>{asset ? conciseLabel(asset.label) : 'Case unavailable'}</h2>
                    <code>{run.caseId}</code>
                  </div>
                  <span
                    role="status"
                    aria-label={`Status ${formatState(displayState)}`}
                    aria-live="polite"
                    aria-atomic="true"
                  >
                    <StatusBadge tone={statusTone(displayState)}>
                      {formatState(displayState)}
                    </StatusBadge>
                  </span>
                  <Link
                    className="workspace-button workspace-button--secondary"
                    to={`/runs/${encodeURIComponent(run.clientRunId)}`}
                  >
                    View details
                  </Link>
                </article>
              )
            })}
          </div>
        ) : (
          <div className="workspace-empty run-empty">
            <h2>No recent simulations</h2>
            <p>Start one from Cases. It will appear here until refresh.</p>
          </div>
        )}
      </section>
    </div>
  )
}
