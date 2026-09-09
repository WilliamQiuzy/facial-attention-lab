import { RotateCcw } from 'lucide-react'
import { Link } from 'react-router-dom'
import { StatusBadge } from './StatusBadge'

export type JobProgressStatus =
  | 'blocked'
  | 'ready'
  | 'draft'
  | 'validating'
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled'

export type JobProgressRow = {
  readonly caseId: string
  readonly label: string
  readonly status: JobProgressStatus
  readonly detail: string
  readonly runId?: string
  readonly attemptCount: number
  readonly parentAttemptId?: string
  readonly retryEligible: boolean
}

function tone(status: JobProgressStatus) {
  if (status === 'succeeded') return 'success' as const
  if (status === 'blocked' || status === 'failed') return 'blocked' as const
  if (status === 'cancelled') return 'warning' as const
  if (status === 'running' || status === 'queued') return 'info' as const
  return 'neutral' as const
}

export function formatJobProgressSummary(
  rows: readonly JobProgressRow[],
): string {
  const summaryOrder: readonly JobProgressStatus[] = [
    'running',
    'queued',
    'succeeded',
    'failed',
    'cancelled',
    'blocked',
    'ready',
    'validating',
    'draft',
  ]
  return summaryOrder.flatMap((status) => {
    const count = rows.filter((row) => row.status === status).length
    return count > 0 ? [`${count} ${status}`] : []
  }).join(' · ')
}

export function JobProgressMatrix({
  rows,
  onRetry,
}: {
  readonly rows: readonly JobProgressRow[]
  readonly onRetry: (runId: string) => void
}) {
  const statusSummary = formatJobProgressSummary(rows)

  return (
    <div className="task5-progress-matrix">
      <p
        className="task5-progress-summary"
        aria-label="Batch status summary"
      >
        {statusSummary}
      </p>
      <div className="task5-progress-matrix__labels" aria-hidden="true">
        <span>Case</span>
        <span>Status and next step</span>
        <span>Action</span>
      </div>
      {rows.map((row) => (
        <article className="task5-progress-row" data-testid="job-progress-row" key={row.caseId}>
          <div>
            <code>{row.caseId}</code>
            <span>{row.label}</span>
          </div>
          <div>
            <StatusBadge tone={tone(row.status)}>{row.status}</StatusBadge>
            <small>{row.detail}</small>
          </div>
          <div>
            {row.status === 'succeeded' && row.runId ? (
              <Link
                className="workspace-button workspace-button--quiet"
                to={`/runs/${encodeURIComponent(row.runId)}`}
              >
                View result
              </Link>
            ) : row.retryEligible && row.runId ? (
              <button
                className="workspace-button workspace-button--quiet"
                type="button"
                onClick={() => onRetry(row.runId!)}
              >
                <RotateCcw aria-hidden="true" /> Retry
              </button>
            ) : (
              <span className="task5-progress-row__locked">No action</span>
            )}
          </div>
        </article>
      ))}
    </div>
  )
}
