import { StatusBadge } from './StatusBadge'
import type { InferenceAttempt, InferenceRun } from '../workbench/types'

type RunTimelineProps = {
  readonly run: InferenceRun
  readonly attemptsById: Readonly<Record<string, InferenceAttempt>>
}

function statusTone(status: InferenceAttempt['status']) {
  if (status === 'succeeded') return 'success' as const
  if (status === 'failed' || status === 'cancelled' || status === 'blocked') {
    return 'blocked' as const
  }
  if (status === 'running' || status === 'queued' || status === 'validating') {
    return 'info' as const
  }
  return 'neutral' as const
}

export function RunTimeline({ run, attemptsById }: RunTimelineProps) {
  return (
    <section className="analysis-timeline workspace-panel" aria-label="Run timeline">
      <div className="workspace-panel__heading">
        <div>
          <p className="workspace-kicker">Immutable attempt history</p>
          <h2>Run timeline</h2>
        </div>
        <code>{run.clientRunId}</code>
      </div>
      <ol>
        {run.attemptIds.map((attemptId, index) => {
          const attempt = attemptsById[attemptId]
          if (!attempt) return null
          return (
            <li key={attempt.id} data-testid="analysis-attempt">
              <span className="analysis-timeline__index">{index + 1}</span>
              <div>
                <code>{attempt.id}</code>
                <span>
                  {attempt.parentAttemptId
                    ? `Parent ${attempt.parentAttemptId}`
                    : 'Initial attempt'}
                </span>
                <small>Token {attempt.attemptToken}</small>
              </div>
              <StatusBadge tone={statusTone(attempt.status)}>
                {attempt.status}
              </StatusBadge>
            </li>
          )
        })}
      </ol>
    </section>
  )
}
