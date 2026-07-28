import { ArrowRight, ShieldCheck } from 'lucide-react'
import { Link } from 'react-router-dom'
import { StatusBadge } from '../components/StatusBadge'
import { listWorkbenchAssets } from '../workbench/catalog'
import { isVerifiedFullImageSourceBinding } from '../workbench/sourceBinding'
import { useWorkspace } from '../workbench/WorkspaceProvider'

function formatState(value: string): string {
  return value.replaceAll('_', ' ')
}

export function WorkspaceDashboardPage() {
  const { state, batchState } = useWorkspace()
  const assets = listWorkbenchAssets()
  const verifiedBindingCount = assets.filter((asset) =>
    isVerifiedFullImageSourceBinding(asset, state.roisByCase[asset.id]),
  ).length
  const bindingRecoveryCount = assets.length - verifiedBindingCount
  const recentRuns = [...state.runOrder]
    .reverse()
    .slice(0, 4)
    .map((runId) => state.runsById[runId])
    .filter((run) => run !== undefined)

  const metrics = [
    ['synthetic-cases', assets.length, 'Synthetic cases'],
    ['binding-verified', verifiedBindingCount, 'Source bindings ready'],
    ['binding-recovery', bindingRecoveryCount, 'Need binding recovery'],
    ['session-runs', state.runOrder.length, 'Session runs'],
    ['session-jobs', batchState.job ? 1 : 0, 'Session jobs'],
    ['session-reviews', state.reviewOrder.length, 'Session reviews'],
  ] as const

  return (
    <div className="workspace-page dashboard-page">
      <header className="workspace-page__header page-shell">
        <div>
          <p className="workspace-kicker">Research operations</p>
          <h1>Facial attention workspace</h1>
          <p>
            Inspect synthetic cases, source-binding readiness, and in-memory inference
            activity from one bounded session.
          </p>
        </div>
        <Link className="workspace-button workspace-button--primary" to="/cases">
          Browse synthetic cases
          <ArrowRight aria-hidden="true" />
        </Link>
      </header>

      <section
        className="workspace-metrics page-shell"
        aria-label="Workspace summary"
      >
        {metrics.map(([id, value, label]) => (
          <div className="workspace-metric" data-testid={`metric-${id}`} key={id}>
            <strong>{value}</strong>
            <span>{label}</span>
          </div>
        ))}
      </section>

      <div className="dashboard-grid page-shell">
        <section className="workspace-panel" aria-labelledby="recent-runs-title">
          <div className="workspace-panel__heading">
            <div>
              <p className="workspace-kicker">Session activity</p>
              <h2 id="recent-runs-title">Recent runs</h2>
            </div>
            <Link to="/runs">View all runs</Link>
          </div>

          {recentRuns.length > 0 ? (
            <ul className="dashboard-run-list">
              {recentRuns.map((run) => (
                <li key={run.clientRunId}>
                  <div>
                    <code>{run.clientRunId}</code>
                    <span>{run.caseId}</span>
                  </div>
                  <StatusBadge tone="info">{formatState(run.status)}</StatusBadge>
                  <Link to={`/runs/${run.clientRunId}`}>Open run</Link>
                </li>
              ))}
            </ul>
          ) : (
            <div className="workspace-empty workspace-empty--compact">
              <h3>No runs in this session</h3>
              <p>Choose a ready synthetic case, then start a research-only simulation.</p>
              <Link to="/cases">Go to cases</Link>
            </div>
          )}
        </section>

        <aside className="workspace-panel safety-panel" aria-labelledby="safety-title">
          <ShieldCheck aria-hidden="true" />
          <div>
            <p className="workspace-kicker">Safety boundary</p>
            <h2 id="safety-title">Session memory only</h2>
            <p>
              No patient records, browser storage, or durable review history are used.
              Clinical decisions are blocked; outputs remain research demonstrations.
            </p>
          </div>
        </aside>
      </div>
    </div>
  )
}
