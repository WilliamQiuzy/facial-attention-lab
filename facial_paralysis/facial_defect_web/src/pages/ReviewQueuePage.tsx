import {
  ArrowRight,
  ClipboardCheck,
  FileSearch,
  LockKeyhole,
} from 'lucide-react'
import { Link } from 'react-router-dom'
import { StatusBadge } from '../components/StatusBadge'
import { getWorkbenchAsset } from '../workbench/catalog'
import {
  listReviewQueueItems,
  reviewRuntimeStateBlockers,
  type ReviewQueueItem,
} from '../workbench/reviewPolicy'
import { useWorkspace } from '../workbench/WorkspaceProvider'
import type { ReviewStatus, WorkspaceState } from '../workbench/types'

function reviewStatusLabel(status: ReviewStatus | undefined): string {
  if (status === 'approved_for_research') return 'Approved for research'
  if (status === 'changes_requested') return 'Changes requested'
  if (status === 'revoked') return 'Approval revoked'
  return 'Awaiting review'
}

type QueueItemContext = {
  readonly item: ReviewQueueItem
  readonly caseId: string
  readonly caseLabel: string
}

function TechnicalDetails({ item }: { readonly item: ReviewQueueItem }) {
  return (
    <details className="task6-technical-details">
      <summary>Technical details</summary>
      <dl>
        <div><dt>Run ID</dt><dd><code>{item.runId}</code></dd></div>
        <div><dt>Attempt ID</dt><dd><code>{item.attemptId}</code></dd></div>
        {item.resultDigest ? (
          <div><dt>Result digest</dt><dd><code>{item.resultDigest}</code></dd></div>
        ) : null}
        {item.inputFingerprint ? (
          <div><dt>Input fingerprint</dt><dd><code>{item.inputFingerprint}</code></dd></div>
        ) : null}
        {item.blockers.length > 0 ? (
          <div>
            <dt>Safety checks</dt>
            <dd>
              <ul>
                {item.blockers.map((entry) => (
                  <li key={entry.code}><code>{entry.code}</code>: {entry.message}</li>
                ))}
              </ul>
            </dd>
          </div>
        ) : null}
      </dl>
    </details>
  )
}

function ItemIdentity({ caseId, caseLabel }: QueueItemContext) {
  return (
    <div className="task6-review-identity">
      <span>Case</span>
      <strong>{caseLabel}</strong>
      <code>{caseId}</code>
    </div>
  )
}

function ReadyItem(context: QueueItemContext) {
  const { item } = context
  return (
    <article className="task6-queue-row task6-queue-row--ready">
      <ItemIdentity {...context} />
      <div className="task6-queue-row__state">
        <StatusBadge tone="success">Ready to review</StatusBadge>
        <span>The current result is ready for an independent decision.</span>
      </div>
      <Link
        className="workspace-button workspace-button--primary"
        to={`/research/reviews/new?run=${encodeURIComponent(item.runId)}&attempt=${encodeURIComponent(item.attemptId)}`}
      >
        Review result <ArrowRight aria-hidden="true" />
      </Link>
      <TechnicalDetails item={item} />
    </article>
  )
}

function ExistingReviewItem({
  reviewStatus,
  ...context
}: QueueItemContext & { readonly reviewStatus?: ReviewStatus }) {
  const { item } = context
  if (!item.reviewId) return null
  return (
    <article className="task6-queue-row">
      <ItemIdentity {...context} />
      <div className="task6-queue-row__state">
        <StatusBadge tone={item.patientPreviewEligible ? 'success' : 'info'}>
          {reviewStatusLabel(reviewStatus)}
        </StatusBadge>
        <span>{item.blockers[0]?.message ?? 'The review is complete for this result.'}</span>
      </div>
      <Link
        className="workspace-button workspace-button--secondary"
        to={`/research/reviews/${encodeURIComponent(item.reviewId)}`}
      >
        Open review <ArrowRight aria-hidden="true" />
      </Link>
      <TechnicalDetails item={item} />
    </article>
  )
}

function BlockedItem(context: QueueItemContext) {
  const { item } = context
  return (
    <article className="task6-queue-row task6-queue-row--blocked">
      <ItemIdentity {...context} />
      <div className="task6-queue-row__blockers">
        <StatusBadge tone="blocked">Needs attention</StatusBadge>
        <div>
          <span>{item.blockers[0]?.message ?? 'This result cannot be reviewed.'}</span>
          {item.blockers.length > 1 ? (
            <small>{item.blockers.length - 1} more safety checks in Technical details.</small>
          ) : null}
        </div>
      </div>
      <TechnicalDetails item={item} />
    </article>
  )
}

function itemContext(state: WorkspaceState, item: ReviewQueueItem): QueueItemContext {
  const run = Object.hasOwn(state.runsById, item.runId)
    ? state.runsById[item.runId]
    : undefined
  const asset = run ? getWorkbenchAsset(run.caseId) : undefined
  return {
    item,
    caseId: run?.caseId ?? 'Case unavailable',
    caseLabel: asset?.label ?? 'Case unavailable',
  }
}

function QueueHeader() {
  return (
    <header className="workspace-page__header page-shell task6-page__header">
      <div>
        <p className="workspace-kicker">Results</p>
        <h1>Review results</h1>
        <p>
          Review each current result, record the evidence limits, and choose the next
          safe step.
        </p>
      </div>
      <div className="task6-boundary-note" role="note">
        <LockKeyhole aria-hidden="true" />
        <div>
          <strong>Research-only review</strong>
          <span>This does not approve clinical use. Review state stays in this session.</span>
        </div>
      </div>
    </header>
  )
}

export function ReviewQueuePage() {
  const { state } = useWorkspace()
  const runtimeBlockers = reviewRuntimeStateBlockers(state)
  if (runtimeBlockers.length > 0) {
    return (
      <section className="workspace-page task6-page task6-queue-page">
        <QueueHeader />
        <div className="page-shell">
          <div className="task6-review-blockers" role="alert">
            <LockKeyhole aria-hidden="true" />
            <div>
              <strong>Review session state is unavailable</strong>
              <ul>
                {runtimeBlockers.map((entry) => (
                  <li key={entry.code}>{entry.message}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </section>
    )
  }
  const items = listReviewQueueItems(state)
  const ready = items.filter((item) => item.canCreateReview)
  const existing = items.filter((item) => item.reviewId !== undefined)
  const blocked = items.filter(
    (item) => !item.canCreateReview && item.reviewId === undefined,
  )

  return (
    <section className="workspace-page task6-page task6-queue-page">
      <QueueHeader />

      <div className="page-shell task6-queue-summary" aria-label="Review queue summary">
        <div><strong>{ready.length}</strong><span>Ready to review</span></div>
        <div><strong>{existing.length}</strong><span>Reviews started</span></div>
        <div><strong>{blocked.length}</strong><span>Need attention</span></div>
      </div>

      <div className="page-shell task6-queue-stack">
        <section
          className="workspace-panel task6-queue-section"
          aria-label="Ready for research review"
        >
          <div className="task6-section-heading">
            <ClipboardCheck aria-hidden="true" />
            <div><p className="workspace-kicker">Current result</p><h2>Ready for review</h2></div>
          </div>
          {ready.length > 0 ? (
            <div>{ready.map((item) => <ReadyItem key={`${item.runId}:${item.attemptId}`} {...itemContext(state, item)} />)}</div>
          ) : (
            <div className="task6-empty-state">
              <FileSearch aria-hidden="true" />
              <p>No session results are ready for review.</p>
              <span>Run an approved synthetic case to add a result here.</span>
              <Link to="/cases">Choose a case</Link>
            </div>
          )}
        </section>

        {existing.length > 0 ? (
          <section className="workspace-panel task6-queue-section" aria-label="Session reviews">
            <div className="task6-section-heading">
              <FileSearch aria-hidden="true" />
              <div><p className="workspace-kicker">Session history</p><h2>Recorded reviews</h2></div>
            </div>
            <div>{existing.map((item) => {
              const review = item.reviewId && Object.hasOwn(state.reviewsById, item.reviewId)
                ? state.reviewsById[item.reviewId]
                : undefined
              return (
                <ExistingReviewItem
                  key={`${item.runId}:${item.attemptId}`}
                  {...itemContext(state, item)}
                  reviewStatus={review?.status}
                />
              )
            })}</div>
          </section>
        ) : null}

        {blocked.length > 0 ? (
          <section
            className="workspace-panel task6-queue-section"
            aria-label="Blocked review candidates"
          >
            <div className="task6-section-heading task6-section-heading--blocked">
              <LockKeyhole aria-hidden="true" />
              <div><p className="workspace-kicker">Fail closed</p><h2>Blocked candidates</h2></div>
            </div>
            <div>{blocked.map((item) => <BlockedItem key={`${item.runId}:${item.attemptId}`} {...itemContext(state, item)} />)}</div>
          </section>
        ) : null}
      </div>
    </section>
  )
}
