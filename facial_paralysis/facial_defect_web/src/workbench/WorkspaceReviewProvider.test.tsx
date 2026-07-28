import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { WorkbenchGateway } from './WorkbenchGateway'
import {
  WorkspaceProvider,
  useWorkspace,
  type WorkspaceRuntime,
} from './WorkspaceProvider'
import {
  AUTHOR_NOTE,
  REVIEWER_NOTE,
  createSucceededReviewTarget,
} from './reviewTestFixtures'
import type { ResearchReviewNote } from './types'

const gateway: WorkbenchGateway = {
  mode: 'mock',
  runInference: async () => {
    throw new Error('not used')
  },
}

const CHANGE_NOTE: ResearchReviewNote = {
  rationale: 'Clarify the research-only scope.',
  limitations: 'Current explanation is too broad.',
}

function ReviewProbe({ runId, attemptId }: { runId: string; attemptId: string }) {
  const { state, actions } = useWorkspace()
  const reviewId = state.reviewOrder[0]
  const review = reviewId ? state.reviewsById[reviewId] : undefined
  return (
    <>
      <button
        type="button"
        onClick={() => actions.createReview({ runId, attemptId, note: AUTHOR_NOTE })}
      >
        Create review
      </button>
      <button
        type="button"
        onClick={() => reviewId && actions.requestReviewChanges(reviewId, CHANGE_NOTE)}
      >
        Request changes
      </button>
      <button
        type="button"
        onClick={() => reviewId && actions.resubmitReview(reviewId, AUTHOR_NOTE)}
      >
        Resubmit review
      </button>
      <button
        type="button"
        onClick={() => reviewId && actions.approveReview(reviewId, REVIEWER_NOTE)}
      >
        Approve review
      </button>
      <button
        type="button"
        onClick={() => reviewId && actions.revokeReview(reviewId, CHANGE_NOTE)}
      >
        Revoke review
      </button>
      <output aria-label="review ID">{reviewId ?? 'none'}</output>
      <output aria-label="review status">{review?.status ?? 'none'}</output>
      <output aria-label="review events">{review?.events.length ?? 0}</output>
    </>
  )
}

describe('WorkspaceProvider result review commands', () => {
  it('owns the review ID and creates an exact awaiting review', async () => {
    const user = userEvent.setup()
    const seeded = createSucceededReviewTarget()
    const runtime: WorkspaceRuntime = {
      nextId: vi.fn((kind) => `${kind}-provider-1`),
    }
    render(
      <WorkspaceProvider gateway={gateway} runtime={runtime} initialState={seeded.state}>
        <ReviewProbe runId={seeded.binding.clientRunId} attemptId={seeded.attemptId} />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Create review' }))

    expect(runtime.nextId).toHaveBeenCalledOnce()
    expect(runtime.nextId).toHaveBeenCalledWith('review')
    expect(screen.getByLabelText('review ID')).toHaveTextContent('review-provider-1')
    expect(screen.getByLabelText('review status')).toHaveTextContent('awaiting_review')
    expect(screen.getByLabelText('review events')).toHaveTextContent('1')
  })

  it('exposes the complete separated-role lifecycle as append-only events', async () => {
    const user = userEvent.setup()
    const seeded = createSucceededReviewTarget()
    render(
      <WorkspaceProvider gateway={gateway} initialState={seeded.state}>
        <ReviewProbe runId={seeded.binding.clientRunId} attemptId={seeded.attemptId} />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Create review' }))
    await user.click(screen.getByRole('button', { name: 'Request changes' }))
    expect(screen.getByLabelText('review status')).toHaveTextContent('changes_requested')
    await user.click(screen.getByRole('button', { name: 'Resubmit review' }))
    expect(screen.getByLabelText('review status')).toHaveTextContent('awaiting_review')
    await user.click(screen.getByRole('button', { name: 'Approve review' }))
    expect(screen.getByLabelText('review status')).toHaveTextContent(
      'approved_for_research',
    )
    await user.click(screen.getByRole('button', { name: 'Revoke review' }))
    expect(screen.getByLabelText('review status')).toHaveTextContent('revoked')
    expect(screen.getByLabelText('review events')).toHaveTextContent('5')
  })
})
