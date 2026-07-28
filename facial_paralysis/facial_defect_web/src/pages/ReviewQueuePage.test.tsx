import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { MockWorkbenchGateway } from '../workbench/MockWorkbenchGateway'
import { WorkspaceProvider } from '../workbench/WorkspaceProvider'
import { listWorkbenchAssets } from '../workbench/catalog'
import type { WorkspaceState } from '../workbench/types'
import { ReviewQueuePage } from './ReviewQueuePage'
import { seedTask6State, TestRenderBoundary } from './task6TestSupport'

function renderQueue(state?: WorkspaceState) {
  return render(
    <TestRenderBoundary>
      <MemoryRouter initialEntries={['/research/reviews']}>
        <WorkspaceProvider
          gateway={new MockWorkbenchGateway()}
          {...(state ? { initialState: state } : {})}
        >
          <ReviewQueuePage />
        </WorkspaceProvider>
      </MemoryRouter>
    </TestRenderBoundary>,
  )
}

type QueueRuntimeCorruptor = (
  state: WorkspaceState,
  runId: string,
  attemptId: string,
) => WorkspaceState

const malformedQueueRuntimeCases: readonly [
  label: string,
  corrupt: QueueRuntimeCorruptor,
][] = [
  ['null runsById map', (state) => ({ ...state, runsById: null }) as unknown as WorkspaceState],
  ['null attemptsById map', (state) => ({ ...state, attemptsById: null }) as unknown as WorkspaceState],
  ['null reviewsById map', (state) => ({ ...state, reviewsById: null }) as unknown as WorkspaceState],
  ['null reviewOrder', (state) => ({ ...state, reviewOrder: null }) as unknown as WorkspaceState],
  ['null roisByCase map', (state) => ({ ...state, roisByCase: null }) as unknown as WorkspaceState],
  [
    'object-shaped attemptIds',
    (state, runId, attemptId) => ({
      ...state,
      runsById: {
        ...state.runsById,
        [runId]: { ...state.runsById[runId], attemptIds: { 0: attemptId, length: 1 } },
      },
    }) as unknown as WorkspaceState,
  ],
  [
    'run missing attemptIds',
    (state, runId) => ({
      ...state,
      runsById: {
        ...state.runsById,
        [runId]: Object.fromEntries(
          Object.entries(state.runsById[runId]!).filter(([key]) => key !== 'attemptIds'),
        ),
      },
    }) as unknown as WorkspaceState,
  ],
  [
    'run with an extra key',
    (state, runId) => ({
      ...state,
      runsById: {
        ...state.runsById,
        [runId]: { ...state.runsById[runId], unexpected: true },
      },
    }) as unknown as WorkspaceState,
  ],
  [
    'attempt missing attemptToken',
    (state, _runId, attemptId) => ({
      ...state,
      attemptsById: {
        ...state.attemptsById,
        [attemptId]: Object.fromEntries(
          Object.entries(state.attemptsById[attemptId]!).filter(
            ([key]) => key !== 'attemptToken',
          ),
        ),
      },
    }) as unknown as WorkspaceState,
  ],
  [
    'attempt with an extra key',
    (state, _runId, attemptId) => ({
      ...state,
      attemptsById: {
        ...state.attemptsById,
        [attemptId]: { ...state.attemptsById[attemptId], unexpected: true },
      },
    }) as unknown as WorkspaceState,
  ],
  [
    'activeAttemptId outside the run attempt list',
    (state, runId) => ({
      ...state,
      runsById: {
        ...state.runsById,
        [runId]: { ...state.runsById[runId], activeAttemptId: 'attempt-not-in-this-run' },
      },
    }) as unknown as WorkspaceState,
  ],
]

function queueState(): WorkspaceState {
  const ready = seedTask6State({ suffix: 'ready', assetIndex: 2 })
  const blocked = seedTask6State({
    suffix: 'blocked',
    assetIndex: 3,
    freshness: 'stale',
  })
  const existing = seedTask6State({
    suffix: 'existing',
    assetIndex: 4,
    reviewStatus: 'awaiting_review',
  })

  return {
    ...ready.state,
    runsById: {
      ...ready.state.runsById,
      ...blocked.state.runsById,
      ...existing.state.runsById,
    },
    runOrder: [ready.runId, blocked.runId, existing.runId],
    attemptsById: {
      ...ready.state.attemptsById,
      ...blocked.state.attemptsById,
      ...existing.state.attemptsById,
    },
    reviewsById: existing.state.reviewsById,
    reviewOrder: existing.state.reviewOrder,
  }
}

describe('result review queue', () => {
  it.each(malformedQueueRuntimeCases)(
    'shows one fail-closed blocker with no result action for %s',
    (_label, corrupt) => {
      const seed = seedTask6State({ suffix: 'malformed-queue' })
      const state = corrupt(seed.state, seed.runId, seed.attemptId)
      const view = renderQueue(state)

      expect(
        screen.getByRole('heading', { name: 'Review results', level: 1 }),
      ).toBeVisible()
      expect(screen.getByRole('alert')).toHaveTextContent(
        /review session state is unavailable/i,
      )
      expect(screen.queryByRole('link')).not.toBeInTheDocument()
      expect(screen.queryByRole('button')).not.toBeInTheDocument()
      expect(view.container.querySelector('img')).toBeNull()
      expect(
        screen.queryByRole('heading', { name: /render crashed/i }),
      ).not.toBeInTheDocument()
    },
  )

  it('separates exact eligible candidates from visible blockers', async () => {
    const user = userEvent.setup()
    const state = queueState()
    renderQueue(state)

    expect(
      screen.getByRole('heading', { name: 'Review results', level: 1 }),
    ).toBeVisible()
    const ready = screen.getByRole('region', { name: 'Ready for research review' })
    const readyAction = within(ready).getByRole('link', { name: 'Review result' })
    expect(readyAction).toHaveAttribute(
      'href',
      '/research/reviews/new?run=run-task6-ready&attempt=attempt-task6-ready',
    )
    const readyRow = readyAction.closest('article')
    expect(readyRow).not.toBeNull()
    expect(within(readyRow!).getByText(listWorkbenchAssets()[2]!.label)).toBeVisible()
    expect(within(readyRow!).getByText('SYN-HNC-CHEEK-TUMOUR')).toBeVisible()
    expect(within(readyRow!).getByText('Ready to review')).toBeVisible()
    expect(within(readyRow!).getByText('run-task6-ready')).not.toBeVisible()

    const readyDetails = within(readyRow!).getByText('Technical details').closest('details')
    expect(readyDetails).not.toBeNull()
    expect(readyDetails).not.toHaveAttribute('open')
    await user.click(within(readyRow!).getByText('Technical details'))
    expect(readyDetails).toHaveAttribute('open')
    expect(within(readyRow!).getByText('run-task6-ready')).toBeVisible()

    const blocked = screen.getByRole('region', { name: 'Blocked review candidates' })
    expect(within(blocked).getByText(listWorkbenchAssets()[3]!.label)).toBeVisible()
    expect(within(blocked).getByText('SYN-HNC-CHEEK-FREEFLAP')).toBeVisible()
    expect(within(blocked).getByText('Needs attention')).toBeVisible()
    expect(within(blocked).getByText('run-task6-blocked')).not.toBeVisible()
    expect(
      within(blocked).getByText(/stale/i, {
        selector: '.task6-queue-row__blockers > div > span',
      }),
    ).toBeVisible()
    expect(within(blocked).queryByRole('link')).not.toBeInTheDocument()

    const existing = screen.getByRole('region', { name: 'Session reviews' })
    const existingAction = within(existing).getByRole('link', { name: 'Open review' })
    expect(existingAction).toHaveAttribute(
      'href',
      '/research/reviews/review-task6-existing',
    )
    expect(within(existing).getByText(listWorkbenchAssets()[4]!.label)).toBeVisible()
    expect(screen.getByText(/research-only review/i)).toBeVisible()
  })

  it('shows a truthful empty state without inventing a candidate', () => {
    renderQueue()

    expect(
      screen.getByRole('heading', { name: 'Review results', level: 1 }),
    ).toBeVisible()
    expect(screen.getByText(/no session results are ready for review/i)).toBeVisible()
    expect(screen.queryByRole('link', { name: 'Review result' })).not.toBeInTheDocument()
  })
})
