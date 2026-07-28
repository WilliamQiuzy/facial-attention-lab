import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { MockWorkbenchGateway } from '../workbench/MockWorkbenchGateway'
import { WorkspaceProvider } from '../workbench/WorkspaceProvider'
import { getWorkbenchAsset } from '../workbench/catalog'
import type { ResultReview, WorkspaceState } from '../workbench/types'
import { ResultReviewPage } from './ResultReviewPage'
import {
  seedTask6State,
  TestRenderBoundary,
} from './task6TestSupport'

function renderReview(path: string, state = seedTask6State().state) {
  return render(
    <TestRenderBoundary>
      <MemoryRouter initialEntries={[path]}>
        <WorkspaceProvider
          gateway={new MockWorkbenchGateway()}
          initialState={state}
        >
          <Routes>
            <Route
              path="/research/reviews/:reviewId"
              element={<ResultReviewPage />}
            />
          </Routes>
        </WorkspaceProvider>
      </MemoryRouter>
    </TestRenderBoundary>,
  )
}

const corruptReviewCases: readonly [
  label: string,
  corrupt: (review: ResultReview) => unknown,
][] = [
  ['null', () => null],
  ['malformed', (review) => ({ ...review, events: null })],
  [
    'null event',
    (review) => ({ ...review, events: [null, ...review.events.slice(1)] }),
  ],
  ['unknown status', (review) => ({ ...review, status: 'clinically_approved' })],
  [
    'prototype-polluted',
    (review) =>
      Object.fromEntries([
        ...Object.entries(review),
        ['__proto__', { exposeSensitiveReview: true }],
      ]),
  ],
]

type DetailRuntimeCorruptor = (
  state: WorkspaceState,
  runId: string,
  attemptId: string,
) => WorkspaceState

const malformedDetailRuntimeCases: readonly [
  label: string,
  corrupt: DetailRuntimeCorruptor,
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

describe('exact result review workflow', () => {
  it.each(['missing-review', 'toString', '__proto__'])(
    'fails closed for unknown or prototype review ID %s',
    (reviewId) => {
      renderReview(`/research/reviews/${reviewId}`)

      expect(
        screen.getByRole('heading', {
          name: 'Review unavailable in this session',
          level: 1,
        }),
      ).toBeVisible()
      expect(screen.getByText(reviewId)).toBeVisible()
    },
  )

  it('does not substitute a result digest for the exact review ID', () => {
    const seed = seedTask6State({ reviewStatus: 'awaiting_review' })
    renderReview(`/research/reviews/${seed.resultDigest}`, seed.state)

    expect(
      screen.getByRole('heading', {
        name: 'Review unavailable in this session',
        level: 1,
      }),
    ).toBeVisible()
    expect(screen.queryByText('awaiting review')).not.toBeInTheDocument()
  })

  it.each(corruptReviewCases)(
    'fails closed without rendering evidence for a %s in-memory review',
    (_label, corrupt) => {
      const seed = seedTask6State({ reviewStatus: 'awaiting_review' })
      const review = seed.state.reviewsById[seed.reviewId]
      const state = {
        ...seed.state,
        reviewsById: {
          ...seed.state.reviewsById,
          [seed.reviewId]: corrupt(review),
        },
      } as unknown as WorkspaceState

      const view = renderReview(
        `/research/reviews/${seed.reviewId}`,
        state,
      )

      expect(
        screen.getByRole('heading', {
          name: 'Review unavailable in this session',
          level: 1,
        }),
      ).toBeVisible()
      expect(screen.queryByText(seed.runId)).not.toBeInTheDocument()
      expect(screen.queryByText(seed.attemptId)).not.toBeInTheDocument()
      expect(screen.queryByText(seed.resultDigest)).not.toBeInTheDocument()
      expect(
        screen.queryByText('Suitable for a synthetic research demonstration.'),
      ).not.toBeInTheDocument()
      expect(
        screen.queryByRole('region', { name: 'Review event history' }),
      ).not.toBeInTheDocument()
      expect(view.container.querySelector('img')).toBeNull()
    },
  )

  it.each(malformedDetailRuntimeCases)(
    'fails closed without review evidence or lifecycle actions when detail state has %s',
    (_label, corrupt) => {
      const seed = seedTask6State({
        suffix: 'malformed-detail',
        reviewStatus: 'awaiting_review',
      })
      const state = corrupt(seed.state, seed.runId, seed.attemptId)
      const view = renderReview(
        `/research/reviews/${seed.reviewId}`,
        state,
      )

      expect(
        screen.getByRole('heading', { name: 'Review target unavailable', level: 1 }),
      ).toBeVisible()
      expect(screen.getByText(/run and attempt session state is malformed/i)).toBeVisible()
      expect(
        screen.queryByRole('form', { name: 'Structured research review note' }),
      ).not.toBeInTheDocument()
      expect(
        screen.queryByRole('link', { name: 'Open gated patient explanation' }),
      ).not.toBeInTheDocument()
      expect(view.container.querySelector('img')).toBeNull()
      expect(
        screen.queryByRole('heading', { name: /render crashed/i }),
      ).not.toBeInTheDocument()
    },
  )

  it.each(malformedDetailRuntimeCases)(
    'fails closed before review creation when target state has %s',
    (_label, corrupt) => {
      const seed = seedTask6State({ suffix: 'malformed-create' })
      const state = corrupt(seed.state, seed.runId, seed.attemptId)
      const view = renderReview(
        `/research/reviews/new?run=${seed.runId}&attempt=${seed.attemptId}`,
        state,
      )

      expect(
        screen.getByRole('heading', { name: 'Review target unavailable', level: 1 }),
      ).toBeVisible()
      expect(screen.getByText(/run and attempt session state is malformed/i)).toBeVisible()
      expect(
        screen.queryByRole('form', { name: 'Structured research review note' }),
      ).not.toBeInTheDocument()
      expect(view.container.querySelector('img')).toBeNull()
      expect(
        screen.queryByRole('heading', { name: /render crashed/i }),
      ).not.toBeInTheDocument()
    },
  )

  it.each([
    ['/research/reviews/new', 'No exact run and attempt were supplied'],
    [
      '/research/reviews/new?run=run-task6-1',
      'No exact run and attempt were supplied',
    ],
    [
      '/research/reviews/new?run=run-task6-1&run=run-task6-1&attempt=attempt-task6-1',
      'Duplicate run or attempt parameters were supplied',
    ],
    [
      '/research/reviews/new?run=%20run-task6-1&attempt=attempt-task6-1',
      'No exact run and attempt were supplied',
    ],
    [
      '/research/reviews/new?run=run-task6-1%20&attempt=attempt-task6-1',
      'No exact run and attempt were supplied',
    ],
    [
      '/research/reviews/new?run=run-task6-1&attempt=%20attempt-task6-1',
      'No exact run and attempt were supplied',
    ],
    [
      '/research/reviews/new?run=&attempt=attempt-task6-1',
      'No exact run and attempt were supplied',
    ],
    [
      '/research/reviews/new?run=run-task6-1&attempt=',
      'No exact run and attempt were supplied',
    ],
    [
      '/research/reviews/new?run=run-task6-1&attempt=attempt-task6-1&review=review-task6-1',
      'No exact run and attempt were supplied',
    ],
  ])('fails closed for malformed creation binding %s', (path, reason) => {
    renderReview(path)

    expect(
      screen.getByRole('heading', { name: 'Review target unavailable', level: 1 }),
    ).toBeVisible()
    expect(screen.getByText(reason)).toBeVisible()
  })

  it('creates a review from one exact immutable run and attempt with structured notes', async () => {
    const seed = seedTask6State()
    const user = userEvent.setup()
    renderReview(
      `/research/reviews/new?run=${seed.runId}&attempt=${seed.attemptId}`,
      seed.state,
    )

    expect(
      screen.getByRole('heading', { name: 'Review result', level: 1 }),
    ).toBeVisible()
    const asset = getWorkbenchAsset(seed.caseId)!
    expect(screen.getByText(asset.label)).toBeVisible()
    expect(screen.getByText(seed.caseId)).toBeVisible()
    expect(screen.getByText('Ready for author note')).toBeVisible()

    const technicalDetails = screen.getByText('Technical details').closest('details')
    expect(technicalDetails).not.toBeNull()
    expect(technicalDetails).not.toHaveAttribute('open')
    expect(screen.getByText(seed.runId)).not.toBeVisible()
    expect(screen.getByText(seed.attemptId)).not.toBeVisible()
    expect(screen.getByText(seed.resultDigest)).not.toBeVisible()
    expect(screen.getByText(seed.modelVersion)).not.toBeVisible()

    const form = screen.getByRole('form', { name: 'Structured research review note' })
    const rationale = within(form).getByLabelText('Rationale')
    const limitations = within(form).getByLabelText('Limitations')
    expect(rationale).toBeRequired()
    expect(limitations).toBeRequired()
    expect(rationale).toHaveAttribute('name', 'rationale')
    expect(limitations).toHaveAttribute('name', 'limitations')
    expect(rationale).toHaveAttribute('autocomplete', 'off')
    expect(limitations).toHaveAttribute('autocomplete', 'off')

    await user.type(rationale, 'Suitable for a synthetic research discussion.')
    await user.type(limitations, 'No human gaze, patient, or clinical inference.')
    await user.click(within(form).getByRole('button', { name: 'Create review' }))

    expect(
      await screen.findByRole('heading', { name: 'Review result', level: 1 }),
    ).toBeVisible()
    expect(screen.getByText('awaiting review')).toBeVisible()
    expect(screen.getByText(seed.resultDigest)).not.toBeVisible()
    expect(screen.getByText('Suitable for a synthetic research discussion.')).toBeVisible()
  })

  it('leads with the case and status while keeping exact lineage in technical details', async () => {
    const seed = seedTask6State({ reviewStatus: 'awaiting_review' })
    const user = userEvent.setup()
    renderReview(`/research/reviews/${seed.reviewId}`, seed.state)

    expect(screen.getByRole('heading', { name: 'Review result', level: 1 })).toBeVisible()
    expect(screen.getByText(getWorkbenchAsset(seed.caseId)!.label)).toBeVisible()
    expect(screen.getByText(seed.caseId)).toBeVisible()
    expect(screen.getByText('awaiting review')).toBeVisible()
    expect(screen.getByText('Next: record an independent decision.')).toBeVisible()

    const details = screen.getByText('Technical details').closest('details')
    expect(details).not.toBeNull()
    expect(details).not.toHaveAttribute('open')
    expect(screen.getByText(seed.runId)).not.toBeVisible()
    expect(screen.getByText(seed.attemptId)).not.toBeVisible()
    expect(screen.getByText(seed.resultDigest)).not.toBeVisible()
    expect(screen.getByText(seed.assetSha256)).not.toBeVisible()
    expect(screen.getByText(seed.modelVersion)).not.toBeVisible()

    await user.click(screen.getByText('Technical details'))
    expect(screen.getByText(seed.runId)).toBeVisible()
    expect(screen.getByText(seed.attemptId)).toBeVisible()
    expect(screen.getByText(seed.resultDigest)).toBeVisible()
    expect(screen.getByText(seed.assetSha256)).toBeVisible()
    expect(screen.getByText(seed.modelVersion)).toBeVisible()
    expect(
      within(details as HTMLElement).getByText('Simulation profile')
        .nextElementSibling,
    ).toHaveTextContent(seed.modelVersion)
    expect(
      within(details as HTMLElement).getByText('Simulation engine version')
        .nextElementSibling,
    ).toHaveTextContent('1')
    expect(
      within(details as HTMLElement).queryByText('Request interface profile'),
    ).not.toBeInTheDocument()
    expect(
      within(details as HTMLElement).queryByText('Model version'),
    ).not.toBeInTheDocument()
    expect(
      within(details as HTMLElement).queryByText('Model mode'),
    ).not.toBeInTheDocument()
  })

  it('labels connected review evidence with request contract and response model identity', async () => {
    const seed = seedTask6State({
      suffix: 'connected-technical',
      origin: 'model_prediction',
      reviewStatus: 'awaiting_review',
    })
    const user = userEvent.setup()
    renderReview(`/research/reviews/${seed.reviewId}`, seed.state)

    const details = screen.getByText('Technical details').closest('details')
    expect(details).not.toBeNull()
    await user.click(screen.getByText('Technical details'))
    const technical = within(details as HTMLElement)

    expect(
      technical.getByText('Connected request contract').nextElementSibling,
    ).toHaveTextContent('synthetic-spatial-contract-rehearsal/1')
    expect(
      technical.getByText('Connected engine version').nextElementSibling,
    ).toHaveTextContent('task6-test')
    expect(
      technical.getByText('Result digest').nextElementSibling,
    ).toHaveTextContent(seed.resultDigest)
    expect(
      technical.getByText('Connected model ID').nextElementSibling,
    ).toHaveTextContent('observer-attention-test')
    expect(
      technical.getByText('Connected model version').nextElementSibling,
    ).toHaveTextContent('test-v1')
    expect(details).toHaveTextContent(/response-reported connected model identity/i)
    expect(technical.queryByText('Model mode')).not.toBeInTheDocument()
    expect(technical.queryByText('Simulation profile')).not.toBeInTheDocument()
    expect(
      technical.queryByText('Simulation engine version'),
    ).not.toBeInTheDocument()
  })

  it('marks every empty structured-note field and focuses the first empty field', async () => {
    const seed = seedTask6State({ reviewStatus: 'awaiting_review' })
    const user = userEvent.setup()
    renderReview(`/research/reviews/${seed.reviewId}`, seed.state)

    const rationale = screen.getByLabelText('Rationale')
    const limitations = screen.getByLabelText('Limitations')
    await user.click(screen.getByRole('button', { name: 'Request changes' }))

    expect(rationale).toHaveAttribute('aria-invalid', 'true')
    expect(limitations).toHaveAttribute('aria-invalid', 'true')
    expect(rationale).toHaveAttribute('aria-describedby')
    expect(limitations).toHaveAttribute('aria-describedby')
    expect(rationale).toHaveFocus()
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Rationale and limitations are both required.',
    )

    await user.type(rationale, 'A rationale is now present.')
    await user.click(screen.getByRole('button', { name: 'Request changes' }))
    expect(rationale).not.toHaveAttribute('aria-invalid')
    expect(limitations).toHaveAttribute('aria-invalid', 'true')
    expect(limitations).toHaveFocus()
  })

  it('clears structured-note validation state after a successful lifecycle action', async () => {
    const seed = seedTask6State({ reviewStatus: 'awaiting_review' })
    const user = userEvent.setup()
    renderReview(`/research/reviews/${seed.reviewId}`, seed.state)

    await user.click(screen.getByRole('button', { name: 'Request changes' }))
    await user.type(screen.getByLabelText('Rationale'), 'Clarification required.')
    await user.type(
      screen.getByLabelText('Limitations'),
      'Keep the synthetic-only boundary explicit.',
    )
    await user.click(screen.getByRole('button', { name: 'Request changes' }))

    expect(screen.getByText('changes requested')).toBeVisible()
    expect(
      screen.queryByText('Rationale and limitations are both required.'),
    ).not.toBeInTheDocument()
    expect(screen.getByLabelText('Rationale')).not.toHaveAttribute('aria-invalid')
    expect(screen.getByLabelText('Rationale')).not.toHaveAttribute('aria-describedby')
    expect(screen.getByLabelText('Limitations')).not.toHaveAttribute('aria-invalid')
    expect(screen.getByLabelText('Limitations')).not.toHaveAttribute('aria-describedby')
    expect(screen.getByLabelText('Rationale')).toHaveValue('')
    expect(screen.getByLabelText('Limitations')).toHaveValue('')
  })

  it('announces lifecycle status and next action once through one polite live region', async () => {
    const seed = seedTask6State({ reviewStatus: 'awaiting_review' })
    const user = userEvent.setup()
    renderReview(`/research/reviews/${seed.reviewId}`, seed.state)

    expect(screen.getAllByRole('status')).toHaveLength(1)
    const announcement = screen.getByRole('status')
    expect(announcement).toHaveAttribute('aria-live', 'polite')
    expect(announcement).toHaveAttribute('aria-atomic', 'true')
    expect(announcement).toHaveTextContent(/awaiting review/i)
    expect(announcement).toHaveTextContent(/1 recorded event/i)
    expect(announcement).toHaveTextContent(/record an independent decision/i)

    await user.type(screen.getByLabelText('Rationale'), 'Clarification required.')
    await user.type(
      screen.getByLabelText('Limitations'),
      'Keep the synthetic-only boundary explicit.',
    )
    await user.click(screen.getByRole('button', { name: 'Request changes' }))

    expect(screen.getAllByRole('status')).toHaveLength(1)
    expect(screen.getByRole('status')).toHaveTextContent(/changes requested/i)
    expect(screen.getByRole('status')).toHaveTextContent(/2 recorded events/i)
    expect(screen.getByRole('status')).toHaveTextContent(/update the rationale/i)
  })

  it('supports request-changes, resubmit, approve, and revoke without losing history', async () => {
    const seed = seedTask6State({ reviewStatus: 'awaiting_review' })
    const user = userEvent.setup()
    renderReview(`/research/reviews/${seed.reviewId}`, seed.state)

    const fillNote = async (rationale: string, limitations: string) => {
      await user.clear(screen.getByLabelText('Rationale'))
      await user.type(screen.getByLabelText('Rationale'), rationale)
      await user.clear(screen.getByLabelText('Limitations'))
      await user.type(screen.getByLabelText('Limitations'), limitations)
    }

    await fillNote('Clarification required.', 'Keep the synthetic boundary explicit.')
    await user.click(screen.getByRole('button', { name: 'Request changes' }))
    expect(screen.getByText('changes requested')).toBeVisible()

    await fillNote('Boundary clarified.', 'Research demo only; no patient use.')
    await user.click(screen.getByRole('button', { name: 'Resubmit for review' }))
    expect(screen.getByText('awaiting review')).toBeVisible()

    await fillNote('Research demo accepted.', 'Clinical use remains blocked.')
    await user.click(screen.getByRole('button', { name: 'Approve for research demo' }))
    expect(screen.getByText('approved for research')).toBeVisible()
    expect(
      screen.getByRole('link', { name: 'Open gated patient explanation' }),
    ).toHaveAttribute('href', `/patient-report?review=${seed.reviewId}`)

    await fillNote('Approval withdrawn.', 'Do not preview or export this result.')
    await user.click(screen.getByRole('button', { name: 'Revoke research approval' }))
    expect(screen.getByText('revoked')).toBeVisible()
    expect(
      screen.queryByRole('link', { name: 'Open gated patient explanation' }),
    ).not.toBeInTheDocument()

    const history = screen.getByRole('region', { name: 'Review event history' })
    expect(within(history).getAllByRole('listitem')).toHaveLength(5)
  })
})
