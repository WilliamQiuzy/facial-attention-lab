import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { readFileSync } from 'node:fs'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import { BatchJobsPage } from './BatchJobsPage'
import { listWorkbenchAssets } from '../workbench/catalog'
import type { WorkbenchGateway } from '../workbench/WorkbenchGateway'
import {
  WorkspaceProvider,
  type WorkspaceRuntime,
} from '../workbench/WorkspaceProvider'
import { runMockEngine } from '../workbench/mockEngine'
import { createInitialWorkspaceState } from '../workbench/reducer'
import type {
  InferenceBinding,
  InferenceOutput,
  RoiAnnotation,
  WorkspaceState,
} from '../workbench/types'

type DeferredRequest = {
  readonly binding: InferenceBinding
  readonly signal: AbortSignal | undefined
  readonly resolve: (output: InferenceOutput) => void
  readonly reject: (error: unknown) => void
}

function createDeferredGateway(mode: WorkbenchGateway['mode'] = 'mock') {
  const requests: DeferredRequest[] = []
  const runInference = vi.fn(
    (binding: InferenceBinding, options?: { readonly signal?: AbortSignal }) =>
      new Promise<InferenceOutput>((resolve, reject) => {
        requests.push({ binding, signal: options?.signal, resolve, reject })
      }),
  )
  const gateway: WorkbenchGateway = { mode, runInference }
  return { gateway, requests, runInference }
}

function renderPage(
  gateway = createDeferredGateway(),
  runtime?: WorkspaceRuntime,
  initialState?: WorkspaceState,
) {
  render(
    <MemoryRouter>
      <WorkspaceProvider
        gateway={gateway.gateway}
        runtime={runtime}
        initialState={initialState}
      >
        <BatchJobsPage />
      </WorkspaceProvider>
    </MemoryRouter>,
  )
  return gateway
}

const catalog = listWorkbenchAssets()
const blockedCases = catalog.slice(0, 2)
const task5Css = readFileSync('src/styles/task5.css', 'utf8')

function createExplicitBlockedState(): WorkspaceState {
  const state = createInitialWorkspaceState()
  const draftDefault = state.roisByCase[blockedCases[0].id]!
  const inReviewDefault = state.roisByCase[blockedCases[1].id]!
  const { reviewerId: _draftReviewer, ...draftBase } = draftDefault
  const { reviewerId: _reviewReviewer, ...inReviewBase } = inReviewDefault
  const draftRoi: RoiAnnotation = { ...draftBase, status: 'draft' }
  const inReviewRoi: RoiAnnotation = { ...inReviewBase, status: 'in_review' }
  return {
    ...state,
    roisByCase: {
      ...state.roisByCase,
      [blockedCases[0].id]: draftRoi,
      [blockedCases[1].id]: inReviewRoi,
    },
  }
}

function renderBlockedPage(
  gateway = createDeferredGateway(),
  runtime?: WorkspaceRuntime,
) {
  return renderPage(gateway, runtime, createExplicitBlockedState())
}

async function selectAllCases(user: ReturnType<typeof userEvent.setup>) {
  for (const checkbox of screen.getAllByRole('checkbox')) {
    await user.click(checkbox)
  }
}

function confirmExclusions() {
  return screen.getByRole('checkbox', {
    name: /I understand that 2 cases need source binding and will not run/i,
  })
}

describe('batch jobs page', () => {
  it('collapses the two-column layout before progress controls can clip', () => {
    const responsiveStart = task5Css.indexOf('@media (max-width: 1260px)')
    expect(responsiveStart).toBeGreaterThan(-1)
    expect(task5Css.slice(responsiveStart, responsiveStart + 180)).toMatch(
      /\.task5-layout\s*{\s*grid-template-columns:\s*1fr/,
    )
    const compactProgressStart = task5Css.indexOf('@media (max-width: 780px)')
    expect(compactProgressStart).toBeGreaterThan(-1)
    expect(task5Css.slice(compactProgressStart, compactProgressStart + 420)).toMatch(
      /\.task5-progress-row\s*{\s*grid-template-columns:\s*1fr/,
    )

    const labelRule = task5Css.match(
      /\.task5-progress-matrix__labels\s*{([^}]*)}/,
    )?.[1]
    expect(labelRule).toMatch(/font-size:\s*\.875rem/)
    expect(task5Css).toMatch(
      /\.task5-progress__heading h2\s*{[^}]*scroll-margin-top:\s*84px/,
    )
  })

  it('uses clinician-facing copy and keeps settings in a native disclosure', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    const storageSpy = vi.spyOn(Storage.prototype, 'setItem')
    const user = userEvent.setup()
    renderBlockedPage()

    expect(screen.getByRole('heading', { name: 'Run several cases', level: 1 })).toBeVisible()
    expect(screen.getAllByRole('checkbox')).toHaveLength(10)
    expect(screen.queryByLabelText(/upload/i)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/file/i)).not.toBeInTheDocument()
    expect(screen.getByText(/memory only/i)).toBeVisible()
    for (const blockedCase of blockedCases) {
      const checklistRow = screen.getByText(blockedCase.id).closest<HTMLElement>(
        '.task5-case-checklist__row',
      )
      expect(checklistRow).not.toBeNull()
      expect(
        within(checklistRow!).getByRole('link', {
          name: 'Restore source binding',
        }),
      ).toHaveAttribute('href', `/cases/${blockedCase.id}/roi`)
    }

    const advancedSettings = screen.getByText('Advanced settings').closest('details')
    expect(advancedSettings).not.toBeNull()
    expect(advancedSettings).not.toHaveAttribute('open')
    expect(within(advancedSettings!).getByRole('combobox', { name: 'Mock model' })).toHaveAttribute(
      'name',
      'modelVersion',
    )
    expect(within(advancedSettings!).getByLabelText('Threshold 45 percent')).toHaveAttribute(
      'name',
      'threshold',
    )
    expect(within(advancedSettings!).getByLabelText('Smoothing 30 percent')).toHaveAttribute(
      'name',
      'smoothing',
    )
    for (const checkbox of within(
      screen.getByRole('complementary', {
        name: 'Batch configuration',
      }),
    ).getAllByRole('checkbox')) {
      expect(checkbox).toHaveAttribute('name', 'caseIds')
    }

    await user.click(screen.getByRole('button', { name: 'Select ready cases' }))
    expect(screen.getByLabelText('Selected cases')).toHaveTextContent('8')
    await user.click(screen.getByRole('button', { name: 'Review selected cases' }))

    expect(screen.getByRole('heading', { name: 'Review selected cases', level: 2 })).toBeVisible()
    const preview = screen.getByRole('region', { name: 'Selected case review' })
    expect(within(preview).getAllByTestId('preflight-item')).toHaveLength(10)
    const exclusions = within(preview).getByRole('region', {
      name: 'Cases excluded from this run',
    })
    expect(within(exclusions).getByRole('checkbox')).toHaveAttribute(
      'name',
      'confirmExclusions',
    )
    for (const blockedCase of blockedCases) {
      expect(within(exclusions).getByText(blockedCase.id)).toBeVisible()
      expect(within(exclusions).getByText(blockedCase.label)).toBeVisible()
    }
    expect(screen.getByRole('button', { name: 'Start 8 simulations' })).toBeDisabled()
    await user.click(confirmExclusions())
    expect(screen.getByRole('button', { name: 'Start 8 simulations' })).toBeEnabled()
    expect(fetchSpy).not.toHaveBeenCalled()
    expect(storageSpy).not.toHaveBeenCalled()
  })

  it('states the actual gateway boundary instead of promising no network in connected mode', () => {
    renderPage(createDeferredGateway('connected'))

    expect(screen.getByText(/Network required for simulations/i)).toBeVisible()
    expect(screen.queryByText(/no upload, network, or storage/i)).not.toBeInTheDocument()
  })

  it('keeps all selected cases visible, names blocked exclusions, and requires confirmation', async () => {
    const user = userEvent.setup()
    renderBlockedPage()

    await selectAllCases(user)
    await user.click(screen.getByRole('button', { name: 'Review selected cases' }))

    const preview = screen.getByRole('region', { name: 'Selected case review' })
    const rows = within(preview).getAllByTestId('preflight-item')
    expect(rows).toHaveLength(10)
    expect(rows.filter((row) => within(row).queryByText('Ready'))).toHaveLength(8)
    expect(
      rows.filter((row) => within(row).queryByText('Needs source binding')),
    ).toHaveLength(2)
    expect(within(preview).queryByText(/ROI review/i)).not.toBeInTheDocument()
    expect(within(preview).getByLabelText('Ready cases')).toHaveTextContent('8')
    expect(within(preview).getByLabelText('Blocked cases')).toHaveTextContent('2')
    for (const asset of listWorkbenchAssets()) {
      expect(rows.some((row) => within(row).queryByText(asset.id))).toBe(true)
    }
    expect(preview.querySelector('.task5-manifest__warning')).toHaveTextContent(
      /2 cases need source binding and will not run/i,
    )
    for (const blockedCase of blockedCases) {
      const row = rows.find((candidate) =>
        within(candidate).queryByText(blockedCase.id),
      )
      expect(row).toBeDefined()
      expect(
        within(row!).queryByRole('link', { name: 'Restore source binding' }),
      ).not.toBeInTheDocument()
    }

    const technicalDetails = within(preview).getByText('Technical details').closest('details')
    expect(technicalDetails).not.toBeNull()
    expect(technicalDetails).not.toHaveAttribute('open')
    expect(within(technicalDetails!).getByLabelText('Manifest hash')).toHaveTextContent(
      /^manifest_[a-f0-9]{16}$/,
    )
    await user.click(within(preview).getByText('Technical details'))
    expect(technicalDetails).toHaveAttribute('open')
    expect(within(technicalDetails!).getByLabelText('Manifest hash')).toBeVisible()

    const exclusions = screen.getByRole('region', { name: 'Cases excluded from this run' })
    for (const blockedCase of blockedCases) {
      expect(within(exclusions).getByText(blockedCase.id)).toBeVisible()
      expect(within(exclusions).getByText(blockedCase.label)).toBeVisible()
      const exclusion = within(exclusions).getByText(blockedCase.id).closest('li')
      expect(exclusion).not.toBeNull()
      expect(
        within(exclusion!).getByRole('link', { name: 'Restore source binding' }),
      ).toHaveAttribute('href', `/cases/${blockedCase.id}/roi`)
    }
    expect(screen.getByRole('button', { name: 'Start 8 simulations' })).toBeDisabled()
    await user.click(confirmExclusions())
    expect(screen.getByRole('button', { name: 'Start 8 simulations' })).toBeEnabled()
  })

  it('resets exclusion confirmation when selection, settings, or the review manifest changes', async () => {
    const user = userEvent.setup()
    renderBlockedPage()

    await selectAllCases(user)
    await user.click(screen.getByRole('button', { name: 'Review selected cases' }))
    await user.click(confirmExclusions())
    expect(screen.getByRole('button', { name: 'Start 8 simulations' })).toBeEnabled()

    const lastCase = catalog.at(-1)!
    await user.click(screen.getByRole('checkbox', { name: new RegExp(lastCase.id) }))
    await user.click(screen.getByRole('checkbox', { name: new RegExp(lastCase.id) }))
    expect(confirmExclusions()).not.toBeChecked()
    expect(screen.getByRole('button', { name: 'Start 8 simulations' })).toBeDisabled()

    await user.click(confirmExclusions())
    fireEvent.change(screen.getByLabelText('Threshold 45 percent'), {
      target: { value: '46' },
    })
    fireEvent.change(screen.getByLabelText('Threshold 46 percent'), {
      target: { value: '45' },
    })
    expect(confirmExclusions()).not.toBeChecked()
    expect(screen.getByRole('button', { name: 'Start 8 simulations' })).toBeDisabled()

    await user.click(confirmExclusions())
    await user.click(screen.getByRole('button', { name: 'Review selected cases' }))
    expect(confirmExclusions()).not.toBeChecked()
    expect(screen.getByRole('button', { name: 'Start 8 simulations' })).toBeDisabled()
  })

  it('invalidates launch when configuration changes after review', async () => {
    const user = userEvent.setup()
    renderBlockedPage()

    await user.click(screen.getByRole('button', { name: 'Select ready cases' }))
    await user.click(screen.getByRole('button', { name: 'Review selected cases' }))
    fireEvent.change(screen.getByLabelText('Threshold 45 percent'), {
      target: { value: '46' },
    })

    expect(screen.getByRole('alert')).toHaveTextContent(/review is out of date/i)
    expect(screen.getByRole('button', { name: 'Start 8 simulations' })).toBeDisabled()
  })

  it('submits only visibly ready items, exposes progress, cancels, and retries with lineage', async () => {
    const user = userEvent.setup()
    const deferred = renderBlockedPage()

    await selectAllCases(user)
    await user.click(screen.getByRole('button', { name: 'Review selected cases' }))
    await user.click(confirmExclusions())
    await user.click(screen.getByRole('button', { name: 'Start 8 simulations' }))

    await waitFor(() => expect(deferred.runInference).toHaveBeenCalledTimes(8))
    const compactConfiguration = screen.getByRole('complementary', {
      name: 'Batch submission summary',
    })
    expect(
      within(compactConfiguration).getByRole('heading', {
        name: '8 simulations started',
        level: 2,
      }),
    ).toBeVisible()
    expect(
      within(compactConfiguration).getByText(
        '2 cases need source binding and were not submitted.',
      ),
    ).toBeVisible()
    expect(
      screen.queryByRole('region', { name: 'Selected case review' }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole('button', { name: 'Start 8 simulations' }),
    ).not.toBeInTheDocument()
    expect(deferred.requests.map((request) => request.binding.caseId)).not.toEqual(
      expect.arrayContaining(blockedCases.map((asset) => asset.id)),
    )
    const progress = screen.getByRole('region', { name: 'Batch progress' })
    expect(
      within(progress).getByRole('heading', {
        name: 'Batch progress',
        level: 2,
      }),
    ).toHaveFocus()
    expect(progress).toHaveAttribute('aria-busy', 'true')
    expect(
      within(progress).getByText('Analyzing 8 cases…'),
    ).toBeVisible()
    expect(within(progress).getAllByTestId('job-progress-row')).toHaveLength(10)
    expect(within(progress).getAllByText('running')).toHaveLength(8)
    expect(within(progress).getAllByText('blocked')).toHaveLength(2)
    expect(within(progress).getByText(/8 submitted · 2 blocked/i)).toBeVisible()
    expect(
      within(progress).queryByRole('heading', { name: /^batch-job-/i }),
    ).not.toBeInTheDocument()
    expect(
      within(progress).getByLabelText('Batch status summary'),
    ).toHaveTextContent(/8 running · 2 blocked/i)
    const batchAnnouncement = screen.getByRole('status', {
      name: 'Batch progress announcement',
    })
    expect(progress).not.toContainElement(batchAnnouncement)
    expect(batchAnnouncement).toHaveTextContent(/8 running · 2 blocked/i)

    const technicalDetails = within(progress).getByText('Technical details').closest('details')
    expect(technicalDetails).not.toBeNull()
    expect(technicalDetails).not.toHaveAttribute('open')
    const firstRunId = deferred.requests[0].binding.clientRunId
    expect(
      within(technicalDetails!).getByText(`Run ID: ${firstRunId}`),
    ).not.toBeVisible()
    const firstReadyRowBeforeRetry = within(progress).getAllByTestId('job-progress-row')[2]
    expect(within(firstReadyRowBeforeRetry).queryByText(/attempt/i)).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Cancel batch' }))
    expect(deferred.requests.every((request) => request.signal?.aborted)).toBe(true)
    expect(progress).toHaveAttribute('aria-busy', 'false')
    expect(within(progress).getAllByText('cancelled')).toHaveLength(8)
    expect(
      within(progress).getByLabelText('Batch status summary'),
    ).toHaveTextContent(/8 cancelled · 2 blocked/i)
    expect(batchAnnouncement).toHaveTextContent(/8 cancelled · 2 blocked/i)

    await user.click(screen.getByRole('button', { name: 'Retry 8 eligible attempts' }))
    await waitFor(() => expect(deferred.runInference).toHaveBeenCalledTimes(16))
    const firstReadyRow = within(progress).getAllByTestId('job-progress-row')[2]
    expect(within(firstReadyRow).queryByText(/attempt/i)).not.toBeInTheDocument()
    expect(within(firstReadyRow).queryByText(/parent attempt-/i)).not.toBeInTheDocument()
    expect(
      within(technicalDetails!).getAllByText(/parent attempt-/i).every(
        (entry) => !entry.matches(':scope > summary') && !entry.closest('details')?.open,
      ),
    ).toBe(true)
    await user.click(within(progress).getByText('Technical details'))
    expect(technicalDetails).toHaveAttribute('open')
    expect(within(technicalDetails!).getByText(`Run ID: ${firstRunId}`)).toBeVisible()
    expect(
      within(technicalDetails!).getAllByText(/parent attempt-/i).every(
        (entry) => entry.closest('details')?.open,
      ),
    ).toBe(true)
  })

  it('uses the provider atomic action so a mid-batch ID failure leaves no progress or request', async () => {
    const user = userEvent.setup()
    const deferred = createDeferredGateway()
    const runtimeIds = ['run-a', 'attempt-a', 'token-a', 'run-b']
    const runtime: WorkspaceRuntime = {
      nextId: vi.fn(() => {
        const next = runtimeIds.shift()
        if (!next) throw new Error('Injected runtime failure')
        return next
      }),
    }
    renderPage(deferred, runtime)

    await user.click(screen.getByRole('checkbox', { name: new RegExp(catalog[2].id) }))
    await user.click(screen.getByRole('checkbox', { name: new RegExp(catalog[3].id) }))
    await user.click(screen.getByRole('button', { name: 'Review selected cases' }))
    await user.click(screen.getByRole('button', { name: 'Start 2 simulations' }))

    expect(screen.queryByRole('region', { name: 'Batch progress' })).not.toBeInTheDocument()
    expect(deferred.runInference).not.toHaveBeenCalled()
  })

  it('retries a failed item with a new attempt while preserving its run', async () => {
    const user = userEvent.setup()
    const deferred = renderPage()
    const approvedCase = listWorkbenchAssets()[2]

    await user.click(screen.getByRole('checkbox', { name: new RegExp(approvedCase.id) }))
    await user.click(screen.getByRole('button', { name: 'Review selected cases' }))
    await user.click(screen.getByRole('button', { name: 'Start 1 simulation' }))
    const firstBinding = deferred.requests[0].binding

    await act(async () => {
      deferred.requests[0].reject(new Error('Synthetic scheduler failure'))
      await Promise.resolve()
    })
    await waitFor(() => expect(screen.getByText('failed')).toBeVisible())

    await user.click(screen.getByRole('button', { name: 'Retry 1 eligible attempt' }))
    expect(deferred.requests).toHaveLength(2)
    expect(deferred.requests[1].binding.clientRunId).toBe(firstBinding.clientRunId)
    expect(deferred.requests[1].binding.attemptToken).not.toBe(firstBinding.attemptToken)
    expect(deferred.requests[1].binding.inputFingerprint).toBe(
      firstBinding.inputFingerprint,
    )

    await act(async () => {
      deferred.requests[1].resolve(runMockEngine(deferred.requests[1].binding))
      await Promise.resolve()
    })
    await waitFor(() => expect(screen.getByText('succeeded')).toBeVisible())
    expect(
      screen.getByRole('heading', {
        name: '1 simulation completed',
        level: 2,
      }),
    ).toBeVisible()
    expect(screen.getByText('The result is ready.')).toBeVisible()
    expect(
      screen.getByRole('link', { name: 'View result' }),
    ).toHaveAttribute('href', `/runs/${firstBinding.clientRunId}`)
    expect(
      screen.queryByRole('button', { name: 'Cancel batch' }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole('button', { name: /Retry .* eligible/ }),
    ).not.toBeInTheDocument()
  })
})
