import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { MemoryRouter } from 'react-router-dom'
import { App } from '../App'
import { listWorkbenchAssets } from '../workbench/catalog'
import { runMockEngine } from '../workbench/mockEngine'
import type { WorkbenchGateway } from '../workbench/WorkbenchGateway'

const approvedCase = listWorkbenchAssets()[2]

function clickPrimary(label: string) {
  fireEvent.click(
    within(screen.getByRole('navigation', { name: 'Primary navigation' })).getByRole(
      'link',
      { name: label },
    ),
  )
}

function clickResearchTool(label: string) {
  clickPrimary('Help')
  fireEvent.click(
    within(screen.getByRole('navigation', { name: 'Research tools' })).getByRole(
      'link',
      { name: label },
    ),
  )
}

function configureSingleCaseBatch() {
  fireEvent.click(
    screen.getByRole('checkbox', { name: new RegExp(approvedCase.id) }),
  )
  fireEvent.click(screen.getByRole('button', { name: 'Review selected cases' }))
  fireEvent.click(screen.getByRole('button', { name: 'Start 1 simulation' }))
}

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('production batch route integration', () => {
  it('preserves the queued job across navigation and cancels before the 700ms launch', () => {
    vi.useFakeTimers()
    const runInference = vi.fn(async (binding) => runMockEngine(binding))
    const gateway: WorkbenchGateway = { mode: 'mock', runInference }
    render(
      <MemoryRouter initialEntries={['/jobs']}>
        <App gateway={gateway} />
      </MemoryRouter>,
    )

    configureSingleCaseBatch()

    expect(screen.getByText('queued')).toBeVisible()
    expect(runInference).not.toHaveBeenCalled()
    const progress = screen.getByRole('region', { name: 'Batch progress' })
    const technicalDetails = within(progress).getByText('Technical details').closest('details')
    expect(technicalDetails).not.toHaveAttribute('open')
    fireEvent.click(within(progress).getByText('Technical details'))
    const manifestHash = within(progress).getByLabelText('Manifest hash').textContent
    expect(within(progress).getByRole('heading', { name: 'Batch progress' })).toBeVisible()
    expect(screen.queryByRole('heading', { name: 'batch-job-1' })).not.toBeInTheDocument()

    clickResearchTool('Runs')
    expect(screen.getByRole('heading', { name: 'Recent simulations' })).toBeVisible()
    clickResearchTool('Jobs')

    const restoredProgress = screen.getByRole('region', { name: 'Batch progress' })
    const restoredTechnicalDetails = within(restoredProgress)
      .getByText('Technical details')
      .closest('details')
    fireEvent.click(within(restoredProgress).getByText('Technical details'))
    expect(restoredTechnicalDetails).toHaveAttribute('open')
    expect(within(restoredProgress).getByLabelText('Manifest hash')).toHaveTextContent(manifestHash!)
    expect(within(restoredProgress).getByRole('heading', { name: 'Batch progress' })).toBeVisible()
    expect(screen.getByText('queued')).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'Cancel batch' }))
    expect(screen.getByText('cancelled')).toBeVisible()

    act(() => vi.advanceTimersByTime(800))
    expect(runInference).not.toHaveBeenCalled()
  })

  it('exposes the real queued window and launches only after 700ms', async () => {
    vi.useFakeTimers()
    const runInference = vi.fn(async (binding) => runMockEngine(binding))
    const gateway: WorkbenchGateway = { mode: 'mock', runInference }
    render(
      <MemoryRouter initialEntries={['/jobs']}>
        <App gateway={gateway} />
      </MemoryRouter>,
    )

    configureSingleCaseBatch()
    expect(screen.getByText('queued')).toBeVisible()

    act(() => vi.advanceTimersByTime(699))
    expect(runInference).not.toHaveBeenCalled()
    expect(screen.getByText('queued')).toBeVisible()

    await act(async () => {
      vi.advanceTimersByTime(1)
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(runInference).toHaveBeenCalledOnce()
    expect(screen.getByText('succeeded')).toBeVisible()
  })

  it('preserves failure and retry lineage across route changes', async () => {
    vi.useFakeTimers()
    const runInference = vi.fn(async () => {
      throw new Error('Synthetic scheduled failure')
    })
    const gateway: WorkbenchGateway = { mode: 'mock', runInference }
    render(
      <MemoryRouter initialEntries={['/jobs']}>
        <App gateway={gateway} />
      </MemoryRouter>,
    )

    configureSingleCaseBatch()
    await act(async () => {
      vi.advanceTimersByTime(700)
      for (let index = 0; index < 8; index += 1) await Promise.resolve()
    })
    expect(screen.getByText('failed')).toBeVisible()

    clickResearchTool('Models')
    expect(screen.getByRole('heading', { name: 'Compare simulation versions' })).toBeVisible()
    clickResearchTool('Jobs')
    expect(screen.getByText('failed')).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: 'Retry 1 eligible attempt' }))
    expect(screen.getByText('queued')).toBeVisible()
    const progress = screen.getByRole('region', { name: 'Batch progress' })
    expect(within(progress).queryByText('2 attempts')).not.toBeInTheDocument()
    const technicalDetails = within(progress).getByText('Technical details').closest('details')
    expect(technicalDetails).not.toHaveAttribute('open')
    expect(within(technicalDetails!).getByText(/Attempts: 2/)).not.toBeVisible()
    expect(within(technicalDetails!).getByText(/Parent attempt-/)).not.toBeVisible()

    clickResearchTool('Runs')
    clickResearchTool('Jobs')
    const restoredProgress = screen.getByRole('region', { name: 'Batch progress' })
    const restoredTechnical = within(restoredProgress).getByText('Technical details').closest('details')
    expect(within(restoredTechnical!).getByText(/Attempts: 2/)).not.toBeVisible()
    expect(screen.getByRole('heading', { name: 'Batch progress' })).toBeVisible()
  })

  it('preserves the app-lifetime batch after returning to the clinician entry', () => {
    vi.useFakeTimers()
    const gateway: WorkbenchGateway = {
      mode: 'mock',
      runInference: vi.fn(async (binding) => runMockEngine(binding)),
    }
    render(
      <MemoryRouter initialEntries={['/jobs']}>
        <App gateway={gateway} />
      </MemoryRouter>,
    )

    configureSingleCaseBatch()
    fireEvent.click(
      screen.getByRole('link', {
        name: /FaceAI/i,
      }),
    )

    expect(
      screen.getByRole('heading', { name: 'Patients', level: 1 }),
    ).toBeVisible()
    expect(screen.queryByTestId('metric-session-jobs')).not.toBeInTheDocument()
    clickResearchTool('Jobs')
    expect(screen.getByRole('heading', { name: 'Batch progress' })).toBeVisible()
    expect(screen.getByText('queued')).toBeVisible()
  })
})
