import { useState } from 'react'
import { act, fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { createBatchManifest, type BatchManifest } from './batchManifest'
import { listWorkbenchAssets } from './catalog'
import { MockWorkbenchGateway } from './MockWorkbenchGateway'
import {
  WorkspaceProvider,
  useWorkspace,
  type WorkspaceRuntime,
} from './WorkspaceProvider'

const catalog = listWorkbenchAssets()
const approvedCase = catalog[2]
const config = { threshold: 0.45, smoothing: 0.3 } as const

function BatchProbe() {
  const { state, actions, batchState, batchActions } = useWorkspace()
  const createJob = () => {
    const manifest = createBatchManifest({
      workspaceState: state,
      selectedCaseIds: [approvedCase.id],
      modelVersion: 'mock-salience-v0.4',
      config,
    })
    batchActions.selectAllCases([approvedCase.id])
    batchActions.setManifest(manifest)
    batchActions.startBatch()
  }

  return (
    <>
      <button type="button" onClick={createJob}>Create batch job</button>
      <button
        type="button"
        onClick={() => actions.supersedeRoi(approvedCase.id)}
      >
        Invalidate batch source binding
      </button>
      <button type="button" onClick={actions.resetSession}>Reset whole session</button>
      <output aria-label="batch job ID">{batchState.job?.id ?? 'none'}</output>
      <output aria-label="batch selection count">{batchState.selectedCaseIds.length}</output>
      <output aria-label="workspace run count">{state.runOrder.length}</output>
      <output aria-label="batch attempt status">
        {Object.values(state.attemptsById)[0]?.status ?? 'none'}
      </output>
      <output aria-label="batch attempt failure">
        {Object.values(state.attemptsById)[0]?.failure?.reason ?? 'none'}
      </output>
    </>
  )
}

function SubmissionBoundaryProbe() {
  const { state, actions, batchState, batchActions } = useWorkspace()
  const [replayResult, setReplayResult] = useState('not-submitted')
  const [staleResult, setStaleResult] = useState('not-submitted')

  const createValidJobAndReplay = () => {
    const manifest = createBatchManifest({
      workspaceState: state,
      selectedCaseIds: [approvedCase.id],
      modelVersion: 'mock-salience-v0.4',
      config,
    })
    batchActions.selectAllCases([approvedCase.id])
    batchActions.setManifest(manifest)
    batchActions.startBatch()
    setReplayResult(batchActions.startBatch() ?? 'rejected')
  }

  const submitStaleManifest = () => {
    const manifest = createBatchManifest({
      workspaceState: state,
      selectedCaseIds: [approvedCase.id],
      modelVersion: 'mock-salience-v0.4',
      config,
    })
    batchActions.selectAllCases([approvedCase.id])
    batchActions.setManifest(manifest)
    actions.supersedeRoi(approvedCase.id)
    setStaleResult(batchActions.startBatch() ?? 'rejected')
  }

  return (
    <>
      <button type="button" onClick={createValidJobAndReplay}>
        Test replay boundaries
      </button>
      <button type="button" onClick={submitStaleManifest}>
        Submit stale manifest
      </button>
      <output aria-label="boundary batch job ID">{batchState.job?.id ?? 'none'}</output>
      <output aria-label="exact replay result">{replayResult}</output>
      <output aria-label="stale manifest result">{staleResult}</output>
      <output aria-label="boundary run count">{state.runOrder.length}</output>
    </>
  )
}

function AtomicBatchProbe({
  caseIds,
  tamperManifest = false,
}: {
  readonly caseIds: readonly string[]
  readonly tamperManifest?: boolean
}) {
  const { state, batchState, batchActions } = useWorkspace()
  const [submissionResult, setSubmissionResult] = useState('not-submitted')

  const submit = () => {
    const canonical = createBatchManifest({
      workspaceState: state,
      selectedCaseIds: caseIds,
      modelVersion: 'mock-salience-v0.4',
      config,
    })
    const manifest = tamperManifest
      ? ({ ...canonical, hash: 'manifest_0000000000000000' } as BatchManifest)
      : canonical
    batchActions.selectAllCases(caseIds)
    batchActions.setManifest(manifest)
    setSubmissionResult(batchActions.startBatch() ?? 'rejected')
  }

  return (
    <>
      <button type="button" onClick={submit}>Submit atomic batch</button>
      <output aria-label="atomic submission result">{submissionResult}</output>
      <output aria-label="atomic batch job ID">{batchState.job?.id ?? 'none'}</output>
      <output aria-label="atomic run count">{state.runOrder.length}</output>
      <output aria-label="atomic attempt count">
        {Object.keys(state.attemptsById).length}
      </output>
    </>
  )
}

function MalformedManifestProbe({
  field,
}: {
  readonly field: 'config' | 'items'
}) {
  const { state, batchState, batchActions } = useWorkspace()
  const [submissionResult, setSubmissionResult] = useState('not-submitted')

  const submit = () => {
    const canonical = createBatchManifest({
      workspaceState: state,
      selectedCaseIds: [approvedCase.id],
      modelVersion: 'mock-salience-v0.4',
      config,
    })
    const manifest = {
      ...canonical,
      [field]: null,
    } as unknown as BatchManifest
    batchActions.selectAllCases([approvedCase.id])
    batchActions.setManifest(manifest)
    setSubmissionResult(batchActions.startBatch() ?? 'rejected')
  }

  return (
    <>
      <button type="button" onClick={submit}>Submit malformed manifest</button>
      <output aria-label="malformed submission result">{submissionResult}</output>
      <output aria-label="malformed batch job ID">{batchState.job?.id ?? 'none'}</output>
      <output aria-label="malformed run count">{state.runOrder.length}</output>
    </>
  )
}

describe('WorkspaceProvider batch session slice', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('resets batch and run state together while preserving monotonic job IDs', async () => {
    const user = userEvent.setup()
    render(
      <WorkspaceProvider
        gateway={new MockWorkbenchGateway()}
        queueDelayMs={10_000}
      >
        <BatchProbe />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Create batch job' }))
    expect(screen.getByLabelText('batch job ID')).toHaveTextContent('batch-job-1')
    expect(screen.getByLabelText('batch selection count')).toHaveTextContent('1')
    expect(screen.getByLabelText('workspace run count')).toHaveTextContent('1')

    await user.click(screen.getByRole('button', { name: 'Reset whole session' }))
    expect(screen.getByLabelText('batch job ID')).toHaveTextContent('none')
    expect(screen.getByLabelText('batch selection count')).toHaveTextContent('0')
    expect(screen.getByLabelText('workspace run count')).toHaveTextContent('0')

    await user.click(screen.getByRole('button', { name: 'Create batch job' }))
    expect(screen.getByLabelText('batch job ID')).toHaveTextContent('batch-job-2')
    expect(screen.getByLabelText('workspace run count')).toHaveTextContent('1')
  })

  it('replays the exact atomic submission without creating another run', async () => {
    const user = userEvent.setup()
    render(
      <WorkspaceProvider
        gateway={new MockWorkbenchGateway()}
        queueDelayMs={10_000}
      >
        <SubmissionBoundaryProbe />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Test replay boundaries' }))
    expect(screen.getByLabelText('boundary batch job ID')).toHaveTextContent('batch-job-1')
    expect(screen.getByLabelText('exact replay result')).toHaveTextContent('batch-job-1')
    expect(screen.getByLabelText('boundary run count')).toHaveTextContent('1')
  })

  it('rejects a manifest that became stale before job submission', async () => {
    const user = userEvent.setup()
    render(
      <WorkspaceProvider
        gateway={new MockWorkbenchGateway()}
        queueDelayMs={10_000}
      >
        <SubmissionBoundaryProbe />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Submit stale manifest' }))
    expect(screen.getByLabelText('stale manifest result')).toHaveTextContent('rejected')
    expect(screen.getByLabelText('boundary batch job ID')).toHaveTextContent('none')
    expect(screen.getByLabelText('boundary run count')).toHaveTextContent('0')
  })

  it('revalidates every queued batch source binding before gateway launch', async () => {
    vi.useFakeTimers()
    const runInference = vi.fn()
    render(
      <WorkspaceProvider
        gateway={{ mode: 'mock', runInference }}
        queueDelayMs={100}
      >
        <BatchProbe />
      </WorkspaceProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Create batch job' }))
    expect(screen.getByLabelText('batch attempt status')).toHaveTextContent(
      'queued',
    )
    fireEvent.click(
      screen.getByRole('button', {
        name: 'Invalidate batch source binding',
      }),
    )
    await act(async () => {
      vi.advanceTimersByTime(100)
      await Promise.resolve()
    })

    expect(runInference).not.toHaveBeenCalled()
    expect(screen.getByLabelText('batch attempt status')).toHaveTextContent(
      'blocked',
    )
    expect(screen.getByLabelText('batch attempt failure')).toHaveTextContent(
      'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
    )
  })

  it('leaves no job, run, attempt, or request when runtime ID generation fails mid-batch', async () => {
    const user = userEvent.setup()
    const runInference = vi.fn()
    const runtimeIds = ['run-a', 'attempt-a', 'token-a', 'run-b']
    const runtime: WorkspaceRuntime = {
      nextId: vi.fn(() => {
        const next = runtimeIds.shift()
        if (!next) throw new Error('Injected runtime failure')
        return next
      }),
    }
    render(
      <WorkspaceProvider
        gateway={{ mode: 'mock', runInference }}
        runtime={runtime}
      >
        <AtomicBatchProbe caseIds={[catalog[2].id, catalog[3].id]} />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Submit atomic batch' }))

    expect(screen.getByLabelText('atomic submission result')).toHaveTextContent('rejected')
    expect(screen.getByLabelText('atomic batch job ID')).toHaveTextContent('none')
    expect(screen.getByLabelText('atomic run count')).toHaveTextContent('0')
    expect(screen.getByLabelText('atomic attempt count')).toHaveTextContent('0')
    expect(runInference).not.toHaveBeenCalled()
  })

  it('leaves no partial state or request when IDs collide within a batch', async () => {
    const user = userEvent.setup()
    const runInference = vi.fn()
    const runtimeIds = [
      'run-a',
      'attempt-a',
      'token-a',
      'run-a',
      'attempt-b',
      'token-b',
    ]
    const runtime: WorkspaceRuntime = {
      nextId: vi.fn(() => runtimeIds.shift() ?? 'unexpected-id'),
    }
    render(
      <WorkspaceProvider
        gateway={{ mode: 'mock', runInference }}
        runtime={runtime}
      >
        <AtomicBatchProbe caseIds={[catalog[2].id, catalog[3].id]} />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Submit atomic batch' }))

    expect(screen.getByLabelText('atomic submission result')).toHaveTextContent('rejected')
    expect(screen.getByLabelText('atomic batch job ID')).toHaveTextContent('none')
    expect(screen.getByLabelText('atomic run count')).toHaveTextContent('0')
    expect(screen.getByLabelText('atomic attempt count')).toHaveTextContent('0')
    expect(runInference).not.toHaveBeenCalled()
  })

  it('rejects cross-kind runtime ID collisions without partially submitting the batch', async () => {
    const user = userEvent.setup()
    const runInference = vi.fn()
    const runtimeIds = [
      'run-a',
      'attempt-a',
      'token-a',
      'run-b',
      'attempt-b',
      'run-a',
    ]
    const runtime: WorkspaceRuntime = {
      nextId: vi.fn(() => runtimeIds.shift() ?? 'unexpected-id'),
    }
    render(
      <WorkspaceProvider
        gateway={{ mode: 'mock', runInference }}
        runtime={runtime}
      >
        <AtomicBatchProbe caseIds={[catalog[2].id, catalog[3].id]} />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Submit atomic batch' }))

    expect(screen.getByLabelText('atomic submission result')).toHaveTextContent('rejected')
    expect(screen.getByLabelText('atomic batch job ID')).toHaveTextContent('none')
    expect(screen.getByLabelText('atomic run count')).toHaveTextContent('0')
    expect(screen.getByLabelText('atomic attempt count')).toHaveTextContent('0')
    expect(runInference).not.toHaveBeenCalled()
  })

  it('rejects a tampered manifest atomically before creating runs or requests', async () => {
    const user = userEvent.setup()
    const runInference = vi.fn()
    render(
      <WorkspaceProvider gateway={{ mode: 'mock', runInference }}>
        <AtomicBatchProbe caseIds={[catalog[2].id]} tamperManifest />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Submit atomic batch' }))

    expect(screen.getByLabelText('atomic submission result')).toHaveTextContent('rejected')
    expect(screen.getByLabelText('atomic batch job ID')).toHaveTextContent('none')
    expect(screen.getByLabelText('atomic run count')).toHaveTextContent('0')
    expect(screen.getByLabelText('atomic attempt count')).toHaveTextContent('0')
    expect(runInference).not.toHaveBeenCalled()
  })

  it.each(['config', 'items'] as const)(
    'audits a manifest with %s set to null before reading its nested fields',
    async (field) => {
      const user = userEvent.setup()
      const runInference = vi.fn()
      render(
        <WorkspaceProvider gateway={{ mode: 'mock', runInference }}>
          <MalformedManifestProbe field={field} />
        </WorkspaceProvider>,
      )

      await expect(
        user.click(screen.getByRole('button', { name: 'Submit malformed manifest' })),
      ).resolves.toBeUndefined()
      expect(screen.getByLabelText('malformed submission result')).toHaveTextContent(
        'rejected',
      )
      expect(screen.getByLabelText('malformed batch job ID')).toHaveTextContent('none')
      expect(screen.getByLabelText('malformed run count')).toHaveTextContent('0')
      expect(runInference).not.toHaveBeenCalled()
    },
  )
})
