import { readFileSync } from 'node:fs'
import { StrictMode, useState } from 'react'
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { listWorkbenchAssets } from './catalog'
import { createInferenceBinding, runMockEngine } from './mockEngine'
import {
  createInitialWorkspaceState,
  getDisplayableResult,
  workspaceReducer,
} from './reducer'
import type { WorkbenchGateway } from './WorkbenchGateway'
import {
  WorkspaceProvider,
  useWorkspace,
  type WorkspaceRuntime,
} from './WorkspaceProvider'
import {
  WorkbenchError,
  type InferenceBinding,
  type InferenceOutput,
  type RoiAnnotation,
  type WorkspaceState,
} from './types'

const approvedCase = listWorkbenchAssets()[2]
const draftCase = listWorkbenchAssets()[0]
const runInput = {
  caseId: approvedCase.id,
  modelVersion: 'mock-salience-v0.3',
  config: { threshold: 0.42, smoothing: 0.27 },
} as const

function createExplicitResearchState(): WorkspaceState {
  const state = createInitialWorkspaceState()
  const inReviewCase = listWorkbenchAssets()[1]
  const draftDefault = state.roisByCase[draftCase.id]!
  const inReviewDefault = state.roisByCase[inReviewCase.id]!
  const { reviewerId: _draftReviewer, ...draftBase } = draftDefault
  const { reviewerId: _reviewReviewer, ...inReviewBase } = inReviewDefault
  const draftRoi: RoiAnnotation = { ...draftBase, status: 'draft' }
  const inReviewRoi: RoiAnnotation = { ...inReviewBase, status: 'in_review' }
  return {
    ...state,
    roisByCase: {
      ...state.roisByCase,
      [draftCase.id]: draftRoi,
      [inReviewCase.id]: inReviewRoi,
    },
  }
}

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

function RunProbe() {
  const { state, actions, gatewayMode, persistence } = useWorkspace()
  const activeRun = state.activeRunId ? state.runsById[state.activeRunId] : undefined
  const activeAttempt = activeRun?.activeAttemptId
    ? state.attemptsById[activeRun.activeAttemptId]
    : undefined

  return (
    <>
      <button type="button" onClick={() => actions.startRun(runInput)}>
        Start
      </button>
      <button
        type="button"
        disabled={!activeRun}
        onClick={() => activeRun && actions.cancelRun(activeRun.clientRunId)}
      >
        Cancel
      </button>
      <button type="button" onClick={() => actions.resetSession()}>
        Reset
      </button>
      <button
        type="button"
        onClick={() => actions.supersedeRoi(approvedCase.id)}
      >
        Invalidate source binding
      </button>
      <button
        type="button"
        disabled={!activeRun || !activeAttempt}
        onClick={() =>
          activeRun &&
          activeAttempt &&
          actions.revokeResult(activeRun.clientRunId, activeAttempt.id)
        }
      >
        Revoke result
      </button>
      <output aria-label="gateway mode">{gatewayMode}</output>
      <output aria-label="persistence">{persistence}</output>
      <output aria-label="run count">{state.runOrder.length}</output>
      <output aria-label="run ID">{activeRun?.clientRunId ?? 'none'}</output>
      <output aria-label="attempt ID">{activeAttempt?.id ?? 'none'}</output>
      <output aria-label="attempt token">
        {activeAttempt?.attemptToken ?? 'none'}
      </output>
      <output aria-label="attempt status">{activeAttempt?.status ?? 'none'}</output>
      <output aria-label="result state">
        {activeAttempt?.result?.freshness ?? 'none'}
      </output>
      <output aria-label="failure reason">
        {activeAttempt?.failure?.reason ?? 'none'}
      </output>
      <output aria-label="failure message">
        {activeAttempt?.failure?.message ?? 'none'}
      </output>
    </>
  )
}

function RoiProbe() {
  const { state, actions } = useWorkspace()
  const roi = state.roisByCase[draftCase.id]
  return (
    <>
      <button
        type="button"
        onClick={() =>
          actions.updateRoi(draftCase.id, {
            x: 0.19,
            y: 0.2,
            width: 0.4,
            height: 0.35,
          })
        }
      >
        Update ROI
      </button>
      <button type="button" onClick={() => actions.submitRoi(draftCase.id)}>
        Submit ROI
      </button>
      <button
        type="button"
        onClick={() => {
          actions.updateRoi(draftCase.id, {
            x: 0.21,
            y: 0.2,
            width: 0.4,
            height: 0.35,
          })
          actions.submitRoi(draftCase.id)
        }}
      >
        Update and submit ROI
      </button>
      <button type="button" onClick={() => actions.approveRoi(draftCase.id)}>
        Approve ROI
      </button>
      <button
        type="button"
        onClick={() => actions.requestRoiChanges(draftCase.id)}
      >
        Request changes
      </button>
      <button type="button" onClick={() => actions.reopenRoi(draftCase.id)}>
        Reopen ROI
      </button>
      <button type="button" onClick={() => actions.supersedeRoi(draftCase.id)}>
        Supersede ROI
      </button>
      <button type="button" onClick={() => actions.resetSession()}>
        Reset session
      </button>
      <output aria-label="ROI status">{roi?.status ?? 'missing'}</output>
      <output aria-label="ROI version">{roi?.version ?? 'missing'}</output>
    </>
  )
}

function RetryProbe() {
  const { state, actions } = useWorkspace()
  const [retryFailure, setRetryFailure] = useState('none')
  const run = state.activeRunId ? state.runsById[state.activeRunId] : undefined
  const activeAttempt = run?.activeAttemptId
    ? state.attemptsById[run.activeAttemptId]
    : undefined
  const lineage =
    run?.attemptIds
      .map((attemptId) => {
        const attempt = state.attemptsById[attemptId]
        return `${attempt.id}:${attempt.status}:${attempt.parentAttemptId ?? '-'}`
      })
      .join('|') ?? 'none'

  return (
    <>
      <button type="button" onClick={() => actions.startRun(runInput)}>
        Start retry run
      </button>
      <button
        type="button"
        onClick={() => run && actions.cancelRun(run.clientRunId)}
      >
        Cancel retry run
      </button>
      <button
        type="button"
        onClick={() => actions.supersedeRoi(approvedCase.id)}
      >
        Invalidate retry source binding
      </button>
      <button
        type="button"
        onClick={() => {
          if (!run) return
          try {
            actions.retryRun(run.clientRunId)
            setRetryFailure('none')
          } catch (error) {
            setRetryFailure(
              error instanceof WorkbenchError ? error.reason : 'unexpected',
            )
          }
        }}
      >
        Retry run
      </button>
      <output aria-label="active attempt ID">{activeAttempt?.id ?? 'none'}</output>
      <output aria-label="active retry status">
        {activeAttempt?.status ?? 'none'}
      </output>
      <output aria-label="active retry failure">
        {activeAttempt?.failure?.reason ?? 'none'}
      </output>
      <output aria-label="attempt lineage">{lineage}</output>
      <output aria-label="displayable digest">
        {getDisplayableResult(state)?.resultDigest ?? 'none'}
      </output>
      <output aria-label="retry failure">{retryFailure}</output>
    </>
  )
}

function PreflightProbe() {
  const { actions, state } = useWorkspace()
  const [failure, setFailure] = useState('none')
  const attempt = (input: Parameters<typeof actions.startRun>[0]) => {
    try {
      actions.startRun(input)
      setFailure('none')
    } catch (error) {
      setFailure(error instanceof WorkbenchError ? error.reason : 'unexpected')
    }
  }

  return (
    <>
      <button
        type="button"
        onClick={() => attempt({ ...runInput, caseId: draftCase.id })}
      >
        Start draft
      </button>
      <button
        type="button"
        onClick={() =>
          attempt({ ...runInput, caseId: listWorkbenchAssets()[1].id })
        }
      >
        Start in review
      </button>
      <button
        type="button"
        onClick={() => attempt({ ...runInput, caseId: 'UNKNOWN-CASE' })}
      >
        Start unknown
      </button>
      <button
        type="button"
        onClick={() =>
          attempt({
            ...runInput,
            config: { ...runInput.config, threshold: Number.NaN },
          })
        }
      >
        Start invalid
      </button>
      <button type="button" onClick={() => attempt(runInput)}>
        Start approved
      </button>
      <output aria-label="preflight failure">{failure}</output>
      <output aria-label="preflight run count">{state.runOrder.length}</output>
    </>
  )
}

function SourceBindingRestoreProbe() {
  const { state, actions } = useWorkspace()
  const sourceBinding = state.roisByCase[approvedCase.id]
  const restore = (
    actions as typeof actions & {
      readonly restoreFullImageSourceBinding: (caseId: string) => void
    }
  ).restoreFullImageSourceBinding

  return (
    <>
      <button type="button" onClick={() => restore(approvedCase.id)}>
        Restore source binding
      </button>
      <button type="button" onClick={() => actions.startRun(runInput)}>
        Run after restore
      </button>
      <output aria-label="restored source status">
        {sourceBinding?.status ?? 'missing'}
      </output>
      <output aria-label="restored source geometry">
        {sourceBinding
          ? [
              sourceBinding.geometry.x,
              sourceBinding.geometry.y,
              sourceBinding.geometry.width,
              sourceBinding.geometry.height,
            ].join(',')
          : 'missing'}
      </output>
      <output aria-label="restore run count">{state.runOrder.length}</output>
    </>
  )
}

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
})

describe('WorkspaceProvider boundary', () => {
  it('throws when useWorkspace is rendered outside a provider', () => {
    function OutsideProbe() {
      useWorkspace()
      return null
    }

    expect(() => render(<OutsideProbe />)).toThrow(
      'useWorkspace must be used within a WorkspaceProvider.',
    )
  })

  it('synchronizes the command snapshot only after commit and preserves sequential prediction', async () => {
    const source = readFileSync('src/workbench/WorkspaceProvider.tsx', 'utf8')
    expect(source).not.toMatch(/\n  stateRef\.current = state\n/)
    expect(source).toContain('useLayoutEffect')

    const user = userEvent.setup()
    const gateway = createDeferredGateway()
    render(
      <WorkspaceProvider
        gateway={gateway.gateway}
        initialState={createExplicitResearchState()}
      >
        <RoiProbe />
      </WorkspaceProvider>,
    )
    const initialVersion = Number(screen.getByLabelText('ROI version').textContent)

    await user.click(screen.getByRole('button', { name: 'Update and submit ROI' }))

    expect(screen.getByLabelText('ROI status')).toHaveTextContent('in_review')
    expect(screen.getByLabelText('ROI version')).toHaveTextContent(
      String(initialVersion + 1),
    )
  })

  it('is StrictMode-safe: mount is inert and one click creates one request and ID sequence', async () => {
    const user = userEvent.setup()
    const deferred = createDeferredGateway('connected')
    let sequence = 0
    const runtime: WorkspaceRuntime = {
      nextId: vi.fn((kind) => `${kind}-${++sequence}`),
    }

    render(
      <StrictMode>
        <WorkspaceProvider gateway={deferred.gateway} runtime={runtime}>
          <RunProbe />
        </WorkspaceProvider>
      </StrictMode>,
    )

    expect(deferred.runInference).not.toHaveBeenCalled()
    expect(runtime.nextId).not.toHaveBeenCalled()
    expect(screen.getByLabelText('gateway mode')).toHaveTextContent('connected')
    expect(screen.getByLabelText('persistence')).toHaveTextContent('memory_only')

    await user.click(screen.getByRole('button', { name: 'Start' }))

    await waitFor(() => {
      expect(deferred.runInference).toHaveBeenCalledOnce()
      expect(screen.getByLabelText('run count')).toHaveTextContent('1')
      expect(screen.getByLabelText('attempt status')).toHaveTextContent('running')
    })
    expect(runtime.nextId).toHaveBeenCalledTimes(3)
    expect(runtime.nextId).toHaveBeenNthCalledWith(1, 'run')
    expect(runtime.nextId).toHaveBeenNthCalledWith(2, 'attempt')
    expect(runtime.nextId).toHaveBeenNthCalledWith(3, 'token')
    expect(deferred.requests[0].binding).toMatchObject({
      clientRunId: 'run-1',
      attemptToken: 'token-3',
      caseId: approvedCase.id,
    })
    expect(deferred.requests[0].signal?.aborted).toBe(false)
  })

  it('revalidates the current full-image source binding at delayed launch before gateway work', async () => {
    vi.useFakeTimers()
    const deferred = createDeferredGateway()

    render(
      <WorkspaceProvider gateway={deferred.gateway} queueDelayMs={100}>
        <RunProbe />
      </WorkspaceProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Start' }))
    expect(screen.getByLabelText('attempt status')).toHaveTextContent('queued')

    fireEvent.click(
      screen.getByRole('button', { name: 'Invalidate source binding' }),
    )
    await act(async () => {
      vi.advanceTimersByTime(100)
      await Promise.resolve()
    })

    expect(deferred.runInference).not.toHaveBeenCalled()
    expect(screen.getByLabelText('attempt status')).toHaveTextContent('blocked')
    expect(screen.getByLabelText('failure reason')).toHaveTextContent(
      'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
    )
  })

  it('aborts a deferred request on cancel and ignores its late resolution', async () => {
    const user = userEvent.setup()
    const deferred = createDeferredGateway()

    render(
      <WorkspaceProvider gateway={deferred.gateway}>
        <RunProbe />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Start' }))
    await waitFor(() => {
      expect(deferred.runInference).toHaveBeenCalledOnce()
      expect(screen.getByLabelText('attempt status')).toHaveTextContent('running')
    })

    const request = deferred.requests[0]
    await user.click(screen.getByRole('button', { name: 'Cancel' }))

    expect(request.signal?.aborted).toBe(true)
    expect(screen.getByLabelText('attempt status')).toHaveTextContent('cancelled')

    await act(async () => {
      request.resolve(runMockEngine(request.binding))
      await Promise.resolve()
    })

    expect(screen.getByLabelText('attempt status')).toHaveTextContent('cancelled')
    expect(screen.getByLabelText('result state')).toHaveTextContent('none')
  })

  it('keeps custom ROI state in one provider lifetime and reset/remount restores approved defaults', async () => {
    const user = userEvent.setup()
    const storageSpies = [
      vi.spyOn(Storage.prototype, 'getItem'),
      vi.spyOn(Storage.prototype, 'setItem'),
      vi.spyOn(Storage.prototype, 'removeItem'),
      vi.spyOn(Storage.prototype, 'clear'),
    ]
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    const deferred = createDeferredGateway()
    const reset = vi.fn()
    let id = 0
    const runtime: WorkspaceRuntime = {
      nextId: (kind) => `${kind}-lifecycle-${++id}`,
      reset,
    }
    const initialState = createExplicitResearchState()
    const initialRoi = initialState.roisByCase[draftCase.id]!
    const first = render(
      <WorkspaceProvider
        gateway={deferred.gateway}
        runtime={runtime}
        initialState={initialState}
      >
        <RoiProbe />
      </WorkspaceProvider>,
    )
    const initialVersion = Number(screen.getByLabelText('ROI version').textContent)

    await user.click(screen.getByRole('button', { name: 'Update ROI' }))
    expect(screen.getByLabelText('ROI status')).toHaveTextContent('draft')
    expect(screen.getByLabelText('ROI version')).toHaveTextContent(
      String(initialVersion + 1),
    )
    expect(initialState.roisByCase[draftCase.id]).toBe(initialRoi)
    expect(initialRoi.version).toBe(initialVersion)

    await user.click(screen.getByRole('button', { name: 'Submit ROI' }))
    await user.click(screen.getByRole('button', { name: 'Request changes' }))
    await user.click(screen.getByRole('button', { name: 'Reopen ROI' }))
    await user.click(screen.getByRole('button', { name: 'Submit ROI' }))
    await user.click(screen.getByRole('button', { name: 'Approve ROI' }))
    await user.click(screen.getByRole('button', { name: 'Supersede ROI' }))
    expect(screen.getByLabelText('ROI status')).toHaveTextContent('superseded')

    await user.click(screen.getByRole('button', { name: 'Reset session' }))
    expect(reset).not.toHaveBeenCalled()
    expect(screen.getByLabelText('ROI status')).toHaveTextContent('approved')
    expect(screen.getByLabelText('ROI version')).toHaveTextContent(String(initialVersion))

    first.unmount()
    render(
      <WorkspaceProvider gateway={deferred.gateway}>
        <RoiProbe />
      </WorkspaceProvider>,
    )
    expect(screen.getByLabelText('ROI status')).toHaveTextContent('approved')
    expect(screen.getByLabelText('ROI version')).toHaveTextContent(String(initialVersion))
    for (const storageSpy of storageSpies) expect(storageSpy).not.toHaveBeenCalled()
    expect(fetchSpy).not.toHaveBeenCalled()
    expect(deferred.runInference).not.toHaveBeenCalled()
  })

  it('aborts every active request on reset and unmount without late state resurrection', async () => {
    const user = userEvent.setup()
    const deferred = createDeferredGateway()
    const mounted = render(
      <WorkspaceProvider gateway={deferred.gateway}>
        <RunProbe />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Start' }))
    const pendingA = deferred.requests[0]
    const attemptA = screen.getByLabelText('attempt ID').textContent
    expect(pendingA.signal?.aborted).toBe(false)

    await user.click(screen.getByRole('button', { name: 'Reset' }))
    expect(pendingA.signal?.aborted).toBe(true)
    expect(screen.getByLabelText('run count')).toHaveTextContent('0')
    expect(screen.getByLabelText('attempt status')).toHaveTextContent('none')

    await user.click(screen.getByRole('button', { name: 'Start' }))
    const replacementRequest = deferred.requests[1]
    const attemptB = screen.getByLabelText('attempt ID').textContent
    expect(replacementRequest.binding.clientRunId).not.toBe(
      pendingA.binding.clientRunId,
    )
    expect(attemptB).not.toBe(attemptA)
    expect(replacementRequest.binding.attemptToken).not.toBe(
      pendingA.binding.attemptToken,
    )
    await act(async () => {
      pendingA.resolve(runMockEngine(pendingA.binding))
      await Promise.resolve()
    })
    expect(replacementRequest.signal?.aborted).toBe(false)
    expect(screen.getByLabelText('run count')).toHaveTextContent('1')
    expect(screen.getByLabelText('attempt status')).toHaveTextContent('running')

    await user.click(screen.getByRole('button', { name: 'Start' }))
    const unmountRequests = deferred.requests.slice(1)
    expect(unmountRequests).toHaveLength(2)

    mounted.unmount()
    expect(unmountRequests.every((request) => request.signal?.aborted === true)).toBe(
      true,
    )
    await act(async () => {
      for (const request of unmountRequests) {
        request.resolve(runMockEngine(request.binding))
      }
      await Promise.resolve()
    })
  })

  it('retries only as a new child attempt, preserves its parent, ignores late parent output, and displays child success', async () => {
    const user = userEvent.setup()
    const deferred = createDeferredGateway()
    let sequence = 0
    const runtime: WorkspaceRuntime = {
      nextId: (kind) => `${kind}-${++sequence}`,
    }
    render(
      <WorkspaceProvider gateway={deferred.gateway} runtime={runtime}>
        <RetryProbe />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Start retry run' }))
    const parentRequest = deferred.requests[0]
    await user.click(screen.getByRole('button', { name: 'Cancel retry run' }))
    expect(screen.getByLabelText('attempt lineage')).toHaveTextContent(
      'attempt-2:cancelled:-',
    )

    await user.click(screen.getByRole('button', { name: 'Retry run' }))
    const childRequest = deferred.requests[1]
    expect(childRequest.binding.inputFingerprint).toBe(
      parentRequest.binding.inputFingerprint,
    )
    expect(screen.getByLabelText('active attempt ID')).toHaveTextContent('attempt-4')
    expect(screen.getByLabelText('active retry status')).toHaveTextContent('running')
    expect(screen.getByLabelText('attempt lineage')).toHaveTextContent(
      'attempt-2:cancelled:-|attempt-4:running:attempt-2',
    )

    await act(async () => {
      parentRequest.resolve(runMockEngine(parentRequest.binding))
      await Promise.resolve()
    })
    expect(screen.getByLabelText('active retry status')).toHaveTextContent('running')
    expect(screen.getByLabelText('displayable digest')).toHaveTextContent('none')

    const childOutput = runMockEngine(childRequest.binding)
    await act(async () => {
      childRequest.resolve(childOutput)
      await Promise.resolve()
    })
    expect(screen.getByLabelText('active retry status')).toHaveTextContent('succeeded')
    expect(screen.getByLabelText('attempt lineage')).toHaveTextContent(
      'attempt-2:cancelled:-|attempt-4:succeeded:attempt-2',
    )
    expect(screen.getByLabelText('displayable digest')).toHaveTextContent(
      childOutput.resultDigest,
    )
  })

  it('revalidates a retry source binding after queueing and before the child gateway call', async () => {
    vi.useFakeTimers()
    const deferred = createDeferredGateway()
    render(
      <WorkspaceProvider gateway={deferred.gateway} queueDelayMs={100}>
        <RetryProbe />
      </WorkspaceProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Start retry run' }))
    await act(async () => {
      vi.advanceTimersByTime(100)
      await Promise.resolve()
    })
    expect(deferred.runInference).toHaveBeenCalledOnce()

    fireEvent.click(screen.getByRole('button', { name: 'Cancel retry run' }))
    fireEvent.click(screen.getByRole('button', { name: 'Retry run' }))
    expect(screen.getByLabelText('active retry status')).toHaveTextContent('queued')

    fireEvent.click(
      screen.getByRole('button', { name: 'Invalidate retry source binding' }),
    )
    await act(async () => {
      vi.advanceTimersByTime(100)
      await Promise.resolve()
    })

    expect(deferred.runInference).toHaveBeenCalledOnce()
    expect(screen.getByLabelText('active retry status')).toHaveTextContent(
      'blocked',
    )
    expect(screen.getByLabelText('active retry failure')).toHaveTextContent(
      'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
    )
  })

  it('rejects unknown, unapproved, and invalid inputs before gateway work but starts an approved case', async () => {
    const user = userEvent.setup()
    const deferred = createDeferredGateway()
    render(
      <WorkspaceProvider
        gateway={deferred.gateway}
        initialState={createExplicitResearchState()}
      >
        <PreflightProbe />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Start draft' }))
    expect(screen.getByLabelText('preflight failure')).toHaveTextContent(
      'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
    )
    expect(deferred.runInference).not.toHaveBeenCalled()

    await user.click(screen.getByRole('button', { name: 'Start in review' }))
    expect(screen.getByLabelText('preflight failure')).toHaveTextContent(
      'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
    )
    expect(deferred.runInference).not.toHaveBeenCalled()

    await user.click(screen.getByRole('button', { name: 'Start unknown' }))
    expect(screen.getByLabelText('preflight failure')).toHaveTextContent('UNKNOWN_CASE')
    expect(deferred.runInference).not.toHaveBeenCalled()

    await user.click(screen.getByRole('button', { name: 'Start invalid' }))
    expect(screen.getByLabelText('preflight failure')).toHaveTextContent(
      'INVALID_CONFIGURATION',
    )
    expect(deferred.runInference).not.toHaveBeenCalled()

    await user.click(screen.getByRole('button', { name: 'Start approved' }))
    expect(screen.getByLabelText('preflight failure')).toHaveTextContent('none')
    expect(deferred.runInference).toHaveBeenCalledOnce()
  })

  it('rejects an approved partial source rectangle before allocating IDs or gateway work', async () => {
    const user = userEvent.setup()
    const deferred = createDeferredGateway()
    const initialState = createInitialWorkspaceState()
    const current = initialState.roisByCase[approvedCase.id]!
    const partialState: WorkspaceState = {
      ...initialState,
      roisByCase: {
        ...initialState.roisByCase,
        [approvedCase.id]: {
          ...current,
          geometry: { x: 0.05, y: 0.05, width: 0.9, height: 0.9 },
        },
      },
    }
    const runtime: WorkspaceRuntime = {
      nextId: vi.fn((kind) => `${kind}-must-not-allocate`),
    }

    render(
      <WorkspaceProvider
        gateway={deferred.gateway}
        runtime={runtime}
        initialState={partialState}
      >
        <PreflightProbe />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Start approved' }))

    expect(screen.getByLabelText('preflight failure')).toHaveTextContent(
      'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
    )
    expect(screen.getByLabelText('preflight run count')).toHaveTextContent('0')
    expect(runtime.nextId).not.toHaveBeenCalled()
    expect(deferred.runInference).not.toHaveBeenCalled()
  })

  it('rejects retry against a partial current source binding before allocating child IDs', async () => {
    const user = userEvent.setup()
    const initial = createInitialWorkspaceState()
    const asset = approvedCase
    const roi = initial.roisByCase[asset.id]!
    const binding = createInferenceBinding({
      clientRunId: 'run-preloaded',
      attemptToken: 'token-preloaded',
      caseId: asset.id,
      assetId: asset.id,
      assetSha256: asset.sha256,
      roi,
      modelVersion: 'mock-salience-v0.3',
      modelMode: 'mock_only',
      config: { threshold: 0.42, smoothing: 0.27 },
    })
    const asyncBinding = {
      runId: binding.clientRunId,
      attemptId: 'attempt-preloaded',
      attemptToken: binding.attemptToken,
      inputFingerprint: binding.inputFingerprint,
    }
    const failed = [
      {
        type: 'run/create',
        runId: binding.clientRunId,
        attemptId: asyncBinding.attemptId,
        binding,
      },
      { type: 'run/validate', ...asyncBinding },
      { type: 'run/queue', ...asyncBinding },
      { type: 'run/start', ...asyncBinding },
      {
        type: 'run/fail',
        ...asyncBinding,
        failure: {
          reason: 'REQUEST_TIMEOUT',
          message: 'Preloaded timeout.',
        },
      },
    ].reduce(
      (state, action) =>
        workspaceReducer(state, action as Parameters<typeof workspaceReducer>[1]),
      initial,
    )
    const partialState: WorkspaceState = {
      ...failed,
      roisByCase: {
        ...failed.roisByCase,
        [asset.id]: {
          ...roi,
          geometry: { x: 0.05, y: 0.05, width: 0.9, height: 0.9 },
        },
      },
    }
    const deferred = createDeferredGateway()
    const runtime: WorkspaceRuntime = {
      nextId: vi.fn((kind) => `${kind}-must-not-allocate`),
    }

    render(
      <WorkspaceProvider
        gateway={deferred.gateway}
        runtime={runtime}
        initialState={partialState}
      >
        <RetryProbe />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Retry run' }))

    expect(screen.getByLabelText('retry failure')).toHaveTextContent(
      'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
    )
    expect(screen.getByLabelText('attempt lineage')).toHaveTextContent(
      'attempt-preloaded:failed:-',
    )
    expect(runtime.nextId).not.toHaveBeenCalled()
    expect(deferred.runInference).not.toHaveBeenCalled()
  })

  it('restores the source binding without allocating IDs or running inference', async () => {
    const user = userEvent.setup()
    const initial = createInitialWorkspaceState()
    const current = initial.roisByCase[approvedCase.id]!
    const partialState: WorkspaceState = {
      ...initial,
      roisByCase: {
        ...initial.roisByCase,
        [approvedCase.id]: {
          ...current,
          status: 'superseded',
          geometry: { x: 0.05, y: 0.05, width: 0.9, height: 0.9 },
        },
      },
    }
    const deferred = createDeferredGateway()
    let sequence = 0
    const runtime: WorkspaceRuntime = {
      nextId: vi.fn((kind) => `${kind}-${++sequence}`),
    }

    render(
      <WorkspaceProvider
        gateway={deferred.gateway}
        runtime={runtime}
        initialState={partialState}
      >
        <SourceBindingRestoreProbe />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Restore source binding' }))

    expect(screen.getByLabelText('restored source status')).toHaveTextContent(
      'approved',
    )
    expect(screen.getByLabelText('restored source geometry')).toHaveTextContent(
      '0,0,1,1',
    )
    expect(screen.getByLabelText('restore run count')).toHaveTextContent('0')
    expect(runtime.nextId).not.toHaveBeenCalled()
    expect(deferred.runInference).not.toHaveBeenCalled()

    await user.click(screen.getByRole('button', { name: 'Run after restore' }))
    await waitFor(() => expect(deferred.runInference).toHaveBeenCalledOnce())
    expect(runtime.nextId).toHaveBeenCalledTimes(3)
  })

  it('rejects invalid runtime operational IDs before creating gateway work', async () => {
    const user = userEvent.setup()
    const deferred = createDeferredGateway()
    const runtime: WorkspaceRuntime = {
      nextId: (kind) => (kind === 'token' ? 'token-valid' : 'duplicate-id'),
    }
    render(
      <WorkspaceProvider gateway={deferred.gateway} runtime={runtime}>
        <PreflightProbe />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Start approved' }))

    expect(screen.getByLabelText('preflight failure')).toHaveTextContent(
      'INVALID_OPERATIONAL_ID',
    )
    expect(deferred.runInference).not.toHaveBeenCalled()
  })

  it('maps gateway rejection to failed while an abort rejection after explicit cancel stays cancelled', async () => {
    const user = userEvent.setup()
    const rejected = createDeferredGateway()
    const first = render(
      <WorkspaceProvider gateway={rejected.gateway}>
        <RunProbe />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Start' }))
    await act(async () => {
      rejected.requests[0].reject(
        new WorkbenchError({
          reason: 'HTTP_ERROR',
          message: 'The gateway returned HTTP 503.',
        }),
      )
      await Promise.resolve()
    })
    await waitFor(() => {
      expect(screen.getByLabelText('attempt status')).toHaveTextContent('failed')
      expect(screen.getByLabelText('failure reason')).toHaveTextContent('HTTP_ERROR')
    })

    await user.click(screen.getByRole('button', { name: 'Start' }))
    await act(async () => {
      rejected.requests[1].reject(new Error('Gateway unavailable.'))
      await Promise.resolve()
    })
    await waitFor(() => {
      expect(screen.getByLabelText('attempt status')).toHaveTextContent('failed')
      expect(screen.getByLabelText('failure reason')).toHaveTextContent('NETWORK_ERROR')
    })

    first.unmount()
    const aborted = createDeferredGateway()
    render(
      <WorkspaceProvider gateway={aborted.gateway}>
        <RunProbe />
      </WorkspaceProvider>,
    )
    await user.click(screen.getByRole('button', { name: 'Start' }))
    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    await act(async () => {
      aborted.requests[0].reject(
        new WorkbenchError({
          reason: 'REQUEST_ABORTED',
          message: 'The request was aborted.',
        }),
      )
      await Promise.resolve()
    })

    expect(screen.getByLabelText('attempt status')).toHaveTextContent('cancelled')
    expect(screen.getByLabelText('failure reason')).toHaveTextContent('none')
  })

  it('fails instead of stranding a run when resolved output is rejected for binding or gate integrity', async () => {
    const user = userEvent.setup()
    const deferred = createDeferredGateway()
    render(
      <WorkspaceProvider gateway={deferred.gateway}>
        <RunProbe />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Start' }))
    const mismatchedRequest = deferred.requests[0]
    const validOutput = runMockEngine(mismatchedRequest.binding)
    const mismatchedOutput = {
      ...validOutput,
      binding: {
        ...validOutput.binding,
        attemptToken: 'token-from-another-attempt',
      },
    } as InferenceOutput
    await act(async () => {
      mismatchedRequest.resolve(mismatchedOutput)
      await Promise.resolve()
    })
    await waitFor(() => {
      expect(screen.getByLabelText('attempt status')).toHaveTextContent('failed')
    })
    expect(screen.getByLabelText('failure reason')).toHaveTextContent(
      'IMMUTABLE_BINDING_MISMATCH',
    )
    expect(screen.getByLabelText('failure message')).toHaveTextContent(
      'The resolved output does not match the active inference binding.',
    )
    expect(screen.getByLabelText('result state')).toHaveTextContent('none')
    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(mismatchedRequest.signal?.aborted).toBe(false)

    await user.click(screen.getByRole('button', { name: 'Start' }))
    const malformedRequest = deferred.requests[1]
    const secondValidOutput = runMockEngine(malformedRequest.binding)
    const malformedOutput = {
      ...secondValidOutput,
      qualityGates: {
        ...secondValidOutput.qualityGates,
        bindingIntegrity: 'failed',
      },
    } as unknown as InferenceOutput
    await act(async () => {
      malformedRequest.resolve(malformedOutput)
      await Promise.resolve()
    })
    await waitFor(() => {
      expect(screen.getByLabelText('attempt status')).toHaveTextContent('failed')
    })
    expect(screen.getByLabelText('failure reason')).toHaveTextContent(
      'MALFORMED_RESPONSE',
    )
    expect(screen.getByLabelText('failure message')).toHaveTextContent(
      'The gateway resolved an invalid inference output.',
    )
    expect(screen.getByLabelText('result state')).toHaveTextContent('none')
    await user.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(malformedRequest.signal?.aborted).toBe(false)
  })

  it('revokes a completed result through the provider action', async () => {
    const user = userEvent.setup()
    const deferred = createDeferredGateway()
    render(
      <WorkspaceProvider gateway={deferred.gateway}>
        <RunProbe />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Start' }))
    const request = deferred.requests[0]
    await act(async () => {
      request.resolve(runMockEngine(request.binding))
      await Promise.resolve()
    })
    await waitFor(() => {
      expect(screen.getByLabelText('attempt status')).toHaveTextContent('succeeded')
      expect(screen.getByLabelText('result state')).toHaveTextContent('current')
    })

    await user.click(screen.getByRole('button', { name: 'Revoke result' }))
    expect(screen.getByLabelText('attempt status')).toHaveTextContent('succeeded')
    expect(screen.getByLabelText('result state')).toHaveTextContent('revoked')
  })
})
