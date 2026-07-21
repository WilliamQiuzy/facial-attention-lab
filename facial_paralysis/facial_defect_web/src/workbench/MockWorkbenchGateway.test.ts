import { describe, expect, it, vi } from 'vitest'
import { getWorkbenchAsset } from './catalog'
import { MockWorkbenchGateway } from './MockWorkbenchGateway'
import { createInferenceBinding } from './mockEngine'
import { WorkbenchError, type InferenceBinding } from './types'

const asset = getWorkbenchAsset('SYN-MOHS-SCC-CHEEK')!

function createBinding(
  operationalIds: {
    readonly clientRunId: string
    readonly attemptToken: string
  } = { clientRunId: 'run-001', attemptToken: 'attempt-001' },
): InferenceBinding {
  return createInferenceBinding({
    ...operationalIds,
    caseId: asset.id,
    assetId: asset.id,
    assetSha256: asset.sha256,
    roi: {
      id: 'roi-primary',
      caseId: asset.id,
      assetId: asset.id,
      version: 3,
      geometry: { x: 0.18, y: 0.24, width: 0.41, height: 0.36 },
      status: 'approved',
      authorId: 'demo_author',
      reviewerId: 'demo_reviewer',
    },
    modelVersion: 'mock-salience-v0.3',
    modelMode: 'mock_only',
    config: { threshold: 0.42, smoothing: 0.27 },
  })
}

function deferred(): {
  readonly promise: Promise<void>
  readonly resolve: () => void
} {
  let resolve!: () => void
  const promise = new Promise<void>((settle) => {
    resolve = settle
  })
  return { promise, resolve }
}

async function expectRequestAborted(request: Promise<unknown>): Promise<void> {
  try {
    await request
  } catch (error) {
    expect(error).toBeInstanceOf(WorkbenchError)
    expect((error as WorkbenchError).reason).toBe('REQUEST_ABORTED')
    return
  }
  throw new Error('Expected the workbench request to be aborted.')
}

describe('MockWorkbenchGateway', () => {
  it('runs deterministic mock inference across operational retries', async () => {
    const gateway = new MockWorkbenchGateway()
    const first = await gateway.runInference(createBinding())
    const retry = await gateway.runInference(
      createBinding({ clientRunId: 'run-002', attemptToken: 'attempt-retry-001' }),
    )

    expect(gateway.mode).toBe('mock')
    expect(retry.resultDigest).toBe(first.resultDigest)
    expect(retry.heatmap).toEqual(first.heatmap)
    expect(retry.metrics).toEqual(first.metrics)
  })

  it('rejects an already-aborted request before scheduling work', async () => {
    const scheduler = vi.fn().mockResolvedValue(undefined)
    const gateway = new MockWorkbenchGateway(scheduler)
    const controller = new AbortController()
    controller.abort()

    const request = gateway.runInference(createBinding(), {
      signal: controller.signal,
    })

    await expectRequestAborted(request)
    expect(scheduler).not.toHaveBeenCalled()
  })

  it('rejects an aborted in-flight schedule without resolving a stale result', async () => {
    const scheduled = deferred()
    const scheduler = vi.fn(() => scheduled.promise)
    const gateway = new MockWorkbenchGateway(scheduler)
    const controller = new AbortController()
    const request = gateway.runInference(createBinding(), {
      signal: controller.signal,
    })

    expect(scheduler).toHaveBeenCalledOnce()
    controller.abort()
    scheduled.resolve()

    await expectRequestAborted(request)
  })

  it('rejects when the scheduler synchronously aborts before returning pending work', async () => {
    const controller = new AbortController()
    const neverResolves = new Promise<void>(() => {})
    const scheduler = vi.fn(() => {
      controller.abort()
      return neverResolves
    })
    const gateway = new MockWorkbenchGateway(scheduler)
    const request = gateway.runInference(createBinding(), {
      signal: controller.signal,
    })

    const outcome = await Promise.race([
      request.then(
        () => ({ kind: 'resolved' as const }),
        (error: unknown) => ({ kind: 'rejected' as const, error }),
      ),
      Promise.resolve()
        .then(() => undefined)
        .then(() => ({ kind: 'still_pending' as const })),
    ])

    expect(scheduler).toHaveBeenCalledOnce()
    if (outcome.kind !== 'rejected') {
      throw new Error(`Expected phase-entry abort rejection, received ${outcome.kind}.`)
    }
    expect(outcome.error).toBeInstanceOf(WorkbenchError)
    expect((outcome.error as WorkbenchError).reason).toBe('REQUEST_ABORTED')
  })
})
