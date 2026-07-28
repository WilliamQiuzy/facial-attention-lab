import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  DEFAULT_HTTP_WORKBENCH_TIMEOUT_MS,
  HttpWorkbenchGateway,
} from './HttpWorkbenchGateway'
import { MockWorkbenchGateway } from './MockWorkbenchGateway'
import type { ConnectedWorkbenchGateway } from './WorkbenchGateway'
import { getWorkbenchAsset } from './catalog'
import { createWorkbenchGateway } from './createWorkbenchGateway'
import { createInferenceBinding } from './mockEngine'
import {
  WorkbenchError,
  type InferenceBinding,
  type WorkbenchFailureReason,
} from './types'

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllEnvs()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

function expectConnectedConfigurationFailure(operation: () => unknown): void {
  try {
    operation()
  } catch (error) {
    expect(error).toBeInstanceOf(WorkbenchError)
    expect((error as WorkbenchError).reason).toBe(
      'CONNECTED_MODE_NOT_CONFIGURED',
    )
    return
  }
  throw new Error('Expected connected mode to fail closed.')
}

function expectConstructionFailure(
  operation: () => unknown,
  reason: string,
): void {
  try {
    operation()
  } catch (error) {
    expect(error).toBeInstanceOf(WorkbenchError)
    expect((error as WorkbenchError).reason).toBe(reason)
    return
  }
  throw new Error(`Expected gateway construction to fail with ${reason}.`)
}

const asset = getWorkbenchAsset('SYN-MOHS-SCC-CHEEK')!

function createBinding(): InferenceBinding {
  return createInferenceBinding({
    clientRunId: 'factory-run-001',
    attemptToken: 'factory-attempt-001',
    caseId: asset.id,
    assetId: asset.id,
    assetSha256: asset.sha256,
    roi: {
      id: 'factory-roi-primary',
      caseId: asset.id,
      assetId: asset.id,
      version: 1,
      geometry: { x: 0, y: 0, width: 1, height: 1 },
      status: 'approved',
      authorId: 'demo_author',
      reviewerId: 'demo_reviewer',
    },
    modelVersion: 'mock-salience-v0.3',
    modelMode: 'mock_only',
    config: { threshold: 0.42, smoothing: 0.27 },
  })
}

async function expectFailure(
  request: Promise<unknown>,
  reason: WorkbenchFailureReason,
): Promise<void> {
  try {
    await request
  } catch (error) {
    expect(error).toBeInstanceOf(WorkbenchError)
    expect((error as WorkbenchError).reason).toBe(reason)
    return
  }
  throw new Error(`Expected the connected request to fail with ${reason}.`)
}

describe('createWorkbenchGateway', () => {
  it('creates the zero-network, zero-storage mock gateway by default and when disabled', () => {
    vi.stubEnv('VITE_ENABLE_CONNECTED_MODE', 'false')
    vi.stubEnv('VITE_ATTENTION_API_URL', 'https://ambient-api.invalid')
    const fetchSpy = vi.fn()
    const storageReadSpy = vi.spyOn(Storage.prototype, 'getItem')
    const storageWriteSpy = vi.spyOn(Storage.prototype, 'setItem')
    vi.stubGlobal('fetch', fetchSpy)

    expect(createWorkbenchGateway().mode).toBe('mock')
    expect(
      createWorkbenchGateway({ enabled: false, apiUrl: 'https://unused.invalid' })
        .mode,
    ).toBe('mock')
    expect(fetchSpy).not.toHaveBeenCalled()
    expect(storageReadSpy).not.toHaveBeenCalled()
    expect(storageWriteSpy).not.toHaveBeenCalled()
  })

  it.each([undefined, '', '   '])(
    'fails closed when connected mode has an incomplete API URL (%s)',
    (apiUrl) => {
      const connectedFactory = vi.fn()

      expectConnectedConfigurationFailure(() =>
        createWorkbenchGateway(
          { enabled: true, apiUrl },
          { connectedFactory },
        ),
      )
      expect(connectedFactory).not.toHaveBeenCalled()
    },
  )

  it('passes a normalized URL and validated timeout to the injectable connected gateway seam', () => {
    const connectedGateway = {
      mode: 'connected',
      runInference: vi.fn(),
    } satisfies ConnectedWorkbenchGateway
    const connectedFactory = vi.fn(() => connectedGateway)

    const gateway = createWorkbenchGateway(
      {
        enabled: true,
        apiUrl: '  https://research-api.invalid/root///  ',
        timeoutMs: 125,
      },
      { connectedFactory },
    )

    expect(gateway).toBe(connectedGateway)
    expect(connectedFactory).toHaveBeenCalledOnce()
    expect(connectedFactory).toHaveBeenCalledWith(
      'https://research-api.invalid/root',
      125,
    )
  })

  it('validates the URL before invoking an injected connected gateway factory', () => {
    const connectedFactory = vi.fn()

    expectConstructionFailure(
      () =>
        createWorkbenchGateway(
          { enabled: true, apiUrl: '/relative-api' },
          { connectedFactory },
        ),
      'INVALID_API_URL',
    )
    expect(connectedFactory).not.toHaveBeenCalled()
  })

  it.each([0, -1, Number.NaN, Number.POSITIVE_INFINITY, '1000']) (
    'rejects an invalid connected timeout before invoking the factory (%s)',
    (timeoutMs) => {
      const connectedFactory = vi.fn()

      expectConstructionFailure(
        () =>
          createWorkbenchGateway(
            {
              enabled: true,
              apiUrl: 'https://research-api.invalid',
              timeoutMs: timeoutMs as number,
            },
            { connectedFactory },
          ),
        'INVALID_TIMEOUT',
      )
      expect(connectedFactory).not.toHaveBeenCalled()
    },
  )

  it('constructs the HTTP gateway by default without network or storage access', () => {
    const fetchSpy = vi.fn()
    const storageReadSpy = vi.spyOn(Storage.prototype, 'getItem')
    const storageWriteSpy = vi.spyOn(Storage.prototype, 'setItem')
    vi.stubGlobal('fetch', fetchSpy)

    const gateway = createWorkbenchGateway({
      enabled: true,
      apiUrl: '  https://research-api.invalid/root/  ',
    })

    expect(gateway).toBeInstanceOf(HttpWorkbenchGateway)
    expect(gateway.mode).toBe('connected')
    expect(fetchSpy).not.toHaveBeenCalled()
    expect(storageReadSpy).not.toHaveBeenCalled()
    expect(storageWriteSpy).not.toHaveBeenCalled()
  })

  it('applies the finite default timeout to a signal-ignoring connected fetch', async () => {
    vi.useFakeTimers()
    const defaultTimeoutMs = DEFAULT_HTTP_WORKBENCH_TIMEOUT_MS
    let requestSignal: AbortSignal | undefined
    const fetchSpy = vi.fn(
      (_input: RequestInfo | URL, init?: RequestInit) =>
        new Promise<Response>(() => {
          requestSignal = init?.signal ?? undefined
        }),
    )
    vi.stubGlobal('fetch', fetchSpy)
    const gateway = createWorkbenchGateway({
      enabled: true,
      apiUrl: 'https://research-api.invalid',
    })
    const outcome = Promise.race([
      gateway.runInference(createBinding()).then(
        () => ({ kind: 'resolved' as const }),
        (error: unknown) => ({ kind: 'rejected' as const, error }),
      ),
      new Promise<{ readonly kind: 'still_pending' }>((resolve) => {
        setTimeout(() => resolve({ kind: 'still_pending' }), defaultTimeoutMs + 1)
      }),
    ])
    let settled = false
    void outcome.then(() => {
      settled = true
    })

    await vi.advanceTimersByTimeAsync(defaultTimeoutMs - 1)
    expect(settled).toBe(false)
    expect(requestSignal).toBeInstanceOf(AbortSignal)
    expect(requestSignal?.aborted).toBe(false)

    await vi.advanceTimersByTimeAsync(2)

    const result = await outcome
    expect(result.kind).toBe('rejected')
    if (result.kind !== 'rejected') {
      throw new Error('The factory-created request exceeded its default deadline.')
    }
    expect(result.error).toBeInstanceOf(WorkbenchError)
    expect((result.error as WorkbenchError).reason).toBe('REQUEST_TIMEOUT')
    expect(requestSignal?.aborted).toBe(true)
  })

  it('honors a positive finite connected timeout override', async () => {
    vi.useFakeTimers()
    const overrideTimeoutMs = 25
    const fetchSpy = vi.fn(
      () => new Promise<Response>(() => undefined),
    )
    vi.stubGlobal('fetch', fetchSpy)
    const gateway = createWorkbenchGateway({
      enabled: true,
      apiUrl: 'https://research-api.invalid',
      timeoutMs: overrideTimeoutMs,
    })
    const outcome = Promise.race([
      gateway.runInference(createBinding()).then(
        () => ({ kind: 'resolved' as const }),
        (error: unknown) => ({ kind: 'rejected' as const, error }),
      ),
      new Promise<{ readonly kind: 'still_pending' }>((resolve) => {
        setTimeout(() => resolve({ kind: 'still_pending' }), overrideTimeoutMs + 1)
      }),
    ])

    await vi.advanceTimersByTimeAsync(overrideTimeoutMs + 1)

    const result = await outcome
    expect(result.kind).toBe('rejected')
    if (result.kind !== 'rejected') {
      throw new Error('The configured connected timeout was not enforced.')
    }
    expect(result.error).toBeInstanceOf(WorkbenchError)
    expect((result.error as WorkbenchError).reason).toBe('REQUEST_TIMEOUT')
  })

  it('never falls back to mock inference after a connected request failure', async () => {
    const fetchSpy = vi.fn().mockRejectedValue(new TypeError('offline'))
    const mockRunSpy = vi.spyOn(MockWorkbenchGateway.prototype, 'runInference')
    vi.stubGlobal('fetch', fetchSpy)
    const gateway = createWorkbenchGateway({
      enabled: true,
      apiUrl: 'https://research-api.invalid',
    })

    await expectFailure(gateway.runInference(createBinding()), 'NETWORK_ERROR')
    expect(fetchSpy).toHaveBeenCalledOnce()
    expect(mockRunSpy).not.toHaveBeenCalled()
  })
})
