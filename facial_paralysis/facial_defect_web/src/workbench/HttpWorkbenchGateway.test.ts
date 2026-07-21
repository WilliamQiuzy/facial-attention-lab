import { afterEach, describe, expect, it, vi } from 'vitest'
import { getWorkbenchAsset } from './catalog'
import {
  HttpWorkbenchGateway,
  type WorkbenchFetch,
} from './HttpWorkbenchGateway'
import { createInferenceBinding } from './mockEngine'
import {
  WorkbenchError,
  type ConnectedInferenceOutput,
  type InferenceBinding,
  type WorkbenchFailureReason,
} from './types'

const asset = getWorkbenchAsset('SYN-MOHS-SCC-CHEEK')!

afterEach(() => {
  vi.useRealTimers()
})

function createBinding(): InferenceBinding {
  return createInferenceBinding({
    clientRunId: 'run-http-001',
    attemptToken: 'attempt-http-001',
    caseId: asset.id,
    assetId: asset.id,
    assetSha256: asset.sha256,
    roi: {
      id: 'roi-http-primary',
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

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

function withChangedPath<T>(
  value: T,
  path: readonly string[],
  replacement: unknown,
): T {
  const changed = clone(value) as unknown as Record<string, unknown>
  let target = changed
  for (const key of path.slice(0, -1)) {
    target = target[key] as Record<string, unknown>
  }
  target[path.at(-1)!] = replacement
  return changed as T
}

function createValidResponse(binding: InferenceBinding): ConnectedInferenceOutput {
  return {
    origin: 'model_prediction',
    capabilityStatus: 'research_unvalidated',
    watermark:
      'MODEL PREDICTION — RESEARCH UNVALIDATED — NOT HUMAN GAZE — CLINICAL USE BLOCKED',
    binding: clone(binding),
    resultDigest: 'connected-result-001',
    heatmap: [
      { x: 0.31, y: 0.37, intensity: 0.72, radius: 0.08 },
      { x: 0.48, y: 0.44, intensity: 0.59, radius: 0.12 },
    ],
    metrics: {
      roiCoverage: 0.42,
      peakIntensity: 0.72,
      meanIntensity: 0.655,
      focusScore: 0.81,
    },
    qualityGates: {
      bindingIntegrity: 'passed',
      roiApproval: 'passed',
      finiteValues: 'passed',
      normalizedBounds: 'passed',
      researchDisplayEligible: true,
      clinicalUseEligible: false,
    },
    provenance: {
      engine: 'connected_model_gateway',
      engineVersion: 'research-model-2026.07',
      canonicalSyntheticAsset: true,
      deterministic: false,
      networkAccessed: true,
      storageAccessed: false,
      humanGazeData: false,
    },
  }
}

function jsonResponse(body: unknown): Awaited<ReturnType<WorkbenchFetch>> {
  return {
    ok: true,
    status: 200,
    json: async () => body,
  }
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
  throw new Error(`Expected the request to fail with ${reason}.`)
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
  throw new Error(`Expected construction to fail with ${reason}.`)
}

describe('HttpWorkbenchGateway', () => {
  it.each([
    ['relative URL', '/research-api'],
    ['non-HTTP protocol', 'ftp://research-api.invalid/root'],
    ['credentials', 'https://user:secret@research-api.invalid/root'],
    ['query parameters', 'https://research-api.invalid/root?tenant=demo'],
    ['URL fragment', 'https://research-api.invalid/root#workbench'],
  ])('rejects an API URL containing %s before fetch', (_label, apiUrl) => {
    const fetchImpl = vi.fn<WorkbenchFetch>()

    expectConstructionFailure(
      () => new HttpWorkbenchGateway(apiUrl, { fetchImpl }),
      'INVALID_API_URL',
    )
    expect(fetchImpl).not.toHaveBeenCalled()
  })

  it.each([0, -1, Number.NaN, Number.POSITIVE_INFINITY])(
    'rejects an invalid request timeout before fetch (%s)',
    (timeoutMs) => {
      const fetchImpl = vi.fn<WorkbenchFetch>()

      expectConstructionFailure(
        () =>
          new HttpWorkbenchGateway('https://research-api.invalid', {
            fetchImpl,
            timeoutMs,
          }),
        'INVALID_TIMEOUT',
      )
      expect(fetchImpl).not.toHaveBeenCalled()
    },
  )

  it('posts the complete binding with the caller signal and accepts a deeply immutable response', async () => {
    const binding = createBinding()
    const responsePayload = createValidResponse(binding)
    const fetchImpl = vi.fn<WorkbenchFetch>().mockResolvedValue(
      jsonResponse(responsePayload),
    )
    const gateway = new HttpWorkbenchGateway(
      'https://research-api.invalid/root///',
      { fetchImpl },
    )
    const controller = new AbortController()

    const result = await gateway.runInference(binding, {
      signal: controller.signal,
    })

    expect(gateway.mode).toBe('connected')
    expect(fetchImpl).toHaveBeenCalledOnce()
    expect(fetchImpl).toHaveBeenCalledWith(
      'https://research-api.invalid/root/api/v1/workbench/inference',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(binding),
        signal: controller.signal,
      }),
    )
    expect(result).toEqual(responsePayload)
    expect(result).not.toBe(responsePayload)
    expect(result.binding).not.toBe(responsePayload.binding)
    expect(Object.isFrozen(result)).toBe(true)
    expect(Object.isFrozen(result.binding)).toBe(true)
    expect(Object.isFrozen(result.binding.roiGeometry)).toBe(true)
    expect(Object.isFrozen(result.binding.config)).toBe(true)
    expect(Object.isFrozen(result.heatmap)).toBe(true)
    expect(Object.isFrozen(result.heatmap[0])).toBe(true)
    expect(Object.isFrozen(result.metrics)).toBe(true)
    expect(Object.isFrozen(result.qualityGates)).toBe(true)
    expect(Object.isFrozen(result.provenance)).toBe(true)

    ;(responsePayload as { binding: InferenceBinding }).binding = {
      ...responsePayload.binding,
      attemptToken: 'mutated-after-validation',
    }
    ;(responsePayload.heatmap as unknown as Array<(typeof responsePayload.heatmap)[number]>)[0] = {
      x: 1,
      y: 1,
      intensity: 1,
      radius: 1,
    }
    expect(result.binding).toEqual(binding)
    expect(result.heatmap[0]).toEqual({
      x: 0.31,
      y: 0.37,
      intensity: 0.72,
      radius: 0.08,
    })
  })

  it.each([
    ['non-string run ID', ['clientRunId'], 123, 'INVALID_OPERATIONAL_ID'],
    ['blank attempt token', ['attemptToken'], '   ', 'INVALID_OPERATIONAL_ID'],
    ['unknown case', ['caseId'], 'UNKNOWN-CASE', 'UNKNOWN_CASE'],
    ['unknown asset', ['assetId'], 'UNKNOWN-ASSET', 'UNKNOWN_ASSET'],
    ['non-canonical asset hash', ['assetSha256'], '0'.repeat(64), 'ASSET_HASH_MISMATCH'],
    ['non-string ROI ID', ['roiId'], 123, 'INVALID_ROI_VERSION'],
    ['non-positive ROI version', ['roiVersion'], 0, 'INVALID_ROI_VERSION'],
    ['unapproved ROI', ['roiStatus'], 'in_review', 'ROI_NOT_APPROVED'],
    ['invalid ROI geometry', ['roiGeometry', 'width'], 0, 'INVALID_ROI_GEOMETRY'],
    ['unknown model', ['modelVersion'], 'unknown-model', 'UNKNOWN_MODEL'],
    ['unsupported model mode', ['modelMode'], 'remote_only', 'UNSUPPORTED_MODEL_MODE'],
    ['invalid threshold', ['config', 'threshold'], Number.NaN, 'INVALID_CONFIGURATION'],
    ['invalid smoothing', ['config', 'smoothing'], 1.01, 'INVALID_CONFIGURATION'],
    [
      'changed configuration hash',
      ['configurationHash'],
      'cfg_changed',
      'BINDING_INTEGRITY_MISMATCH',
    ],
    [
      'changed input fingerprint',
      ['inputFingerprint'],
      'input_changed',
      'BINDING_INTEGRITY_MISMATCH',
    ],
  ] as const)(
    'rejects an invalid caller binding before fetch: %s',
    async (_label, path, replacement, reason) => {
      const invalidBinding = withChangedPath(createBinding(), path, replacement)
      const fetchImpl = vi.fn<WorkbenchFetch>()
      const gateway = new HttpWorkbenchGateway('https://research-api.invalid', {
        fetchImpl,
      })

      await expectFailure(gateway.runInference(invalidBinding), reason)
      expect(fetchImpl).not.toHaveBeenCalled()
    },
  )

  it.each([
    [['clientRunId'], 'server-run'],
    [['attemptToken'], 'server-attempt'],
    [['caseId'], 'SYN-MOHS-NASAL-RECON'],
    [['assetId'], 'SYN-MOHS-NASAL-RECON'],
    [['assetSha256'], '0'.repeat(64)],
    [['roiId'], 'server-roi'],
    [['roiVersion'], 4],
    [['roiGeometry', 'x'], 0.19],
    [['roiGeometry', 'y'], 0.25],
    [['roiGeometry', 'width'], 0.4],
    [['roiGeometry', 'height'], 0.35],
    [['roiStatus'], 'in_review'],
    [['modelVersion'], 'mock-salience-v0.4'],
    [['modelMode'], 'server_only'],
    [['config', 'threshold'], 0.43],
    [['config', 'smoothing'], 0.28],
    [['configurationHash'], 'cfg_server_replaced'],
    [['inputFingerprint'], 'input_server_replaced'],
  ] as const)(
    'rejects a response that changes immutable binding field %s',
    async (path, replacement) => {
      const binding = createBinding()
      const payload = createValidResponse(binding)
      ;(payload as { binding: InferenceBinding }).binding = withChangedPath(
        payload.binding,
        path,
        replacement,
      )
      const fetchImpl = vi
        .fn<WorkbenchFetch>()
        .mockResolvedValue(jsonResponse(payload))
      const gateway = new HttpWorkbenchGateway('https://research-api.invalid', {
        fetchImpl,
      })

      await expectFailure(
        gateway.runInference(binding),
        'IMMUTABLE_BINDING_MISMATCH',
      )
    },
  )

  it.each([
    ['origin', 'mock_simulation', 'ORIGIN_MISMATCH'],
    ['capabilityStatus', 'simulated_ui_only', 'CAPABILITY_MISMATCH'],
  ] as const)(
    'rejects a response with the wrong %s',
    async (field, value, reason) => {
      const binding = createBinding()
      const payload = {
        ...createValidResponse(binding),
        [field]: value,
      }
      const fetchImpl = vi
        .fn<WorkbenchFetch>()
        .mockResolvedValue(jsonResponse(payload))
      const gateway = new HttpWorkbenchGateway('https://research-api.invalid', {
        fetchImpl,
      })

      await expectFailure(gateway.runInference(binding), reason)
    },
  )

  it('rejects a response with the wrong connected safety watermark', async () => {
    const binding = createBinding()
    const payload = {
      ...createValidResponse(binding),
      watermark: 'MODEL PREDICTION',
    }
    const fetchImpl = vi.fn<WorkbenchFetch>().mockResolvedValue(jsonResponse(payload))
    const gateway = new HttpWorkbenchGateway('https://research-api.invalid', {
      fetchImpl,
    })

    await expectFailure(gateway.runInference(binding), 'CAPABILITY_MISMATCH')
  })

  it.each([
    ['non-object envelope', [], undefined],
    ['empty result digest', ['resultDigest'], '   '],
    ['empty heatmap', ['heatmap'], []],
    ['non-finite heatmap x', ['heatmap', '0', 'x'], Number.NaN],
    ['out-of-range heatmap y', ['heatmap', '0', 'y'], -0.01],
    ['out-of-range heatmap intensity', ['heatmap', '0', 'intensity'], 1.01],
    ['non-numeric heatmap radius', ['heatmap', '0', 'radius'], '0.1'],
    ['non-finite ROI coverage', ['metrics', 'roiCoverage'], Number.NaN],
    ['out-of-range peak intensity', ['metrics', 'peakIntensity'], 1.01],
    ['out-of-range mean intensity', ['metrics', 'meanIntensity'], -0.01],
    ['non-numeric focus score', ['metrics', 'focusScore'], '0.8'],
    ['failed binding-integrity gate', ['qualityGates', 'bindingIntegrity'], 'failed'],
    ['failed ROI-approval gate', ['qualityGates', 'roiApproval'], 'failed'],
    ['failed finite-values gate', ['qualityGates', 'finiteValues'], 'failed'],
    ['failed normalized-bounds gate', ['qualityGates', 'normalizedBounds'], 'failed'],
    [
      'non-boolean research-display gate',
      ['qualityGates', 'researchDisplayEligible'],
      'true',
    ],
    [
      'research-display gate not eligible',
      ['qualityGates', 'researchDisplayEligible'],
      false,
    ],
    ['clinical-use eligibility', ['qualityGates', 'clinicalUseEligible'], true],
    ['wrong provenance engine', ['provenance', 'engine'], 'remote_model'],
    ['empty engine version', ['provenance', 'engineVersion'], '  '],
    [
      'non-boolean canonical-asset provenance',
      ['provenance', 'canonicalSyntheticAsset'],
      'true',
    ],
    [
      'non-canonical asset provenance',
      ['provenance', 'canonicalSyntheticAsset'],
      false,
    ],
    ['non-boolean determinism provenance', ['provenance', 'deterministic'], 'false'],
    ['network-not-accessed provenance', ['provenance', 'networkAccessed'], false],
    ['non-boolean storage provenance', ['provenance', 'storageAccessed'], 'false'],
    ['human-gaze provenance', ['provenance', 'humanGazeData'], true],
  ] as const)(
    'rejects malformed connected output: %s',
    async (_label, pathOrEnvelope, replacement) => {
      const binding = createBinding()
      const payload = Array.isArray(pathOrEnvelope) && replacement !== undefined
        ? withChangedPath(createValidResponse(binding), pathOrEnvelope, replacement)
        : pathOrEnvelope
      const fetchImpl = vi
        .fn<WorkbenchFetch>()
        .mockResolvedValue(jsonResponse(payload))
      const gateway = new HttpWorkbenchGateway('https://research-api.invalid', {
        fetchImpl,
      })

      await expectFailure(gateway.runInference(binding), 'MALFORMED_RESPONSE')
    },
  )

  it('rejects a non-OK response without parsing or falling back', async () => {
    const json = vi.fn(async () => createValidResponse(createBinding()))
    const fetchImpl = vi.fn<WorkbenchFetch>().mockResolvedValue({
      ok: false,
      status: 503,
      json,
    })
    const gateway = new HttpWorkbenchGateway('https://research-api.invalid', {
      fetchImpl,
    })

    await expectFailure(gateway.runInference(createBinding()), 'HTTP_ERROR')
    expect(fetchImpl).toHaveBeenCalledOnce()
    expect(json).not.toHaveBeenCalled()
  })

  it('classifies a fetch rejection as a network failure', async () => {
    const fetchImpl = vi
      .fn<WorkbenchFetch>()
      .mockRejectedValue(new TypeError('connection reset'))
    const gateway = new HttpWorkbenchGateway('https://research-api.invalid', {
      fetchImpl,
    })

    await expectFailure(gateway.runInference(createBinding()), 'NETWORK_ERROR')
  })

  it('classifies invalid JSON as a malformed response', async () => {
    const fetchImpl = vi.fn<WorkbenchFetch>().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => {
        throw new SyntaxError('Unexpected token')
      },
    })
    const gateway = new HttpWorkbenchGateway('https://research-api.invalid', {
      fetchImpl,
    })

    await expectFailure(
      gateway.runInference(createBinding()),
      'MALFORMED_RESPONSE',
    )
  })

  it('rejects an already-aborted request before fetch', async () => {
    const binding = createBinding()
    const fetchImpl = vi
      .fn<WorkbenchFetch>()
      .mockResolvedValue(jsonResponse(createValidResponse(binding)))
    const gateway = new HttpWorkbenchGateway('https://research-api.invalid', {
      fetchImpl,
    })
    const controller = new AbortController()
    controller.abort()

    await expectFailure(
      gateway.runInference(binding, { signal: controller.signal }),
      'REQUEST_ABORTED',
    )
    expect(fetchImpl).not.toHaveBeenCalled()
  })

  it('rejects an in-flight request when the caller aborts', async () => {
    const fetchImpl = vi.fn<WorkbenchFetch>()
    fetchImpl.mockImplementation(
      () => new Promise(() => undefined),
    )
    const gateway = new HttpWorkbenchGateway('https://research-api.invalid', {
      fetchImpl,
    })
    const controller = new AbortController()
    const request = gateway.runInference(createBinding(), {
      signal: controller.signal,
    })
    const failure = expectFailure(request, 'REQUEST_ABORTED')

    expect(fetchImpl).toHaveBeenCalledOnce()
    controller.abort()

    await failure
  })

  it('preserves a network failure when caller abort is queued after fetch rejection', async () => {
    let rejectFetch!: (error: unknown) => void
    const fetchImpl = vi.fn<WorkbenchFetch>()
    fetchImpl.mockImplementation(
      () =>
        new Promise((_resolve, reject) => {
          rejectFetch = reject
        }),
    )
    const gateway = new HttpWorkbenchGateway('https://research-api.invalid', {
      fetchImpl,
    })
    const controller = new AbortController()
    const request = gateway.runInference(createBinding(), {
      signal: controller.signal,
    })
    const failure = expectFailure(request, 'NETWORK_ERROR')

    rejectFetch(new TypeError('Network failed first.'))
    queueMicrotask(() => controller.abort())

    await failure
  })

  it('preserves malformed JSON when caller abort is queued after parse rejection', async () => {
    let rejectJson!: (error: unknown) => void
    const json = vi.fn(
      () =>
        new Promise<unknown>((_resolve, reject) => {
          rejectJson = reject
        }),
    )
    const fetchImpl = vi.fn<WorkbenchFetch>().mockResolvedValue({
      ok: true,
      status: 200,
      json,
    })
    const gateway = new HttpWorkbenchGateway('https://research-api.invalid', {
      fetchImpl,
    })
    const controller = new AbortController()
    const request = gateway.runInference(createBinding(), {
      signal: controller.signal,
    })
    const failure = expectFailure(request, 'MALFORMED_RESPONSE')
    await Promise.resolve()
    await Promise.resolve()
    expect(json).toHaveBeenCalledOnce()

    rejectJson(new SyntaxError('JSON failed first.'))
    queueMicrotask(() => controller.abort())

    await failure
  })

  it('rejects immediately when the signal is aborted before JSON-phase waiting', async () => {
    vi.useFakeTimers()
    const controller = new AbortController()
    const json = vi.fn(() => {
      controller.abort()
      return new Promise<unknown>(() => undefined)
    })
    const fetchImpl = vi.fn<WorkbenchFetch>().mockResolvedValue({
      ok: true,
      status: 200,
      json,
    })
    const gateway = new HttpWorkbenchGateway('https://research-api.invalid', {
      fetchImpl,
    })
    const request = gateway.runInference(createBinding(), {
      signal: controller.signal,
    })
    const outcome = Promise.race([
      request.then(
        () => ({ kind: 'resolved' as const }),
        (error: unknown) => ({ kind: 'rejected' as const, error }),
      ),
      new Promise<{ readonly kind: 'still_pending' }>((resolve) => {
        setTimeout(() => resolve({ kind: 'still_pending' }), 1)
      }),
    ])

    await vi.advanceTimersByTimeAsync(1)

    const settled = await outcome
    expect(settled.kind).toBe('rejected')
    if (settled.kind !== 'rejected') {
      throw new Error('The aborted JSON phase remained pending.')
    }
    expect(settled.error).toBeInstanceOf(WorkbenchError)
    expect((settled.error as WorkbenchError).reason).toBe('REQUEST_ABORTED')
    expect(json).toHaveBeenCalledOnce()
  })

  it('aborts and rejects an overdue request with the stable timeout reason', async () => {
    vi.useFakeTimers()
    let requestSignal: AbortSignal | undefined
    const fetchImpl = vi.fn<WorkbenchFetch>()
    fetchImpl.mockImplementation(
      (_input, init) =>
        new Promise(() => {
          requestSignal = init?.signal ?? undefined
        }),
    )
    const gateway = new HttpWorkbenchGateway('https://research-api.invalid', {
      fetchImpl,
      timeoutMs: 50,
    })
    const request = gateway.runInference(createBinding())
    const failure = expectFailure(request, 'REQUEST_TIMEOUT')
    let settled = false
    void failure.then(() => {
      settled = true
    })

    await vi.advanceTimersByTimeAsync(49)
    expect(settled).toBe(false)
    expect(requestSignal?.aborted).toBe(false)

    await vi.advanceTimersByTimeAsync(1)

    await failure
    expect(requestSignal?.aborted).toBe(true)
  })
})
