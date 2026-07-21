import type {
  RunInferenceOptions,
  WorkbenchGateway,
} from './WorkbenchGateway'
import { getWorkbenchAsset } from './catalog'
import { createCanonicalInferenceBindingSnapshot } from './mockEngine'
import {
  WorkbenchError,
  type ConnectedInferenceOutput,
  type InferenceBinding,
  type WorkbenchFailureReason,
} from './types'

export type WorkbenchFetchResponse = {
  readonly ok: boolean
  readonly status: number
  readonly json: () => Promise<unknown>
}

export type WorkbenchFetch = (
  input: RequestInfo | URL,
  init?: RequestInit,
) => Promise<WorkbenchFetchResponse>

export type HttpWorkbenchGatewayDependencies = {
  readonly fetchImpl?: WorkbenchFetch
  readonly timeoutMs?: number
}

export const DEFAULT_HTTP_WORKBENCH_TIMEOUT_MS = 30_000

export function resolveHttpWorkbenchTimeoutMs(
  timeoutMs: unknown = DEFAULT_HTTP_WORKBENCH_TIMEOUT_MS,
): number {
  if (
    typeof timeoutMs !== 'number' ||
    !Number.isFinite(timeoutMs) ||
    timeoutMs <= 0
  ) {
    throw new WorkbenchError({
      reason: 'INVALID_TIMEOUT',
      message: 'Connected workbench timeout must be a positive finite number.',
      field: 'timeoutMs',
    })
  }
  return timeoutMs
}

const CONNECTED_WATERMARK =
  'MODEL PREDICTION — RESEARCH UNVALIDATED — NOT HUMAN GAZE — CLINICAL USE BLOCKED' as const

function fail(reason: WorkbenchFailureReason, message: string): never {
  throw new WorkbenchError({ reason, message })
}

function invalidApiUrl(): never {
  throw new WorkbenchError({
    reason: 'INVALID_API_URL',
    message:
      'Connected workbench API URL must be an absolute HTTP(S) URL without credentials, query parameters, or a fragment.',
    field: 'apiUrl',
  })
}

export function normalizeHttpWorkbenchApiUrl(apiUrl: string): string {
  let parsed: URL
  try {
    parsed = new URL(apiUrl.trim())
  } catch {
    return invalidApiUrl()
  }

  if (
    (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') ||
    parsed.username.length > 0 ||
    parsed.password.length > 0 ||
    parsed.search.length > 0 ||
    parsed.hash.length > 0
  ) {
    return invalidApiUrl()
  }

  const normalizedPath = parsed.pathname.replace(/\/+$/, '')
  return `${parsed.origin}${normalizedPath}`
}

function inferenceEndpoint(apiUrl: string): string {
  const endpoint = new URL(normalizeHttpWorkbenchApiUrl(apiUrl))
  endpoint.pathname = `${endpoint.pathname.replace(/\/+$/, '')}/api/v1/workbench/inference`
  return endpoint.href
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function hasExactKeys(
  value: Record<string, unknown>,
  expectedKeys: readonly string[],
): boolean {
  const actualKeys = Object.keys(value)
  return (
    actualKeys.length === expectedKeys.length &&
    expectedKeys.every((key) => Object.hasOwn(value, key))
  )
}

function isBoundedFinite(value: unknown): value is number {
  return (
    typeof value === 'number' &&
    Number.isFinite(value) &&
    value >= 0 &&
    value <= 1
  )
}

function bindingsMatch(actual: unknown, expected: InferenceBinding): boolean {
  if (!isRecord(actual)) return false

  const roiGeometry = actual.roiGeometry
  const config = actual.config
  return (
    hasExactKeys(actual, [
      'clientRunId',
      'attemptToken',
      'caseId',
      'assetId',
      'assetSha256',
      'roiId',
      'roiVersion',
      'roiGeometry',
      'roiStatus',
      'modelVersion',
      'modelMode',
      'config',
      'configurationHash',
      'inputFingerprint',
    ]) &&
    actual.clientRunId === expected.clientRunId &&
    actual.attemptToken === expected.attemptToken &&
    actual.caseId === expected.caseId &&
    actual.assetId === expected.assetId &&
    actual.assetSha256 === expected.assetSha256 &&
    actual.roiId === expected.roiId &&
    actual.roiVersion === expected.roiVersion &&
    isRecord(roiGeometry) &&
    hasExactKeys(roiGeometry, ['x', 'y', 'width', 'height']) &&
    roiGeometry.x === expected.roiGeometry.x &&
    roiGeometry.y === expected.roiGeometry.y &&
    roiGeometry.width === expected.roiGeometry.width &&
    roiGeometry.height === expected.roiGeometry.height &&
    actual.roiStatus === expected.roiStatus &&
    actual.modelVersion === expected.modelVersion &&
    actual.modelMode === expected.modelMode &&
    isRecord(config) &&
    hasExactKeys(config, ['threshold', 'smoothing']) &&
    config.threshold === expected.config.threshold &&
    config.smoothing === expected.config.smoothing &&
    actual.configurationHash === expected.configurationHash &&
    actual.inputFingerprint === expected.inputFingerprint
  )
}

function hasCanonicalAssetBinding(binding: InferenceBinding): boolean {
  const canonicalCase = getWorkbenchAsset(binding.caseId)
  const canonicalAsset = getWorkbenchAsset(binding.assetId)
  return (
    canonicalCase?.id === binding.caseId &&
    canonicalAsset?.id === binding.assetId &&
    binding.caseId === binding.assetId &&
    binding.assetSha256 === canonicalAsset.sha256
  )
}

function isValidHeatmap(value: unknown): boolean {
  return (
    Array.isArray(value) &&
    value.length > 0 &&
    value.every(
      (point) =>
        isRecord(point) &&
        hasExactKeys(point, ['x', 'y', 'intensity', 'radius']) &&
        isBoundedFinite(point.x) &&
        isBoundedFinite(point.y) &&
        isBoundedFinite(point.intensity) &&
        isBoundedFinite(point.radius),
    )
  )
}

function isValidMetrics(value: unknown): boolean {
  return (
    isRecord(value) &&
    hasExactKeys(value, [
      'roiCoverage',
      'peakIntensity',
      'meanIntensity',
      'focusScore',
    ]) &&
    isBoundedFinite(value.roiCoverage) &&
    isBoundedFinite(value.peakIntensity) &&
    isBoundedFinite(value.meanIntensity) &&
    isBoundedFinite(value.focusScore)
  )
}

function isValidQualityGates(value: unknown): boolean {
  return (
    isRecord(value) &&
    hasExactKeys(value, [
      'bindingIntegrity',
      'roiApproval',
      'finiteValues',
      'normalizedBounds',
      'researchDisplayEligible',
      'clinicalUseEligible',
    ]) &&
    value.bindingIntegrity === 'passed' &&
    value.roiApproval === 'passed' &&
    value.finiteValues === 'passed' &&
    value.normalizedBounds === 'passed' &&
    value.researchDisplayEligible === true &&
    value.clinicalUseEligible === false
  )
}

function isValidConnectedProvenance(value: unknown): boolean {
  return (
    isRecord(value) &&
    hasExactKeys(value, [
      'engine',
      'engineVersion',
      'canonicalSyntheticAsset',
      'deterministic',
      'networkAccessed',
      'storageAccessed',
      'humanGazeData',
    ]) &&
    value.engine === 'connected_model_gateway' &&
    typeof value.engineVersion === 'string' &&
    value.engineVersion.trim().length > 0 &&
    value.canonicalSyntheticAsset === true &&
    typeof value.deterministic === 'boolean' &&
    value.networkAccessed === true &&
    typeof value.storageAccessed === 'boolean' &&
    value.humanGazeData === false
  )
}

function deepCloneAndFreeze<T>(value: T): T {
  if (Array.isArray(value)) {
    return Object.freeze(value.map((entry) => deepCloneAndFreeze(entry))) as T
  }
  if (isRecord(value)) {
    const clone = Object.fromEntries(
      Object.entries(value).map(([key, entry]) => [key, deepCloneAndFreeze(entry)]),
    )
    return Object.freeze(clone) as T
  }
  return value
}

function interruptedRequest(timedOut: boolean): WorkbenchError {
  return new WorkbenchError({
    reason: timedOut ? 'REQUEST_TIMEOUT' : 'REQUEST_ABORTED',
    message: timedOut
      ? 'The connected inference request timed out.'
      : 'The connected inference request was aborted.',
  })
}

function isAbortError(error: unknown): boolean {
  return isRecord(error) && error.name === 'AbortError'
}

type RequestSignalContext = {
  readonly signal: AbortSignal | undefined
  readonly didTimeout: () => boolean
  readonly cleanup: () => void
}

function createRequestSignal(
  callerSignal: AbortSignal | undefined,
  timeoutMs: number | undefined,
): RequestSignalContext {
  if (timeoutMs === undefined) {
    return {
      signal: callerSignal,
      didTimeout: () => false,
      cleanup: () => undefined,
    }
  }

  const controller = new AbortController()
  let timedOut = false
  const onCallerAbort = () => controller.abort()
  callerSignal?.addEventListener('abort', onCallerAbort, { once: true })
  const timeout = setTimeout(() => {
    timedOut = true
    controller.abort()
  }, Math.max(0, timeoutMs))

  return {
    signal: controller.signal,
    didTimeout: () => timedOut,
    cleanup: () => {
      clearTimeout(timeout)
      callerSignal?.removeEventListener('abort', onCallerAbort)
    },
  }
}

function waitForOperation<T>(
  operation: Promise<T>,
  signal: AbortSignal | undefined,
  didTimeout: () => boolean,
): Promise<T> {
  if (!signal) return operation
  if (signal.aborted) return Promise.reject(interruptedRequest(didTimeout()))

  return new Promise<T>((resolve, reject) => {
    const onAbort = () => {
      signal.removeEventListener('abort', onAbort)
      reject(interruptedRequest(didTimeout()))
    }
    const settle = (callback: () => void) => {
      signal.removeEventListener('abort', onAbort)
      callback()
    }

    signal.addEventListener('abort', onAbort, { once: true })
    operation.then(
      (value) => settle(() => resolve(value)),
      (error: unknown) => settle(() => reject(error)),
    )
  })
}

function validateResponse(
  body: unknown,
  expectedBinding: InferenceBinding,
): ConnectedInferenceOutput {
  if (!isRecord(body)) {
    fail('MALFORMED_RESPONSE', 'The connected gateway returned a malformed response.')
  }
  if (body.origin !== 'model_prediction') {
    fail('ORIGIN_MISMATCH', 'The connected gateway returned the wrong result origin.')
  }
  if (body.capabilityStatus !== 'research_unvalidated') {
    fail(
      'CAPABILITY_MISMATCH',
      'The connected gateway returned the wrong capability status.',
    )
  }
  if (body.watermark !== CONNECTED_WATERMARK) {
    fail('CAPABILITY_MISMATCH', 'The connected gateway returned the wrong safety watermark.')
  }
  if (
    !bindingsMatch(body.binding, expectedBinding) ||
    !hasCanonicalAssetBinding(expectedBinding)
  ) {
    fail(
      'IMMUTABLE_BINDING_MISMATCH',
      'The connected result is not bound to the exact inference request.',
    )
  }
  if (
    !hasExactKeys(body, [
      'origin',
      'capabilityStatus',
      'watermark',
      'binding',
      'resultDigest',
      'heatmap',
      'metrics',
      'qualityGates',
      'provenance',
    ]) ||
    typeof body.resultDigest !== 'string' ||
    body.resultDigest.trim().length === 0 ||
    !isValidHeatmap(body.heatmap) ||
    !isValidMetrics(body.metrics) ||
    !isValidQualityGates(body.qualityGates) ||
    !isValidConnectedProvenance(body.provenance)
  ) {
    fail(
      'MALFORMED_RESPONSE',
      'The connected gateway returned an invalid result envelope.',
    )
  }

  return deepCloneAndFreeze(body) as ConnectedInferenceOutput
}

export class HttpWorkbenchGateway implements WorkbenchGateway {
  readonly mode = 'connected' as const

  private readonly fetchImpl: WorkbenchFetch
  private readonly timeoutMs: number | undefined
  private readonly inferenceUrl: string

  constructor(
    apiUrl: string,
    dependencies: HttpWorkbenchGatewayDependencies = {},
  ) {
    this.inferenceUrl = inferenceEndpoint(apiUrl)
    this.fetchImpl = dependencies.fetchImpl ?? (fetch as WorkbenchFetch)
    this.timeoutMs =
      dependencies.timeoutMs === undefined
        ? undefined
        : resolveHttpWorkbenchTimeoutMs(dependencies.timeoutMs)
  }

  async runInference(
    binding: InferenceBinding,
    options: RunInferenceOptions = {},
  ): Promise<ConnectedInferenceOutput> {
    if (options.signal?.aborted) throw interruptedRequest(false)

    const payload = createCanonicalInferenceBindingSnapshot(binding)
    const requestSignal = createRequestSignal(options.signal, this.timeoutMs)

    try {
      let response: WorkbenchFetchResponse
      try {
        response = await waitForOperation(
          this.fetchImpl(
            this.inferenceUrl,
            {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload),
              signal: requestSignal.signal,
            },
          ),
          requestSignal.signal,
          requestSignal.didTimeout,
        )
      } catch (error) {
        if (error instanceof WorkbenchError) throw error
        if (isAbortError(error)) {
          throw interruptedRequest(requestSignal.didTimeout())
        }
        fail('NETWORK_ERROR', 'The connected inference request failed on the network.')
      }

      if (!response.ok) {
        fail(
          'HTTP_ERROR',
          `The connected inference request failed with HTTP status ${response.status}.`,
        )
      }

      let body: unknown
      try {
        body = await waitForOperation(
          response.json(),
          requestSignal.signal,
          requestSignal.didTimeout,
        )
      } catch (error) {
        if (error instanceof WorkbenchError) throw error
        if (isAbortError(error)) {
          throw interruptedRequest(requestSignal.didTimeout())
        }
        fail(
          'MALFORMED_RESPONSE',
          'The connected gateway returned invalid JSON.',
        )
      }

      return validateResponse(body, payload)
    } finally {
      requestSignal.cleanup()
    }
  }
}
