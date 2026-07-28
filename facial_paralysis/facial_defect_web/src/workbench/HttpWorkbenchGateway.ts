import type {
  RunInferenceOptions,
  WorkbenchGateway,
} from './WorkbenchGateway'
import { getWorkbenchAsset } from './catalog'
import { createCanonicalInferenceBindingSnapshot } from './mockEngine'
import {
  CONNECTED_ATTENTION_REQUEST_PROFILE_VERSION,
  MAX_SPATIAL_DISPLAY_POINTS,
  WorkbenchError,
  type ConnectedAttentionRequestV1,
  type ConnectedInferenceOutput,
  type ConnectedModelIdentity,
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

export function createConnectedAttentionRequestV1(
  binding: InferenceBinding,
): ConnectedAttentionRequestV1 {
  const canonical = createCanonicalInferenceBindingSnapshot(binding)
  return Object.freeze({
    requestProfileVersion: CONNECTED_ATTENTION_REQUEST_PROFILE_VERSION,
    clientRunId: canonical.clientRunId,
    attemptToken: canonical.attemptToken,
    caseId: canonical.caseId,
    assetId: canonical.assetId,
    sourceSha256: canonical.assetSha256,
    sourceBinding: Object.freeze({
      id: canonical.roiId,
      version: canonical.roiVersion,
      geometry: Object.freeze({
        x: canonical.roiGeometry.x,
        y: canonical.roiGeometry.y,
        width: canonical.roiGeometry.width,
        height: canonical.roiGeometry.height,
      }),
    }),
  })
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

function requestIdentitiesMatch(
  actual: unknown,
  expected: ConnectedAttentionRequestV1,
): boolean {
  if (!isRecord(actual)) return false
  const sourceBinding = actual.sourceBinding
  if (!isRecord(sourceBinding)) return false
  const geometry = sourceBinding.geometry
  return (
    hasExactKeys(actual, [
      'requestProfileVersion',
      'clientRunId',
      'attemptToken',
      'caseId',
      'assetId',
      'sourceSha256',
      'sourceBinding',
    ]) &&
    actual.requestProfileVersion === expected.requestProfileVersion &&
    actual.clientRunId === expected.clientRunId &&
    actual.attemptToken === expected.attemptToken &&
    actual.caseId === expected.caseId &&
    actual.assetId === expected.assetId &&
    actual.sourceSha256 === expected.sourceSha256 &&
    hasExactKeys(sourceBinding, ['id', 'version', 'geometry']) &&
    sourceBinding.id === expected.sourceBinding.id &&
    sourceBinding.version === expected.sourceBinding.version &&
    isRecord(geometry) &&
    hasExactKeys(geometry, ['x', 'y', 'width', 'height']) &&
    geometry.x === expected.sourceBinding.geometry.x &&
    geometry.y === expected.sourceBinding.geometry.y &&
    geometry.width === expected.sourceBinding.geometry.width &&
    geometry.height === expected.sourceBinding.geometry.height
  )
}

function isValidConnectedModelIdentity(
  value: unknown,
): value is ConnectedModelIdentity {
  return (
    isRecord(value) &&
    hasExactKeys(value, [
      'modelId',
      'modelVersion',
      'artifactSha256',
      'preprocessingVersion',
      'calibrationVersion',
      'displayScaleId',
    ]) &&
    typeof value.modelId === 'string' &&
    value.modelId.trim().length > 0 &&
    typeof value.modelVersion === 'string' &&
    value.modelVersion.trim().length > 0 &&
    typeof value.artifactSha256 === 'string' &&
    /^[0-9a-f]{64}$/.test(value.artifactSha256) &&
    typeof value.preprocessingVersion === 'string' &&
    value.preprocessingVersion.trim().length > 0 &&
    typeof value.calibrationVersion === 'string' &&
    value.calibrationVersion.trim().length > 0 &&
    typeof value.displayScaleId === 'string' &&
    value.displayScaleId.trim().length > 0
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
    value.length <= MAX_SPATIAL_DISPLAY_POINTS &&
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

function isValidQualityGates(value: unknown): boolean {
  return (
    isRecord(value) &&
    hasExactKeys(value, [
      'bindingIntegrity',
      'sourceBindingIntegrity',
      'finiteValues',
      'normalizedBounds',
      'researchDisplayEligible',
      'clinicalUseEligible',
    ]) &&
    value.bindingIntegrity === 'passed' &&
    value.sourceBindingIntegrity === 'passed' &&
    value.finiteValues === 'passed' &&
    value.normalizedBounds === 'passed' &&
    value.researchDisplayEligible === true &&
    value.clinicalUseEligible === false
  )
}

function isValidConnectedAttentionSemantics(value: unknown): boolean {
  if (
    !isRecord(value) ||
    !hasExactKeys(value, [
      'schemaVersion',
      'fieldMeaning',
      'target',
      'interpretation',
      'normalization',
      'clinicalAoi',
    ]) ||
    value.schemaVersion !== 'predicted-observer-attention/1' ||
    value.fieldMeaning !== 'relative_spatial_density' ||
    value.target !== 'predicted_observer_attention' ||
    value.interpretation !== 'population_level' ||
    value.normalization !== 'shared_display_scale_required' ||
    !isRecord(value.clinicalAoi) ||
    !hasExactKeys(value.clinicalAoi, [
      'registration',
      'role',
      'modifiesPrediction',
    ])
  ) {
    return false
  }

  return (
    value.clinicalAoi.registration ===
      'registration_geometry_unavailable_v1' &&
    value.clinicalAoi.role === 'post_inference_summary' &&
    value.clinicalAoi.modifiesPrediction === false
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
      'observedGazePayloadIncluded',
      'trainingDataProvenance',
    ]) &&
    value.engine === 'connected_model_gateway' &&
    typeof value.engineVersion === 'string' &&
    value.engineVersion.trim().length > 0 &&
    value.canonicalSyntheticAsset === true &&
    typeof value.deterministic === 'boolean' &&
    value.networkAccessed === true &&
    typeof value.storageAccessed === 'boolean' &&
    value.observedGazePayloadIncluded === false &&
    value.trainingDataProvenance === 'not_disclosed'
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
  const expectedRequestIdentity =
    createConnectedAttentionRequestV1(expectedBinding)
  if (
    !requestIdentitiesMatch(body.requestIdentity, expectedRequestIdentity) ||
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
      'requestIdentity',
      'modelIdentity',
      'resultDigest',
      'attentionSemantics',
      'heatmap',
      'qualityGates',
      'provenance',
    ]) ||
    typeof body.resultDigest !== 'string' ||
    body.resultDigest.trim().length === 0 ||
    !isValidConnectedModelIdentity(body.modelIdentity) ||
    !isValidConnectedAttentionSemantics(body.attentionSemantics) ||
    !isValidHeatmap(body.heatmap) ||
    !isValidQualityGates(body.qualityGates) ||
    !isValidConnectedProvenance(body.provenance)
  ) {
    fail(
      'MALFORMED_RESPONSE',
      'The connected gateway returned an invalid result envelope.',
    )
  }

  const {
    requestIdentity: _requestIdentity,
    ...validatedWireOutput
  } = body
  return deepCloneAndFreeze({
    ...validatedWireOutput,
    binding: expectedBinding,
  }) as ConnectedInferenceOutput
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

    const canonicalBinding = createCanonicalInferenceBindingSnapshot(binding)
    const payload = createConnectedAttentionRequestV1(canonicalBinding)
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

      return validateResponse(body, canonicalBinding)
    } finally {
      requestSignal.cleanup()
    }
  }
}
