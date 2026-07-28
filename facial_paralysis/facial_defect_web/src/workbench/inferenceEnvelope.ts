import { createCanonicalInferenceBindingSnapshot } from './mockEngine'
import type { WorkbenchGatewayMode } from './WorkbenchGateway'
import type {
  InferenceBinding,
  InferenceOutput,
  WorkbenchFailure,
  WorkbenchFailureReason,
} from './types'
import { MAX_SPATIAL_DISPLAY_POINTS } from './types'

export const MOCK_INFERENCE_WATERMARK = 'SIMULATED — NOT HUMAN GAZE' as const
export const CONNECTED_INFERENCE_WATERMARK =
  'MODEL PREDICTION — RESEARCH UNVALIDATED — NOT HUMAN GAZE — CLINICAL USE BLOCKED' as const

type EnvelopeValidation =
  | { readonly valid: true; readonly output: InferenceOutput }
  | { readonly valid: false; readonly failure: WorkbenchFailure }

function invalid(
  reason: WorkbenchFailureReason,
  message: string,
): EnvelopeValidation {
  return { valid: false, failure: { reason, message } }
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

function bindingMatches(actual: unknown, expected: InferenceBinding): boolean {
  if (!isRecord(actual)) return false
  const geometry = actual.roiGeometry
  const config = actual.config
  if (
    !hasExactKeys(actual, [
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
    ]) ||
    !isRecord(geometry) ||
    !hasExactKeys(geometry, ['x', 'y', 'width', 'height']) ||
    !isRecord(config) ||
    !hasExactKeys(config, ['threshold', 'smoothing'])
  ) {
    return false
  }

  try {
    const received = createCanonicalInferenceBindingSnapshot(
      actual as unknown as InferenceBinding,
    )
    const canonicalExpected = createCanonicalInferenceBindingSnapshot(expected)
    return JSON.stringify(received) === JSON.stringify(canonicalExpected)
  } catch {
    return false
  }
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

function isValidAttentionSemantics(
  value: unknown,
  expectedRegistration:
    | 'synthetic_template_v1'
    | 'registration_geometry_unavailable_v1',
): boolean {
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
    value.clinicalAoi.registration === expectedRegistration &&
    value.clinicalAoi.role === 'post_inference_summary' &&
    value.clinicalAoi.modifiesPrediction === false
  )
}

function isValidMockProvenance(value: unknown): boolean {
  return (
    isRecord(value) &&
    hasExactKeys(value, [
      'engine',
      'engineVersion',
      'modelMode',
      'canonicalSyntheticAsset',
      'deterministic',
      'networkAccessed',
      'storageAccessed',
      'humanGazeData',
    ]) &&
    value.engine === 'deterministic_mock_engine' &&
    value.engineVersion === '1' &&
    value.modelMode === 'mock_only' &&
    value.canonicalSyntheticAsset === true &&
    value.deterministic === true &&
    value.networkAccessed === false &&
    value.storageAccessed === false &&
    value.humanGazeData === false
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

function isValidConnectedModelIdentity(value: unknown): boolean {
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

export function validateInferenceOutputEnvelope(
  value: unknown,
  expectedBinding: InferenceBinding,
  expectedGatewayMode?: WorkbenchGatewayMode,
): EnvelopeValidation {
  if (!isRecord(value)) {
    return invalid(
      'MALFORMED_RESPONSE',
      'The gateway resolved an invalid inference output.',
    )
  }

  const expectedOrigin =
    expectedGatewayMode === 'mock'
      ? 'mock_simulation'
      : expectedGatewayMode === 'connected'
        ? 'model_prediction'
        : undefined
  if (
    (expectedOrigin !== undefined && value.origin !== expectedOrigin) ||
    (value.origin !== 'mock_simulation' && value.origin !== 'model_prediction')
  ) {
    return invalid(
      'ORIGIN_MISMATCH',
      'The resolved output origin does not match the active gateway mode.',
    )
  }

  const mockBranch = value.origin === 'mock_simulation'
  const expectedCapability = mockBranch
    ? 'simulated_ui_only'
    : 'research_unvalidated'
  const expectedWatermark = mockBranch
    ? MOCK_INFERENCE_WATERMARK
    : CONNECTED_INFERENCE_WATERMARK
  if (
    value.capabilityStatus !== expectedCapability ||
    value.watermark !== expectedWatermark
  ) {
    return invalid(
      'CAPABILITY_MISMATCH',
      'The resolved output capability or safety watermark is invalid.',
    )
  }

  if (!bindingMatches(value.binding, expectedBinding)) {
    return invalid(
      'IMMUTABLE_BINDING_MISMATCH',
      'The resolved output does not match the active inference binding.',
    )
  }

  const outputKeys = [
      'origin',
      'capabilityStatus',
      'watermark',
      'binding',
      'resultDigest',
      'attentionSemantics',
      'heatmap',
      'qualityGates',
      'provenance',
    ]
  if (!mockBranch) outputKeys.push('modelIdentity')

  if (
    !hasExactKeys(value, outputKeys) ||
    typeof value.resultDigest !== 'string' ||
    value.resultDigest.trim().length === 0 ||
    !isValidAttentionSemantics(
      value.attentionSemantics,
      mockBranch
        ? 'synthetic_template_v1'
        : 'registration_geometry_unavailable_v1',
    ) ||
    !isValidHeatmap(value.heatmap) ||
    !isValidQualityGates(value.qualityGates) ||
    (mockBranch
      ? !isValidMockProvenance(value.provenance)
      : !isValidConnectedProvenance(value.provenance) ||
        !isValidConnectedModelIdentity(value.modelIdentity))
  ) {
    return invalid(
      'MALFORMED_RESPONSE',
      'The gateway resolved an invalid inference output.',
    )
  }

  return { valid: true, output: value as unknown as InferenceOutput }
}
