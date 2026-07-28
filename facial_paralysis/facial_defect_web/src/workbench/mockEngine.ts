import { getWorkbenchAsset } from './catalog'
import { isVerifiedFullImageSourceBinding } from './sourceBinding'
import {
  WorkbenchError,
  type CreateInferenceBindingInput,
  type InferenceBinding,
  type InferenceConfiguration,
  type MockInferenceOutput,
  type MockModelVersion,
  type NormalizedRoi,
  type ScientificInferenceInput,
  type SpatialAttentionSemantics,
  type WorkbenchFailureReason,
} from './types'

const MODEL_VERSIONS = new Set<MockModelVersion>([
  'mock-salience-v0.3',
  'mock-salience-v0.4',
])

const MOCK_ENGINE_VERSION = '1' as const

const MOCK_QUALITY_GATES = Object.freeze({
  bindingIntegrity: 'passed',
  sourceBindingIntegrity: 'passed',
  finiteValues: 'passed',
  normalizedBounds: 'passed',
  researchDisplayEligible: true,
  clinicalUseEligible: false,
} as const)

const MOCK_PROVENANCE = Object.freeze({
  engine: 'deterministic_mock_engine',
  engineVersion: MOCK_ENGINE_VERSION,
  modelMode: 'mock_only',
  canonicalSyntheticAsset: true,
  deterministic: true,
  networkAccessed: false,
  storageAccessed: false,
  humanGazeData: false,
} as const)

const MOCK_ATTENTION_SEMANTICS = Object.freeze({
  schemaVersion: 'predicted-observer-attention/1',
  fieldMeaning: 'relative_spatial_density',
  target: 'predicted_observer_attention',
  interpretation: 'population_level',
  normalization: 'shared_display_scale_required',
  clinicalAoi: Object.freeze({
    registration: 'synthetic_template_v1',
    role: 'post_inference_summary',
    modifiesPrediction: false,
  }),
} as const satisfies SpatialAttentionSemantics<'synthetic_template_v1'>)

function fail(
  reason: WorkbenchFailureReason,
  message: string,
  field?: string,
): never {
  throw new WorkbenchError({ reason, message, ...(field ? { field } : {}) })
}

function isBoundedFinite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
}

export function validateNormalizedRoi(geometry: unknown): geometry is NormalizedRoi {
  if (!geometry || typeof geometry !== 'object') return false

  const candidate = geometry as Partial<NormalizedRoi>
  if (
    !isBoundedFinite(candidate.x) ||
    !isBoundedFinite(candidate.y) ||
    !isBoundedFinite(candidate.width) ||
    !isBoundedFinite(candidate.height) ||
    candidate.width <= 0 ||
    candidate.height <= 0
  ) {
    return false
  }

  return candidate.x + candidate.width <= 1 && candidate.y + candidate.height <= 1
}

function validateConfiguration(
  config: InferenceConfiguration,
): asserts config is InferenceConfiguration {
  if (!isBoundedFinite(config?.threshold) || !isBoundedFinite(config?.smoothing)) {
    fail(
      'INVALID_CONFIGURATION',
      'Inference threshold and smoothing must be finite values from 0 through 1.',
      'config',
    )
  }
}

function stableSerialize(value: unknown): string {
  if (value === null || typeof value !== 'object') {
    return JSON.stringify(value)
  }
  if (Array.isArray(value)) {
    return `[${value.map(stableSerialize).join(',')}]`
  }

  const record = value as Record<string, unknown>
  return `{${Object.keys(record)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${stableSerialize(record[key])}`)
    .join(',')}}`
}

function hashText(value: string): string {
  let first = 0x811c9dc5
  let second = 0x9e3779b9

  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index)
    first ^= code
    first = Math.imul(first, 0x01000193)
    second ^= code + index
    second = Math.imul(second, 0x85ebca6b)
  }

  return `${(first >>> 0).toString(16).padStart(8, '0')}${(second >>> 0)
    .toString(16)
    .padStart(8, '0')}`
}

export function hashConfiguration(config: InferenceConfiguration): string {
  validateConfiguration(config)
  return `cfg_${hashText(
    stableSerialize({ smoothing: config.smoothing, threshold: config.threshold }),
  )}`
}

export function fingerprintInferenceInput(input: ScientificInferenceInput): string {
  return `input_${hashText(
    stableSerialize({
      assetId: input.assetId,
      assetSha256: input.assetSha256,
      configurationHash: input.configurationHash,
      modelVersion: input.modelVersion,
      roiGeometry: input.roiGeometry,
      roiId: input.roiId,
      roiVersion: input.roiVersion,
    }),
  )}`
}

function freezeBinding(binding: InferenceBinding): InferenceBinding {
  Object.freeze(binding.roiGeometry)
  Object.freeze(binding.config)
  return Object.freeze(binding)
}

export function createInferenceBinding(
  input: CreateInferenceBindingInput,
): InferenceBinding {
  const canonicalCase = getWorkbenchAsset(input.caseId)
  if (!canonicalCase) {
    fail('UNKNOWN_CASE', `Unknown workbench case: ${input.caseId}.`, 'caseId')
  }

  const canonicalAsset = getWorkbenchAsset(input.assetId)
  if (!canonicalAsset) {
    fail('UNKNOWN_ASSET', `Unknown workbench asset: ${input.assetId}.`, 'assetId')
  }
  if (input.caseId !== input.assetId) {
    fail(
      'CASE_ASSET_MISMATCH',
      'The case ID and asset ID must identify the same canonical workbench entry.',
      'assetId',
    )
  }
  if (input.assetSha256 !== canonicalAsset.sha256) {
    fail(
      'ASSET_HASH_MISMATCH',
      'The supplied asset hash does not match the canonical workbench catalog.',
      'assetSha256',
    )
  }
  if (input.roi.caseId !== input.caseId || input.roi.assetId !== input.assetId) {
    fail(
      'ROI_BINDING_MISMATCH',
      'The ROI must be bound to the selected case and canonical asset.',
      'roi',
    )
  }
  if (input.roi.status !== 'approved') {
    fail('ROI_NOT_APPROVED', 'Inference requires an approved ROI.', 'roi.status')
  }
  if (!validateNormalizedRoi(input.roi.geometry)) {
    fail(
      'INVALID_ROI_GEOMETRY',
      'ROI geometry must be finite, normalized, positive-area, and contained in the image.',
      'roi.geometry',
    )
  }
  if (
    !Number.isInteger(input.roi.version) ||
    input.roi.version <= 0 ||
    typeof input.roi.id !== 'string' ||
    input.roi.id.trim().length === 0
  ) {
    fail(
      'INVALID_ROI_VERSION',
      'An approved ROI requires a non-empty ID and positive integer version.',
      'roi.version',
    )
  }
  if (input.roi.authorId !== 'demo_author' || input.roi.reviewerId !== 'demo_reviewer') {
    fail(
      'INVALID_ROI_ACTORS',
      'ROI authorship and review must use the declared demo actors.',
      'roi',
    )
  }
  if (!isVerifiedFullImageSourceBinding(canonicalAsset, input.roi)) {
    fail(
      'FULL_IMAGE_SOURCE_BINDING_REQUIRED',
      'Inference requires a verified full-image source binding; partial rectangles are not inference inputs.',
      'roi.geometry',
    )
  }
  if (!MODEL_VERSIONS.has(input.modelVersion)) {
    fail('UNKNOWN_MODEL', `Unknown mock model: ${input.modelVersion}.`, 'modelVersion')
  }
  if (input.modelMode !== 'mock_only') {
    fail(
      'UNSUPPORTED_MODEL_MODE',
      'This workbench slice supports mock_only execution.',
      'modelMode',
    )
  }
  if (
    typeof input.clientRunId !== 'string' ||
    input.clientRunId.trim().length === 0 ||
    typeof input.attemptToken !== 'string' ||
    input.attemptToken.trim().length === 0
  ) {
    fail(
      'INVALID_OPERATIONAL_ID',
      'Client run and attempt identifiers must be non-empty.',
    )
  }

  validateConfiguration(input.config)
  const config = Object.freeze({
    threshold: input.config.threshold,
    smoothing: input.config.smoothing,
  })
  const roiGeometry = Object.freeze({
    x: input.roi.geometry.x,
    y: input.roi.geometry.y,
    width: input.roi.geometry.width,
    height: input.roi.geometry.height,
  })
  const configurationHash = hashConfiguration(config)
  const scientificInput: ScientificInferenceInput = {
    assetId: canonicalAsset.id,
    assetSha256: canonicalAsset.sha256,
    roiId: input.roi.id,
    roiVersion: input.roi.version,
    roiGeometry,
    modelVersion: input.modelVersion,
    configurationHash,
  }

  return freezeBinding({
    clientRunId: input.clientRunId,
    attemptToken: input.attemptToken,
    caseId: canonicalCase.id,
    assetId: canonicalAsset.id,
    assetSha256: canonicalAsset.sha256,
    roiId: input.roi.id,
    roiVersion: input.roi.version,
    roiGeometry,
    roiStatus: 'approved',
    modelVersion: input.modelVersion,
    modelMode: 'mock_only',
    config,
    configurationHash,
    inputFingerprint: fingerprintInferenceInput(scientificInput),
  })
}

export function createCanonicalInferenceBindingSnapshot(
  binding: InferenceBinding,
): InferenceBinding {
  const rebuilt = createInferenceBinding({
    clientRunId: binding.clientRunId,
    attemptToken: binding.attemptToken,
    caseId: binding.caseId,
    assetId: binding.assetId,
    assetSha256: binding.assetSha256,
    roi: {
      id: binding.roiId,
      caseId: binding.caseId,
      assetId: binding.assetId,
      version: binding.roiVersion,
      geometry: binding.roiGeometry,
      status: binding.roiStatus,
      authorId: 'demo_author',
      reviewerId: 'demo_reviewer',
    },
    modelVersion: binding.modelVersion,
    modelMode: binding.modelMode,
    config: binding.config,
  })

  if (
    rebuilt.configurationHash !== binding.configurationHash ||
    rebuilt.inputFingerprint !== binding.inputFingerprint
  ) {
    fail(
      'BINDING_INTEGRITY_MISMATCH',
      'The inference binding no longer matches its configuration hash and input fingerprint.',
    )
  }

  return rebuilt
}

function seedFromText(value: string): number {
  return Number.parseInt(hashText(value).slice(0, 8), 16) || 0x6d2b79f5
}

function createPrng(seed: number): () => number {
  let state = seed >>> 0
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let value = state
    value = Math.imul(value ^ (value >>> 15), value | 1)
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61)
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296
  }
}

function rounded(value: number): number {
  return Number(value.toFixed(6))
}

function bounded(value: number): number {
  return rounded(Math.min(1, Math.max(0, value)))
}

function boundedWithinRoi(value: number, start: number, size: number): number {
  return Math.min(start + size, Math.max(start, bounded(value)))
}

const SYNTHETIC_FACE_CENTERS = Object.freeze([
  { x: 0.34, y: 0.22 },
  { x: 0.66, y: 0.22 },
  { x: 0.38, y: 0.28 },
  { x: 0.62, y: 0.28 },
  { x: 0.34, y: 0.36 },
  { x: 0.66, y: 0.36 },
  { x: 0.36, y: 0.43 },
  { x: 0.64, y: 0.43 },
  { x: 0.35, y: 0.53 },
  { x: 0.65, y: 0.53 },
  { x: 0.42, y: 0.59 },
  { x: 0.58, y: 0.59 },
  { x: 0.4, y: 0.7 },
  { x: 0.6, y: 0.7 },
  { x: 0.44, y: 0.78 },
  { x: 0.56, y: 0.78 },
] as const)

function createHeatmap(binding: InferenceBinding) {
  const random = createPrng(seedFromText(binding.inputFingerprint))
  const points = SYNTHETIC_FACE_CENTERS.map((center) => {
    const relativeX = center.x + (random() - 0.5) * 0.06
    const relativeY = center.y + (random() - 0.5) * 0.05
    const x =
      binding.roiGeometry.x + relativeX * binding.roiGeometry.width
    const y =
      binding.roiGeometry.y + relativeY * binding.roiGeometry.height
    const rawIntensity = random()
    const intensity =
      rawIntensity < binding.config.threshold
        ? rawIntensity * (1 - binding.config.smoothing)
        : rawIntensity + (1 - rawIntensity) * binding.config.smoothing

    return Object.freeze({
      x: boundedWithinRoi(x, binding.roiGeometry.x, binding.roiGeometry.width),
      y: boundedWithinRoi(y, binding.roiGeometry.y, binding.roiGeometry.height),
      intensity: bounded(intensity),
      radius: bounded(0.025 + random() * (0.16 + binding.config.smoothing * 0.08)),
    })
  })
  return Object.freeze(points)
}

function createResultDigest(
  binding: InferenceBinding,
  heatmap: ReturnType<typeof createHeatmap>,
): string {
  const canonicalResultPayload = {
    engineVersion: MOCK_ENGINE_VERSION,
    bindingScientificFingerprint: binding.inputFingerprint,
    heatmap,
    attentionSemantics: MOCK_ATTENTION_SEMANTICS,
    qualityGates: MOCK_QUALITY_GATES,
    provenance: MOCK_PROVENANCE,
  }

  return `result_${hashText(stableSerialize(canonicalResultPayload))}`
}

export function runMockEngine(binding: InferenceBinding): MockInferenceOutput {
  const canonicalBinding = createCanonicalInferenceBindingSnapshot(binding)
  const heatmap = createHeatmap(canonicalBinding)

  return Object.freeze({
    origin: 'mock_simulation',
    capabilityStatus: 'simulated_ui_only',
    watermark: 'SIMULATED — NOT HUMAN GAZE',
    binding: canonicalBinding,
    resultDigest: createResultDigest(canonicalBinding, heatmap),
    attentionSemantics: MOCK_ATTENTION_SEMANTICS,
    heatmap,
    qualityGates: MOCK_QUALITY_GATES,
    provenance: MOCK_PROVENANCE,
  })
}
