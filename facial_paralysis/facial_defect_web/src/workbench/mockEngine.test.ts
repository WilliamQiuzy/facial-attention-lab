import { describe, expect, expectTypeOf, it, vi } from 'vitest'
import { getWorkbenchAsset } from './catalog'
import {
  createInferenceBinding,
  fingerprintInferenceInput,
  hashConfiguration,
  runMockEngine,
  validateNormalizedRoi,
} from './mockEngine'
import {
  WorkbenchError,
  type BatchJobStatus,
  type ConnectedInferenceOutput,
  type CreateInferenceBindingInput,
  type InferenceBinding,
  type InferenceOutput,
  type MockInferenceOutput,
  type MockModelVersion,
  type NormalizedRoi,
  type ResultFreshness,
  type ReviewStatus,
  type RoiStatus,
  type RunAttemptStatus,
} from './types'

const primaryAsset = getWorkbenchAsset('SYN-MOHS-SCC-CHEEK')!
const alternateAsset = getWorkbenchAsset('SYN-MOHS-NASAL-RECON')!

const fullImageGeometry: NormalizedRoi = {
  x: 0,
  y: 0,
  width: 1,
  height: 1,
}

const validPartialGeometry: NormalizedRoi = {
  x: 0.18,
  y: 0.24,
  width: 0.41,
  height: 0.36,
}

function makeInput(
  overrides: Partial<CreateInferenceBindingInput> = {},
): CreateInferenceBindingInput {
  return {
    clientRunId: 'run-001',
    attemptToken: 'attempt-001',
    caseId: primaryAsset.id,
    assetId: primaryAsset.id,
    assetSha256: primaryAsset.sha256,
    roi: {
      id: 'roi-primary',
      caseId: primaryAsset.id,
      assetId: primaryAsset.id,
      version: 3,
      geometry: fullImageGeometry,
      status: 'approved',
      authorId: 'demo_author',
      reviewerId: 'demo_reviewer',
    },
    modelVersion: 'mock-salience-v0.3',
    modelMode: 'mock_only',
    config: { threshold: 0.42, smoothing: 0.27 },
    ...overrides,
  }
}

function captureFailure(input: CreateInferenceBindingInput): WorkbenchError {
  try {
    createInferenceBinding(input)
  } catch (error) {
    expect(error).toBeInstanceOf(WorkbenchError)
    return error as WorkbenchError
  }
  throw new Error('Expected createInferenceBinding to fail closed.')
}

describe('deterministic mock inference engine', () => {
  it('rejects an approved partial rectangle as a source binding', () => {
    const error = captureFailure(
      makeInput({
        roi: { ...makeInput().roi, geometry: validPartialGeometry },
      }),
    )

    expect(error.reason).toBe('FULL_IMAGE_SOURCE_BINDING_REQUIRED')
    expect(error.message).toMatch(/full-image source binding/i)
  })

  it('keeps workflow state contracts exact', () => {
    expectTypeOf<RunAttemptStatus>().toEqualTypeOf<
      | 'draft'
      | 'validating'
      | 'blocked'
      | 'queued'
      | 'running'
      | 'succeeded'
      | 'failed'
      | 'cancelled'
    >()
    expectTypeOf<BatchJobStatus>().toEqualTypeOf<
      | 'draft'
      | 'preflighting'
      | 'ready'
      | 'blocked'
      | 'queued'
      | 'running'
      | 'completed'
      | 'completed_with_failures'
      | 'failed'
      | 'cancelled'
    >()
    expectTypeOf<ReviewStatus>().toEqualTypeOf<
      | 'awaiting_review'
      | 'approved_for_research'
      | 'changes_requested'
      | 'revoked'
    >()
    expectTypeOf<ResultFreshness>().toEqualTypeOf<
      'current' | 'stale' | 'revoked'
    >()
    expectTypeOf<RoiStatus>().toEqualTypeOf<
      | 'draft'
      | 'in_review'
      | 'approved'
      | 'changes_requested'
      | 'superseded'
    >()
    expectTypeOf<
      MockInferenceOutput['attentionSemantics']['clinicalAoi']['registration']
    >().toEqualTypeOf<'synthetic_template_v1'>()
    expectTypeOf<
      ConnectedInferenceOutput['attentionSemantics']['clinicalAoi']['registration']
    >().toEqualTypeOf<'registration_geometry_unavailable_v1'>()
  })

  it('keeps scientific fingerprints and results identical across operational retries', () => {
    const firstBinding = createInferenceBinding(makeInput())
    const retryBinding = createInferenceBinding(
      makeInput({ clientRunId: 'run-999', attemptToken: 'attempt-retry-2' }),
    )

    expect(firstBinding.clientRunId).not.toBe(retryBinding.clientRunId)
    expect(firstBinding.attemptToken).not.toBe(retryBinding.attemptToken)
    expect(firstBinding.configurationHash).toBe(hashConfiguration(firstBinding.config))
    expect(firstBinding.inputFingerprint).toBe(retryBinding.inputFingerprint)
    expect(
      fingerprintInferenceInput({
        assetId: firstBinding.assetId,
        assetSha256: firstBinding.assetSha256,
        roiId: firstBinding.roiId,
        roiVersion: firstBinding.roiVersion,
        roiGeometry: firstBinding.roiGeometry,
        modelVersion: firstBinding.modelVersion,
        configurationHash: firstBinding.configurationHash,
      }),
    ).toBe(firstBinding.inputFingerprint)

    const first = runMockEngine(firstBinding)
    const retry = runMockEngine(retryBinding)

    expect(first.resultDigest).toBe(retry.resultDigest)
    expect(first.heatmap).toEqual(retry.heatmap)
  })

  it('reconstructs a deeply frozen binding snapshot before execution', () => {
    const canonicalBinding = createInferenceBinding(makeInput())
    const callerOwnedBinding = {
      ...canonicalBinding,
      config: { ...canonicalBinding.config },
      roiGeometry: { ...canonicalBinding.roiGeometry },
    } satisfies InferenceBinding

    const result = runMockEngine(callerOwnedBinding)

    callerOwnedBinding.config.threshold = 0.91
    callerOwnedBinding.roiGeometry.x = 0.73

    expect(result.binding).not.toBe(callerOwnedBinding)
    expect(result.binding).toEqual(canonicalBinding)
    expect(result.binding.config.threshold).toBe(canonicalBinding.config.threshold)
    expect(result.binding.roiGeometry.x).toBe(canonicalBinding.roiGeometry.x)
    expect(Object.isFrozen(result)).toBe(true)
    expect(Object.isFrozen(result.binding)).toBe(true)
    expect(Object.isFrozen(result.binding.config)).toBe(true)
    expect(Object.isFrozen(result.binding.roiGeometry)).toBe(true)
    expect(Object.isFrozen(result.heatmap)).toBe(true)
    expect(Object.isFrozen(result.heatmap[0])).toBe(true)
    expect(Object.isFrozen(result.qualityGates)).toBe(true)
    expect(Object.isFrozen(result.provenance)).toBe(true)
    expect(result.attentionSemantics).toBeDefined()
    expect(Object.isFrozen(result.attentionSemantics)).toBe(true)
    expect(Object.isFrozen(result.attentionSemantics?.clinicalAoi)).toBe(true)
  })

  it('matches the exact engine-v1 golden vector for the canonical input', () => {
    const result = runMockEngine(createInferenceBinding(makeInput()))

    expect({
      engineVersion: result.provenance.engineVersion,
      configurationHash: result.binding.configurationHash,
      inputFingerprint: result.binding.inputFingerprint,
      resultDigest: result.resultDigest,
      heatmap: result.heatmap,
    }).toEqual({
      engineVersion: '1',
      configurationHash: 'cfg_560a98e7c95091ec',
      inputFingerprint: 'input_e1091986e713d7f1',
      resultDigest: 'result_406edfbd6714d9c3',
      heatmap: [
        { x: 0.347147, y: 0.22672, intensity: 0.587918, radius: 0.150106 },
        { x: 0.6829, y: 0.21517, intensity: 0.631632, radius: 0.109231 },
        { x: 0.3582, y: 0.258895, intensity: 0.123662, radius: 0.102727 },
        { x: 0.639281, y: 0.292528, intensity: 0.672541, radius: 0.068695 },
        { x: 0.322669, y: 0.379045, intensity: 0.008687, radius: 0.144545 },
        { x: 0.659595, y: 0.37515, intensity: 0.616381, radius: 0.144152 },
        { x: 0.364898, y: 0.446753, intensity: 0.294998, radius: 0.138897 },
        { x: 0.616203, y: 0.413255, intensity: 0.736642, radius: 0.204754 },
        { x: 0.331808, y: 0.542056, intensity: 0.741663, radius: 0.112948 },
        { x: 0.64037, y: 0.547147, intensity: 0.066566, radius: 0.166378 },
        { x: 0.393255, y: 0.612281, intensity: 0.29272, radius: 0.068245 },
        { x: 0.551444, y: 0.582181, intensity: 0.852715, radius: 0.136274 },
        { x: 0.37544, y: 0.716651, intensity: 0.097371, radius: 0.06257 },
        { x: 0.609844, y: 0.710545, intensity: 0.847758, radius: 0.167345 },
        { x: 0.438857, y: 0.773936, intensity: 0.269143, radius: 0.064075 },
        { x: 0.584918, y: 0.80246, intensity: 0.021256, radius: 0.161606 },
      ],
    })
  })

  it.each([
    [
      'asset identity and canonical hash',
      () =>
        makeInput({
          caseId: alternateAsset.id,
          assetId: alternateAsset.id,
          assetSha256: alternateAsset.sha256,
          roi: {
            ...makeInput().roi,
            caseId: alternateAsset.id,
            assetId: alternateAsset.id,
          },
        }),
    ],
    ['ROI version', () => makeInput({ roi: { ...makeInput().roi, version: 4 } })],
    ['source binding ID', () => makeInput({
      roi: { ...makeInput().roi, id: 'source-binding-alternate' },
    })],
    ['model version', () => makeInput({ modelVersion: 'mock-salience-v0.4' })],
    [
      'configuration',
      () => makeInput({ config: { ...makeInput().config, smoothing: 0.31 } }),
    ],
  ])('changes the fingerprint and result when %s changes', (_label, variation) => {
    const baseline = runMockEngine(createInferenceBinding(makeInput()))
    const changed = runMockEngine(createInferenceBinding(variation()))

    expect(changed.binding.inputFingerprint).not.toBe(
      baseline.binding.inputFingerprint,
    )
    expect(changed.resultDigest).not.toBe(baseline.resultDigest)
    expect(changed.heatmap).not.toEqual(baseline.heatmap)
  })

  it('returns bounded finite values with explicit mock-only safety labels', () => {
    const binding = createInferenceBinding(makeInput())
    const result: InferenceOutput = runMockEngine(binding)

    expect(result.origin).toBe('mock_simulation')
    expect(result.capabilityStatus).toBe('simulated_ui_only')
    expect(result.watermark).toBe('SIMULATED — NOT HUMAN GAZE')
    expect(result.binding).not.toBe(binding)
    expect(result.binding).toEqual(binding)
    expect(result.attentionSemantics).toEqual({
      schemaVersion: 'predicted-observer-attention/1',
      fieldMeaning: 'relative_spatial_density',
      target: 'predicted_observer_attention',
      interpretation: 'population_level',
      normalization: 'shared_display_scale_required',
      clinicalAoi: {
        registration: 'synthetic_template_v1',
        role: 'post_inference_summary',
        modifiesPrediction: false,
      },
    })
    expect(result.heatmap.length).toBeGreaterThan(0)
    for (const point of result.heatmap) {
      for (const value of [point.x, point.y, point.intensity, point.radius]) {
        expect(Number.isFinite(value)).toBe(true)
        expect(value).toBeGreaterThanOrEqual(0)
        expect(value).toBeLessThanOrEqual(1)
      }
    }
    expect(result.qualityGates).toEqual({
      bindingIntegrity: 'passed',
      sourceBindingIntegrity: 'passed',
      finiteValues: 'passed',
      normalizedBounds: 'passed',
      researchDisplayEligible: true,
      clinicalUseEligible: false,
    })
    expect(result.provenance).toEqual({
      engine: 'deterministic_mock_engine',
      engineVersion: '1',
      modelMode: 'mock_only',
      canonicalSyntheticAsset: true,
      deterministic: true,
      networkAccessed: false,
      storageAccessed: false,
      humanGazeData: false,
    })
  })

  it('places deterministic samples around synthetic facial feature centers', () => {
    const result = runMockEngine(createInferenceBinding(makeInput()))
    const relativePoints = result.heatmap.map((point) => ({
      x: (point.x - fullImageGeometry.x) / fullImageGeometry.width,
      y: (point.y - fullImageGeometry.y) / fullImageGeometry.height,
    }))
    const featureBands = [
      ['brow', relativePoints.slice(0, 4), [0.16, 0.34]],
      ['bilateral orbital', relativePoints.slice(4, 8), [0.32, 0.48]],
      ['midface', relativePoints.slice(8, 12), [0.48, 0.66]],
      ['perioral', relativePoints.slice(12, 16), [0.66, 0.84]],
    ] as const

    for (const [label, points, [minY, maxY]] of featureBands) {
      expect(points, label).toHaveLength(4)
      expect(points.some((point) => point.x < 0.5), label).toBe(true)
      expect(points.some((point) => point.x > 0.5), label).toBe(true)
      for (const point of points) {
        expect(point.x, label).toBeGreaterThanOrEqual(0.27)
        expect(point.x, label).toBeLessThanOrEqual(0.73)
        expect(point.y, label).toBeGreaterThanOrEqual(minY)
        expect(point.y, label).toBeLessThanOrEqual(maxY)
      }
    }
  })

  it('accepts sub-micro geometry as normalized but rejects it as an execution source binding', () => {
    const geometry: NormalizedRoi = {
      x: 4e-7,
      y: 4e-7,
      width: 1e-7,
      height: 1e-7,
    }
    const error = captureFailure(
      makeInput({ roi: { ...makeInput().roi, geometry } }),
    )

    expect(validateNormalizedRoi(geometry)).toBe(true)
    expect(error.reason).toBe('FULL_IMAGE_SOURCE_BINDING_REQUIRED')
  })

  it.each([
    [
      'unknown case',
      makeInput({ caseId: 'UNKNOWN-CASE' }),
      'UNKNOWN_CASE',
    ],
    [
      'unknown asset',
      makeInput({ assetId: 'UNKNOWN-ASSET' }),
      'UNKNOWN_ASSET',
    ],
    [
      'case and asset mismatch',
      makeInput({ caseId: alternateAsset.id }),
      'CASE_ASSET_MISMATCH',
    ],
    [
      'asset hash mismatch',
      makeInput({ assetSha256: '0'.repeat(64) }),
      'ASSET_HASH_MISMATCH',
    ],
    [
      'ROI not approved',
      makeInput({ roi: { ...makeInput().roi, status: 'in_review' } }),
      'ROI_NOT_APPROVED',
    ],
    [
      'invalid ROI geometry',
      makeInput({
        roi: {
          ...makeInput().roi,
          geometry: { ...fullImageGeometry, width: 0 },
        },
      }),
      'INVALID_ROI_GEOMETRY',
    ],
    [
      'unknown model',
      makeInput({ modelVersion: 'unregistered-model' as MockModelVersion }),
      'UNKNOWN_MODEL',
    ],
    [
      'invalid configuration',
      makeInput({ config: { threshold: Number.NaN, smoothing: 1.1 } }),
      'INVALID_CONFIGURATION',
    ],
  ] as const)('fails closed with a stable reason for %s', (_label, input, reason) => {
    const error = captureFailure(input as CreateInferenceBindingInput)

    expect(error.reason).toBe(reason)
    expect(error.failure).toMatchObject({ reason })
    expect(error.message.length).toBeGreaterThan(0)
  })

  it.each([
    [fullImageGeometry, true],
    [validPartialGeometry, true],
    [{ ...validPartialGeometry, x: -0.01 }, false],
    [{ ...validPartialGeometry, y: Number.POSITIVE_INFINITY }, false],
    [{ ...validPartialGeometry, width: 0 }, false],
    [{ ...validPartialGeometry, height: 0 }, false],
    [{ ...validPartialGeometry, x: 0.8, width: 0.3 }, false],
    [{ ...validPartialGeometry, y: 0.8, height: 0.3 }, false],
  ])('validates normalized ROI geometry %#', (geometry, expected) => {
    expect(validateNormalizedRoi(geometry)).toBe(expected)
  })

  it('does not use fetch or browser storage and excludes operational IDs from the fingerprint', () => {
    const fetchProbe = vi.fn(() => {
      throw new Error('fetch must not be called by the pure mock engine')
    })
    const localGetProbe = vi.spyOn(Storage.prototype, 'getItem')
    const localSetProbe = vi.spyOn(Storage.prototype, 'setItem')
    vi.stubGlobal('fetch', fetchProbe)

    try {
      const first = createInferenceBinding(makeInput())
      const changedOperationalIds = createInferenceBinding(
        makeInput({ clientRunId: 'another-run', attemptToken: 'another-attempt' }),
      )
      runMockEngine(first)
      runMockEngine(changedOperationalIds)

      expect(changedOperationalIds.inputFingerprint).toBe(first.inputFingerprint)
      expect(first.inputFingerprint).not.toContain(first.clientRunId)
      expect(first.inputFingerprint).not.toContain(first.attemptToken)
      expect(fetchProbe).not.toHaveBeenCalled()
      expect(localGetProbe).not.toHaveBeenCalled()
      expect(localSetProbe).not.toHaveBeenCalled()
    } finally {
      vi.unstubAllGlobals()
      vi.restoreAllMocks()
    }
  })
})
