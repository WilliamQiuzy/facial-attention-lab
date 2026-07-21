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
  type CreateInferenceBindingInput,
  type InferenceBinding,
  type InferenceOutput,
  type MockModelVersion,
  type NormalizedRoi,
  type ResultFreshness,
  type ReviewStatus,
  type RoiStatus,
  type RunAttemptStatus,
} from './types'

const primaryAsset = getWorkbenchAsset('SYN-MOHS-SCC-CHEEK')!
const alternateAsset = getWorkbenchAsset('SYN-MOHS-NASAL-RECON')!

const approvedGeometry: NormalizedRoi = {
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
      geometry: approvedGeometry,
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
    expect(first.metrics).toEqual(retry.metrics)
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
    expect(Object.isFrozen(result.metrics)).toBe(true)
    expect(Object.isFrozen(result.qualityGates)).toBe(true)
    expect(Object.isFrozen(result.provenance)).toBe(true)
  })

  it('matches the exact engine-v1 golden vector for the canonical input', () => {
    const result = runMockEngine(createInferenceBinding(makeInput()))

    expect({
      engineVersion: result.provenance.engineVersion,
      configurationHash: result.binding.configurationHash,
      inputFingerprint: result.binding.inputFingerprint,
      resultDigest: result.resultDigest,
      heatmap: result.heatmap,
      metrics: result.metrics,
    }).toEqual({
      engineVersion: '1',
      configurationHash: 'cfg_560a98e7c95091ec',
      inputFingerprint: 'input_31dacbcb4bf0af1e',
      resultDigest: 'result_c02d34ec4178a9e6',
      heatmap: [
        { x: 0.568864, y: 0.540952, intensity: 0.797848, radius: 0.199183 },
        { x: 0.357449, y: 0.595387, intensity: 0.254804, radius: 0.122143 },
        { x: 0.348713, y: 0.550194, intensity: 0.714423, radius: 0.199951 },
        { x: 0.492627, y: 0.348792, intensity: 0.26855, radius: 0.044487 },
        { x: 0.403277, y: 0.404227, intensity: 0.849792, radius: 0.079589 },
        { x: 0.285073, y: 0.283259, intensity: 0.974416, radius: 0.096441 },
        { x: 0.516874, y: 0.419981, intensity: 0.018861, radius: 0.099955 },
        { x: 0.251337, y: 0.591079, intensity: 0.089975, radius: 0.117029 },
        { x: 0.197058, y: 0.485218, intensity: 0.018493, radius: 0.114175 },
        { x: 0.443899, y: 0.457815, intensity: 0.858915, radius: 0.054206 },
        { x: 0.577864, y: 0.489751, intensity: 0.23627, radius: 0.087002 },
        { x: 0.474379, y: 0.579651, intensity: 0.167714, radius: 0.034113 },
        { x: 0.201255, y: 0.30874, intensity: 0.076872, radius: 0.052386 },
        { x: 0.307677, y: 0.497174, intensity: 0.636596, radius: 0.045938 },
        { x: 0.236918, y: 0.26947, intensity: 0.873897, radius: 0.104208 },
        { x: 0.237404, y: 0.59883, intensity: 0.660868, radius: 0.150112 },
      ],
      metrics: {
        roiCoverage: 0.124072,
        peakIntensity: 0.974416,
        meanIntensity: 0.468643,
        focusScore: 0.890735,
      },
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
    [
      'ROI geometry',
      () =>
        makeInput({
          roi: {
            ...makeInput().roi,
            geometry: { ...approvedGeometry, x: approvedGeometry.x + 0.01 },
          },
        }),
    ],
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
    expect({ heatmap: changed.heatmap, metrics: changed.metrics }).not.toEqual({
      heatmap: baseline.heatmap,
      metrics: baseline.metrics,
    })
  })

  it('returns bounded finite values with explicit mock-only safety labels', () => {
    const binding = createInferenceBinding(makeInput())
    const result: InferenceOutput = runMockEngine(binding)

    expect(result.origin).toBe('mock_simulation')
    expect(result.capabilityStatus).toBe('simulated_ui_only')
    expect(result.watermark).toBe('SIMULATED — NOT HUMAN GAZE')
    expect(result.binding).not.toBe(binding)
    expect(result.binding).toEqual(binding)
    expect(result.heatmap.length).toBeGreaterThan(0)
    for (const point of result.heatmap) {
      for (const value of [point.x, point.y, point.intensity, point.radius]) {
        expect(Number.isFinite(value)).toBe(true)
        expect(value).toBeGreaterThanOrEqual(0)
        expect(value).toBeLessThanOrEqual(1)
      }
    }
    for (const value of Object.values(result.metrics)) {
      expect(Number.isFinite(value)).toBe(true)
      expect(value).toBeGreaterThanOrEqual(0)
      expect(value).toBeLessThanOrEqual(1)
    }
    expect(result.qualityGates).toEqual({
      bindingIntegrity: 'passed',
      roiApproval: 'passed',
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

  it('keeps every heatmap coordinate inside a valid sub-micro ROI', () => {
    const geometry: NormalizedRoi = {
      x: 4e-7,
      y: 4e-7,
      width: 1e-7,
      height: 1e-7,
    }
    const result = runMockEngine(
      createInferenceBinding(
        makeInput({ roi: { ...makeInput().roi, geometry } }),
      ),
    )
    const maxX = geometry.x + geometry.width
    const maxY = geometry.y + geometry.height

    expect(validateNormalizedRoi(geometry)).toBe(true)
    for (const point of result.heatmap) {
      expect(point.x).toBeGreaterThanOrEqual(geometry.x)
      expect(point.x).toBeLessThanOrEqual(maxX)
      expect(point.y).toBeGreaterThanOrEqual(geometry.y)
      expect(point.y).toBeLessThanOrEqual(maxY)
      expect(point.x).toBeGreaterThanOrEqual(0)
      expect(point.x).toBeLessThanOrEqual(1)
      expect(point.y).toBeGreaterThanOrEqual(0)
      expect(point.y).toBeLessThanOrEqual(1)
    }
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
          geometry: { ...approvedGeometry, width: 0 },
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
    [approvedGeometry, true],
    [{ ...approvedGeometry, x: -0.01 }, false],
    [{ ...approvedGeometry, y: Number.POSITIVE_INFINITY }, false],
    [{ ...approvedGeometry, width: 0 }, false],
    [{ ...approvedGeometry, height: 0 }, false],
    [{ ...approvedGeometry, x: 0.8, width: 0.3 }, false],
    [{ ...approvedGeometry, y: 0.8, height: 0.3 }, false],
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
