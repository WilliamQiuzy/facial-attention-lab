import { describe, expect, it, vi } from 'vitest'
import { listWorkbenchAssets } from './catalog'
import {
  createMockModelComparison,
  getExactApprovedComparisonRoi,
  parseStrictModelComparisonQuery,
} from './modelComparison'
import { createInitialWorkspaceState } from './reducer'
import {
  WorkbenchError,
  type RoiAnnotation,
  type WorkspaceState,
} from './types'

const catalog = listWorkbenchAssets()
const approvedCase = catalog[2]

function createExplicitDraftState(): WorkspaceState {
  const state = createInitialWorkspaceState()
  const defaultRoi = state.roisByCase[catalog[0].id]!
  const { reviewerId: _reviewerId, ...draftBase } = defaultRoi
  const draftRoi: RoiAnnotation = { ...draftBase, status: 'draft' }
  return {
    ...state,
    roisByCase: { ...state.roisByCase, [catalog[0].id]: draftRoi },
  }
}

describe('strict same-case model comparison', () => {
  it('accepts exactly one canonical case query and rejects every ambiguous form', () => {
    expect(parseStrictModelComparisonQuery(`?case=${approvedCase.id}`)).toBe(
      approvedCase.id,
    )

    for (const search of [
      '',
      '?case=',
      '?case=UNKNOWN-CASE',
      `?case=${approvedCase.id}&case=${approvedCase.id}`,
      `?case=${approvedCase.id}&case=${catalog[3].id}`,
      `?case=${approvedCase.id}&comparisonCase=${catalog[3].id}`,
    ]) {
      expect(() => parseStrictModelComparisonQuery(search)).toThrow(WorkbenchError)
    }
  })

  it('pins both simulation versions to one exact asset, full-image bound, and configuration', () => {
    const comparison = createMockModelComparison({
      workspaceState: createInitialWorkspaceState(),
      caseId: approvedCase.id,
      config: { threshold: 0.45, smoothing: 0.3 },
    })

    expect(comparison.caseId).toBe(approvedCase.id)
    expect(comparison.assetSha256).toBe(approvedCase.sha256)
    expect(comparison.left.binding).toMatchObject({
      caseId: approvedCase.id,
      assetId: approvedCase.id,
      assetSha256: approvedCase.sha256,
      roiId: comparison.roi.id,
      roiVersion: comparison.roi.version,
      roiGeometry: comparison.roi.geometry,
      modelVersion: 'mock-salience-v0.3',
      config: comparison.config,
    })
    expect(comparison.right.binding).toMatchObject({
      caseId: approvedCase.id,
      assetId: approvedCase.id,
      assetSha256: approvedCase.sha256,
      roiId: comparison.roi.id,
      roiVersion: comparison.roi.version,
      roiGeometry: comparison.roi.geometry,
      modelVersion: 'mock-salience-v0.4',
      config: comparison.config,
    })
    expect(comparison.left.binding.inputFingerprint).not.toBe(
      comparison.right.binding.inputFingerprint,
    )
    expect(Object.isFrozen(comparison)).toBe(true)
    expect(Object.isFrozen(comparison.roi)).toBe(true)
    expect(Object.isFrozen(comparison.clinicalAoiMethod)).toBe(true)
    expect(Object.isFrozen(comparison.clinicalAoiGroups)).toBe(true)
  })

  it('compares center-assigned simulated point weights in three non-interchangeable groups', () => {
    const input = {
      workspaceState: createInitialWorkspaceState(),
      caseId: approvedCase.id,
      config: { threshold: 0.45, smoothing: 0.3 },
    } as const
    const first = createMockModelComparison(input)
    const replay = createMockModelComparison(input)

    expect(replay).toEqual(first)
    expect(first.clinicalAoiMethod).toEqual({
      schemaVersion: 'synthetic-point-weight-aoi/1',
      template: 'fixed_anatomical_template',
      weight: 'point_intensity',
      assignment: 'point_center',
      radiusContribution: 'ignored',
      purpose: 'ui_rehearsal_only',
    })
    expect(
      first.clinicalAoiGroups.map((group) => [
        group.id,
        group.label,
        group.relationship,
      ]),
    ).toEqual([
      ['subsite_partition', 'Facial subsite partition', 'partition_total_1'],
      ['hemiface_partition', 'Hemiface partition', 'partition_total_1'],
      [
        'central_triangle_reference',
        'Overlapping reference',
        'overlapping_non_additive',
      ],
    ])

    const [subsites, hemifaces, centralTriangle] = first.clinicalAoiGroups
    expect(subsites.rows.map((row) => [row.key, row.label])).toEqual([
      ['brow_forehead', 'Brow / forehead'],
      ['orbital', 'Orbital / eyes'],
      ['nasal_midface', 'Nasal / midface'],
      ['perioral', 'Perioral / mouth'],
      ['outside_template', 'Outside fixed template'],
    ])
    expect(hemifaces.rows.map((row) => [row.key, row.label])).toEqual([
      ['patient_left_hemiface', 'Patient-left hemiface'],
      ['patient_right_hemiface', 'Patient-right hemiface'],
    ])
    expect(centralTriangle.rows.map((row) => [row.key, row.label])).toEqual([
      ['central_triangle', 'Central facial triangle'],
    ])

    for (const group of first.clinicalAoiGroups) {
      expect(Object.isFrozen(group)).toBe(true)
      expect(Object.isFrozen(group.rows)).toBe(true)
      for (const row of group.rows) {
        expect(row.versionAShare).toBeGreaterThanOrEqual(0)
        expect(row.versionAShare).toBeLessThanOrEqual(1)
        expect(row.versionBShare).toBeGreaterThanOrEqual(0)
        expect(row.versionBShare).toBeLessThanOrEqual(1)
        expect(row.versionBMinusA).toBeCloseTo(
          row.versionBShare - row.versionAShare,
          6,
        )
        expect(Object.isFrozen(row)).toBe(true)
      }
    }
    for (const partition of [subsites, hemifaces]) {
      expect(
        partition.rows.reduce((total, row) => total + row.versionAShare, 0),
      ).toBeCloseTo(1, 12)
      expect(
        partition.rows.reduce((total, row) => total + row.versionBShare, 0),
      ).toBeCloseTo(1, 12)
    }
    expect(first).not.toHaveProperty('metricDeltas')
    expect(first).not.toHaveProperty('clinicalAoiRows')
    expect(JSON.stringify(first.clinicalAoiGroups)).not.toMatch(
      /roiCoverage|peakIntensity|meanIntensity|focusScore|severity|fixation/i,
    )
  })

  it('uses post-inference AOI semantics without changing either spatial prediction', () => {
    const comparison = createMockModelComparison({
      workspaceState: createInitialWorkspaceState(),
      caseId: approvedCase.id,
      config: { threshold: 0.45, smoothing: 0.3 },
    })

    for (const output of [comparison.left, comparison.right]) {
      expect(output.attentionSemantics).toMatchObject({
        fieldMeaning: 'relative_spatial_density',
        target: 'predicted_observer_attention',
        interpretation: 'population_level',
        clinicalAoi: {
          registration: 'synthetic_template_v1',
          role: 'post_inference_summary',
          modifiesPrediction: false,
        },
      })
    }
  })

  it('fails closed when the exact case source binding is not verified', () => {
    expect(() =>
      createMockModelComparison({
        workspaceState: createExplicitDraftState(),
        caseId: catalog[0].id,
        config: { threshold: 0.45, smoothing: 0.3 },
      }),
    ).toThrowError(
      expect.objectContaining({ reason: 'FULL_IMAGE_SOURCE_BINDING_REQUIRED' }),
    )
  })

  it('fails closed when an approved annotation is not the immutable full-image bound', () => {
    const state = createInitialWorkspaceState()
    const partial = {
      ...state.roisByCase[approvedCase.id]!,
      geometry: { x: 0.1, y: 0.1, width: 0.8, height: 0.8 },
    } as const
    const partialState: WorkspaceState = {
      ...state,
      roisByCase: { ...state.roisByCase, [approvedCase.id]: partial },
    }

    expect(
      getExactApprovedComparisonRoi(partialState, approvedCase.id),
    ).toBeUndefined()
    expect(() =>
      createMockModelComparison({
        workspaceState: partialState,
        caseId: approvedCase.id,
        config: { threshold: 0.45, smoothing: 0.3 },
      }),
    ).toThrowError(
      expect.objectContaining({ reason: 'FULL_IMAGE_SOURCE_BINDING_REQUIRED' }),
    )
  })

  it('does not access fetch or browser storage', () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    const storageSpy = vi.spyOn(Storage.prototype, 'setItem')

    createMockModelComparison({
      workspaceState: createInitialWorkspaceState(),
      caseId: approvedCase.id,
      config: { threshold: 0.45, smoothing: 0.3 },
    })

    expect(fetchSpy).not.toHaveBeenCalled()
    expect(storageSpy).not.toHaveBeenCalled()
    fetchSpy.mockRestore()
    storageSpy.mockRestore()
  })
})
