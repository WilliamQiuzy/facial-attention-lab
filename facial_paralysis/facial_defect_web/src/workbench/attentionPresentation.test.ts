import { describe, expect, it } from 'vitest'
import { deriveAttentionPresentation } from './attentionPresentation'
import type { HeatmapPoint, NormalizedRoi } from './types'

const roi: NormalizedRoi = {
  x: 0.2,
  y: 0.1,
  width: 0.6,
  height: 0.6,
}

function point(
  x: number,
  y: number,
  intensity: number,
  radius = 0.08,
): HeatmapPoint {
  return { x, y, intensity, radius }
}

describe('attention presentation derivation', () => {
  it('aggregates intensity into deterministic ROI-relative cells without using radius', () => {
    const result = deriveAttentionPresentation(
      [
        point(0.25, 0.15, 0.2, 0.01),
        point(0.7, 0.15, 0.8, 0.02),
        point(0.75, 0.2, 0.4, 0.4),
        point(0.5, 0.4, 0.6, 0.08),
      ],
      roi,
    )

    expect(result.ok).toBe(true)
    if (!result.ok) return

    expect(result.cells).toHaveLength(9)
    expect(result.cells.find((cell) => cell.id === 'upper-left')?.score).toBeCloseTo(
      0.2,
    )
    expect(result.cells.find((cell) => cell.id === 'upper-right')?.score).toBeCloseTo(
      1.2,
    )
    expect(result.cells.find((cell) => cell.id === 'middle-center')?.score).toBeCloseTo(
      0.6,
    )
    expect(result.dominantCell?.id).toBe('upper-right')
    expect(result.dominantCell?.label).toBe('Upper right of result field')
  })

  it('uses stable row-major tie breaking and assigns ROI boundaries to final cells', () => {
    const result = deriveAttentionPresentation(
      [point(roi.x, roi.y, 0.7), point(roi.x + roi.width, roi.y + roi.height, 0.7)],
      roi,
    )

    expect(result.ok).toBe(true)
    if (!result.ok) return

    expect(result.dominantCell?.id).toBe('upper-left')
    expect(result.cells.find((cell) => cell.id === 'lower-right')?.score).toBe(0.7)
  })

  it('does not invent a dominant cell when total intensity is zero', () => {
    const result = deriveAttentionPresentation(
      [point(0.25, 0.15, 0), point(0.7, 0.65, 0)],
      roi,
    )

    expect(result.ok).toBe(true)
    if (!result.ok) return

    expect(result.dominantCell).toBeUndefined()
    expect(result.cells.every((cell) => cell.level === 0)).toBe(true)
  })

  it('derives the same summary for every contract-valid display radius, including zero', () => {
    const withoutSmoothing = deriveAttentionPresentation(
      [point(0.25, 0.15, 0.4, 0), point(0.7, 0.65, 0.9, 0)],
      roi,
    )
    const withSmoothing = deriveAttentionPresentation(
      [point(0.25, 0.15, 0.4, 1), point(0.7, 0.65, 0.9, 0.6)],
      roi,
    )

    expect(withoutSmoothing).toEqual(withSmoothing)
  })

  it('does not mutate caller-owned heatmap points or ROI geometry', () => {
    const heatmap = [point(0.25, 0.15, 0.5), point(0.7, 0.65, 0.9)]
    const mutableRoi = { ...roi }
    const heatmapSnapshot = structuredClone(heatmap)
    const roiSnapshot = structuredClone(mutableRoi)

    deriveAttentionPresentation(heatmap, mutableRoi)

    expect(heatmap).toEqual(heatmapSnapshot)
    expect(mutableRoi).toEqual(roiSnapshot)
  })

  it.each([
    ['empty heatmap', [], roi, 'EMPTY_HEATMAP'],
    [
      'non-finite point',
      [point(Number.NaN, 0.2, 0.5)],
      roi,
      'INVALID_POINT',
    ],
    ['negative intensity', [point(0.3, 0.2, -0.1)], roi, 'INVALID_POINT'],
    ['point outside normalized image', [point(1.1, 0.2, 0.5)], roi, 'INVALID_POINT'],
    ['point outside approved ROI', [point(0.1, 0.2, 0.5)], roi, 'POINT_OUTSIDE_ROI'],
    [
      'non-finite ROI',
      [point(0.3, 0.2, 0.5)],
      { ...roi, x: Number.POSITIVE_INFINITY },
      'INVALID_ROI',
    ],
    ['zero-area ROI', [point(0.3, 0.2, 0.5)], { ...roi, width: 0 }, 'INVALID_ROI'],
    [
      'ROI outside normalized image',
      [point(0.3, 0.2, 0.5)],
      { ...roi, x: 0.8, width: 0.3 },
      'INVALID_ROI',
    ],
  ] as const)('fails closed for %s', (_label, heatmap, geometry, reason) => {
    expect(deriveAttentionPresentation(heatmap, geometry)).toEqual({
      ok: false,
      reason,
    })
  })
})
