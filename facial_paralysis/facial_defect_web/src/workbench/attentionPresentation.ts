import type { HeatmapPoint, NormalizedRoi } from './types'

export type AttentionCellId =
  | 'upper-left'
  | 'upper-center'
  | 'upper-right'
  | 'middle-left'
  | 'middle-center'
  | 'middle-right'
  | 'lower-left'
  | 'lower-center'
  | 'lower-right'

export type AttentionCell = {
  readonly id: AttentionCellId
  readonly label: string
  readonly score: number
  readonly level: number
}

export type AttentionPresentationFailureReason =
  | 'EMPTY_HEATMAP'
  | 'INVALID_ROI'
  | 'INVALID_POINT'
  | 'POINT_OUTSIDE_ROI'

export type AttentionPresentation =
  | {
      readonly ok: true
      readonly cells: readonly AttentionCell[]
      readonly dominantCell?: AttentionCell
      readonly pointCount: number
    }
  | {
      readonly ok: false
      readonly reason: AttentionPresentationFailureReason
    }

const cellDefinitions = [
  ['upper-left', 'Upper left of result field'],
  ['upper-center', 'Upper center of result field'],
  ['upper-right', 'Upper right of result field'],
  ['middle-left', 'Middle left of result field'],
  ['middle-center', 'Center of result field'],
  ['middle-right', 'Middle right of result field'],
  ['lower-left', 'Lower left of result field'],
  ['lower-center', 'Lower center of result field'],
  ['lower-right', 'Lower right of result field'],
] as const satisfies readonly (readonly [AttentionCellId, string])[]

function isFiniteNormalized(value: number): boolean {
  return Number.isFinite(value) && value >= 0 && value <= 1
}

function isValidRoi(roi: Readonly<NormalizedRoi>): boolean {
  return (
    isFiniteNormalized(roi.x) &&
    isFiniteNormalized(roi.y) &&
    Number.isFinite(roi.width) &&
    Number.isFinite(roi.height) &&
    roi.width > 0 &&
    roi.height > 0 &&
    roi.width <= 1 &&
    roi.height <= 1 &&
    roi.x + roi.width <= 1 &&
    roi.y + roi.height <= 1
  )
}

function isValidPoint(point: Readonly<HeatmapPoint>): boolean {
  return (
    isFiniteNormalized(point.x) &&
    isFiniteNormalized(point.y) &&
    isFiniteNormalized(point.intensity) &&
    isFiniteNormalized(point.radius)
  )
}

function isPointInsideRoi(
  point: Readonly<HeatmapPoint>,
  roi: Readonly<NormalizedRoi>,
): boolean {
  return (
    point.x >= roi.x &&
    point.x <= roi.x + roi.width &&
    point.y >= roi.y &&
    point.y <= roi.y + roi.height
  )
}

export function deriveAttentionPresentation(
  heatmap: readonly HeatmapPoint[],
  roi: Readonly<NormalizedRoi>,
): AttentionPresentation {
  if (!isValidRoi(roi)) {
    return Object.freeze({ ok: false, reason: 'INVALID_ROI' })
  }

  if (heatmap.length === 0) {
    return Object.freeze({ ok: false, reason: 'EMPTY_HEATMAP' })
  }

  const scores = Array<number>(cellDefinitions.length).fill(0)

  for (const point of heatmap) {
    if (!isValidPoint(point)) {
      return Object.freeze({ ok: false, reason: 'INVALID_POINT' })
    }
    if (!isPointInsideRoi(point, roi)) {
      return Object.freeze({ ok: false, reason: 'POINT_OUTSIDE_ROI' })
    }

    const relativeX = (point.x - roi.x) / roi.width
    const relativeY = (point.y - roi.y) / roi.height
    const column = Math.min(2, Math.floor(relativeX * 3))
    const row = Math.min(2, Math.floor(relativeY * 3))
    scores[row * 3 + column] += point.intensity
  }

  const maxScore = Math.max(...scores)
  const cells = Object.freeze(
    cellDefinitions.map(([id, label], index) =>
      Object.freeze({
        id,
        label,
        score: scores[index],
        level: maxScore === 0 ? 0 : scores[index] / maxScore,
      }),
    ),
  )

  const dominantCell =
    maxScore === 0 ? undefined : cells[scores.findIndex((score) => score === maxScore)]

  return Object.freeze({
    ok: true,
    cells,
    dominantCell,
    pointCount: heatmap.length,
  })
}
