import type { HeatmapPoint, NormalizedRoi } from './types'

export type ClinicalAoiSubsiteId =
  | 'brow_forehead'
  | 'orbital'
  | 'nasal_midface'
  | 'perioral'

export type ClinicalAoiSubsite = Readonly<{
  readonly id: ClinicalAoiSubsiteId
  readonly label: string
  readonly share: number
}>

export type ClinicalAoiPresentationFailureReason =
  | 'EMPTY_FIELD'
  | 'INVALID_BOUNDARY'
  | 'INVALID_POINT'
  | 'POINT_OUTSIDE_BOUNDARY'

export type ClinicalAoiPresentation =
  | Readonly<{
      readonly ok: true
      readonly pointCount: number
      readonly totalMass: number
      readonly orientation: Readonly<{
        readonly viewerLeft: 'patient_right'
        readonly viewerRight: 'patient_left'
      }>
      readonly centralTriangleShare: number
      readonly hemifaces: Readonly<{
        readonly patientLeftShare: number
        readonly patientRightShare: number
        readonly absoluteDifference: number
        readonly dominant:
          | 'patient_left'
          | 'patient_right'
          | 'balanced'
          | 'none'
      }>
      readonly subsites: readonly ClinicalAoiSubsite[]
      readonly outsideTemplateShare: number
      readonly dominantSubsite?: ClinicalAoiSubsite
    }>
  | Readonly<{
      readonly ok: false
      readonly reason: ClinicalAoiPresentationFailureReason
    }>

type SubsiteDefinition = Readonly<{
  readonly id: ClinicalAoiSubsiteId
  readonly label: string
  readonly minimumY: number
  readonly maximumY: number
  readonly includesMinimumY: boolean
}>

type BoundaryRelativePoint = Readonly<{
  readonly x: number
  readonly y: number
}>

type NormalizationUncertainty = Readonly<{
  readonly x: number
  readonly y: number
}>

const SUBSITE_DEFINITIONS = Object.freeze([
  Object.freeze({
    id: 'brow_forehead',
    label: 'Brow / forehead',
    minimumY: 0.16,
    maximumY: 0.34,
    includesMinimumY: true,
  }),
  Object.freeze({
    id: 'orbital',
    label: 'Orbital / eyes',
    minimumY: 0.34,
    maximumY: 0.48,
    includesMinimumY: false,
  }),
  Object.freeze({
    id: 'nasal_midface',
    label: 'Nasal / midface',
    minimumY: 0.48,
    maximumY: 0.66,
    includesMinimumY: false,
  }),
  Object.freeze({
    id: 'perioral',
    label: 'Perioral / mouth',
    minimumY: 0.66,
    maximumY: 0.84,
    includesMinimumY: false,
  }),
] as const satisfies readonly SubsiteDefinition[])

const ORIENTATION = Object.freeze({
  viewerLeft: 'patient_right',
  viewerRight: 'patient_left',
} as const)

const CENTRAL_TRIANGLE = Object.freeze([
  Object.freeze({ x: 0.28, y: 0.34 }),
  Object.freeze({ x: 0.72, y: 0.34 }),
  Object.freeze({ x: 0.5, y: 0.84 }),
] as const)

function isFiniteNormalized(value: number): boolean {
  return Number.isFinite(value) && value >= 0 && value <= 1
}

function isValidBoundary(
  boundary: unknown,
): boundary is Readonly<NormalizedRoi> {
  if (typeof boundary !== 'object' || boundary === null) return false
  const candidate = boundary as Record<string, unknown>
  if (
    typeof candidate.x !== 'number' ||
    typeof candidate.y !== 'number' ||
    typeof candidate.width !== 'number' ||
    typeof candidate.height !== 'number'
  ) {
    return false
  }
  const { x, y, width, height } = candidate
  const maximumX = x + width
  const maximumY = y + height
  return (
    isFiniteNormalized(x) &&
    isFiniteNormalized(y) &&
    Number.isFinite(width) &&
    Number.isFinite(height) &&
    width > 0 &&
    height > 0 &&
    width <= 1 &&
    height <= 1 &&
    width <= 1 - x &&
    height <= 1 - y &&
    maximumX > x &&
    maximumY > y
  )
}

function isValidPoint(point: unknown): point is Readonly<HeatmapPoint> {
  if (typeof point !== 'object' || point === null) return false
  const candidate = point as Record<string, unknown>
  if (
    typeof candidate.x !== 'number' ||
    typeof candidate.y !== 'number' ||
    typeof candidate.intensity !== 'number' ||
    typeof candidate.radius !== 'number'
  ) {
    return false
  }
  return (
    isFiniteNormalized(candidate.x) &&
    isFiniteNormalized(candidate.y) &&
    isFiniteNormalized(candidate.intensity) &&
    isFiniteNormalized(candidate.radius)
  )
}

function isInsideBoundary(
  point: Readonly<HeatmapPoint>,
  boundary: Readonly<NormalizedRoi>,
): boolean {
  return (
    point.x >= boundary.x &&
    point.x <= boundary.x + boundary.width &&
    point.y >= boundary.y &&
    point.y <= boundary.y + boundary.height
  )
}

function findSubsiteIndex(
  relativePoint: BoundaryRelativePoint,
  uncertainty: NormalizationUncertainty,
): number {
  const relativeX = snapToBoundary(
    relativePoint.x,
    [0.25, 0.75],
    uncertainty.x,
  )
  if (relativeX < 0.25 || relativeX > 0.75) {
    return -1
  }

  const relativeY = snapToBoundary(
    relativePoint.y,
    [0.16, 0.34, 0.48, 0.66, 0.84],
    uncertainty.y,
  )
  return SUBSITE_DEFINITIONS.findIndex((definition) => {
    const isAboveMinimum = definition.includesMinimumY
      ? relativeY >= definition.minimumY
      : relativeY > definition.minimumY
    return isAboveMinimum && relativeY <= definition.maximumY
  })
}

function snapToBoundary(
  value: number,
  boundaries: readonly number[],
  tolerance: number,
): number {
  return (
    boundaries.find((boundary) => Math.abs(value - boundary) <= tolerance) ??
    value
  )
}

function crossProduct(
  start: Readonly<{ x: number; y: number }>,
  end: Readonly<{ x: number; y: number }>,
  point: Readonly<{ x: number; y: number }>,
): number {
  return (
    (end.x - start.x) * (point.y - start.y) -
    (end.y - start.y) * (point.x - start.x)
  )
}

function isInsideCentralTriangle(
  relativePoint: BoundaryRelativePoint,
  uncertainty: NormalizationUncertainty,
): boolean {
  const edges = [
    [CENTRAL_TRIANGLE[0], CENTRAL_TRIANGLE[1]],
    [CENTRAL_TRIANGLE[1], CENTRAL_TRIANGLE[2]],
    [CENTRAL_TRIANGLE[2], CENTRAL_TRIANGLE[0]],
  ] as const
  const crosses = edges.map(([start, end]) =>
    crossProduct(start, end, relativePoint),
  )
  const tolerances = edges.map(
    ([start, end]) =>
      Math.abs(end.x - start.x) * uncertainty.y +
      Math.abs(end.y - start.y) * uncertainty.x +
      Number.EPSILON * 8,
  )
  const hasNegative =
    crosses[0] < -tolerances[0] ||
    crosses[1] < -tolerances[1] ||
    crosses[2] < -tolerances[2]
  const hasPositive =
    crosses[0] > tolerances[0] ||
    crosses[1] > tolerances[1] ||
    crosses[2] > tolerances[2]
  return !(hasNegative && hasPositive)
}

function getNormalizationUncertainty(
  boundary: Readonly<NormalizedRoi>,
): NormalizationUncertainty {
  const axisUncertainty = (origin: number, extent: number): number =>
    Math.min(
      0.01,
      (Number.EPSILON *
        Math.max(Math.abs(origin), extent, Number.MIN_VALUE)) /
        extent,
    )
  return {
    x: axisUncertainty(boundary.x, boundary.width),
    y: axisUncertainty(boundary.y, boundary.height),
  }
}

function areNearlyEqual(
  first: number,
  second: number,
  scale: number,
): boolean {
  return (
    Math.abs(first - second) <=
    Number.EPSILON *
      8 *
      Math.max(Math.abs(first), Math.abs(second), Math.abs(scale), Number.MIN_VALUE)
  )
}

function normalizedShare(mass: number, totalMass: number): number {
  if (totalMass === 0) return 0
  return Math.min(1, Math.max(0, mass / totalMass))
}

function failure(
  reason: ClinicalAoiPresentationFailureReason,
): ClinicalAoiPresentation {
  return Object.freeze({ ok: false, reason })
}

export function deriveClinicalAoiPresentation(
  heatmap: readonly HeatmapPoint[],
  imageBoundary: Readonly<NormalizedRoi>,
): ClinicalAoiPresentation {
  if (!isValidBoundary(imageBoundary)) {
    return failure('INVALID_BOUNDARY')
  }
  if (!Array.isArray(heatmap)) {
    return failure('INVALID_POINT')
  }
  if (heatmap.length === 0) {
    return failure('EMPTY_FIELD')
  }

  const subsiteMasses = Array<number>(SUBSITE_DEFINITIONS.length).fill(0)
  let totalMass = 0
  let patientLeftMass = 0
  let patientRightMass = 0
  let centralTriangleMass = 0
  let outsideTemplateMass = 0
  const normalizationUncertainty =
    getNormalizationUncertainty(imageBoundary)

  for (const point of heatmap) {
    if (!isValidPoint(point)) {
      return failure('INVALID_POINT')
    }
    if (!isInsideBoundary(point, imageBoundary)) {
      return failure('POINT_OUTSIDE_BOUNDARY')
    }

    const mass = point.intensity
    const relativePoint = {
      x: (point.x - imageBoundary.x) / imageBoundary.width,
      y: (point.y - imageBoundary.y) / imageBoundary.height,
    }
    totalMass += mass

    const hemifaceX = snapToBoundary(
      relativePoint.x,
      [0.5],
      normalizationUncertainty.x,
    )
    if (hemifaceX >= 0.5) {
      patientLeftMass += mass
    } else {
      patientRightMass += mass
    }

    if (isInsideCentralTriangle(relativePoint, normalizationUncertainty)) {
      centralTriangleMass += mass
    }

    const subsiteIndex = findSubsiteIndex(
      relativePoint,
      normalizationUncertainty,
    )
    if (subsiteIndex === -1) {
      outsideTemplateMass += mass
    } else {
      subsiteMasses[subsiteIndex] += mass
    }
  }

  const patientLeftShare = normalizedShare(patientLeftMass, totalMass)
  const patientRightShare = normalizedShare(patientRightMass, totalMass)
  const hemifaceDominant =
    totalMass === 0
      ? 'none'
      : areNearlyEqual(patientLeftMass, patientRightMass, totalMass)
        ? 'balanced'
        : patientLeftMass > patientRightMass
          ? 'patient_left'
          : 'patient_right'

  const subsites = Object.freeze(
    SUBSITE_DEFINITIONS.map((definition, index) =>
      Object.freeze({
        id: definition.id,
        label: definition.label,
        share: normalizedShare(subsiteMasses[index], totalMass),
      }),
    ),
  )

  const dominantSubsiteIndex = subsiteMasses.reduce(
    (dominantIndex, mass, index) =>
      mass > subsiteMasses[dominantIndex] &&
      !areNearlyEqual(mass, subsiteMasses[dominantIndex], totalMass)
        ? index
        : dominantIndex,
    0,
  )
  const dominantSubsite =
    subsiteMasses[dominantSubsiteIndex] > 0
      ? subsites[dominantSubsiteIndex]
      : undefined

  return Object.freeze({
    ok: true,
    pointCount: heatmap.length,
    totalMass,
    orientation: ORIENTATION,
    centralTriangleShare: normalizedShare(centralTriangleMass, totalMass),
    hemifaces: Object.freeze({
      patientLeftShare,
      patientRightShare,
      absoluteDifference: Math.abs(patientLeftShare - patientRightShare),
      dominant: hemifaceDominant,
    }),
    subsites,
    outsideTemplateShare: normalizedShare(outsideTemplateMass, totalMass),
    dominantSubsite,
  })
}
