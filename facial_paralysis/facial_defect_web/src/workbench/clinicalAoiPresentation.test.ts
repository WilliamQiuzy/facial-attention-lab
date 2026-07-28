import { describe, expect, it } from 'vitest'
import { deriveClinicalAoiPresentation } from './clinicalAoiPresentation'
import type { HeatmapPoint, NormalizedRoi } from './types'

const imageBoundary: NormalizedRoi = {
  x: 0.2,
  y: 0.1,
  width: 0.6,
  height: 0.8,
}

function pointAtRelative(
  relativeX: number,
  relativeY: number,
  intensity: number,
  radius = 0.08,
): HeatmapPoint {
  return {
    x: imageBoundary.x + relativeX * imageBoundary.width,
    y: imageBoundary.y + relativeY * imageBoundary.height,
    intensity,
    radius,
  }
}

describe('clinical AOI presentation derivation', () => {
  it('maps viewer laterality to patient laterality and assigns the exact midline deterministically', () => {
    const result = deriveClinicalAoiPresentation(
      [
        pointAtRelative(0.2, 0.2, 0.1),
        pointAtRelative(0.8, 0.2, 0.3),
        pointAtRelative(0.5, 0.2, 0.2),
      ],
      imageBoundary,
    )

    expect(result.ok).toBe(true)
    if (!result.ok) return

    expect(result.orientation).toEqual({
      viewerLeft: 'patient_right',
      viewerRight: 'patient_left',
    })
    expect(result.hemifaces.patientLeftShare).toBeCloseTo(5 / 6)
    expect(result.hemifaces.patientRightShare).toBeCloseTo(1 / 6)
    expect(result.hemifaces.absoluteDifference).toBeCloseTo(4 / 6)
    expect(result.hemifaces.dominant).toBe('patient_left')
    expect(
      result.hemifaces.patientLeftShare + result.hemifaces.patientRightShare,
    ).toBeCloseTo(1)
  })

  it('returns the four clinical subsites in stable order with normalized mass shares', () => {
    const result = deriveClinicalAoiPresentation(
      [
        pointAtRelative(0.3, 0.2, 0.1),
        pointAtRelative(0.7, 0.4, 0.2),
        pointAtRelative(0.4, 0.55, 0.3),
        pointAtRelative(0.6, 0.75, 0.4),
      ],
      imageBoundary,
    )

    expect(result.ok).toBe(true)
    if (!result.ok) return

    expect(result.subsites).toEqual([
      { id: 'brow_forehead', label: 'Brow / forehead', share: 0.1 },
      { id: 'orbital', label: 'Orbital / eyes', share: 0.2 },
      { id: 'nasal_midface', label: 'Nasal / midface', share: 0.3 },
      { id: 'perioral', label: 'Perioral / mouth', share: 0.4 },
    ])
    expect(result.dominantSubsite).toEqual(result.subsites[3])
    expect(result.pointCount).toBe(4)
    expect(result.totalMass).toBeCloseTo(1)
  })

  it('computes the central triangle as an overlapping point-in-polygon AOI', () => {
    const result = deriveClinicalAoiPresentation(
      [
        pointAtRelative(0.5, 0.5, 0.6),
        pointAtRelative(0.1, 0.5, 0.2),
        pointAtRelative(0.28, 0.34, 0.2),
      ],
      imageBoundary,
    )

    expect(result.ok).toBe(true)
    if (!result.ok) return

    expect(result.centralTriangleShare).toBeCloseTo(0.8)
  })

  it('assigns all non-template mass to the outside remainder without double counting band edges', () => {
    const result = deriveClinicalAoiPresentation(
      [
        pointAtRelative(0.25, 0.34, 0.1),
        pointAtRelative(0.75, 0.48, 0.1),
        pointAtRelative(0.5, 0.66, 0.1),
        pointAtRelative(0.5, 0.84, 0.1),
        pointAtRelative(0.249, 0.2, 0.3),
        pointAtRelative(0.5, 0.159, 0.3),
      ],
      imageBoundary,
    )

    expect(result.ok).toBe(true)
    if (!result.ok) return

    expect(result.subsites.map((subsite) => subsite.share)).toEqual([
      0.1, 0.1, 0.1, 0.1,
    ])
    expect(result.outsideTemplateShare).toBeCloseTo(0.6)
    expect(
      result.subsites.reduce((sum, subsite) => sum + subsite.share, 0) +
        result.outsideTemplateShare,
    ).toBeCloseTo(1)
  })

  it('keeps all-zero fields conservative and does not invent dominance', () => {
    const result = deriveClinicalAoiPresentation(
      [
        pointAtRelative(0.3, 0.2, 0),
        pointAtRelative(0.7, 0.75, 0),
      ],
      imageBoundary,
    )

    expect(result.ok).toBe(true)
    if (!result.ok) return

    expect(result).toMatchObject({
      pointCount: 2,
      totalMass: 0,
      centralTriangleShare: 0,
      outsideTemplateShare: 0,
      hemifaces: {
        patientLeftShare: 0,
        patientRightShare: 0,
        absoluteDifference: 0,
        dominant: 'none',
      },
    })
    expect(result.subsites.every((subsite) => subsite.share === 0)).toBe(true)
    expect(result.dominantSubsite).toBeUndefined()
  })

  it('uses balanced hemifaces and stable ordered subsite tie breaking', () => {
    const result = deriveClinicalAoiPresentation(
      [
        pointAtRelative(0.3, 0.2, 0.5),
        pointAtRelative(0.7, 0.4, 0.5),
      ],
      imageBoundary,
    )

    expect(result.ok).toBe(true)
    if (!result.ok) return

    expect(result.hemifaces.dominant).toBe('balanced')
    expect(result.dominantSubsite?.id).toBe('brow_forehead')
  })

  it('returns frozen presentation data without mutating caller-owned inputs', () => {
    const heatmap = [
      pointAtRelative(0.3, 0.2, 0.4),
      pointAtRelative(0.7, 0.75, 0.6),
    ]
    const boundary = { ...imageBoundary }
    const heatmapSnapshot = structuredClone(heatmap)
    const boundarySnapshot = structuredClone(boundary)

    const result = deriveClinicalAoiPresentation(heatmap, boundary)

    expect(heatmap).toEqual(heatmapSnapshot)
    expect(boundary).toEqual(boundarySnapshot)
    expect(Object.isFrozen(result)).toBe(true)
    if (!result.ok) return
    expect(Object.isFrozen(result.orientation)).toBe(true)
    expect(Object.isFrozen(result.hemifaces)).toBe(true)
    expect(Object.isFrozen(result.subsites)).toBe(true)
    expect(result.subsites.every(Object.isFrozen)).toBe(true)
  })

  it.each([
    [
      'non-finite boundary',
      [pointAtRelative(0.5, 0.5, 0.5)],
      { ...imageBoundary, x: Number.NaN },
      'INVALID_BOUNDARY',
    ],
    [
      'infinite boundary',
      [pointAtRelative(0.5, 0.5, 0.5)],
      { ...imageBoundary, height: Number.POSITIVE_INFINITY },
      'INVALID_BOUNDARY',
    ],
    [
      'negative boundary origin',
      [pointAtRelative(0.5, 0.5, 0.5)],
      { ...imageBoundary, x: -0.01 },
      'INVALID_BOUNDARY',
    ],
    [
      'zero-area boundary',
      [pointAtRelative(0.5, 0.5, 0.5)],
      { ...imageBoundary, width: 0 },
      'INVALID_BOUNDARY',
    ],
    [
      'boundary outside image',
      [pointAtRelative(0.5, 0.5, 0.5)],
      { ...imageBoundary, x: 0.8, width: 0.3 },
      'INVALID_BOUNDARY',
    ],
  ] as const)(
    'fails closed for %s',
    (_label, heatmap, boundary, reason) => {
      expect(deriveClinicalAoiPresentation(heatmap, boundary)).toEqual({
        ok: false,
        reason,
      })
    },
  )

  it.each([
    ['NaN x', { ...pointAtRelative(0.5, 0.5, 0.5), x: Number.NaN }],
    [
      'infinite y',
      { ...pointAtRelative(0.5, 0.5, 0.5), y: Number.POSITIVE_INFINITY },
    ],
    ['negative x', { ...pointAtRelative(0.5, 0.5, 0.5), x: -0.01 }],
    ['x above one', { ...pointAtRelative(0.5, 0.5, 0.5), x: 1.01 }],
    [
      'negative intensity',
      { ...pointAtRelative(0.5, 0.5, 0.5), intensity: -0.01 },
    ],
    [
      'intensity above one',
      { ...pointAtRelative(0.5, 0.5, 0.5), intensity: 1.01 },
    ],
    [
      'negative radius',
      { ...pointAtRelative(0.5, 0.5, 0.5), radius: -0.01 },
    ],
    [
      'radius above one',
      { ...pointAtRelative(0.5, 0.5, 0.5), radius: 1.01 },
    ],
  ] as const)('rejects an invalid point with %s', (_label, invalidPoint) => {
    expect(deriveClinicalAoiPresentation([invalidPoint], imageBoundary)).toEqual({
      ok: false,
      reason: 'INVALID_POINT',
    })
  })

  it('rejects an empty field', () => {
    expect(deriveClinicalAoiPresentation([], imageBoundary)).toEqual({
      ok: false,
      reason: 'EMPTY_FIELD',
    })
  })

  it('rejects a valid normalized point outside the image boundary', () => {
    expect(
      deriveClinicalAoiPresentation(
        [{ x: 0.1, y: 0.5, intensity: 0.5, radius: 0.08 }],
        imageBoundary,
      ),
    ).toEqual({
      ok: false,
      reason: 'POINT_OUTSIDE_BOUNDARY',
    })
  })

  it('handles a micro-boundary and inclusive image-boundary edges', () => {
    const microBoundary = {
      x: 0.4,
      y: 0.4,
      width: 0.000001,
      height: 0.000001,
    }
    const result = deriveClinicalAoiPresentation(
      [
        {
          x: microBoundary.x + microBoundary.width,
          y: microBoundary.y + microBoundary.height,
          intensity: 1,
          radius: 0,
        },
      ],
      microBoundary,
    )

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.hemifaces.dominant).toBe('patient_left')
    expect(result.outsideTemplateShare).toBe(1)
  })

  it('preserves exact AOI band boundaries inside a micro-boundary', () => {
    const microBoundary = {
      x: 0.4,
      y: 0.4,
      width: 0.000001,
      height: 0.000001,
    }
    const atRelative = (relativeY: number): HeatmapPoint => ({
      x: microBoundary.x + 0.5 * microBoundary.width,
      y: microBoundary.y + relativeY * microBoundary.height,
      intensity: 0.25,
      radius: 0,
    })

    const result = deriveClinicalAoiPresentation(
      [atRelative(0.34), atRelative(0.48), atRelative(0.66), atRelative(0.84)],
      microBoundary,
    )

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.subsites.map((subsite) => subsite.share)).toEqual([
      0.25, 0.25, 0.25, 0.25,
    ])
    expect(result.outsideTemplateShare).toBe(0)
  })

  it('keeps central-triangle geometry stable for a tiny non-square boundary', () => {
    const tinyBoundary = {
      x: 0.4,
      y: 0.4,
      width: 0.00000000000001,
      height: 0.00000000000002,
    }
    const atRelative = (
      relativeX: number,
      relativeY: number,
      intensity: number,
    ): HeatmapPoint => ({
      x: tinyBoundary.x + relativeX * tinyBoundary.width,
      y: tinyBoundary.y + relativeY * tinyBoundary.height,
      intensity,
      radius: 0,
    })

    const result = deriveClinicalAoiPresentation(
      [
        atRelative(0.5, 0.5, 0.3),
        atRelative(0.39, 0.59, 0.2),
        atRelative(0.05, 0.75, 0.5),
      ],
      tinyBoundary,
    )

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.centralTriangleShare).toBeCloseTo(0.5)
  })

  it('does not expand the central triangle across a nearby tiny-boundary gap', () => {
    const tinyBoundary = {
      x: 0.4,
      y: 0.4,
      width: 0.00000000000001,
      height: 0.00000000000002,
    }
    const atRelative = (
      relativeX: number,
      relativeY: number,
      intensity: number,
    ): HeatmapPoint => ({
      x: tinyBoundary.x + relativeX * tinyBoundary.width,
      y: tinyBoundary.y + relativeY * tinyBoundary.height,
      intensity,
      radius: 0,
    })

    const result = deriveClinicalAoiPresentation(
      [
        atRelative(0.28, 0.34, 0.3),
        atRelative(0.281, 0.341, 0.2),
        atRelative(0.25, 0.34, 0.5),
      ],
      tinyBoundary,
    )

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.centralTriangleShare).toBeCloseTo(0.5)
  })

  it('rejects positive boundary dimensions whose endpoints are not representable', () => {
    const unrepresentableBoundary = {
      x: 1,
      y: 1,
      width: Number.MIN_VALUE,
      height: Number.MIN_VALUE,
    }

    expect(
      deriveClinicalAoiPresentation(
        [{ x: 1, y: 1, intensity: 0, radius: 0 }],
        unrepresentableBoundary,
      ),
    ).toEqual({
      ok: false,
      reason: 'INVALID_BOUNDARY',
    })
  })

  it('classifies endpoints of the smallest representable boundary in relative space', () => {
    const smallestBoundary = {
      x: 0.4,
      y: 0.4,
      width: Number.EPSILON / 4,
      height: Number.EPSILON / 4,
    }
    const result = deriveClinicalAoiPresentation(
      [
        {
          x: smallestBoundary.x,
          y: smallestBoundary.y,
          intensity: 0.4,
          radius: 0,
        },
        {
          x: smallestBoundary.x + smallestBoundary.width,
          y: smallestBoundary.y + smallestBoundary.height,
          intensity: 0.6,
          radius: 0,
        },
      ],
      smallestBoundary,
    )

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.hemifaces.patientRightShare).toBeCloseTo(0.4)
    expect(result.hemifaces.patientLeftShare).toBeCloseTo(0.6)
    expect(result.hemifaces.dominant).toBe('patient_left')
    expect(result.subsites.every((subsite) => subsite.share === 0)).toBe(true)
    expect(result.outsideTemplateShare).toBe(1)
  })

  it('assigns a tiny-boundary midpoint leftward without expanding across a clear gap', () => {
    const tinyBoundary = {
      x: 0.4,
      y: 0.4,
      width: 0.00000000000001,
      height: 0.00000000000002,
    }
    const atRelative = (
      relativeX: number,
      intensity: number,
    ): HeatmapPoint => ({
      x: tinyBoundary.x + relativeX * tinyBoundary.width,
      y: tinyBoundary.y,
      intensity,
      radius: 0,
    })

    const result = deriveClinicalAoiPresentation(
      [atRelative(0.5, 0.6), atRelative(0.48, 0.4)],
      tinyBoundary,
    )

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.hemifaces.patientLeftShare).toBeCloseTo(0.6)
    expect(result.hemifaces.patientRightShare).toBeCloseTo(0.4)
    expect(result.hemifaces.dominant).toBe('patient_left')
  })

  it('does not let narrow-x uncertainty expand y-based subsites or triangle edges', () => {
    const tallNarrowBoundary = {
      x: 0.4,
      y: 0,
      width: 0.000000000000005,
      height: 1,
    }
    const atRelative = (
      relativeX: number,
      relativeY: number,
      intensity: number,
    ): HeatmapPoint => ({
      x: tallNarrowBoundary.x + relativeX * tallNarrowBoundary.width,
      y: relativeY,
      intensity,
      radius: 0,
    })

    const subsiteResult = deriveClinicalAoiPresentation(
      [atRelative(0.5, 0.151, 0.5), atRelative(0.5, 0.2, 0.5)],
      tallNarrowBoundary,
    )
    expect(subsiteResult.ok).toBe(true)
    if (!subsiteResult.ok) return
    expect(subsiteResult.subsites[0].share).toBeCloseTo(0.5)
    expect(subsiteResult.outsideTemplateShare).toBeCloseTo(0.5)

    const triangleResult = deriveClinicalAoiPresentation(
      [atRelative(0.5, 0.32, 0.5), atRelative(0.5, 0.5, 0.5)],
      tallNarrowBoundary,
    )
    expect(triangleResult.ok).toBe(true)
    if (!triangleResult.ok) return
    expect(triangleResult.centralTriangleShare).toBeCloseTo(0.5)
  })

  it('treats floating-point mathematical mass ties as balanced and stable', () => {
    const result = deriveClinicalAoiPresentation(
      [
        pointAtRelative(0.3, 0.2, 0.3),
        pointAtRelative(0.7, 0.4, 0.1),
        pointAtRelative(0.7, 0.4, 0.2),
      ],
      imageBoundary,
    )

    expect(result.ok).toBe(true)
    if (!result.ok) return
    expect(result.hemifaces.dominant).toBe('balanced')
    expect(result.dominantSubsite?.id).toBe('brow_forehead')
  })

  it('fails closed instead of throwing for malformed runtime containers', () => {
    expect(
      deriveClinicalAoiPresentation(
        null as unknown as readonly HeatmapPoint[],
        imageBoundary,
      ),
    ).toEqual({ ok: false, reason: 'INVALID_POINT' })
    expect(
      deriveClinicalAoiPresentation(
        {} as unknown as readonly HeatmapPoint[],
        imageBoundary,
      ),
    ).toEqual({ ok: false, reason: 'INVALID_POINT' })
    expect(
      deriveClinicalAoiPresentation(
        [null as unknown as HeatmapPoint],
        imageBoundary,
      ),
    ).toEqual({ ok: false, reason: 'INVALID_POINT' })
  })

  it('fails closed instead of throwing for a null runtime boundary', () => {
    expect(
      deriveClinicalAoiPresentation(
        [pointAtRelative(0.5, 0.5, 0.5)],
        null as unknown as NormalizedRoi,
      ),
    ).toEqual({ ok: false, reason: 'INVALID_BOUNDARY' })
  })
})
