import { describe, expect, it } from 'vitest'
import {
  presentationAttentionBySubject,
  presentationAttentionSummary,
} from './presentationAttention'
import { presentationSubjectIds } from '../data/presentationDemoAssets'

function integratedSignal(
  points: readonly {
    x: number
    y: number
    radius: number
    intensity: number
  }[],
  inRegion: (point: { x: number; y: number }) => boolean,
): number {
  return points
    .filter(inRegion)
    .reduce(
      (total, point) =>
        total + Math.PI * point.radius ** 2 * point.intensity,
      0,
    )
}

describe('presentation attention story', () => {
  it.each(presentationSubjectIds)('uses finite aligned sampling geometry for %s', (subjectId) => {
    const before = presentationAttentionBySubject[subjectId].preoperative
    const after = presentationAttentionBySubject[subjectId].postoperative

    expect(after).toHaveLength(before.length)
    expect(
      after.map(({ x, y, radius }) => ({ x, y, radius })),
    ).toEqual(
      before.map(({ x, y, radius }) => ({ x, y, radius })),
    )
    for (const point of [...before, ...after]) {
      expect(Number.isFinite(point.x)).toBe(true)
      expect(Number.isFinite(point.y)).toBe(true)
      expect(Number.isFinite(point.radius)).toBe(true)
      expect(Number.isFinite(point.intensity)).toBe(true)
      expect(point.x).toBeGreaterThanOrEqual(0)
      expect(point.x).toBeLessThanOrEqual(1)
      expect(point.y).toBeGreaterThanOrEqual(0)
      expect(point.y).toBeLessThanOrEqual(1)
      expect(point.radius).toBeGreaterThan(0)
      expect(point.intensity).toBeGreaterThanOrEqual(0)
      expect(point.intensity).toBeLessThanOrEqual(1)
    }
  })

  it.each(presentationSubjectIds)('reduces %s cheek signal by at least 45 percent while keeping a small signal', (subjectId) => {
    const isTargetCheek = ({ x, y }: { x: number; y: number }) =>
      x >= 0.58 && x <= 0.72 && y >= 0.56 && y <= 0.69
    const before = integratedSignal(
      presentationAttentionBySubject[subjectId].preoperative,
      isTargetCheek,
    )
    const after = integratedSignal(
      presentationAttentionBySubject[subjectId].postoperative,
      isTargetCheek,
    )

    expect(after).toBeGreaterThan(0)
    expect(after / before).toBeLessThanOrEqual(0.55)
  })

  it.each(presentationSubjectIds)('keeps %s eye and mouth reference-region signal within ten percent', (subjectId) => {
    const isReference = ({ x, y }: { x: number; y: number }) =>
      (y >= 0.4 && y <= 0.51) ||
      (y >= 0.68 && y <= 0.77 && x >= 0.38 && x <= 0.62)
    const before = integratedSignal(
      presentationAttentionBySubject[subjectId].preoperative,
      isReference,
    )
    const after = integratedSignal(
      presentationAttentionBySubject[subjectId].postoperative,
      isReference,
    )

    expect(Math.abs(after - before) / before).toBeLessThanOrEqual(
      0.1,
    )
  })

  it('labels both fields as hand-authored simulation', () => {
    expect(presentationAttentionSummary.preoperative.provenance).toBe(
      'hand_authored_simulation',
    )
    expect(presentationAttentionSummary.postoperative.provenance).toBe(
      'hand_authored_simulation',
    )
    expect(presentationAttentionSummary.preoperative.cheekSignal).toBe(
      'higher',
    )
    expect(
      presentationAttentionSummary.postoperative.cheekSignal,
    ).toBe('lower')
  })
})
