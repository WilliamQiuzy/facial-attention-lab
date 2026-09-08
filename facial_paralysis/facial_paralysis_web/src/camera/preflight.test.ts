import { describe, expect, it } from 'vitest'

import { assessFraming, measureLuminance, stabilizeHint, type PreflightSample } from './preflight'

const good: PreflightSample = {
  width: 320,
  height: 240,
  faces: [{ x: 96, y: 48, width: 128, height: 144 }],
  luminance: 110,
}

describe('camera framing hints (not movement or health assessment)', () => {
  it('requires a detected face before reporting framing looks good', () => {
    expect(assessFraming({ ...good, faces: [] })).toBe('no-face')
    expect(assessFraming(good)).toBe('ready')
  })

  it('asks a small centered face to come closer', () => {
    expect(assessFraming({ ...good, faces: [{ x: 136, y: 96, width: 48, height: 48 }] })).toBe('closer')
  })

  it('prioritizes dim light over not finding a face', () => {
    expect(assessFraming({ ...good, faces: [], luminance: 20 })).toBe('light')
  })

  it('prioritizes positioning over distance and reports only one hint', () => {
    expect(assessFraming({ ...good, faces: [{ x: 5, y: 5, width: 48, height: 48 }] })).toBe('center')
    expect(assessFraming({ ...good, faces: [{ x: -5, y: 35, width: 260, height: 200 }] })).toBe('center')
  })

  it('asks for one face without selecting a patient from multiple faces', () => {
    expect(assessFraming({ ...good, faces: [good.faces[0], good.faces[0]] })).toBe('one-face')
  })

  it('does not claim valid framing from invalid dimensions or brightness', () => {
    expect(assessFraming({ ...good, width: 0 })).toBe('checking')
    expect(assessFraming({ ...good, luminance: Number.NaN })).toBe('unavailable')
    expect(assessFraming({ ...good, faces: [{ x: Number.NaN, y: 0, width: 50, height: 50 }] })).toBe('no-face')
  })

  it('uses distinct enter and exit thresholds to prevent boundary flicker', () => {
    const near = { ...good, faces: [{ x: 116.8, y: 76.8, width: 86.4, height: 86.4 }] }
    expect(assessFraming(near, 'ready')).toBe('ready')
    expect(assessFraming(near, 'closer')).toBe('closer')
    expect(assessFraming({ ...good, luminance: 50 }, 'ready')).toBe('ready')
    expect(assessFraming({ ...good, luminance: 50 }, 'light')).toBe('light')
    expect(assessFraming({ ...good, luminance: 65 }, 'light')).toBe('ready')
  })

  it('requires three consecutive samples before changing the displayed hint', () => {
    let stable = stabilizeHint(undefined, 'closer')
    expect(stable.hint).toBe('checking')
    stable = stabilizeHint(stable, 'closer')
    expect(stable.hint).toBe('checking')
    stable = stabilizeHint(stable, 'closer')
    expect(stable.hint).toBe('closer')
    stable = stabilizeHint(stable, 'ready')
    stable = stabilizeHint(stable, 'closer')
    expect(stable.hint).toBe('closer')
  })
})

describe('local luminance measurement', () => {
  it('measures detector face ROI instead of a bright background', () => {
    const pixels = new Uint8ClampedArray(4 * 4 * 4).fill(255)
    for (const pixel of [5, 6, 9, 10]) pixels.fill(20, pixel * 4, pixel * 4 + 3)
    expect(measureLuminance(pixels, 4, 4, { x: 1, y: 1, width: 2, height: 2 })).toBeCloseTo(20)
    expect(measureLuminance(pixels, 4, 4)).toBeCloseTo(20)
  })

  it('clips the ROI and rejects empty, malformed, or out-of-frame samples', () => {
    const pixels = new Uint8ClampedArray(4 * 4 * 4).fill(100)
    expect(measureLuminance(pixels, 4, 4, { x: -1, y: -1, width: 3, height: 3 })).toBeCloseTo(100)
    expect(measureLuminance(pixels, 4, 4, { x: 20, y: 20, width: 3, height: 3 })).toBeNull()
    expect(measureLuminance(new Uint8ClampedArray(), 4, 4)).toBeNull()
  })
})
