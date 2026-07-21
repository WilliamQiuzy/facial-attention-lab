import { describe, expect, it } from 'vitest'
import { compareAttention } from './metrics'

describe('compareAttention', () => {
  it('computes a simulated layout delta without overstating causality', () => {
    const result = compareAttention(
      { scarGazePercent: 38, timeToFirstFixationMs: 420 },
      { scarGazePercent: 21, timeToFirstFixationMs: 760 },
      'mock_simulation',
      'unpaired_demo',
    )

    expect(result.scarGazeChangePoints).toBe(-17)
    expect(result.relativeReductionPercent).toBe(45)
    expect(result.interpretation).toContain('simulated')
    expect(result.interpretation).toContain('unpaired')
    expect(result.interpretation).not.toMatch(/treatment|improved|surgery caused/i)
  })
})
