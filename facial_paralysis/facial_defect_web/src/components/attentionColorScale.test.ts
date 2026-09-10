import { describe, expect, it } from 'vitest'
import {
  ATTENTION_COLOR_SCALE_LABEL,
  attentionColorRgb,
} from './attentionColorScale'

describe('classic heatmap color scale', () => {
  it('uses the familiar blue-cyan-green-yellow-red anchors', () => {
    expect(attentionColorRgb(0)).toBe('0 0 255')
    expect(attentionColorRgb(0.25)).toBe('0 255 255')
    expect(attentionColorRgb(0.5)).toBe('0 255 0')
    expect(attentionColorRgb(0.75)).toBe('255 255 0')
    expect(attentionColorRgb(1)).toBe('255 0 0')
    expect(ATTENTION_COLOR_SCALE_LABEL).toBe(
      'Attention scale: blue indicates less attention and red indicates more attention',
    )
  })

  it('still clamps invalid and out-of-range intensity values', () => {
    expect(attentionColorRgb(Number.NaN)).toBe('0 0 255')
    expect(attentionColorRgb(-1)).toBe('0 0 255')
    expect(attentionColorRgb(2)).toBe('255 0 0')
  })
})
