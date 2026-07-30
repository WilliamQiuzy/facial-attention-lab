type Rgb = readonly [red: number, green: number, blue: number]

type AttentionColorStop = {
  readonly intensity: number
  readonly rgb: Rgb
}

const ATTENTION_COLOR_STOPS = [
  { intensity: 0, rgb: [25, 70, 148] },
  { intensity: 0.25, rgb: [0, 174, 193] },
  { intensity: 0.5, rgb: [255, 215, 58] },
  { intensity: 0.75, rgb: [255, 118, 31] },
  { intensity: 1, rgb: [207, 16, 32] },
] as const satisfies readonly AttentionColorStop[]

export const ATTENTION_COLOR_SCALE_LABEL =
  'Relative density color scale: blue low, cyan, yellow, orange, red peak'

function clampIntensity(intensity: number): number {
  if (!Number.isFinite(intensity)) return 0
  return Math.min(1, Math.max(0, intensity))
}

function interpolateChannel(start: number, end: number, progress: number) {
  return Math.round(start + (end - start) * progress)
}

export function attentionColorRgb(intensity: number): string {
  const normalized = clampIntensity(intensity)
  const upperIndex = ATTENTION_COLOR_STOPS.findIndex(
    (stop) => stop.intensity >= normalized,
  )

  if (upperIndex <= 0) {
    return ATTENTION_COLOR_STOPS[0].rgb.join(' ')
  }

  const upper = ATTENTION_COLOR_STOPS[upperIndex]
  const lower = ATTENTION_COLOR_STOPS[upperIndex - 1]
  const progress =
    (normalized - lower.intensity) / (upper.intensity - lower.intensity)

  return lower.rgb
    .map((channel, index) =>
      interpolateChannel(channel, upper.rgb[index], progress),
    )
    .join(' ')
}
