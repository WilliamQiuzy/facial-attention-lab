import type {
  PatientAttentionPoint,
  PatientRunBinding,
  PatientSimulationOutput,
} from './types'

const POINT_COUNT = 24
const UINT32_RANGE = 0x1_0000_0000
const FACE_TEMPLATE_ANCHORS = Object.freeze([
  Object.freeze({ x: 0.37, y: 0.28 }),
  Object.freeze({ x: 0.5, y: 0.25 }),
  Object.freeze({ x: 0.63, y: 0.28 }),
  Object.freeze({ x: 0.35, y: 0.33 }),
  Object.freeze({ x: 0.65, y: 0.33 }),
  Object.freeze({ x: 0.35, y: 0.4 }),
  Object.freeze({ x: 0.43, y: 0.41 }),
  Object.freeze({ x: 0.57, y: 0.41 }),
  Object.freeze({ x: 0.65, y: 0.4 }),
  Object.freeze({ x: 0.5, y: 0.45 }),
  Object.freeze({ x: 0.5, y: 0.52 }),
  Object.freeze({ x: 0.42, y: 0.54 }),
  Object.freeze({ x: 0.58, y: 0.54 }),
  Object.freeze({ x: 0.36, y: 0.56 }),
  Object.freeze({ x: 0.64, y: 0.56 }),
  Object.freeze({ x: 0.68, y: 0.59 }),
  Object.freeze({ x: 0.62, y: 0.62 }),
  Object.freeze({ x: 0.39, y: 0.62 }),
  Object.freeze({ x: 0.42, y: 0.69 }),
  Object.freeze({ x: 0.5, y: 0.7 }),
  Object.freeze({ x: 0.58, y: 0.69 }),
  Object.freeze({ x: 0.45, y: 0.75 }),
  Object.freeze({ x: 0.55, y: 0.75 }),
  Object.freeze({ x: 0.5, y: 0.8 }),
] as const)

if (FACE_TEMPLATE_ANCHORS.length !== POINT_COUNT) {
  throw new Error('The display simulation template must contain 24 anchors.')
}

function seedFromCaptureSha256(captureSha256: string): number {
  const normalizedSha256 = captureSha256.trim().toLowerCase()

  let hash = 0x811c9dc5
  for (let index = 0; index < normalizedSha256.length; index += 1) {
    hash ^= normalizedSha256.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
  }
  return hash >>> 0 || 0x6d2b79f5
}

function createDeterministicUnitGenerator(
  initialSeed: number,
): () => number {
  let state = initialSeed
  return () => {
    state ^= state << 13
    state ^= state >>> 17
    state ^= state << 5
    return (state >>> 0) / UINT32_RANGE
  }
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value))
}

export function createDemoPatientResult(
  binding: PatientRunBinding,
): PatientSimulationOutput {
  const nextUnit = createDeterministicUnitGenerator(
    seedFromCaptureSha256(binding.captureSha256),
  )
  const points: PatientAttentionPoint[] = []

  for (const anchor of FACE_TEMPLATE_ANCHORS) {
    const x = clamp(
      anchor.x + (nextUnit() - 0.5) * 0.024,
      0.26,
      0.74,
    )
    const y = clamp(
      anchor.y + (nextUnit() - 0.5) * 0.024,
      0.18,
      0.84,
    )
    const illustrativeCheek =
      x >= 0.58 && y >= 0.5 && y <= 0.64
    const [minimumIntensity, maximumIntensity] = illustrativeCheek
      ? [0.78, 0.96]
      : y >= 0.34 && y <= 0.48
        ? [0.2, 0.44]
        : y > 0.64 && y <= 0.76
          ? [0.16, 0.38]
          : [0.08, 0.3]
    points.push(
      Object.freeze({
        x,
        y,
        intensity:
          minimumIntensity +
          nextUnit() * (maximumIntensity - minimumIntensity),
        radius:
          (illustrativeCheek ? 0.05 : 0.035) +
          nextUnit() * (illustrativeCheek ? 0.025 : 0.03),
      }),
    )
  }

  return Object.freeze({
    origin: 'workflow_simulation',
    points: Object.freeze(points),
  })
}
