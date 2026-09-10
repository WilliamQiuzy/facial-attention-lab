import type {
  PatientAttentionPoint,
  PatientFaceRegistration,
  PatientRunBinding,
  PatientSimulationOutput,
} from './types'
import { findPatientSamplePhotoAssetBySha256 } from '../data/patientSamplePhotoPair'

const POINT_COUNT = 24
const UINT32_RANGE = 0x1_0000_0000
const CANONICAL_FACE_BOUNDS = Object.freeze({
  minimumX: 0.21,
  maximumX: 0.79,
  minimumY: 0.07,
  maximumY: 0.93,
})
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
const SAMPLE_FOCUS_OFFSETS = Object.freeze([
  Object.freeze({ x: 0, y: 0, intensityScale: 1, radiusScale: 1 }),
  Object.freeze({ x: -0.028, y: -0.018, intensityScale: 0.9, radiusScale: 0.82 }),
  Object.freeze({ x: 0.027, y: -0.012, intensityScale: 0.86, radiusScale: 0.78 }),
  Object.freeze({ x: 0.012, y: 0.03, intensityScale: 0.8, radiusScale: 0.74 }),
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

function detectedFaceBounds(
  registration: PatientFaceRegistration,
) {
  const oval = registration.paths.find(
    (path) => path.feature === 'face_oval',
  )
  if (!oval || oval.points.length === 0) return undefined

  return {
    minimumX: Math.min(...oval.points.map((point) => point.x)),
    maximumX: Math.max(...oval.points.map((point) => point.x)),
    minimumY: Math.min(...oval.points.map((point) => point.y)),
    maximumY: Math.max(...oval.points.map((point) => point.y)),
  }
}

function registerPoint(
  point: Readonly<{ x: number; y: number; radius: number }>,
  registration?: PatientFaceRegistration,
) {
  if (!registration) return point
  const bounds = detectedFaceBounds(registration)
  if (!bounds) return point

  const canonicalWidth =
    CANONICAL_FACE_BOUNDS.maximumX -
    CANONICAL_FACE_BOUNDS.minimumX
  const canonicalHeight =
    CANONICAL_FACE_BOUNDS.maximumY -
    CANONICAL_FACE_BOUNDS.minimumY
  const detectedWidth = bounds.maximumX - bounds.minimumX
  const detectedHeight = bounds.maximumY - bounds.minimumY
  const relativeX = clamp(
    (point.x - CANONICAL_FACE_BOUNDS.minimumX) / canonicalWidth,
    0,
    1,
  )
  const relativeY = clamp(
    (point.y - CANONICAL_FACE_BOUNDS.minimumY) /
      canonicalHeight,
    0,
    1,
  )
  const radiusScale =
    (detectedWidth / canonicalWidth +
      detectedHeight / canonicalHeight) /
    2

  return {
    x: bounds.minimumX + relativeX * detectedWidth,
    y: bounds.minimumY + relativeY * detectedHeight,
    radius: clamp(point.radius * radiusScale, 0.015, 0.15),
  }
}

export function createDemoPatientResult(
  binding: PatientRunBinding,
  faceRegistration?: PatientFaceRegistration,
): PatientSimulationOutput {
  const nextUnit = createDeterministicUnitGenerator(
    seedFromCaptureSha256(binding.captureSha256),
  )
  const sampleAsset = findPatientSamplePhotoAssetBySha256(
    binding.captureSha256,
  )
  const points: PatientAttentionPoint[] = []

  if (sampleAsset) {
    for (const offset of SAMPLE_FOCUS_OFFSETS) {
      const registered = registerPoint(
        {
          x: sampleAsset.attentionProfile.focus.x + offset.x,
          y: sampleAsset.attentionProfile.focus.y + offset.y,
          radius:
            sampleAsset.attentionProfile.focusRadius *
            offset.radiusScale,
        },
        faceRegistration,
      )
      points.push(
        Object.freeze({
          x: registered.x,
          y: registered.y,
          intensity:
            sampleAsset.attentionProfile.focusIntensity *
            offset.intensityScale,
          radius: registered.radius,
        }),
      )
    }

    for (const anchor of FACE_TEMPLATE_ANCHORS.slice(
      0,
      POINT_COUNT - SAMPLE_FOCUS_OFFSETS.length,
    )) {
      const x = clamp(
        anchor.x + (nextUnit() - 0.5) * 0.02,
        0.26,
        0.74,
      )
      const y = clamp(
        anchor.y + (nextUnit() - 0.5) * 0.02,
        0.18,
        0.84,
      )
      const featureEmphasis =
        y >= 0.34 && y <= 0.48
          ? 0.36
          : y > 0.64 && y <= 0.76
            ? 0.3
            : 0.2
      const registered = registerPoint(
        {
          x,
          y,
          radius: 0.035 + nextUnit() * 0.026,
        },
        faceRegistration,
      )
      points.push(
        Object.freeze({
          x: registered.x,
          y: registered.y,
          intensity: featureEmphasis + nextUnit() * 0.16,
          radius: registered.radius,
        }),
      )
    }

    return Object.freeze({
      origin: 'workflow_simulation',
      points: Object.freeze(points),
    })
  }

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
    const registered = registerPoint(
      {
        x,
        y,
        radius:
          (illustrativeCheek ? 0.05 : 0.035) +
          nextUnit() * (illustrativeCheek ? 0.025 : 0.03),
      },
      faceRegistration,
    )
    points.push(
      Object.freeze({
        x: registered.x,
        y: registered.y,
        intensity:
          minimumIntensity +
          nextUnit() * (maximumIntensity - minimumIntensity),
        radius: registered.radius,
      }),
    )
  }

  return Object.freeze({
    origin: 'workflow_simulation',
    points: Object.freeze(points),
  })
}
