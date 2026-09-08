export type FaceBox = { x: number; y: number; width: number; height: number }

export type PreflightHint =
  | 'checking'
  | 'unavailable'
  | 'no-face'
  | 'one-face'
  | 'center'
  | 'closer'
  | 'light'
  | 'ready'

export type PreflightSample = {
  width: number
  height: number
  faces: FaceBox[]
  luminance: number | null
}

export type StableHint = {
  hint: PreflightHint
  candidate: PreflightHint
  consecutive: number
}

function validBox(box: FaceBox) {
  return [box.x, box.y, box.width, box.height].every(Number.isFinite)
    && box.width > 0 && box.height > 0
}

/** Composition heuristics only. No landmarks, expression, movement, or clinical scoring. */
export function assessFraming(sample: PreflightSample, previous: PreflightHint = 'checking'): PreflightHint {
  const { width, height, luminance } = sample
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return 'checking'
  if (luminance === null || !Number.isFinite(luminance)) return 'unavailable'
  // Conservative severe-darkness threshold, not exposure/skin-tone calibration.
  if (luminance < (previous === 'light' ? 60 : 42)) return 'light'
  const faces = sample.faces.filter((face) => validBox(face)
    && face.x < width && face.y < height && face.x + face.width > 0 && face.y + face.height > 0)
  if (!faces.length) return 'no-face'
  if (faces.length > 1) return 'one-face'
  const face = faces[0]
  const margin = previous === 'center' ? 0.13 : 0.17
  const offCenter = Math.abs((face.x + face.width / 2) / width - 0.5) > margin
    || Math.abs((face.y + face.height / 2) / height - 0.5) > margin
  const clipped = face.x < width * 0.02 || face.y < height * 0.02
    || face.x + face.width > width * 0.98 || face.y + face.height > height * 0.98
  if (offCenter || clipped) return 'center'
  if (Math.min(face.width / width, face.height / height) < (previous === 'closer' ? 0.30 : 0.24)) return 'closer'
  return 'ready'
}

/** Change visible text only after three matching samples (about 1.5 seconds). */
export function stabilizeHint(previous: StableHint | undefined, next: PreflightHint): StableHint {
  const state = previous ?? { hint: 'checking', candidate: 'checking', consecutive: 0 }
  const consecutive = state.candidate === next ? Math.min(state.consecutive + 1, 3) : 1
  return { hint: consecutive >= 3 ? next : state.hint, candidate: next, consecutive }
}

/** In-memory sRGB luma, in [0,255], from the face box or central half of the frame. */
export function measureLuminance(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  face?: FaceBox,
): number | null {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0
    || rgba.length !== width * height * 4) return null
  const roi = face ?? { x: width / 4, y: height / 4, width: width / 2, height: height / 2 }
  if (!validBox(roi)) return null
  const left = Math.max(0, Math.floor(roi.x))
  const top = Math.max(0, Math.floor(roi.y))
  const right = Math.min(width, Math.ceil(roi.x + roi.width))
  const bottom = Math.min(height, Math.ceil(roi.y + roi.height))
  let total = 0
  let count = 0
  for (let y = top; y < bottom; y += 1) {
    for (let x = left; x < right; x += 1) {
      const index = (y * width + x) * 4
      total += rgba[index] * 0.2126 + rgba[index + 1] * 0.7152 + rgba[index + 2] * 0.0722
      count += 1
    }
  }
  return count ? total / count : null
}
