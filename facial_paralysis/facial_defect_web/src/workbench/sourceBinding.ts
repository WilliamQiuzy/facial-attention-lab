import { getWorkbenchAsset } from './catalog'

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

export function isVerifiedFullImageSourceBinding(
  asset: unknown,
  annotation: unknown,
): boolean {
  try {
    if (!isRecord(asset) || !isRecord(annotation)) return false
    if (typeof asset.id !== 'string' || typeof asset.sha256 !== 'string') {
      return false
    }

    const canonicalAsset = getWorkbenchAsset(asset.id)
    if (!canonicalAsset || asset.sha256 !== canonicalAsset.sha256) return false
    if (
      annotation.caseId !== canonicalAsset.id ||
      annotation.assetId !== canonicalAsset.id ||
      typeof annotation.id !== 'string' ||
      annotation.id.trim().length === 0 ||
      !Number.isInteger(annotation.version) ||
      (annotation.version as number) <= 0 ||
      annotation.status !== 'approved' ||
      annotation.authorId !== 'demo_author' ||
      annotation.reviewerId !== 'demo_reviewer'
    ) {
      return false
    }

    const geometry = annotation.geometry
    return (
      isRecord(geometry) &&
      geometry.x === 0 &&
      geometry.y === 0 &&
      geometry.width === 1 &&
      geometry.height === 1
    )
  } catch {
    return false
  }
}
