import {
  workbenchAssetDefinitions,
  type WorkbenchAssetDefinition,
} from './workbenchAssetDefinitions'
import { workbenchAssetUrls } from './workbenchAssetUrls'

export type ApprovedAsset = WorkbenchAssetDefinition & {
  readonly url: string
}

export const approvedAssets: readonly ApprovedAsset[] = workbenchAssetDefinitions.map(
  (definition) => ({
    ...definition,
    url: workbenchAssetUrls[definition.id],
  }),
)

function duplicateValues(values: readonly string[]): string[] {
  const seen = new Set<string>()
  const duplicates = new Set<string>()

  for (const value of values) {
    if (seen.has(value)) duplicates.add(value)
    seen.add(value)
  }

  return [...duplicates]
}

export function validateApprovedAssets(
  assets: readonly ApprovedAsset[],
): { valid: boolean; errors: string[] } {
  const errors: string[] = []

  if (assets.length !== workbenchAssetDefinitions.length) {
    errors.push('The workbench allowlist must contain exactly ten standalone assets.')
  }

  const duplicateIds = duplicateValues(assets.map((asset) => asset.id))
  const duplicatePaths = duplicateValues(assets.map((asset) => asset.sourcePath))
  const duplicateHashes = duplicateValues(assets.map((asset) => asset.sha256))
  if (duplicateIds.length) errors.push(`Duplicate asset ID: ${duplicateIds.join(', ')}.`)
  if (duplicatePaths.length) errors.push(`Duplicate source path: ${duplicatePaths.join(', ')}.`)
  if (duplicateHashes.length) errors.push(`Duplicate SHA-256: ${duplicateHashes.join(', ')}.`)

  for (const [index, asset] of assets.entries()) {
    const normalizedPath = asset.sourcePath.replaceAll('\\', '/').toLowerCase()
    const normalizedUrl = asset.url.replaceAll('\\', '/').toLowerCase()
    const normalizedCategory = String(asset.category).toLowerCase()

    if (!normalizedPath.includes('/output/synthetic/')) {
      errors.push(`${asset.id} falls outside the synthetic-only source boundary.`)
    }
    if (normalizedPath.includes('/output/real/')) {
      errors.push(`${asset.id} references an output/real image path.`)
    }
    if (normalizedUrl.includes('/output/real/')) {
      errors.push(`${asset.id} references an output/real image URL.`)
    }
    if (
      normalizedPath.includes('facial_paralysis') ||
      normalizedUrl.includes('facial_paralysis') ||
      normalizedCategory.includes('facial_paralysis')
    ) {
      errors.push(`${asset.id} references the excluded facial_paralysis scope.`)
    }
    if (asset.sourceClass !== 'synthetic_ai_generated') {
      errors.push(`${asset.id} is not declared synthetic_ai_generated.`)
    }
    if (asset.relationship !== 'unpaired_demo') {
      errors.push(`${asset.id} must remain an unpaired_demo standalone case.`)
    }
    if (asset.allowedUse !== 'simulated_ui_demo') {
      errors.push(`${asset.id} is not approved for simulated_ui_demo use.`)
    }
    if (!/AI-generated.+synthetic.+independent.+unpaired/i.test(asset.disclosure)) {
      errors.push(`${asset.id} is missing its independent synthetic, unpaired disclosure.`)
    }
    if (!/^[a-f0-9]{64}$/.test(asset.sha256)) {
      errors.push(`${asset.id} has an invalid SHA-256 digest.`)
    }

    const expected = workbenchAssetDefinitions[index]
    if (
      !expected ||
      asset.id !== expected.id ||
      asset.sourcePath !== expected.sourcePath ||
      asset.sha256 !== expected.sha256
    ) {
      errors.push(`${asset.id} does not match the canonical allowlist tuple at index ${index}.`)
    }
    if (expected && asset.url !== workbenchAssetUrls[expected.id]) {
      errors.push(`${asset.id} does not match its canonical rendered URL.`)
    }
  }

  return { valid: errors.length === 0, errors }
}
