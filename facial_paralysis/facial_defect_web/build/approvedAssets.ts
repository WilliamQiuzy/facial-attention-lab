import { createHash } from 'node:crypto'
import { readFileSync, statSync } from 'node:fs'
import path from 'node:path'
import {
  workbenchAssetDefinitions,
  type WorkbenchAssetDefinition,
} from '../src/data/workbenchAssetDefinitions'

export type ApprovedSourceAsset = Omit<WorkbenchAssetDefinition, 'sourcePath'> & {
  readonly sourcePath: string
}

export const APPROVED_SOURCE_ASSETS: readonly ApprovedSourceAsset[] =
  workbenchAssetDefinitions.map((definition) => ({
    ...definition,
    sourcePath: `../../${definition.sourcePath}`,
  }))

function duplicateValues(values: readonly string[]): string[] {
  const seen = new Set<string>()
  const duplicates = new Set<string>()

  for (const value of values) {
    if (seen.has(value)) duplicates.add(value)
    seen.add(value)
  }

  return [...duplicates]
}

export function assertApprovedSourceAssetBoundary(
  assets: readonly ApprovedSourceAsset[],
  appRoot = process.cwd(),
): void {
  for (const asset of assets) {
    const sourcePath = asset.sourcePath.replaceAll('\\', '/').toLowerCase()
    const category = String(asset.category).toLowerCase()
    const filePath = path.resolve(appRoot, asset.sourcePath)
    const resolvedPath = filePath.split(path.sep).join('/').toLowerCase()

    if (sourcePath.includes('/output/real/') || resolvedPath.includes('/output/real/')) {
      throw new Error(`${asset.id} references a forbidden output/real image path.`)
    }
    if (
      sourcePath.includes('facial_paralysis') ||
      resolvedPath.includes('facial_paralysis') ||
      category.includes('facial_paralysis')
    ) {
      throw new Error(`${asset.id} references the excluded facial_paralysis scope.`)
    }
    if (
      !resolvedPath.includes('/facial_defect_synthesis/output/synthetic/') ||
      asset.sourceClass !== 'synthetic_ai_generated' ||
      asset.relationship !== 'unpaired_demo' ||
      asset.allowedUse !== 'simulated_ui_demo'
    ) {
      throw new Error(`${asset.id} falls outside the synthetic-only source boundary.`)
    }
  }

  if (assets.length !== workbenchAssetDefinitions.length) {
    throw new Error('The production asset allowlist must contain exactly ten files.')
  }

  if (
    duplicateValues(assets.map((asset) => asset.id)).length ||
    duplicateValues(assets.map((asset) => asset.sourcePath)).length ||
    duplicateValues(assets.map((asset) => asset.sha256)).length
  ) {
    throw new Error('The production asset allowlist contains a duplicate ID, path, or SHA-256.')
  }

  for (const [index, asset] of assets.entries()) {
    const expected = APPROVED_SOURCE_ASSETS[index]
    if (
      !expected ||
      asset.id !== expected.id ||
      asset.sourcePath !== expected.sourcePath ||
      asset.sha256 !== expected.sha256 ||
      asset.category !== expected.category ||
      asset.generationState !== expected.generationState ||
      asset.sourceClass !== expected.sourceClass ||
      asset.relationship !== expected.relationship ||
      asset.allowedUse !== expected.allowedUse ||
      asset.disclosure !== expected.disclosure
    ) {
      throw new Error(`${asset.id} does not match the canonical build verifier input.`)
    }
  }
}

export function verifyApprovedAssetFiles(appRoot = process.cwd()): string[] {
  assertApprovedSourceAssetBoundary(APPROVED_SOURCE_ASSETS, appRoot)

  return APPROVED_SOURCE_ASSETS.map((asset) => {
    const filePath = path.resolve(appRoot, asset.sourcePath)
    if (!statSync(filePath).isFile()) {
      throw new Error(`${asset.id} is not a regular file.`)
    }

    const digest = createHash('sha256').update(readFileSync(filePath)).digest('hex')
    if (digest !== asset.sha256) {
      throw new Error(
        `${asset.id} failed SHA-256 verification: expected ${asset.sha256}, received ${digest}.`,
      )
    }

    return filePath
  })
}
