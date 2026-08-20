import { createHash } from 'node:crypto'
import { readFileSync, statSync } from 'node:fs'
import path from 'node:path'
import {
  presentationDemoManifest,
  type PresentationDemoAssetMetadata,
} from '../src/data/presentationDemoManifest'

export type PresentationSourceAsset = Omit<
  PresentationDemoAssetMetadata,
  'sourcePath'
> & {
  readonly sourcePath: string
}

export const PRESENTATION_SOURCE_ASSETS: readonly PresentationSourceAsset[] =
  Object.values(presentationDemoManifest).map(
    ({ sourcePath, ...asset }) => ({
      ...asset,
      sourcePath: `../../${sourcePath}`,
    }),
  )

export function assertPresentationSourceAssetBoundary(
  assets: readonly PresentationSourceAsset[],
  appRoot = process.cwd(),
): void {
  if (assets.length !== 2) {
    throw new Error(
      'The presentation manifest must contain one two-image synthetic pair.',
    )
  }

  const expectedAssets = Object.values(presentationDemoManifest)
  const timepoints = new Set(assets.map((asset) => asset.timepoint))
  const identityIds = new Set(
    assets.map((asset) => asset.derivedSyntheticIdentityId),
  )
  if (
    !timepoints.has('preoperative') ||
    !timepoints.has('postoperative') ||
    timepoints.size !== 2 ||
    identityIds.size !== 1
  ) {
    throw new Error(
      'The presentation manifest must contain one derived synthetic identity at both timepoints.',
    )
  }

  for (const [index, asset] of assets.entries()) {
    const sourcePath = asset.sourcePath
      .replaceAll('\\', '/')
      .toLowerCase()
    const resolvedPath = path
      .resolve(appRoot, asset.sourcePath)
      .split(path.sep)
      .join('/')
      .toLowerCase()
    if (
      !resolvedPath.includes(
        '/facial_defect_synthesis/output/synthetic/',
      ) ||
      sourcePath.includes('/output/real/') ||
      /patient|mrn|medical-record/i.test(
        asset.derivedSyntheticIdentityId,
      )
    ) {
      throw new Error(
        `${asset.id} falls outside the synthetic presentation boundary.`,
      )
    }
    if (
      asset.relationship !== 'paired_synthetic_presentation' ||
      asset.allowedUse !== 'simulated_presentation_demo'
    ) {
      throw new Error(
        `${asset.id} has an unapproved presentation relationship.`,
      )
    }
    if (!/^[a-f0-9]{64}$/.test(asset.sha256)) {
      throw new Error(`${asset.id} has an invalid SHA-256 digest.`)
    }

    const expected = expectedAssets[index]
    if (
      !expected ||
      asset.id !== expected.id ||
      asset.timepoint !== expected.timepoint ||
      asset.sha256 !== expected.sha256 ||
      asset.sourcePath !== `../../${expected.sourcePath}` ||
      asset.derivedSyntheticIdentityId !==
        expected.derivedSyntheticIdentityId
    ) {
      throw new Error(
        `${asset.id} does not match the canonical presentation manifest.`,
      )
    }
  }
}

export function verifyPresentationAssetFiles(
  appRoot = process.cwd(),
): string[] {
  assertPresentationSourceAssetBoundary(
    PRESENTATION_SOURCE_ASSETS,
    appRoot,
  )

  return PRESENTATION_SOURCE_ASSETS.map((asset) => {
    const filePath = path.resolve(appRoot, asset.sourcePath)
    if (!statSync(filePath).isFile()) {
      throw new Error(`${asset.id} is not a regular file.`)
    }
    const digest = createHash('sha256')
      .update(readFileSync(filePath))
      .digest('hex')
    if (digest !== asset.sha256) {
      throw new Error(
        `${asset.id} failed SHA-256 verification: expected ${asset.sha256}, received ${digest}.`,
      )
    }
    return filePath
  })
}
