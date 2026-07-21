import {
  approvedAssets,
  validateApprovedAssets,
  type ApprovedAsset,
} from '../data/approvedAssetManifest'

export type WorkbenchCatalogEntry = ApprovedAsset & {
  readonly roiReadiness: 'not_started'
  readonly reviewStatus: 'not_reviewed'
}

export const workbenchCatalog: readonly WorkbenchCatalogEntry[] = approvedAssets.map(
  (asset) => ({
    ...asset,
    roiReadiness: 'not_started',
    reviewStatus: 'not_reviewed',
  }),
)

export function listWorkbenchAssets(): readonly WorkbenchCatalogEntry[] {
  return workbenchCatalog
}

export function getWorkbenchAsset(id: string): WorkbenchCatalogEntry | undefined {
  return workbenchCatalog.find((entry) => entry.id === id)
}

export function validateWorkbenchCatalog(
  entries: readonly WorkbenchCatalogEntry[],
): { valid: boolean; errors: string[] } {
  const errors = [...validateApprovedAssets(entries).errors]

  for (const entry of entries) {
    if (entry.roiReadiness !== 'not_started') {
      errors.push(`${entry.id} must begin with ROI readiness not_started.`)
    }
    if (entry.reviewStatus !== 'not_reviewed') {
      errors.push(`${entry.id} must begin with review status not_reviewed.`)
    }
    if ('salienceBand' in entry || 'salience' in entry || 'patientId' in entry) {
      errors.push(`${entry.id} contains inferred or patient-linked catalog metadata.`)
    }
  }

  return { valid: errors.length === 0, errors }
}
