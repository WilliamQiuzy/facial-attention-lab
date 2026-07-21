import { describe, expect, it } from 'vitest'
import {
  getWorkbenchAsset,
  listWorkbenchAssets,
  validateWorkbenchCatalog,
  workbenchCatalog,
  type WorkbenchCatalogEntry,
} from './catalog'

const EXPECTED_IDS = [
  'SYN-MOHS-SCC-CHEEK',
  'SYN-MOHS-NASAL-RECON',
  'SYN-HNC-CHEEK-TUMOUR',
  'SYN-HNC-CHEEK-FREEFLAP',
  'SYN-TRAUMA-CHEEK-SCAR',
  'SYN-TRAUMA-FRACTURE-RECON',
  'SYN-RHINOPHYMA-SEVERE',
  'SYN-BURN-GRAFT-RECON',
  'SYN-VASCULAR-PWS',
  'SYN-NEVUS-CHEEK',
] as const

describe('standalone synthetic workbench catalog', () => {
  it('lists ten independent cases with only auditable catalog metadata', () => {
    const listed = listWorkbenchAssets()

    expect(listed).toBe(workbenchCatalog)
    expect(listed.map((entry) => entry.id)).toEqual(EXPECTED_IDS)
    expect(validateWorkbenchCatalog(listed)).toEqual({ valid: true, errors: [] })

    for (const entry of listed) {
      expect(entry.sourceClass).toBe('synthetic_ai_generated')
      expect(entry.relationship).toBe('unpaired_demo')
      expect(entry.allowedUse).toBe('simulated_ui_demo')
      expect(entry.disclosure).toMatch(/AI-generated.+synthetic.+independent.+unpaired/i)
      expect(entry.roiReadiness).toBe('not_started')
      expect(entry.reviewStatus).toBe('not_reviewed')
      expect(entry).not.toHaveProperty('salienceBand')
      expect(entry).not.toHaveProperty('salience')
      expect(entry).not.toHaveProperty('patientId')
    }
  })

  it('returns the exact requested case and fails closed for every unknown string ID', () => {
    expect(getWorkbenchAsset('SYN-VASCULAR-PWS')?.id).toBe('SYN-VASCULAR-PWS')
    expect(getWorkbenchAsset('not-a-case')).toBeUndefined()
    expect(getWorkbenchAsset('')).toBeUndefined()
  })

  it.each([
    [
      'duplicate entry',
      [
        ...workbenchCatalog.slice(0, -1),
        workbenchCatalog[0],
      ] as readonly WorkbenchCatalogEntry[],
      /duplicate/i,
    ],
    [
      'paired relationship',
      workbenchCatalog.map((entry, index) =>
        index === 0 ? { ...entry, relationship: 'paired_research' } : entry,
      ) as unknown as readonly WorkbenchCatalogEntry[],
      /unpaired/i,
    ],
    [
      'real path',
      workbenchCatalog.map((entry, index) =>
        index === 0
          ? { ...entry, sourcePath: 'facial_defect_synthesis/output/real/patient.png' }
          : entry,
      ) as readonly WorkbenchCatalogEntry[],
      /real|synthetic-only/i,
    ],
    [
      'facial-paralysis category',
      workbenchCatalog.map((entry, index) =>
        index === 0 ? { ...entry, category: 'facial_paralysis' } : entry,
      ) as unknown as readonly WorkbenchCatalogEntry[],
      /facial_paralysis/i,
    ],
  ])('rejects a catalog containing a %s', (_name, catalog, expectedError) => {
    const result = validateWorkbenchCatalog(catalog)
    expect(result.valid).toBe(false)
    expect(result.errors.join(' ')).toMatch(expectedError)
  })
})
