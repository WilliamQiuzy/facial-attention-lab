import { describe, expect, expectTypeOf, it } from 'vitest'
import type { InferenceBinding } from '../workbench/types'
import {
  approvedAssets,
  validateApprovedAssets,
  type ApprovedAsset,
} from './approvedAssetManifest'
import { workbenchAssetDefinitions } from './workbenchAssetDefinitions'
import type { WorkbenchAssetId } from './workbenchAssetDefinitions'

const EXPECTED_ASSET_TUPLES = [
  [
    'SYN-MOHS-SCC-CHEEK',
    'facial_defect_synthesis/output/synthetic/mohs/mohs_preop_scc-cheek_middle-aged-black-female_a905e7.png',
    '1c43951ed068dc7b88a28e1a0e68d724f1f8b649067b81323ceb30fe2cc5eb30',
  ],
  [
    'SYN-MOHS-NASAL-RECON',
    'facial_defect_synthesis/output/synthetic/mohs/mohs_healed_nasal-recon-faint_elderly-hispanic-female_29a140.png',
    '0992e19d0a12e43cc685507672d5cc781f834d8325da29d9b0ef0eb121abd70d',
  ],
  [
    'SYN-HNC-CHEEK-TUMOUR',
    'facial_defect_synthesis/output/synthetic/hn_cancer/hn_cancer_preop_cheek-tumour_middle-aged-middle-eastern-female_277781.png',
    '8220f72236d545b644964319ac61a4cdc1d3bb08fca86d18e38d8aeb0b57563c',
  ],
  [
    'SYN-HNC-CHEEK-FREEFLAP',
    'facial_defect_synthesis/output/synthetic/hn_cancer/hn_cancer_healed_cheek-freeflap-healed_middle-aged-white-female_529016.png',
    'ad36c7c046c1a83584e0c7e60e78f2749ef08ceb33cd3a5a81f3535a57606dd7',
  ],
  [
    'SYN-TRAUMA-CHEEK-SCAR',
    'facial_defect_synthesis/output/synthetic/trauma/trauma_preop_old-cheek-laceration-scar_middle-aged-middle-eastern-male_49cfd8.png',
    'e977d5406f2065ff88cd22e1dfb45e406714c96f00e2b34a77b10a494ad9cd6d',
  ],
  [
    'SYN-TRAUMA-FRACTURE-RECON',
    'facial_defect_synthesis/output/synthetic/trauma/trauma_healed_fracture-recon-faint_young-adult-south-asian-male_fedb1f.png',
    'f5a71df09aa8fa646dc9152cb4f936281b8cd15ad03bb2cc8181ec1fb61247ba',
  ],
  [
    'SYN-RHINOPHYMA-SEVERE',
    'facial_defect_synthesis/output/synthetic/rhinophyma/rhinophyma_preop_severe-bulbous_elderly-middle-eastern-female_073b01.png',
    '0d355a1f2a3ebe6b84cfc0111248e673e3a6e2714d2eb8862ea4256eca652fef',
  ],
  [
    'SYN-BURN-GRAFT-RECON',
    'facial_defect_synthesis/output/synthetic/burns/burns_healed_graft-reconstructed_middle-aged-black-male_ca2269.png',
    'e4e5cbd9914bc73da0f9261acec1d2a523efcde9de652a25bb951245d7feb257',
  ],
  [
    'SYN-VASCULAR-PWS',
    'facial_defect_synthesis/output/synthetic/vascular/vascular_preop_port-wine-stain_young-adult-white-female_ad3581.png',
    '59c98bb105d9523b7601e9f1a8d087db99a09e95cd6259bd6f8f008f2a944bd9',
  ],
  [
    'SYN-NEVUS-CHEEK',
    'facial_defect_synthesis/output/synthetic/nevus/nevus_preop_cheek-patch_young-adult-white-female_3fa4d7.png',
    '14909f75c0ba2ae4ab5e4ac1cf976f261d6d61eff09ae5804d865e2e6229d374',
  ],
] as const

function tuples(
  assets: readonly Pick<ApprovedAsset, 'id' | 'sourcePath' | 'sha256'>[],
) {
  return assets.map(({ id, sourcePath, sha256 }) => [id, sourcePath, sha256])
}

describe('approved synthetic asset boundary', () => {
  it('owns the exact ordered ten-case allowlist in one pure canonical tuple', () => {
    expect(tuples(workbenchAssetDefinitions)).toEqual(EXPECTED_ASSET_TUPLES)
    expect(tuples(approvedAssets)).toEqual(EXPECTED_ASSET_TUPLES)
  })

  it('uses the canonical ID union anywhere an inference result binds an asset', () => {
    expectTypeOf<InferenceBinding['assetId']>().toEqualTypeOf<WorkbenchAssetId>()
  })

  it('allows ten unique hash-pinned standalone synthetic demo cases', () => {
    expect(approvedAssets).toHaveLength(10)
    expect(new Set(approvedAssets.map((asset) => asset.id)).size).toBe(10)
    expect(new Set(approvedAssets.map((asset) => asset.sourcePath)).size).toBe(10)
    expect(new Set(approvedAssets.map((asset) => asset.sha256)).size).toBe(10)

    for (const asset of approvedAssets) {
      expect(asset.sourceClass).toBe('synthetic_ai_generated')
      expect(asset.relationship).toBe('unpaired_demo')
      expect(asset.allowedUse).toBe('simulated_ui_demo')
      expect(asset.disclosure).toMatch(/AI-generated.+synthetic.+independent.+unpaired/i)
      expect(asset.sourcePath).toContain('/output/synthetic/')
      expect(asset.sourcePath).not.toMatch(/\/output\/real\/|facial_paralysis/i)
      expect(asset.category).not.toMatch(/facial_paralysis/i)
      expect(asset.sha256).toMatch(/^[a-f0-9]{64}$/)
    }

    expect(validateApprovedAssets(approvedAssets)).toEqual({ valid: true, errors: [] })
  })

  it.each([
    [
      'real-image path',
      { sourcePath: 'facial_defect_synthesis/output/real/patient.png' },
      /real|synthetic-only/i,
    ],
    [
      'facial-paralysis path',
      {
        sourcePath:
          'facial_defect_synthesis/output/synthetic/facial_paralysis/excluded.png',
      },
      /facial_paralysis/i,
    ],
    [
      'facial-paralysis category',
      { category: 'facial_paralysis' },
      /facial_paralysis/i,
    ],
    ['real-image URL', { url: '/output/real/patient.png' }, /real|synthetic-only/i],
    [
      'facial-paralysis URL',
      { url: '/output/synthetic/facial_paralysis/excluded.png' },
      /facial_paralysis/i,
    ],
    [
      'unregistered synthetic URL',
      { url: '/output/synthetic/mohs/unregistered.png' },
      /canonical.+URL/i,
    ],
    ['patient-pair claim', { relationship: 'paired_research' }, /unpaired/i],
    [
      'missing independent unpaired disclosure',
      { disclosure: 'Synthetic demo image.' },
      /disclosure/i,
    ],
  ])('rejects a %s', (_name, mutation, expectedError) => {
    const unsafe = approvedAssets.map((asset, index) =>
      index === 0 ? { ...asset, ...mutation } : asset,
    ) as unknown as readonly ApprovedAsset[]

    const result = validateApprovedAssets(unsafe)
    expect(result.valid).toBe(false)
    expect(result.errors.join(' ')).toMatch(expectedError)
  })

  it('rejects duplicate IDs, paths, and hashes instead of accepting ten slots', () => {
    const duplicate = [
      ...approvedAssets.slice(0, -1),
      approvedAssets[0],
    ] as readonly ApprovedAsset[]

    const result = validateApprovedAssets(duplicate)
    expect(result.valid).toBe(false)
    expect(result.errors.join(' ')).toMatch(/duplicate.+ID/i)
    expect(result.errors.join(' ')).toMatch(/duplicate.+path/i)
    expect(result.errors.join(' ')).toMatch(/duplicate.+SHA/i)
  })

  it('rejects a different synthetic tuple even when its shape looks safe', () => {
    const changed = approvedAssets.map((asset, index) =>
      index === 0
        ? {
            ...asset,
            sourcePath:
              'facial_defect_synthesis/output/synthetic/mohs/unregistered.png',
          }
        : asset,
    ) as readonly ApprovedAsset[]

    const result = validateApprovedAssets(changed)
    expect(result.valid).toBe(false)
    expect(result.errors.join(' ')).toMatch(/canonical allowlist/i)
  })
})
