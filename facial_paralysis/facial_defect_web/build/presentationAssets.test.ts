// @vitest-environment node
import { describe, expect, it } from 'vitest'
import presentationAudit from '../audits/presentation-synthetic-pair.json'
import { approvedAssets } from '../src/data/approvedAssetManifest'
import { presentationDemoAssets } from '../src/data/presentationDemoAssets'
import { workbenchCatalog } from '../src/workbench/catalog'
import {
  PRESENTATION_SOURCE_ASSETS,
  assertPresentationSourceAssetBoundary,
  verifyPresentationAssetFiles,
  type PresentationSourceAsset,
} from './presentationAssets'

const POSTOPERATIVE_HASH =
  '72d0ee02fa6313b9ddb3c6b4ccf3c1f8c277c98b51a3b4453152948f21e7a58b'

describe('paired synthetic presentation assets', () => {
  it('binds exactly one derived synthetic pair to the provenance audit', () => {
    expect(PRESENTATION_SOURCE_ASSETS).toHaveLength(2)
    expect(Object.keys(presentationDemoAssets)).toEqual([
      'preoperative',
      'postoperative',
    ])
    expect(
      new Set(
        PRESENTATION_SOURCE_ASSETS.map(
          (asset) => asset.derivedSyntheticIdentityId,
        ),
      ),
    ).toEqual(new Set([presentationAudit.derivedSyntheticIdentityId]))
    expect(presentationDemoAssets.preoperative.sha256).toBe(
      presentationAudit.source.sha256,
    )
    expect(presentationDemoAssets.preoperative.sourceAssetId).toBe(
      presentationAudit.source.approvedAssetId,
    )
    expect(presentationDemoAssets.postoperative.sha256).toBe(
      presentationAudit.derivedEdit.sha256,
    )
    expect(presentationDemoAssets.postoperative.sha256).toBe(
      POSTOPERATIVE_HASH,
    )
    expect(presentationDemoAssets.preoperative.relationship).toBe(
      presentationAudit.relationship,
    )
    expect(presentationDemoAssets.postoperative.allowedUse).toBe(
      presentationAudit.allowedUse,
    )
    expect(presentationDemoAssets.postoperative.disclosure).toBe(
      presentationAudit.boundary,
    )
  })

  it('verifies both exact synthetic image files by SHA-256', () => {
    expect(verifyPresentationAssetFiles()).toHaveLength(2)
  })

  it('keeps the derived edit outside every ten-asset workbench path', () => {
    const derivedEdit = presentationDemoAssets.postoperative

    expect(approvedAssets).toHaveLength(10)
    expect(workbenchCatalog).toHaveLength(10)
    expect(
      approvedAssets.some(
        (asset) =>
          asset.id === derivedEdit.id ||
          asset.sourcePath === derivedEdit.sourcePath,
      ),
    ).toBe(false)
    expect(
      workbenchCatalog.some((asset) => asset.id === derivedEdit.id),
    ).toBe(false)
    expect(
      new Set([
        ...approvedAssets.map((asset) => asset.sourcePath),
        presentationDemoAssets.postoperative.sourcePath,
      ]).size,
    ).toBe(11)
  })

  it.each([
    [
      'real-image path',
      {
        ...PRESENTATION_SOURCE_ASSETS[1],
        sourcePath: '../../facial_defect_synthesis/output/real/patient.png',
      },
    ],
    [
      'patient identifier',
      {
        ...PRESENTATION_SOURCE_ASSETS[1],
        derivedSyntheticIdentityId: 'patient-12345',
      },
    ],
    [
      'unapproved relationship',
      {
        ...PRESENTATION_SOURCE_ASSETS[1],
        relationship: 'paired_patient_longitudinal',
      },
    ],
  ])('rejects a %s before reading image bytes', (_name, unsafe) => {
    const candidate = PRESENTATION_SOURCE_ASSETS.map((asset, index) =>
      index === 1 ? unsafe : asset,
    ) as readonly PresentationSourceAsset[]

    expect(() =>
      assertPresentationSourceAssetBoundary(candidate),
    ).toThrow(/synthetic|patient|relationship|canonical|real/i)
  })
})
