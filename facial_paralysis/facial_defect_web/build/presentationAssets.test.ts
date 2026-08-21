// @vitest-environment node
import { describe, expect, it } from 'vitest'
import presentationAudit from '../audits/presentation-synthetic-pair.json'
import packageManifest from '../presentation-assets/manifest.json'
import { approvedAssets } from '../src/data/approvedAssetManifest'
import {
  presentationDemoAssets,
  presentationSubjectIds,
  type PresentationSubjectId,
} from '../src/data/presentationDemoAssets'
import { workbenchCatalog } from '../src/workbench/catalog'
import {
  PRESENTATION_SOURCE_ASSETS,
  assertPresentationSourceAssetBoundary,
  verifyPresentationAssetFiles,
  type PresentationSourceAsset,
} from './presentationAssets'

describe('paired synthetic presentation assets', () => {
  it('publishes the standalone Teams file with the FaceAI name', () => {
    expect(packageManifest.offlineHtml.path).toBe('FaceAI-Demo.html')
  })

  it('binds two complete synthetic pairs to the provenance audit', () => {
    expect(PRESENTATION_SOURCE_ASSETS).toHaveLength(4)
    expect(Object.keys(presentationDemoAssets)).toEqual(presentationSubjectIds)
    expect(
      new Set(
        PRESENTATION_SOURCE_ASSETS.map(
          (asset) => asset.derivedSyntheticIdentityId,
        ),
      ),
    ).toEqual(
      new Set(
        presentationAudit.subjects.map(
          (subject) => subject.derivedSyntheticIdentityId,
        ),
      ),
    )

    for (const auditedSubject of presentationAudit.subjects) {
      const pair = presentationDemoAssets[
        auditedSubject.subjectId as PresentationSubjectId
      ]
      expect(pair.preoperative.sha256).toBe(auditedSubject.source.sha256)
      expect(pair.preoperative.sourceAssetId).toBe(
        auditedSubject.source.approvedAssetId,
      )
      expect(pair.postoperative.sha256).toBe(
        auditedSubject.derivedEdit.sha256,
      )
      expect(pair.preoperative.relationship).toBe(
        presentationAudit.relationship,
      )
      expect(pair.postoperative.allowedUse).toBe(
        presentationAudit.allowedUse,
      )
      expect(pair.postoperative.disclosure).toBe(
        presentationAudit.boundary,
      )
    }
  })

  it('verifies all four exact synthetic image files by SHA-256', () => {
    expect(verifyPresentationAssetFiles()).toHaveLength(4)
  })

  it('keeps both derived edits outside every ten-asset workbench path', () => {
    const derivedEdits = presentationSubjectIds.map(
      (subjectId) => presentationDemoAssets[subjectId].postoperative,
    )

    expect(approvedAssets).toHaveLength(10)
    expect(workbenchCatalog).toHaveLength(10)
    for (const derivedEdit of derivedEdits) {
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
    }
    expect(
      new Set([
        ...approvedAssets.map((asset) => asset.sourcePath),
        ...derivedEdits.map((asset) => asset.sourcePath),
      ]).size,
    ).toBe(12)
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
