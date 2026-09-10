import { describe, expect, it } from 'vitest'
import {
  findPatientSamplePhotoAsset,
  findPatientSamplePhotoAssetBySha256,
  patientSamplePhotoAssetForPatientTimepoint,
  patientSamplePhotoAssetForTimepoint,
  patientSamplePhotoAssetList,
  patientSamplePhotoAssets,
  patientSamplePhotoPairs,
} from './patientSamplePhotoPair'

describe('patientSamplePhotoPair', () => {
  it('declares a same-subject trauma pair for the two comparison timepoints', () => {
    const { preoperative, postoperative } = patientSamplePhotoAssets

    expect(preoperative.pairId).toBe(postoperative.pairId)
    expect(preoperative.relationship).toBe('same_subject_paired_demo')
    expect(postoperative.relationship).toBe('same_subject_paired_demo')
    expect(preoperative.timepoint).toBe('preoperative')
    expect(postoperative.timepoint).toBe('postoperative')
    expect(preoperative.sourcePath).toMatch(
      /patient-samples\/trauma_preoperative/i,
    )
    expect(postoperative.sourcePath).toMatch(/patient-samples\/trauma_postop/i)
    expect(preoperative.sourcePath).not.toMatch(/mohs|tumou?r|lesion/i)
    expect(postoperative.sourcePath).not.toMatch(/mohs|tumou?r|lesion/i)
  })

  it('resolves only the declared timepoint asset and preserves hash metadata', () => {
    expect(patientSamplePhotoAssetList).toHaveLength(6)
    expect(
      patientSamplePhotoAssetForTimepoint('preoperative'),
    ).toBe(patientSamplePhotoAssets.preoperative)
    expect(
      patientSamplePhotoAssetForTimepoint('postoperative'),
    ).toBe(patientSamplePhotoAssets.postoperative)
    expect(patientSamplePhotoAssetForTimepoint('follow_up')).toBeUndefined()

    for (const asset of patientSamplePhotoAssetList) {
      expect(asset.sha256).toMatch(/^[a-f0-9]{64}$/)
      expect(findPatientSamplePhotoAsset(asset.id)).toBe(asset)
      expect(findPatientSamplePhotoAssetBySha256(asset.sha256)).toBe(
        asset,
      )
    }
  })

  it('assigns each sample record its own same-subject photo pair', () => {
    const patientIds = [
      'patient-demo-001',
      'patient-demo-002',
      'patient-demo-003',
    ] as const
    const pairs = patientIds.map(
      (patientId) => patientSamplePhotoPairs[patientId],
    )

    expect(new Set(pairs.map((pair) => pair.pairId)).size).toBe(3)
    expect(new Set(pairs.map((pair) => pair.subjectProfileId)).size).toBe(
      3,
    )
    expect(
      new Set(patientSamplePhotoAssetList.map((asset) => asset.sha256))
        .size,
    ).toBe(6)

    for (const patientId of patientIds) {
      const preoperative =
        patientSamplePhotoAssetForPatientTimepoint(
          patientId,
          'preoperative',
        )
      const postoperative =
        patientSamplePhotoAssetForPatientTimepoint(
          patientId,
          'postoperative',
        )

      expect(preoperative?.pairId).toBe(postoperative?.pairId)
      expect(preoperative?.subjectProfileId).toBe(
        postoperative?.subjectProfileId,
      )
      expect(preoperative?.url).not.toBe(postoperative?.url)
    }
  })

  it('keeps each paired attention target on one side and makes postoperative attention visibly lower', () => {
    for (const pair of Object.values(patientSamplePhotoPairs)) {
      const preoperative = pair.preoperative.attentionProfile
      const postoperative = pair.postoperative.attentionProfile

      expect(preoperative.focus).toEqual(postoperative.focus)
      expect(preoperative.viewerSide).toBe(postoperative.viewerSide)
      expect(preoperative.focus.x < 0.5).toBe(
        preoperative.viewerSide === 'viewer_left',
      )
      expect(
        preoperative.focusIntensity - postoperative.focusIntensity,
      ).toBeGreaterThanOrEqual(0.5)
      expect(postoperative.focusRadius).toBeLessThan(
        preoperative.focusRadius,
      )
    }
  })
})
