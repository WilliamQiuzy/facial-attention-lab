import { describe, expect, it } from 'vitest'
import { getWorkbenchAsset } from './catalog'
import { isVerifiedFullImageSourceBinding } from './sourceBinding'
import type { RoiAnnotation } from './types'

const asset = getWorkbenchAsset('SYN-MOHS-SCC-CHEEK')!

function verifiedAnnotation(
  overrides: Partial<RoiAnnotation> = {},
): RoiAnnotation {
  return {
    id: 'source-binding-primary',
    caseId: asset.id,
    assetId: asset.id,
    version: 3,
    geometry: { x: 0, y: 0, width: 1, height: 1 },
    status: 'approved',
    authorId: 'demo_author',
    reviewerId: 'demo_reviewer',
    ...overrides,
  }
}

describe('verified full-image source binding', () => {
  it('accepts only the canonical asset identity and the exact approved full image', () => {
    expect(isVerifiedFullImageSourceBinding(asset, verifiedAnnotation())).toBe(true)
    expect(
      isVerifiedFullImageSourceBinding(
        { ...asset },
        {
          ...verifiedAnnotation(),
          geometry: { ...verifiedAnnotation().geometry },
        },
      ),
    ).toBe(true)
  })

  it.each([
    ['unknown asset', { ...asset, id: 'UNKNOWN-CASE' }, verifiedAnnotation()],
    ['wrong asset hash', { ...asset, sha256: '0'.repeat(64) }, verifiedAnnotation()],
    [
      'case mismatch',
      asset,
      verifiedAnnotation({ caseId: 'SYN-MOHS-NASAL-RECON' }),
    ],
    [
      'asset mismatch',
      asset,
      verifiedAnnotation({ assetId: 'SYN-MOHS-NASAL-RECON' }),
    ],
    ['blank annotation ID', asset, verifiedAnnotation({ id: '   ' })],
    ['zero version', asset, verifiedAnnotation({ version: 0 })],
    ['fractional version', asset, verifiedAnnotation({ version: 1.5 })],
    ['draft status', asset, verifiedAnnotation({ status: 'draft' })],
    [
      'wrong author',
      asset,
      { ...verifiedAnnotation(), authorId: 'other_author' },
    ],
    [
      'missing reviewer',
      asset,
      { ...verifiedAnnotation(), reviewerId: undefined },
    ],
    [
      'wrong reviewer',
      asset,
      { ...verifiedAnnotation(), reviewerId: 'other_reviewer' },
    ],
    [
      'partial image',
      asset,
      verifiedAnnotation({
        geometry: { x: 0.05, y: 0.05, width: 0.9, height: 0.9 },
      }),
    ],
    [
      'almost full image',
      asset,
      verifiedAnnotation({
        geometry: {
          x: Number.EPSILON,
          y: 0,
          width: 1 - Number.EPSILON,
          height: 1,
        },
      }),
    ],
  ])('rejects %s', (_label, candidateAsset, annotation) => {
    expect(
      isVerifiedFullImageSourceBinding(candidateAsset, annotation),
    ).toBe(false)
  })

  it.each([
    ['null asset', null, verifiedAnnotation()],
    ['array asset', [], verifiedAnnotation()],
    ['null annotation', asset, null],
    ['array annotation', asset, []],
    ['missing geometry', asset, { ...verifiedAnnotation(), geometry: undefined }],
    ['null geometry', asset, { ...verifiedAnnotation(), geometry: null }],
    [
      'non-numeric geometry',
      asset,
      {
        ...verifiedAnnotation(),
        geometry: { x: '0', y: 0, width: 1, height: 1 },
      },
    ],
  ])('fails false for malformed runtime input: %s', (_label, candidateAsset, annotation) => {
    expect(() =>
      isVerifiedFullImageSourceBinding(candidateAsset, annotation),
    ).not.toThrow()
    expect(
      isVerifiedFullImageSourceBinding(candidateAsset, annotation),
    ).toBe(false)
  })
})
