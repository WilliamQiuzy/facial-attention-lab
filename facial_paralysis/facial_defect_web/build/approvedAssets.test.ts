// @vitest-environment node
import { describe, expect, it } from 'vitest'
import { workbenchAssetDefinitions } from '../src/data/workbenchAssetDefinitions'
import {
  APPROVED_SOURCE_ASSETS,
  assertApprovedSourceAssetBoundary,
  verifyApprovedAssetFiles,
  type ApprovedSourceAsset,
} from './approvedAssets'

function runtimeTuples() {
  return workbenchAssetDefinitions.map(({ id, sourcePath, sha256 }) => ({
    id,
    sourcePath,
    sha256,
  }))
}

function buildTuples() {
  return APPROVED_SOURCE_ASSETS.map(({ id, sourcePath, sha256 }) => ({
    id,
    sourcePath: sourcePath.replace(/^(?:\.\.\/){2}/, ''),
    sha256,
  }))
}

describe('production asset byte integrity', () => {
  it('derives the exact build verifier tuple from the canonical runtime definitions', () => {
    expect(APPROVED_SOURCE_ASSETS).toHaveLength(10)
    expect(buildTuples()).toEqual(runtimeTuples())
    expect(new Set(APPROVED_SOURCE_ASSETS.map((asset) => asset.id)).size).toBe(10)
    expect(new Set(APPROVED_SOURCE_ASSETS.map((asset) => asset.sourcePath)).size).toBe(10)
    expect(new Set(APPROVED_SOURCE_ASSETS.map((asset) => asset.sha256)).size).toBe(10)
  })

  it('verifies all ten exact source files against their canonical SHA-256 digests', () => {
    const verified = verifyApprovedAssetFiles()

    expect(verified).toHaveLength(10)
  })

  it.each([
    [
      'a real-image path',
      {
        ...APPROVED_SOURCE_ASSETS[0],
        sourcePath: '../../facial_defect_synthesis/output/real/patient.png',
      },
      /synthetic-only|real/i,
    ],
    [
      'a facial-paralysis path',
      {
        ...APPROVED_SOURCE_ASSETS[0],
        sourcePath:
          '../../facial_defect_synthesis/output/synthetic/facial_paralysis/excluded.png',
      },
      /facial_paralysis/i,
    ],
    [
      'a facial-paralysis category',
      { ...APPROVED_SOURCE_ASSETS[0], category: 'facial_paralysis' },
      /facial_paralysis/i,
    ],
  ])('hard-blocks %s before reading bytes', (_name, unsafeAsset, expectedError) => {
    expect(() =>
      assertApprovedSourceAssetBoundary([
        unsafeAsset as unknown as ApprovedSourceAsset,
      ]),
    ).toThrow(expectedError)
  })

  it.each([
    ['unknown ID', { ...APPROVED_SOURCE_ASSETS[0], id: 'SYN-UNKNOWN' }],
    [
      'swapped canonical ID',
      { ...APPROVED_SOURCE_ASSETS[0], id: APPROVED_SOURCE_ASSETS[1].id },
    ],
    ['paired claim', { ...APPROVED_SOURCE_ASSETS[0], relationship: 'paired_research' }],
  ])('rejects a noncanonical %s verifier input', (_name, unsafeAsset) => {
    const unsafe = APPROVED_SOURCE_ASSETS.map((asset, index) =>
      index === 0 ? unsafeAsset : asset,
    ) as unknown as readonly ApprovedSourceAsset[]

    expect(() => assertApprovedSourceAssetBoundary(unsafe)).toThrow(
      /canonical|unpaired|duplicate|synthetic-only/i,
    )
  })

  it('rejects duplicate verifier entries', () => {
    const duplicate = [
      ...APPROVED_SOURCE_ASSETS.slice(0, -1),
      APPROVED_SOURCE_ASSETS[0],
    ]

    expect(() => assertApprovedSourceAssetBoundary(duplicate)).toThrow(/duplicate|canonical/i)
  })
})
