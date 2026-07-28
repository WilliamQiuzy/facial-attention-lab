import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  createPatientExportManifest,
  downloadPatientExport,
} from './reviewExport'
import { createApprovedReviewState } from './reviewTestFixtures'

afterEach(() => vi.restoreAllMocks())

function readBlob(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.addEventListener('load', () => resolve(String(reader.result)))
    reader.addEventListener('error', () => reject(reader.error))
    reader.readAsText(blob)
  })
}

describe('patient preview JSON export', () => {
  it('emits only the explicit non-PHI whitelist and excludes workflow IDs, actors, and notes', () => {
    const approved = createApprovedReviewState()

    const result = createPatientExportManifest(approved.state, approved.reviewId)

    expect(result.eligible).toBe(true)
    if (!result.eligible) throw new Error('Expected approved mock export.')
    expect(Object.keys(result.manifest)).toEqual([
      'schemaVersion',
      'origin',
      'capabilityStatus',
      'asset',
      'model',
      'roi',
      'result',
      'review',
      'quality',
      'disclaimers',
    ])
    const serialized = JSON.stringify(result.manifest)
    expect(serialized).toContain(approved.asset.sha256)
    expect(serialized).toContain(approved.reference.resultDigest)
    expect(serialized).not.toContain(approved.binding.clientRunId)
    expect(serialized).not.toContain(approved.attemptId)
    expect(serialized).not.toContain(approved.reviewId)
    expect(serialized).not.toContain('demo_author')
    expect(serialized).not.toContain('demo_reviewer')
    expect(serialized).not.toContain('Independent research review completed.')
    expect(serialized).not.toContain('rationale')
    expect(serialized).not.toContain('limitations')
  })

  it('returns blockers and no manifest when eligibility fails', () => {
    const approved = createApprovedReviewState()
    const revoked = {
      ...approved.state,
      attemptsById: {
        ...approved.state.attemptsById,
        [approved.attemptId]: {
          ...approved.state.attemptsById[approved.attemptId]!,
          result: {
            ...approved.state.attemptsById[approved.attemptId]!.result!,
            freshness: 'revoked' as const,
          },
        },
      },
    }

    const result = createPatientExportManifest(revoked, approved.reviewId)

    expect(result.eligible).toBe(false)
    expect('manifest' in result).toBe(false)
  })

  it('downloads application/json and always revokes the object URL', () => {
    const approved = createApprovedReviewState()
    const result = createPatientExportManifest(approved.state, approved.reviewId)
    if (!result.eligible) throw new Error('Expected approved mock export.')
    const createUrl = vi.fn((_blob: Blob) => 'blob:review-export')
    const revokeUrl = vi.fn()
    Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: createUrl })
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revokeUrl })
    const click = vi.fn()
    vi.spyOn(document, 'createElement').mockReturnValue({
      click,
      href: '',
      download: '',
    } as unknown as HTMLAnchorElement)

    downloadPatientExport(result.manifest)

    expect(createUrl).toHaveBeenCalledOnce()
    expect(createUrl.mock.calls[0][0]).toBeInstanceOf(Blob)
    expect((createUrl.mock.calls[0][0] as Blob).type).toBe('application/json')
    expect(click).toHaveBeenCalledOnce()
    expect(revokeUrl).toHaveBeenCalledWith('blob:review-export')
  })

  it('revokes the object URL even when the browser download click throws', () => {
    const approved = createApprovedReviewState()
    const result = createPatientExportManifest(approved.state, approved.reviewId)
    if (!result.eligible) throw new Error('Expected approved mock export.')
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: vi.fn(() => 'blob:throwing-export'),
    })
    const revokeUrl = vi.fn()
    Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: revokeUrl })
    vi.spyOn(document, 'createElement').mockReturnValue({
      click: () => {
        throw new Error('download blocked')
      },
      href: '',
      download: '',
    } as unknown as HTMLAnchorElement)

    expect(() => downloadPatientExport(result.manifest)).toThrow('download blocked')
    expect(revokeUrl).toHaveBeenCalledWith('blob:throwing-export')
  })

  it('rebuilds the exact whitelist before Blob creation and ignores extra keys and toJSON', async () => {
    const approved = createApprovedReviewState()
    const result = createPatientExportManifest(approved.state, approved.reviewId)
    if (!result.eligible) throw new Error('Expected approved mock export.')
    let capturedBlob: Blob | undefined
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: vi.fn((value: Blob | MediaSource) => {
        if (value instanceof Blob) capturedBlob = value
        return 'blob:sanitized-export'
      }),
    })
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      value: vi.fn(),
    })
    vi.spyOn(document, 'createElement').mockReturnValue({
      click: vi.fn(),
      href: '',
      download: '',
    } as unknown as HTMLAnchorElement)
    const poisoned = {
      ...result.manifest,
      patientName: 'Jane Secret',
      asset: { ...result.manifest.asset, note: 'private note' },
      review: { ...result.manifest.review, rationale: 'private rationale' },
      toJSON: () => ({ patientName: 'toJSON secret' }),
    } as unknown as typeof result.manifest

    downloadPatientExport(poisoned)

    expect(capturedBlob).toBeInstanceOf(Blob)
    if (!capturedBlob) throw new Error('Expected a captured Blob.')
    const serialized = await readBlob(capturedBlob)
    expect(Object.keys(JSON.parse(serialized) as object)).toEqual([
      'schemaVersion',
      'origin',
      'capabilityStatus',
      'asset',
      'model',
      'roi',
      'result',
      'review',
      'quality',
      'disclaimers',
    ])
    expect(serialized).not.toMatch(/Jane Secret|private note|private rationale|toJSON secret/)
  })
})
