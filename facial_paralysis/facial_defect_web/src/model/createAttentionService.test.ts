import { afterEach, describe, expect, it, vi } from 'vitest'
import { createAttentionService } from './createAttentionService'
import { HttpAttentionService } from './httpAttentionService'

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('attention service boundary', () => {
  const imageAId = 'SYN-MOHS-SCC-CHEEK' as const
  const imageASha256 = '1c43951ed068dc7b88a28e1a0e68d724f1f8b649067b81323ceb30fe2cc5eb30'

  it('uses a zero-network, zero-persistence demo provider by default', async () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal('fetch', fetchSpy)
    const localStorageSpy = vi.spyOn(Storage.prototype, 'setItem')
    const sessionStorageSpy = vi.spyOn(Storage.prototype, 'setItem')

    const service = createAttentionService({
      enableConnectedMode: false,
      apiUrl: undefined,
    })
    expect(service.mode).toBe('demo')
    if (service.mode !== 'demo') throw new Error('Expected the demo provider.')
    const analysis = await service.getDemoAnalysis()

    expect(analysis.origin).toBe('mock_simulation')
    expect(analysis.capabilityStatus).toBe('simulated_ui_only')
    expect(analysis.imageRelationship).toBe('unpaired_demo')
    expect(fetchSpy).not.toHaveBeenCalled()
    expect(localStorageSpy).not.toHaveBeenCalled()
    expect(sessionStorageSpy).not.toHaveBeenCalled()
  })

  it('fails closed when connected mode lacks an API URL', () => {
    expect(() =>
      createAttentionService({ enableConnectedMode: true, apiUrl: undefined }),
    ).toThrow(/explicit API URL/i)
  })

  it('rejects an observed-gaze payload returned by the prediction endpoint', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          analysisId: 'unsafe-origin',
          origin: 'observed_gaze',
          capabilityStatus: 'research_unvalidated',
        }),
      }),
    )
    const service = new HttpAttentionService('https://research-api.invalid')

    await expect(
      service.requestModelPrediction({ assetId: imageAId }),
    ).rejects.toThrow(/origin mismatch/i)
  })

  it('rejects a connected result whose region has not been reviewed', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          analysisId: 'unreviewed-region',
          origin: 'model_prediction',
          capabilityStatus: 'research_unvalidated',
          assetId: imageAId,
          assetSha256: imageASha256,
          roi: { reviewStatus: 'demo_placeholder', version: 'roi-1' },
          quality: { status: 'eligible' },
          model: { name: 'future-salience-model', version: '0.1.0' },
        }),
      }),
    )
    const service = new HttpAttentionService('https://research-api.invalid')

    await expect(
      service.requestModelPrediction({ assetId: imageAId }),
    ).rejects.toThrow(/reviewed ROI/i)
  })

  it('rejects observed gaze when the eligible sample is below the protocol minimum', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          analysisId: 'underpowered-observed-result',
          origin: 'observed_gaze',
          capabilityStatus: 'research_unvalidated',
          assetId: imageAId,
          assetSha256: imageASha256,
          roi: { reviewStatus: 'reviewed', version: 'roi-1' },
          quality: {
            status: 'eligible',
            eligibleSampleCount: 8,
            protocolMinimum: 12,
          },
          model: null,
        }),
      }),
    )
    const service = new HttpAttentionService('https://research-api.invalid')

    await expect(
      service.requestObservedAnalysis({ assetId: imageAId }),
    ).rejects.toThrow(/minimum eligible sample/i)
  })

  it('accepts a prediction only when provenance and promotion gates are explicit', async () => {
    const fetchSpy = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        analysisId: 'validated-envelope',
        origin: 'model_prediction',
        capabilityStatus: 'research_unvalidated',
        assetId: imageAId,
        assetSha256: imageASha256,
        roi: { reviewStatus: 'reviewed', version: 'roi-1' },
        quality: { status: 'eligible' },
        model: { name: 'future-salience-model', version: '0.1.0' },
      }),
    })
    vi.stubGlobal('fetch', fetchSpy)
    const service = new HttpAttentionService('https://research-api.invalid')

    await expect(
      service.requestModelPrediction({ assetId: imageAId }),
    ).resolves.toMatchObject({
      analysisId: 'validated-envelope',
      origin: 'model_prediction',
      assetId: imageAId,
      assetSha256: imageASha256,
      roi: { reviewStatus: 'reviewed', version: 'roi-1' },
      quality: { status: 'eligible' },
    })
    expect(fetchSpy).toHaveBeenCalledWith(
      'https://research-api.invalid/api/v1/salience-predictions',
      expect.objectContaining({
        body: JSON.stringify({
          assetId: imageAId,
          assetSha256: imageASha256,
        }),
      }),
    )
  })

  it('rejects an asset outside the approved synthetic registry before fetch', async () => {
    const fetchSpy = vi.fn()
    vi.stubGlobal('fetch', fetchSpy)
    const service = new HttpAttentionService('https://research-api.invalid')

    await expect(
      service.requestModelPrediction({
        assetId: 'not-in-the-approved-registry' as typeof imageAId,
      }),
    ).rejects.toThrow(/approved synthetic asset/i)
    expect(fetchSpy).not.toHaveBeenCalled()
  })

  it('rejects a response that is not bound to the requested asset and hash', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({
          analysisId: 'mismatched-asset',
          origin: 'model_prediction',
          capabilityStatus: 'research_unvalidated',
          assetId: 'SYN-MOHS-NASAL-RECON',
          assetSha256: '0992e19d0a12e43cc685507672d5cc781f834d8325da29d9b0ef0eb121abd70d',
          roi: { reviewStatus: 'reviewed', version: 'roi-1' },
          quality: { status: 'eligible' },
          model: { name: 'future-salience-model', version: '0.1.0' },
        }),
      }),
    )
    const service = new HttpAttentionService('https://research-api.invalid')

    await expect(
      service.requestModelPrediction({ assetId: imageAId }),
    ).rejects.toThrow(/asset provenance mismatch/i)
  })
})
