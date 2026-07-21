import { getWorkbenchAsset } from '../workbench/catalog'
import type {
  ApprovedAssetId,
  ConnectedAnalysisEnvelope,
  ConnectedAttentionService,
} from './types'

type AssetRequest = { assetId: ApprovedAssetId }

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

export class HttpAttentionService implements ConnectedAttentionService {
  readonly mode = 'connected' as const

  constructor(private readonly apiUrl: string) {}

  requestObservedAnalysis(payload: AssetRequest): Promise<ConnectedAnalysisEnvelope> {
    return this.request(
      '/api/v1/attention-analyses',
      payload,
      'observed_gaze',
    )
  }

  requestModelPrediction(payload: AssetRequest): Promise<ConnectedAnalysisEnvelope> {
    return this.request(
      '/api/v1/salience-predictions',
      payload,
      'model_prediction',
    )
  }

  private async request(
    path: string,
    payload: AssetRequest,
    expectedOrigin: ConnectedAnalysisEnvelope['origin'],
  ): Promise<ConnectedAnalysisEnvelope> {
    const approvedAsset = getWorkbenchAsset(payload.assetId)
    if (!approvedAsset) {
      throw new Error('Connected requests require an approved synthetic asset.')
    }

    const response = await fetch(`${this.apiUrl.replace(/\/$/, '')}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        assetId: approvedAsset.id,
        assetSha256: approvedAsset.sha256,
      }),
    })

    if (!response.ok) {
      throw new Error(`Research service request failed with status ${response.status}.`)
    }

    const body: unknown = await response.json()
    if (!isRecord(body)) {
      throw new Error('Research service returned a malformed response.')
    }

    const envelope = body as Partial<ConnectedAnalysisEnvelope>
    if (envelope.origin !== expectedOrigin) {
      throw new Error(
        `Research source origin mismatch: expected ${expectedOrigin}, received ${String(envelope.origin)}.`,
      )
    }
    if (
      typeof envelope.analysisId !== 'string' ||
      envelope.analysisId.trim().length === 0 ||
      envelope.capabilityStatus !== 'research_unvalidated'
    ) {
      throw new Error('Research service response failed provenance validation.')
    }

    if (
      envelope.assetId !== approvedAsset.id ||
      envelope.assetSha256 !== approvedAsset.sha256
    ) {
      throw new Error('Research service response asset provenance mismatch.')
    }

    if (
      !isRecord(envelope.roi) ||
      envelope.roi.reviewStatus !== 'reviewed' ||
      typeof envelope.roi.version !== 'string' ||
      envelope.roi.version.trim().length === 0
    ) {
      throw new Error('Research service response requires a reviewed ROI and version.')
    }

    if (!isRecord(envelope.quality) || envelope.quality.status !== 'eligible') {
      throw new Error('Research service response failed the quality eligibility gate.')
    }

    if (expectedOrigin === 'observed_gaze') {
      const eligibleSampleCount = envelope.quality.eligibleSampleCount
      const protocolMinimum = envelope.quality.protocolMinimum
      if (
        typeof eligibleSampleCount !== 'number' ||
        !Number.isFinite(eligibleSampleCount) ||
        typeof protocolMinimum !== 'number' ||
        !Number.isFinite(protocolMinimum) ||
        eligibleSampleCount < protocolMinimum
      ) {
        throw new Error('Observed gaze did not meet the minimum eligible sample gate.')
      }
    }

    if (
      expectedOrigin === 'model_prediction' &&
      (!isRecord(envelope.model) ||
        typeof envelope.model.name !== 'string' ||
        envelope.model.name.trim().length === 0 ||
        typeof envelope.model.version !== 'string' ||
        envelope.model.version.trim().length === 0)
    ) {
      throw new Error('Model prediction requires an explicit model name and version.')
    }

    return envelope as ConnectedAnalysisEnvelope
  }
}
