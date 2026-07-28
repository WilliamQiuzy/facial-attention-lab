import { evaluatePatientReportEligibility, type ReviewPolicyBlocker } from './reviewPolicy'
import type { WorkspaceState } from './types'

export const PATIENT_EXPORT_DISCLAIMERS = Object.freeze([
  'SIMULATED — NOT HUMAN GAZE',
  'MOCK ONLY · UNVALIDATED · NOT A PATIENT RESULT',
  'Clinical use blocked',
] as const)

export type PatientExportManifest = {
  readonly schemaVersion: 'facial-attention-patient-preview/v1'
  readonly origin: 'mock_simulation'
  readonly capabilityStatus: 'simulated_ui_only'
  readonly asset: { readonly sha256: string }
  readonly model: { readonly version: string }
  readonly roi: { readonly version: number }
  readonly result: { readonly digest: string }
  readonly review: {
    readonly decision: 'approved_for_research'
    readonly event: {
      readonly sequence: number
      readonly decision: 'approved_for_research'
    }
  }
  readonly quality: {
    readonly bindingIntegrity: 'passed'
    readonly sourceBindingIntegrity: 'passed'
    readonly finiteValues: 'passed'
    readonly normalizedBounds: 'passed'
    readonly researchDisplayEligible: true
    readonly clinicalUseEligible: false
  }
  readonly disclaimers: typeof PATIENT_EXPORT_DISCLAIMERS
}

export type PatientExportResult =
  | { readonly eligible: true; readonly manifest: PatientExportManifest }
  | { readonly eligible: false; readonly blockers: readonly ReviewPolicyBlocker[] }

export function createPatientExportManifest(
  state: WorkspaceState,
  reviewId: string,
): PatientExportResult {
  const eligibility = evaluatePatientReportEligibility(state, reviewId)
  if (!eligibility.eligible) {
    return { eligible: false, blockers: eligibility.blockers }
  }
  const latestEvent = eligibility.review.events[eligibility.review.events.length - 1]!
  return {
    eligible: true,
    manifest: {
      schemaVersion: 'facial-attention-patient-preview/v1',
      origin: 'mock_simulation',
      capabilityStatus: 'simulated_ui_only',
      asset: { sha256: eligibility.output.binding.assetSha256 },
      model: { version: eligibility.output.binding.modelVersion },
      roi: { version: eligibility.output.binding.roiVersion },
      result: { digest: eligibility.output.resultDigest },
      review: {
        decision: 'approved_for_research',
        event: {
          sequence: latestEvent.sequence,
          decision: 'approved_for_research',
        },
      },
      quality: {
        bindingIntegrity: 'passed',
        sourceBindingIntegrity: 'passed',
        finiteValues: 'passed',
        normalizedBounds: 'passed',
        researchDisplayEligible: true,
        clinicalUseEligible: false,
      },
      disclaimers: PATIENT_EXPORT_DISCLAIMERS,
    },
  }
}

export function downloadPatientExport(manifest: PatientExportManifest): void {
  const safeManifest = rebuildPatientExportWhitelist(manifest)
  const blob = new Blob([JSON.stringify(safeManifest, null, 2)], {
    type: 'application/json',
  })
  const objectUrl = URL.createObjectURL(blob)
  try {
    const anchor = document.createElement('a')
    anchor.href = objectUrl
    anchor.download = 'facial-attention-patient-preview.json'
    anchor.click()
  } finally {
    URL.revokeObjectURL(objectUrl)
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function rebuildPatientExportWhitelist(
  candidate: PatientExportManifest,
): PatientExportManifest {
  const value: unknown = candidate
  if (!isRecord(value)) throw new TypeError('Invalid patient export manifest.')
  const asset = value.asset
  const model = value.model
  const roi = value.roi
  const result = value.result
  const review = value.review
  const quality = value.quality
  if (
    value.schemaVersion !== 'facial-attention-patient-preview/v1' ||
    value.origin !== 'mock_simulation' ||
    value.capabilityStatus !== 'simulated_ui_only' ||
    !isRecord(asset) ||
    typeof asset.sha256 !== 'string' ||
    !/^[0-9a-f]{64}$/.test(asset.sha256) ||
    !isRecord(model) ||
    (model.version !== 'mock-salience-v0.3' &&
      model.version !== 'mock-salience-v0.4') ||
    !isRecord(roi) ||
    typeof roi.version !== 'number' ||
    !Number.isSafeInteger(roi.version) ||
    roi.version < 1 ||
    !isRecord(result) ||
    typeof result.digest !== 'string' ||
    !/^result_[0-9a-f]{16}$/.test(result.digest) ||
    !isRecord(review) ||
    review.decision !== 'approved_for_research' ||
    !isRecord(review.event) ||
    typeof review.event.sequence !== 'number' ||
    !Number.isSafeInteger(review.event.sequence) ||
    review.event.sequence < 1 ||
    review.event.decision !== 'approved_for_research' ||
    !isRecord(quality) ||
    quality.bindingIntegrity !== 'passed' ||
    quality.sourceBindingIntegrity !== 'passed' ||
    quality.finiteValues !== 'passed' ||
    quality.normalizedBounds !== 'passed' ||
    quality.researchDisplayEligible !== true ||
    quality.clinicalUseEligible !== false
  ) {
    throw new TypeError('Invalid patient export manifest.')
  }

  return {
    schemaVersion: 'facial-attention-patient-preview/v1',
    origin: 'mock_simulation',
    capabilityStatus: 'simulated_ui_only',
    asset: { sha256: asset.sha256 },
    model: { version: model.version },
    roi: { version: roi.version },
    result: { digest: result.digest },
    review: {
      decision: 'approved_for_research',
      event: {
        sequence: review.event.sequence,
        decision: 'approved_for_research',
      },
    },
    quality: {
      bindingIntegrity: 'passed',
      sourceBindingIntegrity: 'passed',
      finiteValues: 'passed',
      normalizedBounds: 'passed',
      researchDisplayEligible: true,
      clinicalUseEligible: false,
    },
    disclaimers: PATIENT_EXPORT_DISCLAIMERS,
  }
}
