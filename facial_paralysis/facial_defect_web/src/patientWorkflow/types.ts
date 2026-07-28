import type { WorkbenchAssetId } from '../data/workbenchAssetDefinitions'

declare const patientIdBrand: unique symbol
declare const patientVisitIdBrand: unique symbol
declare const authorizationSnapshotIdBrand: unique symbol
declare const captureAssetIdBrand: unique symbol
declare const patientRunIdBrand: unique symbol
declare const patientResultIdBrand: unique symbol
declare const patientReviewIdBrand: unique symbol

export type PatientId = string & {
  readonly [patientIdBrand]: 'PatientId'
}
export type PatientVisitId = string & {
  readonly [patientVisitIdBrand]: 'PatientVisitId'
}
export type AuthorizationSnapshotId = string & {
  readonly [authorizationSnapshotIdBrand]: 'AuthorizationSnapshotId'
}
export type CaptureAssetId = string & {
  readonly [captureAssetIdBrand]: 'CaptureAssetId'
}
export type PatientRunId = string & {
  readonly [patientRunIdBrand]: 'PatientRunId'
}
export type PatientResultId = string & {
  readonly [patientResultIdBrand]: 'PatientResultId'
}
export type PatientReviewId = string & {
  readonly [patientReviewIdBrand]: 'PatientReviewId'
}

declare const sessionMediaHandleBrand: unique symbol

export type SessionMediaHandle = string & {
  readonly [sessionMediaHandleBrand]: 'SessionMediaHandle'
}

export type PatientRecordKind = 'synthetic_demo' | 'session_test'

export type PatientRecord = {
  readonly id: PatientId
  readonly displayName: string
  readonly recordNumber: string
  readonly dateOfBirth: string
  readonly carePathway: string
  readonly recordKind: PatientRecordKind
  readonly createdAt: string
}

export type PatientTimepoint =
  | 'preoperative'
  | 'postoperative'
  | 'follow_up'

export type PatientVisit = {
  readonly id: PatientVisitId
  readonly patientId: PatientId
  readonly timepoint: PatientTimepoint
  readonly visitDate: string
  readonly createdAt: string
}

export type AuthorizationStatus = 'documented' | 'withdrawn'

export type AuthorizationSnapshot = {
  readonly id: AuthorizationSnapshotId
  readonly patientId: PatientId
  readonly visitId: PatientVisitId
  readonly revision: number
  readonly status: AuthorizationStatus
  readonly recordedAt: string
}

export type CaptureSource = 'camera' | 'upload' | 'synthetic_demo'

export type CaptureProtocol = 'frontal_relaxed_non_mirrored_v1'

export type CaptureQualityChecks = {
  readonly faceVisibleAndCentered: boolean
  readonly focusLightingAndOcclusionAcceptable: boolean
  readonly orientationConfirmed: boolean
  readonly authorizationDocumented: boolean
}

export type CaptureAsset = {
  readonly id: CaptureAssetId
  readonly patientId: PatientId
  readonly visitId: PatientVisitId
  readonly version: number
  readonly status: 'current' | 'superseded'
  readonly source: CaptureSource
  /**
   * Opaque key owned by the injected session media vault. It is not a URL and
   * the reducer never stores the underlying Blob or byte buffer.
   */
  readonly mediaHandle: SessionMediaHandle
  readonly sha256: string
  readonly mimeType: 'image/jpeg' | 'image/png' | 'image/webp'
  readonly sizeBytes: number
  readonly width: number
  readonly height: number
  readonly captureProtocol: CaptureProtocol
  readonly qualityChecks: Readonly<CaptureQualityChecks>
  readonly capturedAt: string
  readonly qualityConfirmedAt?: string
  readonly syntheticSourceAssetId?: WorkbenchAssetId
  readonly supersededByCaptureId?: CaptureAssetId
}

export type PatientRunBinding = {
  readonly patientId: PatientId
  readonly visitId: PatientVisitId
  readonly captureId: CaptureAssetId
  readonly captureVersion: number
  readonly captureSha256: string
  readonly mediaHandle: SessionMediaHandle
  readonly authorizationRevision: number
  readonly captureProtocol: CaptureProtocol
}

export type PatientRunStatus =
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'

export type PatientWorkflowFailureCode =
  | 'INVALID_PATIENT'
  | 'DUPLICATE_PATIENT_ID'
  | 'DUPLICATE_RECORD_NUMBER'
  | 'UNKNOWN_PATIENT'
  | 'INVALID_VISIT'
  | 'DUPLICATE_VISIT_ID'
  | 'UNKNOWN_VISIT'
  | 'VISIT_OWNERSHIP_MISMATCH'
  | 'INVALID_AUTHORIZATION'
  | 'DUPLICATE_AUTHORIZATION_ID'
  | 'INVALID_AUTHORIZATION_REVISION'
  | 'INVALID_CAPTURE'
  | 'DUPLICATE_CAPTURE_ID'
  | 'UNKNOWN_CAPTURE'
  | 'INVALID_CAPTURE_VERSION'
  | 'CAPTURE_NOT_CURRENT'
  | 'CAPTURE_QUALITY_INCOMPLETE'
  | 'CAPTURE_QUALITY_LOCKED'
  | 'INVALID_RUN_BINDING'
  | 'INVALID_RETRY_BINDING'
  | 'DUPLICATE_RUN_ID'
  | 'UNKNOWN_RUN'
  | 'INVALID_RUN_TRANSITION'
  | 'INVALID_RUN_FAILURE'
  | 'ANALYSIS_FAILED'
  | 'INVALID_RESULT'
  | 'DUPLICATE_RESULT_ID'
  | 'INVALID_REVIEW'
  | 'DUPLICATE_REVIEW_ID'
  | 'REPEAT_REASON_REQUIRED'

export type PatientWorkflowFailure = {
  readonly code: PatientWorkflowFailureCode
  readonly message: string
  readonly field?: string
}

export type PatientRun = {
  readonly id: PatientRunId
  readonly status: PatientRunStatus
  readonly binding: Readonly<PatientRunBinding>
  readonly createdAt: string
  readonly retryOfRunId?: PatientRunId
  readonly failure?: PatientWorkflowFailure
}

export type PatientAttentionPoint = {
  readonly x: number
  readonly y: number
  readonly intensity: number
  readonly radius: number
}

export type PatientSimulationOutput = {
  readonly origin: 'workflow_simulation'
  readonly points: readonly PatientAttentionPoint[]
}

export type PatientResult = {
  readonly id: PatientResultId
  readonly runId: PatientRunId
  readonly binding: Readonly<PatientRunBinding>
  readonly freshness: 'current' | 'stale'
  readonly createdAt: string
  readonly output: Readonly<PatientSimulationOutput>
}

export type PatientReviewDecision = 'reviewed' | 'repeat_photo'

export type PatientReview = {
  readonly id: PatientReviewId
  readonly patientId: PatientId
  readonly visitId: PatientVisitId
  readonly resultId: PatientResultId
  readonly captureId: CaptureAssetId
  readonly decision: PatientReviewDecision
  readonly note?: string
  readonly completedAt: string
}

export type PatientWorkflowState = {
  readonly patientsById: Readonly<
    Partial<Record<string, PatientRecord>>
  >
  readonly patientOrder: readonly PatientId[]
  readonly visitsById: Readonly<
    Partial<Record<string, PatientVisit>>
  >
  readonly visitOrder: readonly PatientVisitId[]
  readonly authorizationsById: Readonly<
    Partial<Record<string, AuthorizationSnapshot>>
  >
  readonly authorizationOrder: readonly AuthorizationSnapshotId[]
  readonly capturesById: Readonly<
    Partial<Record<string, CaptureAsset>>
  >
  readonly captureOrder: readonly CaptureAssetId[]
  readonly runsById: Readonly<
    Partial<Record<string, PatientRun>>
  >
  readonly runOrder: readonly PatientRunId[]
  readonly resultsById: Readonly<
    Partial<Record<string, PatientResult>>
  >
  readonly resultOrder: readonly PatientResultId[]
  readonly reviewsById: Readonly<
    Partial<Record<string, PatientReview>>
  >
  readonly reviewOrder: readonly PatientReviewId[]
  readonly lastFailure?: PatientWorkflowFailure
}

export type PatientWorkflowAction =
  | {
      readonly type: 'patient/create'
      readonly patient: PatientRecord
      readonly trustedToday: string
      readonly syntheticTestAttestation: boolean
    }
  | {
      readonly type: 'visit/create'
      readonly visit: PatientVisit
      readonly trustedToday: string
    }
  | {
      readonly type: 'authorization/record'
      readonly authorization: AuthorizationSnapshot
    }
  | {
      readonly type: 'capture/add'
      readonly capture: CaptureAsset
    }
  | {
      readonly type: 'capture/quality/set'
      readonly captureId: CaptureAssetId
      readonly checks: CaptureQualityChecks
      readonly confirmedAt?: string
    }
  | {
      readonly type: 'run/create'
      readonly run: PatientRun
    }
  | {
      readonly type: 'run/status/set'
      readonly runId: PatientRunId
      readonly status: PatientRunStatus
      readonly failure?: PatientWorkflowFailure
    }
  | {
      readonly type: 'result/record'
      readonly result: PatientResult
    }
  | {
      readonly type: 'review/record'
      readonly review: PatientReview
    }

export type VisitNextAction =
  | 'capture_photo'
  | 'confirm_quality'
  | 'run_analysis'
  | 'processing'
  | 'retry_analysis'
  | 'review_result'
  | 'visit_complete'
  | 'retake'

export type PatientNextAction = 'start_visit' | VisitNextAction
