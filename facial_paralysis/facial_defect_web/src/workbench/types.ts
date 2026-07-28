import type { WorkbenchAssetId } from '../data/workbenchAssetDefinitions'

export type WorkbenchActor =
  | { readonly id: 'demo_author'; readonly role: 'author' }
  | { readonly id: 'demo_reviewer'; readonly role: 'reviewer' }

export type WorkbenchActorId = WorkbenchActor['id']
export type WorkbenchActorRole = WorkbenchActor['role']

export const WORKBENCH_ACTORS = {
  demo_author: { id: 'demo_author', role: 'author' },
  demo_reviewer: { id: 'demo_reviewer', role: 'reviewer' },
} as const satisfies Record<WorkbenchActorId, WorkbenchActor>

export type NormalizedRoi = {
  readonly x: number
  readonly y: number
  readonly width: number
  readonly height: number
}

export type RoiStatus =
  | 'draft'
  | 'in_review'
  | 'approved'
  | 'changes_requested'
  | 'superseded'

export type RoiAnnotation = {
  readonly id: string
  readonly caseId: string
  readonly assetId: string
  readonly version: number
  readonly geometry: NormalizedRoi
  readonly status: RoiStatus
  readonly authorId: 'demo_author'
  readonly reviewerId?: 'demo_reviewer'
}

export type ApprovedRoiAnnotation = Omit<
  RoiAnnotation,
  'status' | 'reviewerId'
> & {
  readonly status: 'approved'
  readonly reviewerId: 'demo_reviewer'
}

export type MockModelVersion = 'mock-salience-v0.3' | 'mock-salience-v0.4'
export type ModelMode = 'mock_only'

export type InferenceConfiguration = {
  readonly threshold: number
  readonly smoothing: number
}

export type CreateInferenceBindingInput = {
  readonly clientRunId: string
  readonly attemptToken: string
  readonly caseId: string
  readonly assetId: string
  readonly assetSha256: string
  readonly roi: RoiAnnotation
  readonly modelVersion: MockModelVersion
  readonly modelMode: ModelMode
  readonly config: InferenceConfiguration
}

export type ScientificInferenceInput = {
  readonly assetId: string
  readonly assetSha256: string
  readonly roiId: string
  readonly roiVersion: number
  readonly roiGeometry: NormalizedRoi
  readonly modelVersion: MockModelVersion
  readonly configurationHash: string
}

export type InferenceBinding = {
  readonly clientRunId: string
  readonly attemptToken: string
  readonly caseId: WorkbenchAssetId
  readonly assetId: WorkbenchAssetId
  readonly assetSha256: string
  readonly roiId: string
  readonly roiVersion: number
  readonly roiGeometry: Readonly<NormalizedRoi>
  readonly roiStatus: 'approved'
  readonly modelVersion: MockModelVersion
  readonly modelMode: ModelMode
  readonly config: Readonly<InferenceConfiguration>
  readonly configurationHash: string
  readonly inputFingerprint: string
}

export const CONNECTED_ATTENTION_REQUEST_PROFILE_VERSION =
  'synthetic-spatial-contract-rehearsal/1' as const

export type ConnectedAttentionRequestV1 = {
  readonly requestProfileVersion: typeof CONNECTED_ATTENTION_REQUEST_PROFILE_VERSION
  readonly clientRunId: string
  readonly attemptToken: string
  readonly caseId: WorkbenchAssetId
  readonly assetId: WorkbenchAssetId
  readonly sourceSha256: string
  readonly sourceBinding: Readonly<{
    readonly id: string
    readonly version: number
    readonly geometry: Readonly<NormalizedRoi>
  }>
}

export type ConnectedModelIdentity = Readonly<{
  readonly modelId: string
  readonly modelVersion: string
  readonly artifactSha256: string
  readonly preprocessingVersion: string
  readonly calibrationVersion: string
  readonly displayScaleId: string
}>

export type HeatmapPoint = {
  readonly x: number
  readonly y: number
  readonly intensity: number
  readonly radius: number
}

export const MAX_SPATIAL_DISPLAY_POINTS = 4_096

export type ClinicalAoiRegistration =
  | 'synthetic_template_v1'
  | 'registration_geometry_unavailable_v1'

export type SpatialAttentionSemantics<
  TRegistration extends ClinicalAoiRegistration = ClinicalAoiRegistration,
> = Readonly<{
  readonly schemaVersion: 'predicted-observer-attention/1'
  readonly fieldMeaning: 'relative_spatial_density'
  readonly target: 'predicted_observer_attention'
  readonly interpretation: 'population_level'
  readonly normalization: 'shared_display_scale_required'
  readonly clinicalAoi: Readonly<{
    readonly registration: TRegistration
    readonly role: 'post_inference_summary'
    readonly modifiesPrediction: false
  }>
}>

export type InferenceQualityGates = {
  readonly bindingIntegrity: 'passed'
  readonly sourceBindingIntegrity: 'passed'
  readonly finiteValues: 'passed'
  readonly normalizedBounds: 'passed'
  readonly researchDisplayEligible: boolean
  readonly clinicalUseEligible: false
}

export type MockInferenceProvenance = {
  readonly engine: 'deterministic_mock_engine'
  readonly engineVersion: '1'
  readonly modelMode: 'mock_only'
  readonly canonicalSyntheticAsset: true
  readonly deterministic: true
  readonly networkAccessed: false
  readonly storageAccessed: false
  readonly humanGazeData: false
}

export type ConnectedInferenceProvenance = {
  readonly engine: 'connected_model_gateway'
  readonly engineVersion: string
  readonly canonicalSyntheticAsset: boolean
  readonly deterministic: boolean
  readonly networkAccessed: boolean
  readonly storageAccessed: boolean
  readonly observedGazePayloadIncluded: false
  readonly trainingDataProvenance: 'not_disclosed'
}

export type InferenceOutputCore<
  TProvenance,
  TRegistration extends ClinicalAoiRegistration,
> = {
  readonly binding: InferenceBinding
  readonly resultDigest: string
  readonly attentionSemantics: SpatialAttentionSemantics<TRegistration>
  readonly heatmap: readonly HeatmapPoint[]
  readonly qualityGates: InferenceQualityGates
  readonly provenance: TProvenance
}

export type MockInferenceOutput = InferenceOutputCore<
  MockInferenceProvenance,
  'synthetic_template_v1'
> & {
  readonly origin: 'mock_simulation'
  readonly capabilityStatus: 'simulated_ui_only'
  readonly watermark: 'SIMULATED — NOT HUMAN GAZE'
}

export type ConnectedInferenceOutput =
  InferenceOutputCore<
    ConnectedInferenceProvenance,
    'registration_geometry_unavailable_v1'
  > & {
    readonly origin: 'model_prediction'
    readonly capabilityStatus: 'research_unvalidated'
    readonly watermark: 'MODEL PREDICTION — RESEARCH UNVALIDATED — NOT HUMAN GAZE — CLINICAL USE BLOCKED'
    readonly modelIdentity: ConnectedModelIdentity
  }

export type InferenceOutput = MockInferenceOutput | ConnectedInferenceOutput

export type WorkbenchFailureReason =
  | 'UNKNOWN_CASE'
  | 'UNKNOWN_ASSET'
  | 'CASE_ASSET_MISMATCH'
  | 'ASSET_HASH_MISMATCH'
  | 'ROI_BINDING_MISMATCH'
  | 'ROI_NOT_APPROVED'
  | 'FULL_IMAGE_SOURCE_BINDING_REQUIRED'
  | 'INVALID_ROI_GEOMETRY'
  | 'INVALID_ROI_VERSION'
  | 'INVALID_ROI_ACTORS'
  | 'UNKNOWN_MODEL'
  | 'UNSUPPORTED_MODEL_MODE'
  | 'INVALID_CONFIGURATION'
  | 'INVALID_OPERATIONAL_ID'
  | 'BINDING_INTEGRITY_MISMATCH'
  | 'REQUEST_ABORTED'
  | 'REQUEST_TIMEOUT'
  | 'NETWORK_ERROR'
  | 'HTTP_ERROR'
  | 'MALFORMED_RESPONSE'
  | 'ORIGIN_MISMATCH'
  | 'CAPABILITY_MISMATCH'
  | 'IMMUTABLE_BINDING_MISMATCH'
  | 'INVALID_API_URL'
  | 'INVALID_TIMEOUT'
  | 'CONNECTED_MODE_NOT_CONFIGURED'
  | 'CONNECTED_GATEWAY_NOT_IMPLEMENTED'
  | 'SELF_REVIEW_FORBIDDEN'
  | 'ACTOR_NOT_AUTHORIZED'
  | 'INVALID_REVIEW_NOTE'
  | 'UNKNOWN_REVIEW'
  | 'UNKNOWN_REVIEW_TARGET'
  | 'ILLEGAL_TRANSITION'

export type WorkbenchFailure = {
  readonly reason: WorkbenchFailureReason
  readonly message: string
  readonly field?: string
}

export class WorkbenchError extends Error {
  readonly failure: WorkbenchFailure

  constructor(failure: WorkbenchFailure) {
    super(failure.message)
    this.name = 'WorkbenchError'
    this.failure = Object.freeze({ ...failure })
  }

  get reason(): WorkbenchFailureReason {
    return this.failure.reason
  }
}

export type RunAttemptStatus =
  | 'draft'
  | 'validating'
  | 'blocked'
  | 'queued'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'cancelled'

export type InferenceAttempt = {
  readonly id: string
  readonly clientRunId: string
  readonly attemptToken: string
  readonly parentAttemptId?: string
  readonly status: RunAttemptStatus
  readonly binding?: InferenceBinding
  readonly result?: StoredInferenceResult
  readonly failure?: WorkbenchFailure
}

export type InferenceRun = {
  readonly clientRunId: string
  readonly caseId: WorkbenchAssetId
  readonly assetId: WorkbenchAssetId
  readonly status: RunAttemptStatus
  readonly attemptIds: readonly string[]
  readonly activeAttemptId?: string
}

export type BatchJobStatus =
  | 'draft'
  | 'preflighting'
  | 'ready'
  | 'blocked'
  | 'queued'
  | 'running'
  | 'completed'
  | 'completed_with_failures'
  | 'failed'
  | 'cancelled'

export type InferenceBatch = {
  readonly id: string
  readonly status: BatchJobStatus
  readonly attemptIds: readonly string[]
}

export type ReviewStatus =
  | 'awaiting_review'
  | 'approved_for_research'
  | 'changes_requested'
  | 'revoked'

export type ResearchReviewNote = {
  readonly rationale: string
  readonly limitations: string
}

export type ResultReviewEvent = {
  readonly sequence: number
  readonly decision: ReviewStatus
  readonly actorId: WorkbenchActorId
  readonly note: ResearchReviewNote
}

export type ResultReview = {
  readonly id: string
  readonly runId: string
  readonly attemptId: string
  readonly resultDigest: string
  readonly inputFingerprint: string
  readonly authorId: 'demo_author'
  readonly reviewerId: 'demo_reviewer'
  readonly status: ReviewStatus
  readonly decision: ReviewStatus
  readonly events: readonly ResultReviewEvent[]
}

export type ResultFreshness = 'current' | 'stale' | 'revoked'

export type StoredInferenceResult = {
  readonly output: InferenceOutput
  readonly freshness: ResultFreshness
}

export type WorkspaceState = {
  readonly roisByCase: Readonly<Partial<Record<WorkbenchAssetId, RoiAnnotation>>>
  readonly runsById: Readonly<Record<string, InferenceRun>>
  readonly runOrder: readonly string[]
  readonly attemptsById: Readonly<Record<string, InferenceAttempt>>
  readonly reviewsById: Readonly<Record<string, ResultReview>>
  readonly reviewOrder: readonly string[]
  readonly activeRunId?: string
  readonly lastFailure?: WorkbenchFailure
}

export type AsyncRunActionBinding = {
  readonly runId: string
  readonly attemptId: string
  readonly attemptToken: string
  readonly inputFingerprint: string
}

export type WorkspaceAction =
  | { readonly type: 'session/reset' }
  | {
      readonly type: 'sourceBinding/restoreFullImage'
      readonly caseId: string
    }
  | {
      readonly type: 'roi/updateGeometry'
      readonly caseId: string
      readonly actorId: string
      readonly geometry: NormalizedRoi
    }
  | {
      readonly type:
        | 'roi/submitReview'
        | 'roi/approve'
        | 'roi/requestChanges'
        | 'roi/reopenDraft'
        | 'roi/supersede'
      readonly caseId: string
      readonly actorId: string
    }
  | {
      readonly type: 'run/create'
      readonly runId: string
      readonly attemptId: string
      readonly binding: InferenceBinding
    }
  | ({ readonly type: 'run/validate' | 'run/queue' | 'run/start' } &
      AsyncRunActionBinding)
  | ({ readonly type: 'run/block' | 'run/fail'; readonly failure: WorkbenchFailure } &
      AsyncRunActionBinding)
  | ({ readonly type: 'run/cancel' } & AsyncRunActionBinding)
  | ({ readonly type: 'run/succeed'; readonly output: InferenceOutput } &
      AsyncRunActionBinding)
  | {
      readonly type: 'run/retry'
      readonly runId: string
      readonly attemptId: string
      readonly parentAttemptId: string
      readonly binding: InferenceBinding
    }
  | {
      readonly type: 'result/revoke'
      readonly runId: string
      readonly attemptId: string
    }
  | {
      readonly type: 'review/create'
      readonly reviewId: string
      readonly runId: string
      readonly attemptId: string
      readonly resultDigest: string
      readonly inputFingerprint: string
      readonly actorId: string
      readonly note: ResearchReviewNote
    }
  | {
      readonly type:
        | 'review/approve'
        | 'review/requestChanges'
        | 'review/resubmit'
        | 'review/revoke'
      readonly reviewId: string
      readonly actorId: string
      readonly note: ResearchReviewNote
    }
