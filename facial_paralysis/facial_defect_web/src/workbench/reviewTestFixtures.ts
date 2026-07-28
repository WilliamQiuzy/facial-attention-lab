import { listWorkbenchAssets } from './catalog'
import { createInferenceBinding, runMockEngine } from './mockEngine'
import { createInitialWorkspaceState, getCaseRoi, workspaceReducer } from './reducer'
import type {
  ApprovedRoiAnnotation,
  InferenceBinding,
  ResearchReviewNote,
  WorkspaceAction,
  WorkspaceState,
} from './types'

export const AUTHOR_NOTE: ResearchReviewNote = {
  rationale: 'Suitable for a synthetic research explanation.',
  limitations: 'Simulation only; no patient or clinical inference.',
}

export const REVIEWER_NOTE: ResearchReviewNote = {
  rationale: 'Independent research review completed.',
  limitations: 'Not validated for patient-specific or clinical use.',
}

function reduce(state: WorkspaceState, actions: readonly WorkspaceAction[]) {
  return actions.reduce(workspaceReducer, state)
}

export function createSucceededReviewTarget() {
  let state = createInitialWorkspaceState()
  const asset = listWorkbenchAssets()[2]
  const roi = getCaseRoi(state, asset.id) as ApprovedRoiAnnotation
  const binding = createInferenceBinding({
    clientRunId: 'run-policy-001',
    attemptToken: 'token-policy-001',
    caseId: asset.id,
    assetId: asset.id,
    assetSha256: asset.sha256,
    roi,
    modelVersion: 'mock-salience-v0.3',
    modelMode: 'mock_only',
    config: { threshold: 0.42, smoothing: 0.27 },
  })
  const attemptId = 'attempt-policy-001'
  const asyncBinding = {
    runId: binding.clientRunId,
    attemptId,
    attemptToken: binding.attemptToken,
    inputFingerprint: binding.inputFingerprint,
  }
  state = reduce(state, [
    { type: 'run/create', runId: binding.clientRunId, attemptId, binding },
    { type: 'run/validate', ...asyncBinding },
    { type: 'run/queue', ...asyncBinding },
    { type: 'run/start', ...asyncBinding },
    { type: 'run/succeed', ...asyncBinding, output: runMockEngine(binding) },
  ])
  return { state, binding, attemptId, asset, roi }
}

export function targetReference(binding: InferenceBinding, attemptId: string) {
  return {
    runId: binding.clientRunId,
    attemptId,
    resultDigest: runMockEngine(binding).resultDigest,
    inputFingerprint: binding.inputFingerprint,
  }
}

export function createApprovedReviewState() {
  const target = createSucceededReviewTarget()
  const reference = targetReference(target.binding, target.attemptId)
  const awaiting = workspaceReducer(target.state, {
    type: 'review/create',
    reviewId: 'review-1',
    ...reference,
    actorId: 'demo_author',
    note: AUTHOR_NOTE,
  })
  const state = workspaceReducer(awaiting, {
    type: 'review/approve',
    reviewId: 'review-1',
    actorId: 'demo_reviewer',
    note: REVIEWER_NOTE,
  })
  return { ...target, state, reviewId: 'review-1', reference }
}
