import type { ReactNode } from 'react'
import { Component } from 'react'
import { CONNECTED_INFERENCE_WATERMARK } from '../workbench/inferenceEnvelope'
import { createInferenceBinding, runMockEngine } from '../workbench/mockEngine'
import {
  createInitialWorkspaceState,
  getCaseRoi,
  workspaceReducer,
} from '../workbench/reducer'
import { listWorkbenchAssets } from '../workbench/catalog'
import type {
  ApprovedRoiAnnotation,
  InferenceOutput,
  ReviewStatus,
  WorkspaceAction,
  WorkspaceState,
} from '../workbench/types'

type SeedOptions = {
  readonly suffix?: string
  readonly assetIndex?: number
  readonly origin?: InferenceOutput['origin']
  readonly reviewStatus?: ReviewStatus
  readonly freshness?: 'current' | 'stale' | 'revoked'
}

export type SeededTask6State = {
  readonly state: WorkspaceState
  readonly runId: string
  readonly attemptId: string
  readonly reviewId: string
  readonly resultDigest: string
  readonly caseId: string
  readonly assetSha256: string
  readonly modelVersion: string
  readonly roiVersion: number
}

function reduce(state: WorkspaceState, actions: readonly WorkspaceAction[]) {
  return actions.reduce(workspaceReducer, state)
}

function connectedOutput(output: ReturnType<typeof runMockEngine>): InferenceOutput {
  return {
    ...output,
    attentionSemantics: {
      ...output.attentionSemantics,
      clinicalAoi: {
        ...output.attentionSemantics.clinicalAoi,
        registration: 'registration_geometry_unavailable_v1',
      },
    },
    origin: 'model_prediction',
    capabilityStatus: 'research_unvalidated',
    watermark: CONNECTED_INFERENCE_WATERMARK,
    resultDigest: `connected_${output.resultDigest}`,
    modelIdentity: {
      modelId: 'observer-attention-test',
      modelVersion: 'test-v1',
      artifactSha256: 'a'.repeat(64),
      preprocessingVersion: 'preprocess-v1',
      calibrationVersion: 'calibration-v1',
      displayScaleId: 'display-scale-v1',
    },
    provenance: {
      engine: 'connected_model_gateway',
      engineVersion: 'task6-test',
      canonicalSyntheticAsset: true,
      deterministic: true,
      networkAccessed: true,
      storageAccessed: false,
      observedGazePayloadIncluded: false,
      trainingDataProvenance: 'not_disclosed',
    },
  }
}

export function seedTask6State(options: SeedOptions = {}): SeededTask6State {
  const suffix = options.suffix ?? '1'
  const asset = listWorkbenchAssets()[options.assetIndex ?? 2]
  const initial = createInitialWorkspaceState()
  const roi = getCaseRoi(initial, asset.id) as ApprovedRoiAnnotation
  const runId = `run-task6-${suffix}`
  const attemptId = `attempt-task6-${suffix}`
  const reviewId = `review-task6-${suffix}`
  const binding = createInferenceBinding({
    clientRunId: runId,
    attemptToken: `token-task6-${suffix}`,
    caseId: asset.id,
    assetId: asset.id,
    assetSha256: asset.sha256,
    roi,
    modelVersion: 'mock-salience-v0.4',
    modelMode: 'mock_only',
    config: { threshold: 0.42, smoothing: 0.27 },
  })
  const output = options.origin === 'model_prediction'
    ? connectedOutput(runMockEngine(binding))
    : runMockEngine(binding)
  const asyncBinding = {
    runId,
    attemptId,
    attemptToken: binding.attemptToken,
    inputFingerprint: binding.inputFingerprint,
  }
  let state = reduce(initial, [
    { type: 'run/create', runId, attemptId, binding },
    { type: 'run/validate', ...asyncBinding },
    { type: 'run/queue', ...asyncBinding },
    { type: 'run/start', ...asyncBinding },
    { type: 'run/succeed', ...asyncBinding, output },
  ])

  if (options.reviewStatus) {
    state = workspaceReducer(state, {
      type: 'review/create',
      reviewId,
      runId,
      attemptId,
      resultDigest: output.resultDigest,
      inputFingerprint: binding.inputFingerprint,
      actorId: 'demo_author',
      note: {
        rationale: 'Suitable for a synthetic research demonstration.',
        limitations: 'Simulated attention only; no clinical inference.',
      },
    })

    if (options.reviewStatus === 'changes_requested') {
      state = workspaceReducer(state, {
        type: 'review/requestChanges',
        reviewId,
        actorId: 'demo_reviewer',
        note: {
          rationale: 'Clarify the synthetic-only boundary.',
          limitations: 'No patient or human gaze evidence is present.',
        },
      })
    }
    if (
      options.reviewStatus === 'approved_for_research' ||
      options.reviewStatus === 'revoked'
    ) {
      state = workspaceReducer(state, {
        type: 'review/approve',
        reviewId,
        actorId: 'demo_reviewer',
        note: {
          rationale: 'Approved for this research demonstration only.',
          limitations: 'Clinical and patient interpretation remain blocked.',
        },
      })
    }
    if (options.reviewStatus === 'revoked') {
      state = workspaceReducer(state, {
        type: 'review/revoke',
        reviewId,
        actorId: 'demo_reviewer',
        note: {
          rationale: 'Research display approval withdrawn.',
          limitations: 'Patient preview and export must remain unavailable.',
        },
      })
    }
  }

  if (options.freshness && options.freshness !== 'current') {
    const attempt = state.attemptsById[attemptId]
    if (attempt?.result) {
      state = {
        ...state,
        attemptsById: {
          ...state.attemptsById,
          [attemptId]: {
            ...attempt,
            result: { ...attempt.result, freshness: options.freshness },
          },
        },
      }
    }
  }

  return {
    state,
    runId,
    attemptId,
    reviewId,
    resultDigest: output.resultDigest,
    caseId: asset.id,
    assetSha256: asset.sha256,
    modelVersion: binding.modelVersion,
    roiVersion: binding.roiVersion,
  }
}

type BoundaryState = { readonly crashed: boolean }

export class TestRenderBoundary extends Component<
  { readonly children: ReactNode },
  BoundaryState
> {
  state: BoundaryState = { crashed: false }

  static getDerivedStateFromError(): BoundaryState {
    return { crashed: true }
  }

  render() {
    return this.state.crashed
      ? <h1>Render crashed instead of failing closed</h1>
      : this.props.children
  }
}
