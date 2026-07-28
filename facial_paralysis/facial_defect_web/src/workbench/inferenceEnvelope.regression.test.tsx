import { useState } from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { listWorkbenchAssets } from './catalog'
import { createInferenceBinding, runMockEngine } from './mockEngine'
import {
  createInitialWorkspaceState,
  getCaseRoi,
  workspaceReducer,
} from './reducer'
import type { WorkbenchGateway } from './WorkbenchGateway'
import { WorkspaceProvider, useWorkspace } from './WorkspaceProvider'
import type {
  ApprovedRoiAnnotation,
  InferenceBinding,
  InferenceOutput,
  WorkspaceAction,
  WorkspaceState,
} from './types'

const asset = listWorkbenchAssets()[2]

function bindingFor(state: WorkspaceState): InferenceBinding {
  return createInferenceBinding({
    clientRunId: 'run-envelope',
    attemptToken: 'token-envelope',
    caseId: asset.id,
    assetId: asset.id,
    assetSha256: asset.sha256,
    roi: getCaseRoi(state, asset.id) as ApprovedRoiAnnotation,
    modelVersion: 'mock-salience-v0.3',
    modelMode: 'mock_only',
    config: { threshold: 0.42, smoothing: 0.27 },
  })
}

function reduce(state: WorkspaceState, actions: readonly WorkspaceAction[]) {
  return actions.reduce(workspaceReducer, state)
}

function runningState() {
  const initial = createInitialWorkspaceState()
  const binding = bindingFor(initial)
  const attemptId = 'attempt-envelope'
  const asyncBinding = {
    runId: binding.clientRunId,
    attemptId,
    attemptToken: binding.attemptToken,
    inputFingerprint: binding.inputFingerprint,
  }
  const state = reduce(initial, [
    { type: 'run/create', runId: binding.clientRunId, attemptId, binding },
    { type: 'run/validate', ...asyncBinding },
    { type: 'run/queue', ...asyncBinding },
    { type: 'run/start', ...asyncBinding },
  ])
  return { state, binding, attemptId, asyncBinding }
}

function outputWithLegacyMetrics(binding: InferenceBinding): InferenceOutput {
  const valid = runMockEngine(binding)
  return {
    ...valid,
    metrics: {
      roiCoverage: 0.5,
      peakIntensity: 0.5,
      meanIntensity: 0.5,
      focusScore: 0.5,
    },
  } as unknown as InferenceOutput
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

function withChangedPath<T>(
  value: T,
  path: readonly string[],
  replacement: unknown,
): T {
  const changed = clone(value) as unknown as Record<string, unknown>
  let target = changed
  for (const key of path.slice(0, -1)) {
    target = target[key] as Record<string, unknown>
  }
  target[path.at(-1)!] = replacement
  return changed as T
}

function withoutPath<T>(value: T, path: readonly string[]): T {
  const changed = clone(value) as unknown as Record<string, unknown>
  let target = changed
  for (const key of path.slice(0, -1)) {
    target = target[key] as Record<string, unknown>
  }
  delete target[path.at(-1)!]
  return changed as T
}

function mockOutputWithSemantics(binding: InferenceBinding): InferenceOutput {
  return {
    ...runMockEngine(binding),
    attentionSemantics: {
      schemaVersion: 'predicted-observer-attention/1',
      fieldMeaning: 'relative_spatial_density',
      target: 'predicted_observer_attention',
      interpretation: 'population_level',
      normalization: 'shared_display_scale_required',
      clinicalAoi: {
        registration: 'synthetic_template_v1',
        role: 'post_inference_summary',
        modifiesPrediction: false,
      },
    },
  } as unknown as InferenceOutput
}

function ProviderProbe() {
  const { state, actions } = useWorkspace()
  const [runId, setRunId] = useState<string>()
  const run = runId ? state.runsById[runId] : undefined
  const attempt = run?.activeAttemptId
    ? state.attemptsById[run.activeAttemptId]
    : undefined
  return (
    <>
      <button
        type="button"
        onClick={() =>
          setRunId(
            actions.startRun({
              caseId: asset.id,
              modelVersion: 'mock-salience-v0.3',
              config: { threshold: 0.42, smoothing: 0.27 },
            }).runId,
          )
        }
      >
        Start validation run
      </button>
      <output aria-label="validated attempt status">{attempt?.status ?? 'none'}</output>
      <output aria-label="validated failure reason">{attempt?.failure?.reason ?? 'none'}</output>
    </>
  )
}

describe('complete inference envelope validation', () => {
  it('fails a reducer attempt instead of storing matching-binding malformed numeric output', () => {
    const { state, binding, attemptId, asyncBinding } = runningState()
    const next = workspaceReducer(state, {
      type: 'run/succeed',
      ...asyncBinding,
      output: outputWithLegacyMetrics(binding),
    })

    expect(next.attemptsById[attemptId].status).toBe('failed')
    expect(next.attemptsById[attemptId].failure?.reason).toBe('MALFORMED_RESPONSE')
    expect(next.attemptsById[attemptId].result).toBeUndefined()
  })

  it('fails a reducer attempt for branch-inconsistent provenance without throwing', () => {
    const { state, binding, attemptId, asyncBinding } = runningState()
    const valid = runMockEngine(binding)
    const malformed = {
      ...valid,
      provenance: { ...valid.provenance, networkAccessed: true },
    } as unknown as InferenceOutput

    expect(() =>
      workspaceReducer(state, {
        type: 'run/succeed',
        ...asyncBinding,
        output: malformed,
      }),
    ).not.toThrow()
    const next = workspaceReducer(state, {
      type: 'run/succeed',
      ...asyncBinding,
      output: malformed,
    })
    expect(next.attemptsById[attemptId].status).toBe('failed')
    expect(next.attemptsById[attemptId].failure?.reason).toBe('MALFORMED_RESPONSE')
  })

  it.each([
    ['missing semantics', (output: InferenceOutput) => withoutPath(output, ['attentionSemantics'])],
    [
      'extra semantics key',
      (output: InferenceOutput) =>
        withChangedPath(output, ['attentionSemantics', 'unexpected'], 'value'),
    ],
    [
      'missing clinical AOI key',
      (output: InferenceOutput) =>
        withoutPath(output, ['attentionSemantics', 'clinicalAoi', 'role']),
    ],
    [
      'extra clinical AOI key',
      (output: InferenceOutput) =>
        withChangedPath(output, ['attentionSemantics', 'clinicalAoi', 'unexpected'], true),
    ],
    [
      'altered schema',
      (output: InferenceOutput) =>
        withChangedPath(
          output,
          ['attentionSemantics', 'schemaVersion'],
          'predicted-observer-attention/2',
        ),
    ],
    [
      'prediction-modifying clinical AOI',
      (output: InferenceOutput) =>
        withChangedPath(
          output,
          ['attentionSemantics', 'clinicalAoi', 'modifiesPrediction'],
          true,
        ),
    ],
    [
      'unknown clinical AOI registration',
      (output: InferenceOutput) =>
        withChangedPath(
          output,
          ['attentionSemantics', 'clinicalAoi', 'registration'],
          'unknown_registration',
        ),
    ],
    [
      'connected registration on a mock origin',
      (output: InferenceOutput) =>
        withChangedPath(
          output,
          ['attentionSemantics', 'clinicalAoi', 'registration'],
          'registration_geometry_unavailable_v1',
        ),
    ],
  ])('fails a reducer attempt for malformed spatial semantics: %s', (_label, mutate) => {
    const { state, binding, attemptId, asyncBinding } = runningState()
    const malformed = mutate(mockOutputWithSemantics(binding)) as InferenceOutput
    const next = workspaceReducer(state, {
      type: 'run/succeed',
      ...asyncBinding,
      output: malformed,
    })

    expect(next.attemptsById[attemptId].status).toBe('failed')
    expect(next.attemptsById[attemptId].failure?.reason).toBe('MALFORMED_RESPONSE')
    expect(next.attemptsById[attemptId].result).toBeUndefined()
  })

  it('enforces gateway mode to origin and capability before storing output', async () => {
    const user = userEvent.setup()
    const gateway: WorkbenchGateway = {
      mode: 'connected',
      runInference: async (binding) => runMockEngine(binding),
    }
    render(
      <WorkspaceProvider gateway={gateway} queueDelayMs={0}>
        <ProviderProbe />
      </WorkspaceProvider>,
    )

    await user.click(screen.getByRole('button', { name: 'Start validation run' }))
    await waitFor(() =>
      expect(screen.getByLabelText('validated attempt status')).toHaveTextContent('failed'),
    )
    expect(screen.getByLabelText('validated failure reason')).toHaveTextContent(
      'ORIGIN_MISMATCH',
    )
  })
})
