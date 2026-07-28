import { readFileSync } from 'node:fs'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import { App } from '../App'
import { listWorkbenchAssets } from '../workbench/catalog'
import { CONNECTED_INFERENCE_WATERMARK } from '../workbench/inferenceEnvelope'
import { createInferenceBinding, runMockEngine } from '../workbench/mockEngine'
import {
  createInitialWorkspaceState,
  getCaseRoi,
  workspaceReducer,
} from '../workbench/reducer'
import type {
  ApprovedRoiAnnotation,
  ConnectedInferenceOutput,
  InferenceBinding,
  InferenceAttempt,
  InferenceOutput,
  WorkspaceAction,
  WorkspaceState,
} from '../workbench/types'

function reduce(state: WorkspaceState, actions: readonly WorkspaceAction[]) {
  return actions.reduce(workspaceReducer, state)
}

function connectedOutput(binding: InferenceBinding): ConnectedInferenceOutput {
  const mock = runMockEngine(binding)
  return {
    ...mock,
    attentionSemantics: {
      ...mock.attentionSemantics,
      clinicalAoi: {
        ...mock.attentionSemantics.clinicalAoi,
        registration: 'registration_geometry_unavailable_v1',
      },
    },
    origin: 'model_prediction',
    capabilityStatus: 'research_unvalidated',
    watermark: CONNECTED_INFERENCE_WATERMARK,
    resultDigest: `connected_${mock.resultDigest}`,
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
      engineVersion: 'runs-page-test',
      canonicalSyntheticAsset: true,
      deterministic: false,
      networkAccessed: true,
      storageAccessed: false,
      observedGazePayloadIncluded: false,
      trainingDataProvenance: 'not_disclosed',
    },
  }
}

function replaceActiveResult(
  state: WorkspaceState,
  runId: string,
  update: (output: InferenceOutput) => InferenceOutput,
): WorkspaceState {
  const run = state.runsById[runId]!
  const attempt = state.attemptsById[run.activeAttemptId!]!
  return {
    ...state,
    attemptsById: {
      ...state.attemptsById,
      [attempt.id]: {
        ...attempt,
        result: { ...attempt.result!, output: update(attempt.result!.output) },
      },
    },
  }
}

function replaceActiveStoredResult(
  state: WorkspaceState,
  runId: string,
  update: (result: NonNullable<InferenceAttempt['result']>) => unknown,
): WorkspaceState {
  const run = state.runsById[runId]!
  const attempt = state.attemptsById[run.activeAttemptId!]!
  return {
    ...state,
    attemptsById: {
      ...state.attemptsById,
      [attempt.id]: {
        ...attempt,
        result: update(attempt.result!) as InferenceAttempt['result'],
      },
    },
  }
}

function bindingFor(
  state: WorkspaceState,
  catalogIndex: number,
  runId: string,
  attemptToken: string,
): InferenceBinding {
  const asset = listWorkbenchAssets()[catalogIndex]
  const roi = getCaseRoi(state, asset.id) as ApprovedRoiAnnotation
  return createInferenceBinding({
    clientRunId: runId,
    attemptToken,
    caseId: asset.id,
    assetId: asset.id,
    assetSha256: asset.sha256,
    roi,
    modelVersion: 'mock-salience-v0.3',
    modelMode: 'mock_only',
    config: { threshold: 0.42, smoothing: 0.27 },
  })
}

function asyncActions(
  binding: InferenceBinding,
  attemptId: string,
): readonly WorkspaceAction[] {
  const asyncBinding = {
    runId: binding.clientRunId,
    attemptId,
    attemptToken: binding.attemptToken,
    inputFingerprint: binding.inputFingerprint,
  }
  return [
    { type: 'run/validate', ...asyncBinding },
    { type: 'run/queue', ...asyncBinding },
    { type: 'run/start', ...asyncBinding },
  ]
}

function createSeededRunState() {
  let state = createInitialWorkspaceState()
  const oldBinding = bindingFor(state, 2, 'run-seeded-001', 'token-seeded-001')
  const oldAttemptId = 'attempt-seeded-001'
  state = reduce(state, [
    {
      type: 'run/create',
      runId: oldBinding.clientRunId,
      attemptId: oldAttemptId,
      binding: oldBinding,
    },
    ...asyncActions(oldBinding, oldAttemptId),
    {
      type: 'run/fail',
      runId: oldBinding.clientRunId,
      attemptId: oldAttemptId,
      attemptToken: oldBinding.attemptToken,
      inputFingerprint: oldBinding.inputFingerprint,
      failure: { reason: 'NETWORK_ERROR', message: 'Seeded transient failure.' },
    },
  ])

  const retryBinding = bindingFor(
    state,
    2,
    oldBinding.clientRunId,
    'token-seeded-002',
  )
  const retryAttemptId = 'attempt-seeded-002'
  state = reduce(state, [
    {
      type: 'run/retry',
      runId: retryBinding.clientRunId,
      attemptId: retryAttemptId,
      parentAttemptId: oldAttemptId,
      binding: retryBinding,
    },
    ...asyncActions(retryBinding, retryAttemptId),
    {
      type: 'run/succeed',
      runId: retryBinding.clientRunId,
      attemptId: retryAttemptId,
      attemptToken: retryBinding.attemptToken,
      inputFingerprint: retryBinding.inputFingerprint,
      output: runMockEngine(retryBinding),
    },
  ])

  const newestBinding = bindingFor(state, 3, 'run-newest-002', 'token-newest-001')
  state = reduce(state, [
    {
      type: 'run/create',
      runId: newestBinding.clientRunId,
      attemptId: 'attempt-newest-001',
      binding: newestBinding,
    },
    {
      type: 'run/validate',
      runId: newestBinding.clientRunId,
      attemptId: 'attempt-newest-001',
      attemptToken: newestBinding.attemptToken,
      inputFingerprint: newestBinding.inputFingerprint,
    },
    {
      type: 'run/queue',
      runId: newestBinding.clientRunId,
      attemptId: 'attempt-newest-001',
      attemptToken: newestBinding.attemptToken,
      inputFingerprint: newestBinding.inputFingerprint,
    },
  ])

  return { state, successfulBinding: retryBinding }
}

function createRunStateAt(
  status: 'queued' | 'running' | 'failed' | 'cancelled' | 'blocked',
) {
  let state = createInitialWorkspaceState()
  const binding = bindingFor(state, 2, `run-${status}`, `token-${status}`)
  const attemptId = `attempt-${status}`
  const exact = {
    runId: binding.clientRunId,
    attemptId,
    attemptToken: binding.attemptToken,
    inputFingerprint: binding.inputFingerprint,
  }
  const actions: WorkspaceAction[] = [
    { type: 'run/create', runId: binding.clientRunId, attemptId, binding },
    { type: 'run/validate', ...exact },
  ]

  if (status === 'blocked') {
    actions.push({
      type: 'run/block',
      ...exact,
      failure: { reason: 'ROI_NOT_APPROVED', message: 'Seeded blocked run.' },
    })
  } else {
    actions.push({ type: 'run/queue', ...exact })
    if (status !== 'queued') actions.push({ type: 'run/start', ...exact })
    if (status === 'failed') {
      actions.push({
        type: 'run/fail',
        ...exact,
        failure: { reason: 'NETWORK_ERROR', message: 'Seeded failed run.' },
      })
    }
    if (status === 'cancelled') {
      actions.push({ type: 'run/cancel', ...exact })
    }
  }

  state = reduce(state, actions)
  return { state, binding, attemptId }
}

function conciseLabel(label: string) {
  return label.replace(/^Standalone synthetic case — /, '')
}

function renderRoute(path: string, initialState = createInitialWorkspaceState()) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App initialState={initialState} queueDelayMs={60_000} />
    </MemoryRouter>,
  )
}

describe('session runs', () => {
  it('starts empty with one clear header action and no duplicate empty-state action', () => {
    renderRoute('/runs')

    expect(screen.getByRole('heading', { name: 'Recent simulations', level: 1 })).toBeVisible()
    expect(screen.getByText('Available until refresh.')).toBeVisible()
    expect(screen.getAllByRole('link', { name: 'Start simulation' })).toHaveLength(1)
    expect(screen.getByRole('link', { name: 'Start simulation' })).toHaveAttribute(
      'href',
      '/cases',
    )
    expect(screen.getByText('No recent simulations')).toBeVisible()
    expect(screen.queryAllByTestId('run-row')).toHaveLength(0)
  })

  it('lists newest cases first with status and one detail action but no visible technical identity', () => {
    const { state } = createSeededRunState()
    renderRoute('/runs', state)

    const rows = screen.getAllByTestId('run-row')
    const assets = listWorkbenchAssets()
    expect(rows).toHaveLength(2)
    expect(within(rows[0]).getByRole('heading', { name: conciseLabel(assets[3].label) })).toBeVisible()
    expect(within(rows[1]).getByRole('heading', { name: conciseLabel(assets[2].label) })).toBeVisible()

    expect(within(rows[0]).getByText('queued')).toBeVisible()
    const queuedStatus = within(rows[0]).getByRole('status', { name: 'Status queued' })
    expect(queuedStatus).toHaveAttribute('aria-live', 'polite')
    expect(queuedStatus).toHaveAttribute('aria-atomic', 'true')
    expect(within(rows[0]).getByText('SYN-HNC-CHEEK-FREEFLAP')).toBeVisible()
    expect(within(rows[0]).queryByText('run-newest-002')).not.toBeInTheDocument()
    expect(within(rows[0]).queryByText('mock-salience-v0.3')).not.toBeInTheDocument()
    expect(within(rows[0]).queryByText(/attempt/i)).not.toBeInTheDocument()
    expect(within(rows[0]).getAllByRole('link')).toHaveLength(1)
    expect(within(rows[0]).getByRole('link', { name: 'View details' })).toHaveAttribute(
      'href',
      '/runs/run-newest-002',
    )

    expect(within(rows[1]).getByText('result ready')).toBeVisible()
    expect(
      within(rows[1]).getByRole('status', { name: 'Status result ready' }),
    ).toBeVisible()
    expect(within(rows[1]).queryByText('succeeded')).not.toBeInTheDocument()
    expect(within(rows[1]).getByText('SYN-HNC-CHEEK-TUMOUR')).toBeVisible()
    expect(within(rows[1]).queryByText('run-seeded-001')).not.toBeInTheDocument()
    expect(within(rows[1]).queryByText(/attempt/i)).not.toBeInTheDocument()
  })

  it.each(['stale', 'revoked'] as const)(
    'shows a succeeded run with a %s result as unusable in the list',
    (freshness) => {
      const seeded = createSeededRunState()
      const run = seeded.state.runsById['run-seeded-001']!
      const attempt = seeded.state.attemptsById[run.activeAttemptId!]!
      const state: WorkspaceState = {
        ...seeded.state,
        attemptsById: {
          ...seeded.state.attemptsById,
          [attempt.id]: {
            ...attempt,
            result: { ...attempt.result!, freshness },
          },
        },
      }

      renderRoute('/runs', state)

      const row = screen.getAllByTestId('run-row')[1]
      expect(within(row).getByLabelText(`Status ${freshness}`)).toBeVisible()
      expect(within(row).queryByText('succeeded')).not.toBeInTheDocument()
    },
  )

  it('distinguishes integrity-unavailable and result-unavailable succeeded runs', () => {
    const integritySeed = createSeededRunState()
    const integrityState = replaceActiveResult(
      integritySeed.state,
      'run-seeded-001',
      (output) => ({ ...output, watermark: 'unsafe' }) as unknown as InferenceOutput,
    )
    const integrityView = renderRoute('/runs', integrityState)
    expect(
      within(screen.getAllByTestId('run-row')[1]).getByLabelText(
        'Status integrity unavailable',
      ),
    ).toBeVisible()
    integrityView.unmount()

    const unavailableSeed = createSeededRunState()
    const run = unavailableSeed.state.runsById['run-seeded-001']!
    const attempt = unavailableSeed.state.attemptsById[run.activeAttemptId!]!
    const unavailableState: WorkspaceState = {
      ...unavailableSeed.state,
      attemptsById: {
        ...unavailableSeed.state.attemptsById,
        [attempt.id]: { ...attempt, result: undefined },
      },
    }
    renderRoute('/runs', unavailableState)
    expect(
      within(screen.getAllByTestId('run-row')[1]).getByLabelText(
        'Status result unavailable',
      ),
    ).toBeVisible()
  })

  it.each([
    ['null output', (result: NonNullable<InferenceAttempt['result']>) => ({
      ...result,
      output: null,
    })],
    ['non-record output', (result: NonNullable<InferenceAttempt['result']>) => ({
      ...result,
      output: 'corrupt-output',
    })],
    ['missing provenance', (result: NonNullable<InferenceAttempt['result']>) => {
      const { provenance: _provenance, ...output } = result.output
      return { ...result, output }
    }],
    ['null provenance', (result: NonNullable<InferenceAttempt['result']>) => ({
      ...result,
      output: { ...result.output, provenance: null },
    })],
  ] as const)(
    'renders list integrity unavailable without throwing for %s',
    (_field, corrupt) => {
      const seeded = createSeededRunState()
      const state = replaceActiveStoredResult(
        seeded.state,
        'run-seeded-001',
        corrupt,
      )

      expect(() => renderRoute('/runs', state)).not.toThrow()
      expect(
        within(screen.getAllByTestId('run-row')[1]).getByRole('status', {
          name: 'Status integrity unavailable',
        }),
      ).toBeVisible()
    },
  )

  it.each([
    ['null digest', (result: NonNullable<InferenceAttempt['result']>) => ({
      ...result,
      output: { ...result.output, resultDigest: null },
    })],
    ['null freshness', (result: NonNullable<InferenceAttempt['result']>) => ({
      ...result,
      freshness: null,
    })],
  ] as const)(
    'renders list integrity unavailable for malformed %s',
    (_field, corrupt) => {
      const seeded = createSeededRunState()
      const state = replaceActiveStoredResult(
        seeded.state,
        'run-seeded-001',
        corrupt,
      )

      expect(() => renderRoute('/runs', state)).not.toThrow()
      expect(
        within(screen.getAllByTestId('run-row')[1]).getByRole('status', {
          name: 'Status integrity unavailable',
        }),
      ).toBeVisible()
    },
  )

  it('encodes reserved characters in a run detail route', () => {
    let state = createInitialWorkspaceState()
    const binding = bindingFor(state, 2, 'run/id?view#detail', 'token-reserved')
    const attemptId = 'attempt-reserved'
    state = reduce(state, [
      { type: 'run/create', runId: binding.clientRunId, attemptId, binding },
      ...asyncActions(binding, attemptId),
    ])

    renderRoute('/runs', state)

    expect(screen.getByRole('link', { name: 'View details' })).toHaveAttribute(
      'href',
      '/runs/run%2Fid%3Fview%23detail',
    )
  })

  it('shows a calm case-first detail with review and Analysis actions while technical evidence stays disclosed', async () => {
    const user = userEvent.setup()
    const { state, successfulBinding } = createSeededRunState()
    const { container } = renderRoute('/runs/run-seeded-001', state)
    const asset = listWorkbenchAssets()[2]

    expect(
      screen.getByRole('heading', { name: conciseLabel(asset.label), level: 1 }),
    ).toBeVisible()
    const story = screen.getByRole('region', { name: 'Simulated attention result' })
    expect(story).toHaveAttribute('data-layout', 'clinician-stack')
    const source = within(story).getByRole('heading', { name: 'Source image' })
    const signal = within(story).getByRole('heading', {
      name: 'Simulated attention-density field',
    })
    const overlay = within(story).getByRole('heading', {
      name: 'Density overlay',
    })
    const summary = within(story).getByRole('heading', {
      name: 'Clinical AOI summary',
    })
    expect(source.compareDocumentPosition(signal) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(signal.compareDocumentPosition(overlay) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(overlay.compareDocumentPosition(summary) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(within(story).queryAllByRole('radio')).toHaveLength(0)
    expect(within(story).getAllByRole('img')).toHaveLength(3)
    expect(within(story).queryByRole('checkbox')).not.toBeInTheDocument()
    expect(story).not.toHaveTextContent(/selected region|ROI/i)
    expect(
      screen.queryByRole('img', { name: `${conciseLabel(asset.label)} synthetic preview` }),
    ).not.toBeInTheDocument()
    const nextStep = screen.getByRole('heading', { name: 'Next step' })
    expect(story.compareDocumentPosition(nextStep) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(container.querySelector('.run-primary')).toHaveClass('run-primary--result')
    const progress = screen.getByRole('status', { name: 'Run progress' })
    expect(progress).toHaveAttribute('aria-live', 'polite')
    expect(progress).toHaveAttribute('aria-atomic', 'true')
    expect(within(progress).getByLabelText('Run status succeeded')).toBeVisible()
    expect(within(progress).getByText('Result: Current')).toBeVisible()
    expect(screen.getAllByText(successfulBinding.caseId)[0]).toBeVisible()
    expect(screen.getByText('Result: Current')).toBeVisible()
    expect(screen.getByText(/ready for research review/i)).toBeVisible()
    expect(screen.getByRole('link', { name: 'Review result' })).toHaveAttribute(
      'href',
      '/research/reviews/new?run=run-seeded-001&attempt=attempt-seeded-002',
    )
    expect(screen.getByRole('link', { name: 'Return to Analysis' })).toHaveAttribute(
      'href',
      `/analysis?case=${asset.id}&run=run-seeded-001`,
    )

    const disclosure = screen.getByText('Technical details').closest('details')
    expect(disclosure).not.toHaveAttribute('open')
    expect(screen.getByText(successfulBinding.modelVersion)).not.toBeVisible()
    expect(screen.getByText(successfulBinding.inputFingerprint)).not.toBeVisible()
    await user.click(screen.getByText('Technical details'))
    expect(screen.getByText(successfulBinding.modelVersion)).toBeVisible()
    expect(screen.getByText(successfulBinding.inputFingerprint)).toBeVisible()
    expect(within(disclosure!).getByText('run-seeded-001')).toBeVisible()
    const bindingEvidence = screen.getByRole('region', { name: 'Run binding' })
    expect(
      within(bindingEvidence).getByText('Simulation profile').nextElementSibling,
    ).toHaveTextContent(successfulBinding.modelVersion)
    expect(within(bindingEvidence).queryByText('Model')).not.toBeInTheDocument()
    expect(
      within(bindingEvidence).getByText('Full-image source binding')
        .nextElementSibling,
    ).toHaveTextContent(`version ${successfulBinding.roiVersion}`)

    const timeline = screen.getByRole('region', { name: 'Attempt timeline' })
    const attempts = within(timeline).getAllByTestId('attempt-row')
    expect(attempts).toHaveLength(2)
    expect(within(attempts[0]).getByText('attempt-seeded-001')).toBeVisible()
    expect(within(attempts[0]).getByText('failed')).toBeVisible()
    expect(within(attempts[1]).getByText('attempt-seeded-002')).toBeVisible()
    expect(within(attempts[1]).getByText('succeeded')).toBeVisible()

    const result = screen.getByRole('region', { name: 'Result availability' })
    expect(within(result).getByText('Result available')).toBeVisible()
    expect(within(result).getByText(/result_[a-f0-9]+/i)).toBeVisible()

    const provenance = screen.getByRole('region', { name: 'Run provenance' })
    expect(within(provenance).getByText('deterministic_mock_engine')).toBeVisible()
    expect(within(provenance).getByText('mock_simulation')).toBeVisible()
    expect(
      within(provenance).getByText('Simulation engine version').nextElementSibling,
    ).toHaveTextContent('1')
    expect(within(provenance).getByTestId('provenance-network')).toHaveTextContent(
      'Network accessedNo',
    )
    expect(within(provenance).getByTestId('provenance-storage')).toHaveTextContent(
      'Persistent storage accessedNo',
    )
    expect(within(provenance).getByTestId('provenance-human-gaze')).toHaveTextContent(
      'Human gaze used by simulationNo',
    )
  })

  it.each(['queued', 'running'] as const)(
    'offers an exact Cancel request for a %s active attempt',
    async (status) => {
      const user = userEvent.setup()
      const { state } = createRunStateAt(status)
      renderRoute(`/runs/run-${status}`, state)

      await user.click(screen.getByRole('button', { name: 'Cancel request' }))
      expect(screen.getByLabelText('Run status cancelled')).toBeVisible()
      expect(screen.queryByRole('button', { name: 'Cancel request' })).not.toBeInTheDocument()
      expect(screen.getByRole('button', { name: 'Retry exact input' })).toBeVisible()
    },
  )

  it.each(['failed', 'cancelled', 'blocked'] as const)(
    'retries the exact scientific input for a %s active attempt',
    async (status) => {
      const user = userEvent.setup()
      const { state, binding, attemptId } = createRunStateAt(status)
      renderRoute(`/runs/run-${status}`, state)

      await user.click(screen.getByRole('button', { name: 'Retry exact input' }))
      expect(screen.getByLabelText('Run status queued')).toBeVisible()
      await user.click(screen.getByText('Technical details'))
      expect(screen.getByText(binding.inputFingerprint)).toBeVisible()
      const attempts = within(
        screen.getByRole('region', { name: 'Attempt timeline' }),
      ).getAllByTestId('attempt-row')
      expect(attempts).toHaveLength(2)
      expect(within(attempts[1]).getByText(`Retry of ${attemptId}`)).toBeVisible()
    },
  )

  it.each([
    [
      'binding',
      (output: InferenceOutput) => ({
        ...output,
        binding: { ...output.binding, roiVersion: output.binding.roiVersion + 1 },
      }) as InferenceOutput,
    ],
    [
      'envelope',
      (output: InferenceOutput) => (
        { ...output, watermark: 'unsafe' }
      ) as unknown as InferenceOutput,
    ],
  ] as const)(
    'fails closed at run detail for a corrupt %s',
    (_field, corrupt) => {
      const seeded = createSeededRunState()
      const state = replaceActiveResult(seeded.state, 'run-seeded-001', corrupt)

      renderRoute('/runs/run-seeded-001', state)

      expect(screen.getByText('Result: Integrity unavailable')).toBeVisible()
      expect(screen.queryByRole('link', { name: 'Review result' })).not.toBeInTheDocument()
      expect(
        screen.queryByRole('region', { name: 'Simulated attention result' }),
      ).not.toBeInTheDocument()
      expect(
        screen.getByRole('img', { name: /synthetic preview/i }),
      ).toBeVisible()
    },
  )

  it.each([
    ['null output', (result: NonNullable<InferenceAttempt['result']>) => ({
      ...result,
      output: null,
    })],
    ['non-record output', (result: NonNullable<InferenceAttempt['result']>) => ({
      ...result,
      output: 'corrupt-output',
    })],
    ['missing provenance', (result: NonNullable<InferenceAttempt['result']>) => {
      const { provenance: _provenance, ...output } = result.output
      return { ...result, output }
    }],
    ['null provenance', (result: NonNullable<InferenceAttempt['result']>) => ({
      ...result,
      output: { ...result.output, provenance: null },
    })],
  ] as const)(
    'keeps the exact detail shell fail-closed without throwing for %s',
    (_field, corrupt) => {
      const seeded = createSeededRunState()
      const state = replaceActiveStoredResult(
        seeded.state,
        'run-seeded-001',
        corrupt,
      )

      expect(() => renderRoute('/runs/run-seeded-001', state)).not.toThrow()
      expect(
        screen.getByRole('heading', { name: 'head and neck cheek tumour', level: 1 }),
      ).toBeVisible()
      expect(screen.getByText('Result: Integrity unavailable')).toBeVisible()
      expect(screen.queryByRole('link', { name: 'Review result' })).not.toBeInTheDocument()
      expect(screen.getByText(/provenance is unavailable/i)).toBeInTheDocument()
      expect(
        screen.queryByRole('region', { name: 'Simulated attention result' }),
      ).not.toBeInTheDocument()
      expect(screen.getByRole('img', { name: /synthetic preview/i })).toBeVisible()
    },
  )

  it.each([
    ['null digest', (result: NonNullable<InferenceAttempt['result']>) => ({
      ...result,
      output: { ...result.output, resultDigest: null },
    })],
    ['null freshness', (result: NonNullable<InferenceAttempt['result']>) => ({
      ...result,
      freshness: null,
    })],
  ] as const)(
    'keeps detail review unavailable for malformed %s',
    (_field, corrupt) => {
      const seeded = createSeededRunState()
      const state = replaceActiveStoredResult(
        seeded.state,
        'run-seeded-001',
        corrupt,
      )

      expect(() => renderRoute('/runs/run-seeded-001', state)).not.toThrow()
      expect(screen.getByText('Result: Integrity unavailable')).toBeVisible()
      expect(screen.queryByRole('link', { name: 'Review result' })).not.toBeInTheDocument()
      expect(
        screen.queryByRole('region', { name: 'Simulated attention result' }),
      ).not.toBeInTheDocument()
      expect(screen.getByRole('img', { name: /synthetic preview/i })).toBeVisible()
    },
  )

  it('shows a connected exact result without inventing AOI registration and uses response model identity', async () => {
    const user = userEvent.setup()
    const seeded = createSeededRunState()
    const state = replaceActiveResult(
      seeded.state,
      'run-seeded-001',
      () => connectedOutput(seeded.successfulBinding),
    )

    const { container } = renderRoute('/runs/run-seeded-001', state)

    expect(screen.getByText('Result: Current')).toBeVisible()
    expect(
      screen.getByRole('region', { name: 'Research model attention result' }),
    ).toHaveAttribute('data-layout', 'clinician-stack')
    const connectedStory = screen.getByRole('region', {
      name: 'Research model attention result',
    })
    expect(
      within(connectedStory).getByRole('heading', {
        name: 'AOI summary unavailable',
      }),
    ).toBeVisible()
    expect(connectedStory).toHaveTextContent(
      'Registration geometry was not supplied with this connected result.',
    )
    expect(
      within(connectedStory).queryByRole('heading', {
        name: 'Clinical AOI summary',
      }),
    ).not.toBeInTheDocument()
    expect(connectedStory).not.toHaveTextContent(
      /patient left|patient right|central triangle|dominant anatomical AOI/i,
    )
    expect(container.querySelector('.run-detail-page')).not.toHaveTextContent(
      /\bsimulation\b/i,
    )
    expect(screen.getByRole('region', { name: 'Selected run' })).toBeVisible()
    expect(screen.getByRole('status', { name: 'Run progress' })).toBeVisible()
    expect(screen.getByLabelText('Run status succeeded')).toBeVisible()
    expect(
      screen.queryByRole('status', { name: 'Simulation progress' }),
    ).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Review result' })).toHaveAttribute(
      'href',
      '/research/reviews/new?run=run-seeded-001&attempt=attempt-seeded-002',
    )

    await user.click(screen.getByText('Technical details'))
    const bindingEvidence = screen.getByRole('region', { name: 'Run binding' })
    expect(
      within(bindingEvidence).getByText('Connected request contract')
        .nextElementSibling,
    ).toHaveTextContent('synthetic-spatial-contract-rehearsal/1')
    expect(
      within(bindingEvidence).queryByText('Simulation profile'),
    ).not.toBeInTheDocument()

    const provenance = screen.getByRole('region', { name: 'Run provenance' })
    expect(
      within(provenance).getByText('Connected engine version').nextElementSibling,
    ).toHaveTextContent('runs-page-test')
    expect(
      within(provenance).getByText('Result digest').nextElementSibling,
    ).toHaveTextContent(/^connected_result_[a-f0-9]+$/)
    expect(
      within(provenance).getByText('Connected model ID').nextElementSibling,
    ).toHaveTextContent('observer-attention-test')
    expect(
      within(provenance).getByText('Connected model version').nextElementSibling,
    ).toHaveTextContent('test-v1')
    expect(
      within(provenance).getByText('Observed gaze in result payload')
        .nextElementSibling,
    ).toHaveTextContent('No')
    expect(
      within(provenance).getByText('Training-data provenance').nextElementSibling,
    ).toHaveTextContent('Not disclosed')
    expect(provenance).toHaveTextContent(/response-reported connected model identity/i)
    expect(
      within(provenance).queryByText('Simulation engine version'),
    ).not.toBeInTheDocument()
  })

  it('preserves an uncropped square coordinate plane for the selected-case preview', () => {
    const css = readFileSync('src/styles/workbench.css', 'utf8')

    expect(css).toMatch(
      /\.run-primary\.run-primary--result\s*\{[^}]*grid-template-columns:\s*1fr/s,
    )
    expect(css).toMatch(/\.run-primary__preview\s*\{[^}]*aspect-ratio:\s*1\s*\/\s*1/s)
    expect(css).toMatch(/\.run-primary__preview img\s*\{[^}]*object-fit:\s*contain/s)
    expect(css).not.toMatch(/\.run-primary__preview img\s*\{[^}]*(?:object-fit:\s*cover|min-height:\s*\d+px)/s)
  })

  it('fails closed for an unknown or refresh-lost run without substituting a fixture', () => {
    renderRoute('/runs/run-from-old-session')

    expect(
      screen.getByRole('heading', {
        name: 'Run unavailable in this session',
        level: 1,
      }),
    ).toBeVisible()
    expect(screen.getByText('run-from-old-session')).toBeVisible()
    expect(screen.getByText(/refresh or session expiry/i)).toBeVisible()
    expect(screen.getByRole('link', { name: 'Back to session runs' })).toHaveAttribute(
      'href',
      '/runs',
    )
    expect(screen.queryByText('run-seeded-001')).not.toBeInTheDocument()
    expect(screen.queryByText(/result_[a-f0-9]+/i)).not.toBeInTheDocument()
  })

  it.each(['stale', 'revoked'] as const)(
    'keeps a %s result unavailable for review while preserving technical evidence',
    (freshness) => {
      const { state } = createSeededRunState()
      const run = state.runsById['run-seeded-001']
      const attemptId = run.activeAttemptId!
      const attempt = state.attemptsById[attemptId]
      const guardedState = {
        ...state,
        attemptsById: {
          ...state.attemptsById,
          [attemptId]: {
            ...attempt,
            result: { ...attempt.result!, freshness },
          },
        },
      }

      renderRoute('/runs/run-seeded-001', guardedState)

      expect(screen.getByText(/cannot be reviewed/i)).toBeVisible()
      expect(screen.getByText(`Result: ${freshness}`)).toBeVisible()
      expect(screen.queryByRole('link', { name: 'Review result' })).not.toBeInTheDocument()
      expect(screen.getByRole('link', { name: 'Return to Analysis' })).toHaveAttribute(
        'href',
        '/analysis?case=SYN-HNC-CHEEK-TUMOUR&run=run-seeded-001',
      )
      expect(
        screen.queryByRole('region', { name: 'Simulated attention result' }),
      ).not.toBeInTheDocument()
      expect(screen.getByRole('img', { name: /synthetic preview/i })).toBeVisible()
      const result = screen.getByTestId('result-availability')
      expect(within(result).getByText('Result unavailable')).not.toBeVisible()
      expect(within(result).getAllByText(new RegExp(freshness, 'i'))).toHaveLength(2)
      for (const freshnessEvidence of within(result).getAllByText(new RegExp(freshness, 'i'))) {
        expect(freshnessEvidence).not.toBeVisible()
      }
      expect(within(result).getByText(/result_[a-f0-9]+/i)).not.toBeVisible()
    },
  )

  it('never renders a historical succeeded output when the active attempt failed', () => {
    const seeded = createSeededRunState()
    const retryBinding = bindingFor(
      seeded.state,
      2,
      'run-seeded-001',
      'token-seeded-003',
    )
    const failedAttemptId = 'attempt-seeded-003'
    const priorRun = seeded.state.runsById['run-seeded-001']!
    const failedAttempt: InferenceAttempt = {
      id: failedAttemptId,
      clientRunId: retryBinding.clientRunId,
      attemptToken: retryBinding.attemptToken,
      parentAttemptId: 'attempt-seeded-002',
      status: 'failed',
      binding: retryBinding,
      failure: { reason: 'NETWORK_ERROR', message: 'Seeded later failure.' },
    }
    const state: WorkspaceState = {
      ...seeded.state,
      runsById: {
        ...seeded.state.runsById,
        [priorRun.clientRunId]: {
          ...priorRun,
          activeAttemptId: failedAttemptId,
          attemptIds: [...priorRun.attemptIds, failedAttemptId],
          status: 'failed',
        },
      },
      attemptsById: {
        ...seeded.state.attemptsById,
        [failedAttemptId]: failedAttempt,
      },
    }

    renderRoute('/runs/run-seeded-001', state)

    expect(screen.getByLabelText('Run status failed')).toBeVisible()
    expect(
      screen.queryByRole('region', { name: 'Simulated attention result' }),
    ).not.toBeInTheDocument()
    expect(screen.getByRole('img', { name: /synthetic preview/i })).toBeVisible()
    expect(screen.queryByRole('link', { name: 'Review result' })).not.toBeInTheDocument()
  })

  it.each(['toString', '__proto__', 'constructor'])(
    'fails closed for prototype-key run ID %s',
    (runId) => {
      renderRoute(`/runs/${runId}`)

      expect(
        screen.getByRole('heading', {
          name: 'Run unavailable in this session',
          level: 1,
        }),
      ).toBeVisible()
      expect(screen.getByText(runId)).toBeVisible()
    },
  )
})
