import { readFileSync } from 'node:fs'
import { StrictMode } from 'react'
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation, useNavigate } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from '../App'
import { listWorkbenchAssets } from '../workbench/catalog'
import { deriveClinicalAoiPresentation } from '../workbench/clinicalAoiPresentation'
import {
  createInferenceBinding,
  runMockEngine,
} from '../workbench/mockEngine'
import { createInitialWorkspaceState } from '../workbench/reducer'
import type { RunInferenceOptions, WorkbenchGateway } from '../workbench/WorkbenchGateway'
import type {
  ConnectedInferenceOutput,
  InferenceBinding,
  InferenceConfiguration,
  InferenceOutput,
  MockModelVersion,
  WorkspaceState,
} from '../workbench/types'

const CONNECTED_WATERMARK =
  'MODEL PREDICTION — RESEARCH UNVALIDATED — NOT HUMAN GAZE — CLINICAL USE BLOCKED'
const originalScrollIntoView = Object.getOwnPropertyDescriptor(
  HTMLElement.prototype,
  'scrollIntoView',
)

type DeferredRequest = {
  readonly binding: InferenceBinding
  readonly signal?: AbortSignal
  readonly resolve: (output: InferenceOutput) => void
}

function createDeferredGateway() {
  const requests: DeferredRequest[] = []
  const runInference = vi.fn(
    (binding: InferenceBinding, options?: RunInferenceOptions) =>
      new Promise<InferenceOutput>((resolve) => {
        requests.push({ binding, signal: options?.signal, resolve })
      }),
  )
  const gateway: WorkbenchGateway = { mode: 'mock', runInference }
  return { gateway, requests, runInference }
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
    watermark: CONNECTED_WATERMARK,
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
      engineVersion: 'test-1',
      canonicalSyntheticAsset: true,
      deterministic: true,
      networkAccessed: true,
      storageAccessed: false,
      observedGazePayloadIncluded: false,
      trainingDataProvenance: 'not_disclosed',
    },
  }
}

function TestNavigation({
  targets,
}: {
  readonly targets: Readonly<Record<string, string>>
}) {
  const location = useLocation()
  const navigate = useNavigate()

  return (
    <div>
      <span data-testid="test-location">{location.pathname}{location.search}</span>
      {Object.entries(targets).map(([label, target]) => (
        <button key={label} type="button" onClick={() => navigate(target)}>
          {label}
        </button>
      ))}
      <button type="button" onClick={() => navigate(-1)}>Test browser back</button>
    </div>
  )
}

function renderAnalysis(
  path: string,
  gateway?: WorkbenchGateway,
  strict = false,
  queueDelayMs = 120,
  navigationTargets: Readonly<Record<string, string>> = {},
  initialState?: WorkspaceState,
) {
  const app = (
    <MemoryRouter initialEntries={[path]}>
      <TestNavigation targets={navigationTargets} />
      <App
        gateway={gateway}
        initialState={initialState}
        queueDelayMs={queueDelayMs}
      />
    </MemoryRouter>
  )
  return render(strict ? <StrictMode>{app}</StrictMode> : app)
}

function createSucceededFixture(
  caseIndex = 2,
  overrides: {
    readonly modelVersion?: MockModelVersion
    readonly config?: InferenceConfiguration
  } = {},
): {
  readonly caseId: string
  readonly runId: string
  readonly attemptId: string
  readonly state: WorkspaceState
} {
  const initial = createInitialWorkspaceState()
  const asset = listWorkbenchAssets()[caseIndex]
  const roi = initial.roisByCase[asset.id]
  if (!roi) throw new Error('Expected fixture ROI')
  const runId = 'run-fixture'
  const attemptId = 'attempt-fixture'
  const binding = createInferenceBinding({
    clientRunId: runId,
    attemptToken: 'token-fixture',
    caseId: asset.id,
    assetId: asset.id,
    assetSha256: asset.sha256,
    roi,
    modelVersion: overrides.modelVersion ?? 'mock-salience-v0.3',
    modelMode: 'mock_only',
    config: overrides.config ?? { threshold: 0.42, smoothing: 0.27 },
  })
  const output = runMockEngine(binding)
  const state: WorkspaceState = {
    ...initial,
    runsById: {
      [runId]: {
        clientRunId: runId,
        caseId: asset.id,
        assetId: asset.id,
        status: 'succeeded',
        attemptIds: [attemptId],
        activeAttemptId: attemptId,
      },
    },
    runOrder: [runId],
    attemptsById: {
      [attemptId]: {
        id: attemptId,
        clientRunId: runId,
        attemptToken: binding.attemptToken,
        status: 'succeeded',
        binding,
        result: { output, freshness: 'current' },
      },
    },
    activeRunId: runId,
  }
  return { caseId: asset.id, runId, attemptId, state }
}

type BindingCorruption =
  | 'run case'
  | 'asset hash'
  | 'ROI identity'
  | 'ROI version'
  | 'ROI geometry'
  | 'attempt binding'
  | 'output binding'

function corruptSucceededFixture(kind: BindingCorruption): ReturnType<typeof createSucceededFixture> {
  const fixture = createSucceededFixture()
  const run = fixture.state.runsById[fixture.runId]
  const attempt = fixture.state.attemptsById[fixture.attemptId]
  const binding = attempt.binding
  const result = attempt.result
  if (!run || !binding || !result) throw new Error('Expected complete succeeded fixture')
  const otherCase = listWorkbenchAssets()[3]

  if (kind === 'run case') {
    return {
      ...fixture,
      state: {
        ...fixture.state,
        runsById: { ...fixture.state.runsById, [fixture.runId]: { ...run, caseId: otherCase.id } },
      },
    }
  }

  if (kind === 'attempt binding') {
    return {
      ...fixture,
      state: {
        ...fixture.state,
        attemptsById: {
          ...fixture.state.attemptsById,
          [fixture.attemptId]: { ...attempt, attemptToken: 'different-attempt-token' },
        },
      },
    }
  }

  if (kind === 'output binding') {
    return {
      ...fixture,
      state: {
        ...fixture.state,
        attemptsById: {
          ...fixture.state.attemptsById,
          [fixture.attemptId]: {
            ...attempt,
            result: {
              ...result,
              output: {
                ...result.output,
                binding: { ...binding, attemptToken: 'different-output-token' },
              },
            },
          },
        },
      },
    }
  }

  const corruptedBinding: InferenceBinding = {
    ...binding,
    ...(kind === 'asset hash' ? { assetSha256: '0'.repeat(64) } : {}),
    ...(kind === 'ROI identity' ? { roiId: `${binding.roiId}-other` } : {}),
    ...(kind === 'ROI version' ? { roiVersion: binding.roiVersion + 1 } : {}),
    ...(kind === 'ROI geometry'
      ? { roiGeometry: { ...binding.roiGeometry, x: binding.roiGeometry.x + 0.01 } }
      : {}),
  }
  return {
    ...fixture,
    state: {
      ...fixture.state,
      attemptsById: {
        ...fixture.state.attemptsById,
        [fixture.attemptId]: { ...attempt, binding: corruptedBinding },
      },
    },
  }
}

function fixtureAtStatus(
  status: 'queued' | 'running' | 'succeeded',
): ReturnType<typeof createSucceededFixture> {
  const fixture = createSucceededFixture()
  if (status === 'succeeded') return fixture
  const run = fixture.state.runsById[fixture.runId]
  const attempt = fixture.state.attemptsById[fixture.attemptId]
  return {
    ...fixture,
    state: {
      ...fixture.state,
      runsById: {
        ...fixture.state.runsById,
        [fixture.runId]: { ...run, status },
      },
      attemptsById: {
        ...fixture.state.attemptsById,
        [fixture.attemptId]: { ...attempt, status, result: undefined },
      },
    },
  }
}

function createRoiChangedStaleFixture() {
  const fixture = createSucceededFixture()
  const currentRoi = Object.values(fixture.state.roisByCase).find(
    (candidate) => candidate?.caseId === fixture.caseId,
  )
  const attempt = fixture.state.attemptsById[fixture.attemptId]
  if (!currentRoi || !attempt.result) {
    throw new Error('Expected current ROI and stored result')
  }
  const changedRoi = {
    ...currentRoi,
    version: currentRoi.version + 1,
    geometry: { x: 0.05, y: 0.05, width: 0.9, height: 0.9 },
  }

  return {
    ...fixture,
    changedRoi,
    state: {
      ...fixture.state,
      roisByCase: {
        ...fixture.state.roisByCase,
        [fixture.caseId]: changedRoi,
      },
      attemptsById: {
        ...fixture.state.attemptsById,
        [fixture.attemptId]: {
          ...attempt,
          result: { ...attempt.result, freshness: 'stale' as const },
        },
      },
    },
  }
}

function createApprovedPartialCurrentResultFixture() {
  const fixture = createSucceededFixture()
  const attempt = fixture.state.attemptsById[fixture.attemptId]
  const currentRoi = Object.values(fixture.state.roisByCase).find(
    (candidate) => candidate?.caseId === fixture.caseId,
  )
  if (!attempt.binding || !attempt.result || !currentRoi) {
    throw new Error('Expected complete succeeded fixture')
  }
  const partialRoi = {
    ...currentRoi,
    version: currentRoi.version + 1,
    geometry: { x: 0.05, y: 0.05, width: 0.9, height: 0.9 },
  }

  return {
    ...fixture,
    partialRoi,
    state: {
      ...fixture.state,
      roisByCase: {
        ...fixture.state.roisByCase,
        [fixture.caseId]: partialRoi,
      },
      attemptsById: {
        ...fixture.state.attemptsById,
        [fixture.attemptId]: {
          ...attempt,
          result: { ...attempt.result, freshness: 'current' as const },
        },
      },
    },
  }
}

function createOlderAssetStaleFixture() {
  const fixture = createSucceededFixture()
  const attempt = fixture.state.attemptsById[fixture.attemptId]
  if (!attempt.binding || !attempt.result) {
    throw new Error('Expected historical binding and stored result')
  }
  const historicalSha256 = '0'.repeat(64)
  const historicalBinding: InferenceBinding = {
    ...attempt.binding,
    assetSha256: historicalSha256,
  }

  return {
    ...fixture,
    historicalSha256,
    state: {
      ...fixture.state,
      attemptsById: {
        ...fixture.state.attemptsById,
        [fixture.attemptId]: {
          ...attempt,
          binding: historicalBinding,
          result: {
            ...attempt.result,
            freshness: 'stale' as const,
            output: {
              ...attempt.result.output,
              binding: {
                ...attempt.result.output.binding,
                assetSha256: historicalSha256,
              },
            },
          },
        },
      },
    },
  }
}

function installResultDiscoverySpies(reducedMotion: boolean) {
  const scrollIntoView = vi.fn()
  Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
    configurable: true,
    value: scrollIntoView,
  })
  vi.stubGlobal(
    'matchMedia',
    vi.fn((query: string) => ({
      matches: query === '(prefers-reduced-motion: reduce)' && reducedMotion,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(() => true),
    })),
  )
  return scrollIntoView
}

afterEach(() => {
  vi.useRealTimers()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  if (originalScrollIntoView) {
    Object.defineProperty(
      HTMLElement.prototype,
      'scrollIntoView',
      originalScrollIntoView,
    )
  } else {
    Reflect.deleteProperty(HTMLElement.prototype, 'scrollIntoView')
  }
})

describe('single-case simulated inference', () => {
  it.each([
    ['/analysis', 'No case ID was supplied'],
    ['/analysis?case=UNKNOWN-CASE', 'UNKNOWN-CASE'],
    [
      `/analysis?case=${listWorkbenchAssets()[2].id}&case=${listWorkbenchAssets()[3].id}`,
      'More than one case ID was supplied',
    ],
  ])('fails closed for non-authoritative query %s', (path, expected) => {
    renderAnalysis(path)

    expect(
      screen.getByRole('heading', { name: 'Case unavailable', level: 1 }),
    ).toBeVisible()
    expect(screen.getByText(expected)).toBeVisible()
    expect(screen.queryByRole('button', { name: 'Run simulation' })).not.toBeInTheDocument()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
  })

  it('routes a recognized case with a missing binding to one explicit recovery action', () => {
    const deferred = createDeferredGateway()
    const asset = listWorkbenchAssets()[2]
    const initialState = createInitialWorkspaceState()
    const { [asset.id]: _missingBinding, ...remainingBindings } =
      initialState.roisByCase
    const missingBindingState: WorkspaceState = {
      ...initialState,
      roisByCase: remainingBindings,
    }

    renderAnalysis(
      `/analysis?case=${asset.id}`,
      deferred.gateway,
      false,
      0,
      {},
      missingBindingState,
    )

    expect(
      screen.getByRole('heading', {
        name: 'Source image binding required',
        level: 1,
      }),
    ).toBeVisible()
    expect(screen.getByText(asset.id)).toBeVisible()
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Restore the internal full-image binding before running this case.',
    )
    expect(
      screen.getByRole('link', { name: 'Restore source binding' }),
    ).toHaveAttribute('href', `/cases/${asset.id}/roi`)
    expect(screen.queryByRole('button', { name: 'Run simulation' })).not.toBeInTheDocument()
    expect(deferred.runInference).not.toHaveBeenCalled()
  })

  it('selects a successful run in the URL and never carries its output onto another case', async () => {
    const user = userEvent.setup()
    const caseA = listWorkbenchAssets()[2]
    const caseB = listWorkbenchAssets()[3]
    renderAnalysis(
      `/analysis?case=${caseA.id}`,
      undefined,
      false,
      0,
      { 'Navigate to case B': `/analysis?case=${caseB.id}` },
    )

    await user.click(screen.getByRole('button', { name: 'Run simulation' }))
    expect(await screen.findByRole('region', { name: 'Simulation result' })).toBeVisible()
    expect(screen.getByTestId('test-location')).toHaveTextContent(
      new RegExp(`^/analysis\\?case=${caseA.id}&run=run-`),
    )

    await user.click(screen.getByRole('button', { name: 'Navigate to case B' }))

    expect(screen.getByText(caseB.sha256)).toBeInTheDocument()
    expect(screen.getByLabelText('Active attempt status')).toHaveTextContent('not started')
    expect(screen.queryByRole('region', { name: 'Simulation result' })).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Result digest')).not.toBeInTheDocument()
  })

  it.each(['queued', 'running', 'succeeded'] as const)(
    'preserves the exact %s run through detail and browser back',
    async (status) => {
      const user = userEvent.setup()
      const fixture = fixtureAtStatus(status)
      renderAnalysis(
        `/analysis?case=${fixture.caseId}&run=${fixture.runId}`,
        undefined,
        false,
        120,
        {},
        fixture.state,
      )

      expect(screen.getByLabelText('Active attempt status')).toHaveTextContent(status)
      await user.click(screen.getByRole('link', { name: 'Open exact run detail' }))
      const asset = listWorkbenchAssets().find((entry) => entry.id === fixture.caseId)!
      expect(
        screen.getByRole('heading', {
          name: asset.label.replace(/^Standalone synthetic case — /, ''),
          level: 1,
        }),
      ).toBeVisible()
      expect(screen.getByLabelText(`Run status ${status}`)).toBeVisible()
      await user.click(screen.getByRole('button', { name: 'Test browser back' }))

      expect(screen.getByTestId('test-location')).toHaveTextContent(
        `/analysis?case=${fixture.caseId}&run=${fixture.runId}`,
      )
      expect(screen.getByLabelText('Active attempt status')).toHaveTextContent(status)
      if (status === 'succeeded') {
        expect(screen.getByRole('region', { name: 'Simulation result' })).toBeVisible()
      }
    },
  )

  it('keeps an exact non-default historical result reviewable with read-only provenance', async () => {
    const user = userEvent.setup()
    const fixture = createSucceededFixture(2, {
      modelVersion: 'mock-salience-v0.4',
      config: { threshold: 0.63, smoothing: 0.18 },
    })
    renderAnalysis(
      `/analysis?case=${fixture.caseId}&run=${fixture.runId}`,
      undefined,
      false,
      120,
      {},
      fixture.state,
    )

    const assertHistoricalProvenance = () => {
      expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
      expect(
        screen.queryByRole('slider', { name: 'Inference threshold' }),
      ).not.toBeInTheDocument()
      expect(
        screen.queryByRole('slider', { name: 'Inference smoothing' }),
      ).not.toBeInTheDocument()
      expect(screen.getByRole('link', { name: 'Review this result' })).toBeVisible()
      fireEvent.click(screen.getByText('Technical details'))
      expect(screen.getByText('mock-salience-v0.4')).toBeVisible()
      const attempt = fixture.state.attemptsById[fixture.attemptId]
      const provenance = screen.getByLabelText('Result provenance')
      expect(
        within(provenance).getByText('Simulation profile').nextElementSibling,
      ).toHaveTextContent('mock-salience-v0.4')
      expect(
        within(provenance).getByText('Simulation engine version').nextElementSibling,
      ).toHaveTextContent(attempt.result!.output.provenance.engineVersion)
      expect(within(provenance).queryByText('Model')).not.toBeInTheDocument()
      expect(screen.getByText(attempt.binding!.configurationHash)).toBeVisible()
      expect(screen.queryByText(/settings changed/i)).not.toBeInTheDocument()
    }

    assertHistoricalProvenance()
    await user.click(screen.getByRole('link', { name: 'Open exact run detail' }))
    await user.click(screen.getByRole('button', { name: 'Test browser back' }))
    assertHistoricalProvenance()
  })

  it('uses fixed defaults for a new case after navigating from a custom historical run', async () => {
    const user = userEvent.setup()
    const deferred = createDeferredGateway()
    const fixture = createSucceededFixture(2, {
      modelVersion: 'mock-salience-v0.4',
      config: { threshold: 0.63, smoothing: 0.18 },
    })
    const nextCase = listWorkbenchAssets()[3]
    renderAnalysis(
      `/analysis?case=${fixture.caseId}&run=${fixture.runId}`,
      deferred.gateway,
      false,
      120,
      { 'Navigate to fresh case': `/analysis?case=${nextCase.id}` },
      fixture.state,
    )

    await user.click(screen.getByRole('button', { name: 'Navigate to fresh case' }))

    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    expect(screen.queryByRole('slider', { name: 'Inference threshold' })).not.toBeInTheDocument()
    expect(screen.queryByRole('slider', { name: 'Inference smoothing' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Run simulation' }))
    await waitFor(() => expect(deferred.runInference).toHaveBeenCalledOnce())
    expect(deferred.requests[0].binding.modelVersion).toBe('mock-salience-v0.3')
    expect(deferred.requests[0].binding.config).toEqual({
      threshold: 0.42,
      smoothing: 0.27,
    })
  })

  it('fails closed for unknown, duplicate, empty, padded, and case-mismatched run queries', async () => {
    const user = userEvent.setup()
    const fixture = createSucceededFixture()
    const otherCase = listWorkbenchAssets()[3]
    renderAnalysis(
      `/analysis?case=${fixture.caseId}&run=unknown-run`,
      undefined,
      false,
      120,
      {
        'Use mismatched run': `/analysis?case=${otherCase.id}&run=${fixture.runId}`,
        'Use duplicate run': `/analysis?case=${fixture.caseId}&run=${fixture.runId}&run=other`,
        'Use empty run': `/analysis?case=${fixture.caseId}&run=`,
        'Use padded run': `/analysis?case=${fixture.caseId}&run=%20${fixture.runId}%20`,
      },
      fixture.state,
    )

    expect(screen.getByRole('heading', { name: 'Run unavailable', level: 1 })).toBeVisible()
    expect(screen.queryByRole('region', { name: 'Simulation result' })).not.toBeInTheDocument()

    for (const label of [
      'Use mismatched run',
      'Use duplicate run',
      'Use empty run',
      'Use padded run',
    ]) {
      await user.click(screen.getByRole('button', { name: label }))
      expect(screen.getByRole('heading', { name: 'Run unavailable', level: 1 })).toBeVisible()
      expect(screen.queryByRole('region', { name: 'Simulation result' })).not.toBeInTheDocument()
      expect(screen.queryByLabelText('Result digest')).not.toBeInTheDocument()
    }
  })

  it('renders a selected result only when every canonical run, attempt, and output binding matches', () => {
    const fixture = createSucceededFixture()
    renderAnalysis(
      `/analysis?case=${fixture.caseId}&run=${fixture.runId}`,
      undefined,
      false,
      120,
      {},
      fixture.state,
    )

    expect(screen.getByRole('region', { name: 'Simulation result' })).toBeVisible()
  })

  it('rejects an envelope-valid stored mock result that differs from deterministic replay', () => {
    const fixture = createSucceededFixture()
    const attempt = fixture.state.attemptsById[fixture.attemptId]
    const storedResult = attempt?.result
    if (!attempt || !storedResult || storedResult.output.origin !== 'mock_simulation') {
      throw new Error('Expected stored mock result fixture')
    }
    const firstPoint = storedResult.output.heatmap[0]
    if (!firstPoint) throw new Error('Expected deterministic heatmap point')
    const tamperedState: WorkspaceState = {
      ...fixture.state,
      attemptsById: {
        ...fixture.state.attemptsById,
        [fixture.attemptId]: {
          ...attempt,
          result: {
            ...storedResult,
            output: {
              ...storedResult.output,
              heatmap: [
                {
                  ...firstPoint,
                  intensity: firstPoint.intensity === 1 ? 0.99 : 1,
                },
                ...storedResult.output.heatmap.slice(1),
              ],
            },
          },
        },
      },
    }

    renderAnalysis(
      `/analysis?case=${fixture.caseId}&run=${fixture.runId}`,
      undefined,
      false,
      120,
      {},
      tamperedState,
    )

    expect(screen.queryByRole('region', { name: 'Simulation result' })).not.toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Result integrity validation failed.',
    )
  })

  it.each([
    'run case',
    'asset hash',
    'ROI identity',
    'ROI version',
    'ROI geometry',
    'attempt binding',
    'output binding',
  ] as const)('does not render a result with a mismatched %s', (kind) => {
    const fixture = corruptSucceededFixture(kind)
    renderAnalysis(
      `/analysis?case=${fixture.caseId}&run=${fixture.runId}`,
      undefined,
      false,
      120,
      {},
      fixture.state,
    )

    expect(screen.queryByRole('region', { name: 'Simulation result' })).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Result digest')).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Review this result' })).not.toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Result integrity validation failed.',
    )
    expect(screen.getByRole('link', { name: 'Open exact run detail' })).toHaveAttribute(
      'href',
      `/runs/${fixture.runId}`,
    )
    expect(
      screen.queryByText(
        'The source image binding changed. The previous result is available in run details. Run again to create a result for the current image.',
      ),
    ).not.toBeInTheDocument()
  })

  it.each([
    [false, 'smooth'],
    [true, 'auto'],
  ] as const)(
    'focuses and scrolls a newly succeeded selected result with %s reduced motion',
    async (reducedMotion, behavior) => {
      const user = userEvent.setup()
      const deferred = createDeferredGateway()
      const approvedCase = listWorkbenchAssets()[2]
      const scrollIntoView = installResultDiscoverySpies(reducedMotion)
      renderAnalysis(`/analysis?case=${approvedCase.id}`, deferred.gateway, false, 0)

      await user.click(screen.getByRole('button', { name: 'Run simulation' }))
      await waitFor(() => expect(deferred.runInference).toHaveBeenCalledOnce())
      expect(scrollIntoView).not.toHaveBeenCalled()

      await act(async () => {
        const request = deferred.requests[0]
        request.resolve(runMockEngine(request.binding))
        await Promise.resolve()
      })

      const result = await screen.findByRole('region', { name: 'Simulation result' })
      expect(result).toBeVisible()
      const resultHeading = screen.getByRole('heading', {
        name: 'Result',
        level: 2,
      })
      expect(resultHeading).toHaveAttribute('tabindex', '-1')
      await waitFor(() => expect(resultHeading).toHaveFocus())
      expect(scrollIntoView).toHaveBeenCalledOnce()
      expect(scrollIntoView).toHaveBeenCalledWith({ behavior, block: 'start' })
    },
  )

  it('does not focus or scroll a completed attempt after another case supersedes its selection', async () => {
    const user = userEvent.setup()
    const deferred = createDeferredGateway()
    const caseA = listWorkbenchAssets()[2]
    const caseB = listWorkbenchAssets()[3]
    const scrollIntoView = installResultDiscoverySpies(false)
    renderAnalysis(
      `/analysis?case=${caseA.id}`,
      deferred.gateway,
      false,
      0,
      { 'Navigate to case B': `/analysis?case=${caseB.id}` },
    )

    await user.click(screen.getByRole('button', { name: 'Run simulation' }))
    await waitFor(() => expect(deferred.runInference).toHaveBeenCalledOnce())
    const request = deferred.requests[0]
    await user.click(screen.getByRole('button', { name: 'Navigate to case B' }))
    scrollIntoView.mockClear()

    await act(async () => {
      request.resolve(runMockEngine(request.binding))
      await Promise.resolve()
    })

    expect(screen.getByText(caseB.sha256)).toBeInTheDocument()
    expect(screen.queryByRole('region', { name: 'Simulation result' })).not.toBeInTheDocument()
    expect(scrollIntoView).not.toHaveBeenCalled()
  })

  it('never exposes mutable inference controls or settings-change branches', async () => {
    const user = userEvent.setup()
    const approvedCase = listWorkbenchAssets()[2]
    renderAnalysis(`/analysis?case=${approvedCase.id}`, undefined, false, 0)

    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    expect(screen.queryByRole('slider', { name: 'Inference threshold' })).not.toBeInTheDocument()
    expect(screen.queryByRole('slider', { name: 'Inference smoothing' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Run simulation' }))
    expect(await screen.findByRole('region', { name: 'Simulation result' })).toBeVisible()
    expect(screen.getByRole('link', { name: 'Review this result' })).toBeVisible()
    expect(screen.queryByRole('button', { name: 'Run changed settings' })).not.toBeInTheDocument()
    expect(screen.queryByText(/settings changed/i)).not.toBeInTheDocument()
  })

  it('blocks rerun when an approved stale binding is only a partial image rectangle', async () => {
    const user = userEvent.setup()
    const deferred = createDeferredGateway()
    const fixture = createRoiChangedStaleFixture()
    renderAnalysis(
      `/analysis?case=${fixture.caseId}&run=${fixture.runId}`,
      deferred.gateway,
      false,
      0,
      {},
      fixture.state,
    )

    expect(screen.queryByRole('region', { name: 'Simulation result' })).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Result digest')).not.toBeInTheDocument()
    expect(
      screen.getByText(
        'The source image binding changed. The previous result is available in run details. Run again to create a result for the current image.',
      ),
    ).toBeVisible()
    expect(screen.getByRole('link', { name: 'Open exact run detail' })).toHaveAttribute(
      'href',
      `/runs/${fixture.runId}`,
    )
    expect(screen.queryByRole('link', { name: 'Review this result' })).not.toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Full-image source binding unavailable.',
    )
    expect(screen.getByRole('button', { name: 'Run with current image' })).toBeDisabled()
    await user.click(screen.getByText('Technical details'))
    expect(screen.queryByText(/Verified full image/)).not.toBeInTheDocument()
    expect(screen.getByText(`Unavailable · v${fixture.changedRoi.version}`)).toBeVisible()
    const preflight = screen.getByRole('region', { name: 'Inference preflight' })
    expect(within(preflight).getByText('Full-image source binding unavailable')).toBeVisible()
    expect(
      within(preflight).getByRole('link', { name: 'Restore source binding' }),
    ).toHaveAttribute('href', `/cases/${fixture.caseId}/roi`)
    expect(deferred.runInference).not.toHaveBeenCalled()
  })

  it('does not display a current result bound to an approved partial image rectangle', () => {
    const fixture = createApprovedPartialCurrentResultFixture()
    renderAnalysis(
      `/analysis?case=${fixture.caseId}&run=${fixture.runId}`,
      undefined,
      false,
      0,
      {},
      fixture.state,
    )

    expect(screen.queryByRole('region', { name: 'Simulation result' })).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Result digest')).not.toBeInTheDocument()
    expect(screen.queryByRole('link', { name: 'Review this result' })).not.toBeInTheDocument()
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Full-image source binding unavailable.',
    )
    expect(screen.getByRole('button', { name: 'Run with current image' })).toBeDisabled()
  })

  it('offers a rerun for an exact stale historical record whose asset SHA predates the catalog', () => {
    const fixture = createOlderAssetStaleFixture()
    renderAnalysis(
      `/analysis?case=${fixture.caseId}&run=${fixture.runId}`,
      undefined,
      false,
      0,
      {},
      fixture.state,
    )

    expect(fixture.historicalSha256).not.toBe(listWorkbenchAssets()[2].sha256)
    expect(screen.queryByRole('region', { name: 'Simulation result' })).not.toBeInTheDocument()
    expect(screen.queryByLabelText('Result digest')).not.toBeInTheDocument()
    expect(
      screen.getByText(
        'The source image binding changed. The previous result is available in run details. Run again to create a result for the current image.',
      ),
    ).toBeVisible()
    expect(screen.getByRole('button', { name: 'Run with current image' })).toBeEnabled()
    expect(screen.queryByRole('button', { name: 'Run changed settings' })).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open exact run detail' })).toHaveAttribute(
      'href',
      `/runs/${fixture.runId}`,
    )
    expect(screen.queryByRole('link', { name: 'Review this result' })).not.toBeInTheDocument()
  })

  it('locks the draft while active and exposes only the lifecycle primary action', async () => {
    const user = userEvent.setup()
    const deferred = createDeferredGateway()
    const approvedCase = listWorkbenchAssets()[2]
    renderAnalysis(`/analysis?case=${approvedCase.id}`, deferred.gateway, false, 5_000)

    await user.click(screen.getByRole('button', { name: 'Run simulation' }))

    expect(screen.getByLabelText('Active attempt status')).toHaveTextContent('queued')
    expect(screen.getByRole('button', { name: 'Cancel run' })).toBeVisible()
    expect(screen.queryByRole('button', { name: 'Run simulation' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Retry exact input' })).not.toBeInTheDocument()
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    expect(screen.queryByRole('slider', { name: 'Inference threshold' })).not.toBeInTheDocument()
    expect(screen.queryByRole('slider', { name: 'Inference smoothing' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Cancel run' }))

    expect(screen.getByLabelText('Active attempt status')).toHaveTextContent('cancelled')
    expect(screen.getByRole('button', { name: 'Retry exact input' })).toBeVisible()
    expect(screen.queryByRole('button', { name: 'Run simulation' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Cancel run' })).not.toBeInTheDocument()
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
  })

  it('preflights an exact draft case and stays zero-call until its internal image bound is available', () => {
    const draftCase = listWorkbenchAssets()[0]
    const deferred = createDeferredGateway()
    const initialState = createInitialWorkspaceState()
    const approvedRoi = initialState.roisByCase[draftCase.id]
    if (approvedRoi?.status !== 'approved') throw new Error('Expected approved default binding')
    const { reviewerId: _reviewerId, ...draftBase } = approvedRoi
    const draftState: WorkspaceState = {
      ...initialState,
      roisByCase: {
        ...initialState.roisByCase,
        [draftCase.id]: { ...draftBase, status: 'draft' },
      },
    }
    renderAnalysis(
      `/analysis?case=${draftCase.id}`,
      deferred.gateway,
      false,
      120,
      {},
      draftState,
    )

    expect(screen.getByText(draftCase.id)).toBeInTheDocument()
    expect(screen.getAllByRole('img')).toHaveLength(1)
    expect(screen.getByRole('img')).toHaveAttribute('width', '1024')
    expect(screen.getByRole('img')).toHaveAttribute('height', '1024')
    expect(screen.getByRole('img')).toHaveAttribute('loading', 'eager')
    expect(screen.getByRole('img')).toHaveAttribute('decoding', 'async')
    expect(screen.getByRole('img')).toHaveAttribute('fetchpriority', 'high')
    expect(screen.getByRole('button', { name: 'Run simulation' })).toBeDisabled()
    expect(screen.getByRole('alert')).toHaveTextContent('Full-image source binding unavailable.')
    fireEvent.click(screen.getByText('Technical details'))
    const gates = screen.getByRole('region', { name: 'Inference preflight' })
    expect(within(gates).getByText('Full-image source binding unavailable')).toBeVisible()
    expect(
      within(gates).getByRole('link', { name: 'Restore source binding' }),
    ).toHaveAttribute('href', `/cases/${draftCase.id}/roi`)
    expect(deferred.runInference).not.toHaveBeenCalled()
  })

  it('shows one simple result task and keeps research evidence under one closed disclosure', () => {
    const fixture = createSucceededFixture()
    const asset = listWorkbenchAssets().find((entry) => entry.id === fixture.caseId)!
    const { container } = renderAnalysis(
      `/analysis?case=${fixture.caseId}&run=${fixture.runId}`,
      undefined,
      false,
      120,
      {},
      fixture.state,
    )

    const result = screen.getByRole('region', { name: 'Simulation result' })
    const story = within(result).getByRole('region', { name: 'Simulated attention result' })
    expect(story).toHaveAttribute('data-layout', 'clinician-stack')
    const source = within(story).getByRole('heading', { name: 'Source image' })
    const signal = within(story).getByRole('heading', {
      name: 'Simulated attention-density field',
    })
    const overlay = within(story).getByRole('heading', { name: 'Density overlay' })
    const summary = within(story).getByRole('heading', { name: 'Clinical AOI summary' })
    expect(source.compareDocumentPosition(signal) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(signal.compareDocumentPosition(overlay) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(overlay.compareDocumentPosition(summary) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(within(story).queryAllByRole('radio')).toHaveLength(0)
    expect(within(story).getAllByRole('img')).toHaveLength(3)
    expect(story).not.toHaveTextContent(/selected region|ROI/i)
    expect(
      within(result).getByText(
        'Simulated attention-density interface structure · not observed or human gaze · not a patient prediction or result',
      ),
    ).toBeVisible()

    const technicalSummary = screen.getByText('Technical details')
    const technical = technicalSummary.closest('details')
    expect(technical).not.toHaveAttribute('open')
    const nextStep = screen.getByRole('heading', { name: 'Next step' })
    expect(result.compareDocumentPosition(nextStep) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(nextStep.compareDocumentPosition(technicalSummary) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(summary.compareDocumentPosition(technicalSummary) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(container.querySelector('.inference-layout')).toHaveClass('inference-layout--result')
    expect(technical).toContainElement(screen.getByText(asset.sha256))
    expect(screen.getAllByText(/Verified full image · v\d+/)).toHaveLength(2)
    expect(screen.getAllByText('Full-image source bound')).toHaveLength(2)
    expect(technical).toContainElement(screen.getByText('Internal image binding passed'))
    expect(technical).toContainElement(
      screen.getByRole('region', { name: 'Inference preflight' }),
    )
    expect(technical).toContainElement(screen.getByRole('region', { name: 'Run timeline' }))
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    const evidence = screen.getByRole('region', { name: 'Clinical AOI evidence' })
    expect(technical).toContainElement(evidence)
    const storedOutput = fixture.state.attemptsById[fixture.attemptId].result!.output
    const expectedEvidence = deriveClinicalAoiPresentation(
      storedOutput.heatmap,
      storedOutput.binding.roiGeometry,
    )
    expect(expectedEvidence.ok).toBe(true)
    if (!expectedEvidence.ok) throw new Error('Expected AOI evidence')
    expect(within(evidence).getByText('Central triangle share').nextElementSibling).toHaveTextContent(
      `${Math.round(expectedEvidence.centralTriangleShare * 100)}%`,
    )
    expect(within(evidence).getByText('Patient-left share').nextElementSibling).toHaveTextContent(
      `${Math.round(expectedEvidence.hemifaces.patientLeftShare * 100)}%`,
    )
    expect(within(evidence).getByText('Patient-right share').nextElementSibling).toHaveTextContent(
      `${Math.round(expectedEvidence.hemifaces.patientRightShare * 100)}%`,
    )
    expect(within(evidence).getByText('Dominant anatomical AOI').nextElementSibling).toHaveTextContent(
      expectedEvidence.dominantSubsite?.label ?? 'None available',
    )
    expect(evidence).toHaveTextContent(
      'Shares of simulated point weights assigned by the fixed anatomical template; not gaze duration or clinical measurements.',
    )
    for (const genericMetric of [
      'ROI coverage',
      'Peak intensity',
      'Mean intensity',
      'Focus score',
    ]) {
      expect(screen.queryByText(genericMetric)).not.toBeInTheDocument()
    }

    const environment = screen.getByRole('status', { name: 'Workspace environment' })
    expect(
      within(environment).getByText(
        'Research prototype · sample data only · session data resets on refresh',
      ),
    ).toBeVisible()
    expect(screen.queryByRole('button', { name: 'Start a new run' })).not.toBeInTheDocument()
    expect(
      container.querySelectorAll(
        '.run-command-panel__actions .workspace-button--primary',
      ),
    ).toHaveLength(1)
  })

  it('runs one standalone asset through one primary action with fixed defaults', async () => {
    const user = userEvent.setup()
    const approvedCase = listWorkbenchAssets()[2]
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    const { container } = renderAnalysis(`/analysis?case=${approvedCase.id}`)

    expect(
      screen.getByRole('heading', {
        name: 'Simulated observer-attention density',
        level: 1,
      }),
    ).toBeVisible()
    expect(screen.getAllByRole('button', { name: 'Run simulation' })).toHaveLength(1)
    expect(screen.getAllByRole('img')).toHaveLength(1)
    expect(screen.getByText(approvedCase.sha256)).toBeInTheDocument()
    expect(screen.getByText('AI-generated synthetic source image')).toBeVisible()
    expect(container).not.toHaveTextContent(/before[- /]?after|cross-case|treatment outcome/i)

    await user.click(screen.getByText('Technical details'))
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    expect(screen.queryByRole('slider', { name: 'Inference threshold' })).not.toBeInTheDocument()
    expect(screen.queryByRole('slider', { name: 'Inference smoothing' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: 'Run simulation' }))
    const result = await screen.findByRole('region', { name: 'Simulation result' })
    const story = within(result).getByRole('region', { name: 'Simulated attention result' })
    expect(story).toHaveAttribute('data-layout', 'clinician-stack')
    expect(within(story).queryAllByRole('radio')).toHaveLength(0)
    expect(within(story).getByRole('heading', { name: 'Source image' })).toBeVisible()
    expect(
      within(story).getByRole('heading', { name: 'Simulated attention-density field' }),
    ).toBeVisible()
    expect(within(story).getByRole('heading', { name: 'Density overlay' })).toBeVisible()
    expect(within(story).getByRole('heading', { name: 'Clinical AOI summary' })).toBeVisible()
    expect(
      within(result).getByText(
        'Simulated attention-density interface structure · not observed or human gaze · not a patient prediction or result',
      ),
    ).toBeVisible()
    expect(screen.getByText('deterministic_mock_engine')).toBeVisible()
    expect(screen.getByText('mock_simulation')).toBeVisible()
    const digest = screen.getByLabelText('Result digest').textContent
    expect(digest).toMatch(/^result_[a-f0-9]+$/)
    expect(screen.getByRole('link', { name: 'Open exact run detail' })).toHaveAttribute(
      'href',
      expect.stringMatching(/^\/runs\/run-/),
    )
    expect(
      screen.getByRole('link', { name: 'Review this result' }),
    ).toHaveAttribute(
      'href',
      expect.stringMatching(
        /^\/research\/reviews\/new\?run=run-[^&]+&attempt=attempt-[^&]+$/,
      ),
    )
    expect(fetchSpy).not.toHaveBeenCalled()

    expect(within(result).getAllByRole('img')).toHaveLength(3)
    for (const image of within(result).getAllByRole('img')) {
      expect(image).toHaveAttribute('width', '1024')
      expect(image).toHaveAttribute('height', '1024')
      expect(image).toHaveAttribute('decoding', 'async')
    }
    expect(within(result).getByText('SIMULATED — NOT HUMAN GAZE')).toBeVisible()
    await user.click(within(result).getByText('Display options'))
    const opacity = within(result).getByRole('slider', { name: 'Overlay opacity' })
    fireEvent.change(opacity, { target: { value: '40' } })
    expect(opacity).toHaveValue('40')
    expect(within(result).queryByRole('checkbox')).not.toBeInTheDocument()

    expect(screen.queryByRole('button', { name: 'Run changed settings' })).not.toBeInTheDocument()
    expect(screen.queryByText(/settings changed/i)).not.toBeInTheDocument()
  })

  it('preserves uncropped square image planes for source, signal, and overlay', () => {
    const css = readFileSync('src/styles/workbench.css', 'utf8')

    expect(css).toMatch(/\.inference-source-preview\s*\{[^}]*aspect-ratio:\s*1\s*\/\s*1/s)
    expect(css).toMatch(/\.inference-source-preview img\s*\{[^}]*object-fit:\s*contain/s)
    expect(css).toMatch(/\.inference-visual \.attention-frame\s*\{[^}]*aspect-ratio:\s*1\s*\/\s*1/s)
    expect(css).toMatch(/\.attention-result-view \.attention-frame img\s*\{[^}]*object-fit:\s*contain/s)
    expect(css).not.toMatch(/\.inference-source-preview[\s\S]{0,180}(?:min-height|height):\s*\d+px/)
    expect(css).toMatch(
      /\.run-command-panel__detail\s*\{[^}]*display:\s*inline-flex;[^}]*min-height:\s*44px;[^}]*align-items:\s*center/s,
    )
  })

  it('stacks the result before the next step on narrow screens', () => {
    const css = readFileSync('src/styles/workbench.css', 'utf8')

    expect(css).toMatch(
      /\.inference-layout\.inference-layout--result\s*\{[^}]*grid-template-columns:\s*1fr/s,
    )
    expect(css).toMatch(
      /@media \(max-width: 904px\)\s*\{[^@]*\.inference-visual-column\s*\{[^}]*grid-column:\s*1;[^}]*grid-row:\s*1;[^}]*\}[^@]*\.inference-command-column\s*\{[^}]*grid-column:\s*1;[^}]*grid-row:\s*2;[^}]*\}/s,
    )
  })

  it('derives connected preflight, header, and result boundaries from gateway mode and output origin', async () => {
    const user = userEvent.setup()
    const approvedCase = listWorkbenchAssets()[2]
    const runInference = vi.fn(async (binding: InferenceBinding) =>
      connectedOutput(binding),
    )
    const gateway: WorkbenchGateway = { mode: 'connected', runInference }
    renderAnalysis(`/analysis?case=${approvedCase.id}`, gateway)

    expect(
      screen.getByRole('heading', {
        name: 'Research observer-attention prediction',
        level: 1,
      }),
    ).toBeVisible()
    expect(screen.getByText('head and neck cheek tumour')).toBeVisible()
    expect(
      screen.queryByRole('heading', { name: 'Simulated observer-attention density' }),
    ).not.toBeInTheDocument()
    const boundary = screen.getByRole('region', { name: 'Execution boundary' })
    expect(within(boundary).getByText(CONNECTED_WATERMARK)).toBeVisible()
    await user.click(screen.getByText('Technical details'))
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    expect(
      screen.getByText(/research-unvalidated observer-attention prediction/i),
    ).toBeVisible()
    expect(
      screen.queryByText(/both versions are deterministic UI simulators/i),
    ).not.toBeInTheDocument()
    expect(
      within(boundary).queryByText('MOCK ONLY · UNVALIDATED · NOT A PATIENT RESULT'),
    ).not.toBeInTheDocument()
    const preflight = screen.getByRole('region', { name: 'Inference preflight' })
    expect(
      within(preflight).getByText('Connected research gateway · network request required'),
    ).toBeVisible()
    expect(within(preflight).queryByText(/mock-only · no network/i)).not.toBeInTheDocument()

    expect(screen.queryByRole('button', { name: 'Run simulation' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Run research prediction' }))
    const result = await screen.findByRole('region', {
      name: 'Research observer-attention prediction result',
    })
    const connectedEvidence = screen.getByRole('region', {
      name: 'Clinical AOI evidence',
    })
    expect(connectedEvidence).toBeVisible()
    expect(
      within(connectedEvidence).getByRole('heading', {
        name: 'AOI summary unavailable',
      }),
    ).toBeVisible()
    expect(connectedEvidence).toHaveTextContent(
      'Registration geometry was not supplied with this connected result.',
    )
    for (const unsupportedAoiReadout of [
      'Central triangle share',
      'Patient-left share',
      'Patient-right share',
      'Dominant anatomical AOI',
    ]) {
      expect(
        within(connectedEvidence).queryByText(unsupportedAoiReadout),
      ).not.toBeInTheDocument()
    }
    expect(
      screen.queryByRole('region', { name: 'Simulation result' }),
    ).not.toBeInTheDocument()
    expect(
      within(result).queryByText('ROI coverage'),
    ).not.toBeInTheDocument()
    expect(within(result).queryByText(/mock_only/i)).not.toBeInTheDocument()
    expect(
      within(result).getByRole('region', { name: 'Research model attention result' }),
    ).toHaveAttribute('data-layout', 'clinician-stack')
    expect(within(result).queryAllByRole('radio')).toHaveLength(0)
    expect(screen.getByText('model_prediction')).toBeVisible()
    expect(screen.getByText('research_unvalidated')).toBeVisible()
    expect(screen.getByText('connected_model_gateway')).toBeVisible()
    const provenance = screen.getByLabelText('Result provenance')
    expect(
      within(provenance).getByText('Connected request contract').nextElementSibling,
    ).toHaveTextContent('synthetic-spatial-contract-rehearsal/1')
    expect(
      within(provenance).getByText('Connected engine version').nextElementSibling,
    ).toHaveTextContent('test-1')
    expect(
      within(provenance).getByText('Connected model ID').nextElementSibling,
    ).toHaveTextContent('observer-attention-test')
    expect(
      within(provenance).getByText('Connected model version').nextElementSibling,
    ).toHaveTextContent('test-v1')
    expect(
      within(provenance).getByText('Artifact SHA-256').nextElementSibling,
    ).toHaveTextContent('a'.repeat(64))
    expect(
      within(provenance).getByText('Observed gaze in result payload')
        .nextElementSibling,
    ).toHaveTextContent('No')
    expect(
      within(provenance).getByText('Training-data provenance').nextElementSibling,
    ).toHaveTextContent('Not disclosed')
    expect(
      within(provenance).queryByText('Request configuration hash'),
    ).not.toBeInTheDocument()
    const transparencyNote = screen.getByText(
      /synthetic spatial contract rehearsal/i,
    )
    expect(transparencyNote).toHaveTextContent(
      /response-reported connected model identity/i,
    )
    expect(transparencyNote).toHaveTextContent(
      /research provenance, not clinical certification/i,
    )
  })

  it('rejects a facial-paralysis severity payload without inventing a spatial result', async () => {
    const user = userEvent.setup()
    const approvedCase = listWorkbenchAssets()[2]
    const runInference = vi.fn(async (binding: InferenceBinding) => {
      const { heatmap: _heatmap, ...connectedEnvelope } = connectedOutput(binding)
      return {
        ...connectedEnvelope,
        house_brackmann_logits: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        binary_palsy_logits: [0.7, 0.3],
        coarse_severity_logits: [0.2, 0.5, 0.3],
        eyes_logits: [0.1, 0.4, 0.5],
        mouth_logits: [0.2, 0.3, 0.5],
      } as unknown as InferenceOutput
    })
    const gateway: WorkbenchGateway = { mode: 'connected', runInference }
    renderAnalysis(`/analysis?case=${approvedCase.id}`, gateway)

    await user.click(screen.getByRole('button', { name: 'Run research prediction' }))

    await waitFor(() => {
      expect(screen.getByLabelText('Active attempt status')).toHaveTextContent('failed')
    })
    expect(screen.getByRole('alert')).toHaveTextContent('MALFORMED_RESPONSE')
    expect(
      screen.queryByRole('region', {
        name: 'Research observer-attention prediction result',
      }),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole('region', { name: 'Simulated attention result' }),
    ).not.toBeInTheDocument()
    expect(screen.getAllByRole('img')).toHaveLength(1)
    expect(
      screen.queryByRole('heading', { name: 'Predicted observer-attention density' }),
    ).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Density overlay' })).not.toBeInTheDocument()
  })

  it('shows running, cancellation, immutable retry lineage, and ignores a late parent response', async () => {
    vi.useFakeTimers()
    const approvedCase = listWorkbenchAssets()[2]
    const deferred = createDeferredGateway()
    renderAnalysis(`/analysis?case=${approvedCase.id}`, deferred.gateway, true)

    fireEvent.click(screen.getByRole('button', { name: 'Run simulation' }))
    expect(deferred.runInference).not.toHaveBeenCalled()
    const commandRegion = screen.getByRole('region', {
      name: 'Run command',
    })
    expect(commandRegion).toHaveAttribute('aria-busy', 'true')
    expect(
      within(commandRegion).getByText('Starting analysis…'),
    ).toBeVisible()
    const liveStatus = screen.getByLabelText('Active attempt status')
    expect(liveStatus).not.toHaveAttribute('aria-live')
    expect(liveStatus).toHaveTextContent('queued')
    const statusAnnouncement = screen.getByRole('status', {
      name: 'Analysis status announcement',
    })
    expect(commandRegion).not.toContainElement(statusAnnouncement)
    expect(statusAnnouncement).toHaveTextContent('queued')
    await act(async () => {
      vi.advanceTimersByTime(120)
      await Promise.resolve()
    })
    expect(deferred.runInference).toHaveBeenCalledOnce()
    expect(screen.getByLabelText('Active attempt status')).toHaveTextContent('running')
    expect(statusAnnouncement).toHaveTextContent('running')
    expect(
      within(commandRegion).getByText('Preparing result…'),
    ).toBeVisible()
    const parent = deferred.requests[0]

    fireEvent.click(screen.getByRole('button', { name: 'Cancel run' }))
    expect(parent.signal?.aborted).toBe(true)
    expect(screen.getByLabelText('Active attempt status')).toHaveTextContent('cancelled')

    fireEvent.click(screen.getByRole('button', { name: 'Retry exact input' }))
    expect(screen.getByLabelText('Active attempt status')).toHaveTextContent('queued')
    await act(async () => {
      vi.advanceTimersByTime(120)
      await Promise.resolve()
    })
    expect(deferred.runInference).toHaveBeenCalledTimes(2)
    const child = deferred.requests[1]
    expect(child.binding.inputFingerprint).toBe(parent.binding.inputFingerprint)
    const timeline = screen.getByRole('region', { name: 'Run timeline' })
    const attempts = within(timeline).getAllByTestId('analysis-attempt')
    expect(attempts).toHaveLength(2)
    expect(attempts[1]).toHaveTextContent(/parent attempt-/i)
    expect(screen.getByLabelText('Active attempt status')).toHaveTextContent('running')

    await act(async () => {
      parent.resolve(runMockEngine(parent.binding))
      await Promise.resolve()
    })
    expect(screen.getByLabelText('Active attempt status')).toHaveTextContent('running')
    expect(screen.queryByRole('region', { name: 'Simulation result' })).not.toBeInTheDocument()

    await act(async () => {
      child.resolve(runMockEngine(child.binding))
      await Promise.resolve()
    })
    expect(screen.getByLabelText('Active attempt status')).toHaveTextContent('succeeded')
    expect(commandRegion).toHaveAttribute('aria-busy', 'false')
    expect(screen.getByRole('region', { name: 'Simulation result' })).toBeVisible()
  })

  it('can cancel while queued without starting gateway work later', async () => {
    const user = userEvent.setup()
    const deferred = createDeferredGateway()
    const approvedCase = listWorkbenchAssets()[2]
    renderAnalysis(`/analysis?case=${approvedCase.id}`, deferred.gateway, false, 5_000)

    await user.click(screen.getByRole('button', { name: 'Run simulation' }))
    expect(screen.getByLabelText('Active attempt status')).toHaveTextContent('queued')
    await user.click(screen.getByRole('button', { name: 'Cancel run' }))
    expect(screen.getByLabelText('Active attempt status')).toHaveTextContent('cancelled')

    await new Promise((resolve) => setTimeout(resolve, 180))
    expect(deferred.runInference).not.toHaveBeenCalled()
  })

  it('aborts the active request when the route unmounts', async () => {
    const user = userEvent.setup()
    const deferred = createDeferredGateway()
    const approvedCase = listWorkbenchAssets()[2]
    const mounted = renderAnalysis(`/analysis?case=${approvedCase.id}`, deferred.gateway)

    await user.click(screen.getByRole('button', { name: 'Run simulation' }))
    await waitFor(() => expect(deferred.runInference).toHaveBeenCalledOnce())
    const request = deferred.requests[0]
    expect(request.signal?.aborted).toBe(false)
    mounted.unmount()
    expect(request.signal?.aborted).toBe(true)
  })
})
