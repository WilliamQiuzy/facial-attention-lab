import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createHash } from 'node:crypto'

import { App } from './App'
import { InferenceContractError, type ResearchInferenceResult } from './model/inference'

const readyEndpoint = vi.fn().mockResolvedValue(undefined)
const pendingEndpoint = vi.fn(() => new Promise<void>(() => undefined))

async function uploadVideo(user: ReturnType<typeof userEvent.setup>) {
  const file = new File(['synthetic-video'], 'faces-session.webm', { type: 'video/webm' })
  await user.upload(screen.getByLabelText('Choose LifeLink Face video'), file)
  return file
}

async function uploadTimeline(user: ReturnType<typeof userEvent.setup>, includeOptional = true) {
  const digest = createHash('sha256').update('synthetic-video').digest('hex')
  const ids = [
    'neutral_repose', 'eyebrow_raise', 'gentle_eye_closure', 'tight_eye_squeeze',
    'relaxed_smile', 'lip_pucker', 'lower_teeth_show', 'reanimated_smile',
  ]
  const selectedIds = ids.slice(0, includeOptional ? 8 : 7)
  const sidecar = new File([JSON.stringify({
    schema_version: 'faces-action-timeline/v1',
    script_version: 'faces-script/24-004956-v1',
    recording_sha256: digest,
    timing_source: 'capture_event_log',
    recording_duration_ms: selectedIds.length * 4_000,
    actions: selectedIds.map((action, index) => ({
      action,
      status: 'completed',
      prompt_start_ms: index * 4_000,
      hold_start_ms: index * 4_000 + 500,
      hold_end_ms: index * 4_000 + 3_500,
      completion_ms: index * 4_000 + 3_750,
    })),
  })], 'faces-session.timeline.json', { type: 'application/json' })
  await user.upload(screen.getByLabelText('Choose FACES action timeline'), sidecar)
}

function acceptedResult(includeOptional = true): ResearchInferenceResult {
  const ids = [
    'eyebrow_raise', 'gentle_eye_closure', 'tight_eye_squeeze',
    'relaxed_smile', 'lip_pucker', 'lower_teeth_show', 'reanimated_smile',
  ] as const
  const count = includeOptional ? 7 : 6
  const metrics = [
    ['brow_height_asymmetry_iod', 'brow_height_change_from_rest_iod'],
    ['eye_aperture_asymmetry_iod', 'residual_eye_aperture_iod', 'eye_closure_change_from_rest_iod'],
    ['eye_aperture_asymmetry_iod', 'residual_eye_aperture_iod', 'eye_closure_change_from_rest_iod'],
    ['mouth_corner_vertical_asymmetry_iod', 'mouth_corner_vertical_change_from_rest_iod'],
    ['mouth_corner_horizontal_asymmetry_iod', 'mouth_width_change_from_rest_iod'],
    ['mouth_corner_vertical_asymmetry_iod', 'lower_lip_change_from_rest_iod', 'mouth_open_change_from_rest_iod'],
    ['mouth_corner_vertical_asymmetry_iod', 'mouth_corner_vertical_change_from_rest_iod'],
  ]
  return {
    mode: 'research-inference',
    model: {
      modelId: 'broad_literature_shared_v9_blv9_009_ensemble',
      candidateId: 'BLV9-009',
      releaseManifestSha256: '81e396954090a0da6b99519909c1af15b6df5d1585ba27a642539352fe0a0c64',
      ensembleMembers: 3,
    },
    preprocessing: {
      version: 'faces-to-shared-v9/v1',
      faceLandmarkerSha256: '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff',
      mirrorMethod: 'horizontal_flip_and_redetect',
      protocol: 'cue_aligned_action',
      timingSource: 'capture_event_log',
    },
    quality: {
      eligible: true,
      actionsUsed: count,
      optionalActionsUnavailable: includeOptional ? [] : ['reanimated_smile'],
      actions: ids.slice(0, count).map((id, index) => ({
        id,
        v9Action: ['BROW_RAISE', 'EYE_GENTLE', 'EYE_FORCEFUL', 'SMILE_GENTLE', 'LIP_PUCKER', 'SHOW_BOTTOM_TEETH', 'SMILE_FULL'][index],
        holdStartMs: index * 4_000 + 4_500,
        holdEndMs: index * 4_000 + 7_500,
        validSamples: 32,
      })),
    },
    prediction: {
      probability: 0.73,
      memberProbabilities: [0.71, 0.74, 0.74],
      predictedClass: 1,
      threshold: 0.5,
      interpretation: 'class_1_research_score_only',
      endpointSemantics: 'meei_facial_palsy_vs_healthy_control_development_head',
      class0Label: 'meei_healthy_control',
      class1Label: 'meei_facial_palsy',
    },
    reportEvidence: {
      normalization: 'original_view_centered_eye_axis_aligned_interocular_scaled',
      interpretation: 'measured_movement_observation_not_causal_or_severity',
      contextFrameMethod: 'registered_hold_midpoint_not_model_selected',
      attribution: {
        method: 'integrated_gradients_shared_action_tokens',
        baseline: 'within_recording_neutral_clinical_zero_dense_response',
        scope: 'action_region_model_influence_not_landmark_causality',
        integrationSteps: 32,
        maxCompletenessError: 0.005,
      },
      actions: ids.slice(0, count).map((id, index) => ({
        id,
        region: (index === 0 ? 'brow' : index < 3 ? 'eye' : 'mouth') as 'brow' | 'eye' | 'mouth',
        contextFrameMs: index * 4_000 + 6_000,
        observations: metrics[index].map((metric, metricIndex) => ({
          metric,
          value: 0.01 * (metricIndex + 1),
          unit: 'interocular_distance' as const,
        })),
        modelInfluence: {
          status: 'stable' as const,
          direction: 'toward_class_1' as const,
          strength: (index < 2 ? 'strong' : index < 4 ? 'moderate' : 'smaller') as 'strong' | 'moderate' | 'smaller',
          relativeMagnitude: Math.max(0.1, 1 - index * 0.15),
        },
        stability: {
          ensembleSignAgreement: 3,
          mirrorConsistent: true,
          temporalChecksPassed: 2,
        },
      })),
    },
    clinicalUseEligible: false,
  }
}

describe('App', () => {
  beforeEach(() => {
    window.history.replaceState(null, '', '#analysis')
  })
  it('uses an internal clinical-review product message without exposing the model release', () => {
    const { container } = render(<App demonstrationEnabled checkEndpoint={pendingEndpoint} />)
    expect(screen.getByText('Research use only')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: /capture the full facial movement story/i })).toBeInTheDocument()
    expect(screen.getByText(/FACES protocol · Source script v0.01/i)).toBeInTheDocument()
    expect(screen.queryByText(/IRB/i)).not.toBeInTheDocument()
    expect(screen.queryByText('Interpretation boundary')).not.toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: /research interface, not a diagnosis/i })).not.toBeInTheDocument()
    expect(screen.queryByText(/Eye or mouth severity, House-Brackmann/i)).not.toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Designed to support clinician review.' })).toBeInTheDocument()
    expect(screen.getByText(/FACES AI summarizes standardized facial movement recordings/i)).toBeInTheDocument()
    expect(screen.getByText('Analysis pipeline')).toBeInTheDocument()
    expect(screen.getByText(/timeline, geometry, and response checks/i)).toBeInTheDocument()
    expect(container.textContent).not.toMatch(/BLV9-009|Shared V9|Target release/i)
    expect(screen.queryByText(/heatmap/i)).not.toBeInTheDocument()
  })

  it('runs demonstration only after a separate explicit user action', async () => {
    const user = userEvent.setup()
    render(<App demonstrationEnabled checkEndpoint={pendingEndpoint} />)
    await uploadVideo(user)

    const demoButton = screen.getByRole('button', { name: 'Preview demonstration results' })
    await user.click(demoButton)

    expect(screen.getByText('DEMONSTRATION - NOT MODEL OUTPUT')).toBeInTheDocument()
    expect(screen.getByRole('status')).toHaveTextContent(/no model processed this video/i)
    expect(screen.getByText('Demonstration probability layout')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Movement summary' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Eye region' })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Mouth region' })).toBeInTheDocument()
    expect(screen.queryByText('Binary model output')).not.toBeInTheDocument()
    expect(screen.queryByText('Ordinal region output')).not.toBeInTheDocument()
  })

  it('clears session media and results when starting a new session', async () => {
    const user = userEvent.setup()
    render(<App demonstrationEnabled checkEndpoint={pendingEndpoint} />)
    await uploadVideo(user)
    await user.click(screen.getByRole('button', { name: 'Preview demonstration results' }))

    await user.click(screen.getByRole('button', { name: 'Start a new session' }))

    expect(screen.queryByText('faces-session.webm')).not.toBeInTheDocument()
    expect(screen.queryByText('DEMONSTRATION - NOT MODEL OUTPUT')).not.toBeInTheDocument()
    expect(screen.getByText('Choose a LifeLink Face recording')).toBeInTheDocument()
  })

  it('offers an explicit clear action before analysis or results exist', async () => {
    const user = userEvent.setup()
    render(<App checkEndpoint={pendingEndpoint} />)
    await uploadVideo(user)

    const analyzeButton = screen.getByRole('button', { name: 'Run research analysis' })
    const clearButton = screen.getByRole('button', { name: 'Clear recording and start over' })
    expect(screen.getByRole('button', { name: 'Download recorded video' })).toBeEnabled()
    expect(analyzeButton.parentElement).toBe(clearButton.parentElement)
    expect(analyzeButton.parentElement).toHaveClass('analysis-button-stack')

    await user.click(clearButton)

    expect(screen.queryByText('faces-session.webm')).not.toBeInTheDocument()
    expect(screen.getByText('Choose a LifeLink Face recording')).toBeInTheDocument()
  })

  it('avoids smooth scrolling on reset when reduced motion is requested', async () => {
    const user = userEvent.setup()
    vi.spyOn(window, 'matchMedia').mockReturnValue({ matches: true } as MediaQueryList)
    render(<App demonstrationEnabled checkEndpoint={pendingEndpoint} />)
    await uploadVideo(user)
    await user.click(screen.getByRole('button', { name: 'Preview demonstration results' }))
    await user.click(screen.getByRole('button', { name: 'Start a new session' }))

    expect(window.scrollTo).toHaveBeenCalledWith({ top: 0, behavior: 'auto' })
  })

  it('warns before identifiable video can be sent to a research endpoint', async () => {
    const user = userEvent.setup()
    render(<App apiEndpoint="https://research.example.test/infer" demonstrationEnabled checkEndpoint={pendingEndpoint} />)
    await uploadVideo(user)
    expect(screen.getByText(/facial video is identifiable/i)).toBeInTheDocument()
    expect(screen.getByRole('checkbox', { name: /authorized research endpoint/i })).not.toBeChecked()
    expect(screen.getByRole('button', { name: 'Run research analysis' })).toBeDisabled()
  })

  it('shows API failure without silently producing demonstration output', async () => {
    const user = userEvent.setup()
    const analyze = vi.fn().mockRejectedValue(new Error('Segmentation coverage incomplete'))
    render(
      <App
        apiEndpoint="https://research.example.test/infer"
        demonstrationEnabled
        analyze={analyze}
        checkEndpoint={readyEndpoint}
      />,
    )
    await uploadVideo(user)
    await uploadTimeline(user)
    await user.click(screen.getByRole('checkbox', { name: /authorized research endpoint/i }))
    await user.click(screen.getByRole('button', { name: 'Run research analysis' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(/segmentation coverage incomplete/i)
    expect(screen.queryByText('DEMONSTRATION - NOT MODEL OUTPUT')).not.toBeInTheDocument()
  })

  it('retries a transient model failure with the same retained recording', async () => {
    const user = userEvent.setup()
    const analyze = vi.fn()
      .mockRejectedValueOnce(new InferenceContractError('The model service could not complete this request. Retry the same recording.', true))
      .mockResolvedValueOnce(acceptedResult())
    render(<App apiEndpoint="https://research.example.test/infer" analyze={analyze} checkEndpoint={readyEndpoint} />)
    await uploadVideo(user)
    await uploadTimeline(user)
    await user.click(screen.getByRole('checkbox', { name: /authorized research endpoint/i }))

    const run = screen.getByRole('button', { name: 'Run research analysis' })
    await user.click(run)
    expect(await screen.findByRole('alert')).toHaveTextContent(/retry the same recording/i)
    expect(screen.getByText('faces-session.webm')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Clear recording and start over' })).toBeEnabled()

    await user.click(run)
    expect(await screen.findByText('Research report ready')).toBeInTheDocument()
    expect(analyze).toHaveBeenCalledTimes(2)
  })

  it('enables research analysis for the medically valid seven-step script', async () => {
    const user = userEvent.setup()
    const analyze = vi.fn().mockResolvedValue(acceptedResult(false))
    render(<App apiEndpoint="https://research.example.test/infer" analyze={analyze} checkEndpoint={readyEndpoint} />)
    await uploadVideo(user)
    await uploadTimeline(user, false)
    await user.click(screen.getByRole('checkbox', { name: /authorized research endpoint/i }))

    const button = screen.getByRole('button', { name: 'Run research analysis' })
    expect(button).toBeEnabled()
    await user.click(button)

    await waitFor(() => expect(analyze).toHaveBeenCalledTimes(1))
    expect(analyze.mock.calls[0][1].reanimatedSmileApplicable).toBe(false)
  })

  it('discards a stale response after the recording changes', async () => {
    const user = userEvent.setup()
    let resolveAnalysis: ((result: ResearchInferenceResult) => void) | undefined
    const analyze = vi.fn(
      () =>
        new Promise<ResearchInferenceResult>((resolve) => {
          resolveAnalysis = resolve
        }),
    )
    render(<App apiEndpoint="https://research.example.test/infer" analyze={analyze} checkEndpoint={readyEndpoint} />)
    await uploadVideo(user)
    await uploadTimeline(user)
    await user.click(screen.getByRole('checkbox', { name: /authorized research endpoint/i }))
    expect(screen.getByRole('button', { name: 'Run research analysis' })).toBeEnabled()
    await user.click(screen.getByRole('button', { name: 'Run research analysis' }))

    const replacement = new File(['replacement'], 'replacement.webm', { type: 'video/webm' })
    await user.upload(screen.getByLabelText('Choose LifeLink Face video'), replacement)
    resolveAnalysis?.(acceptedResult())

    await waitFor(() => expect(screen.getByText('replacement.webm')).toBeInTheDocument())
    expect(screen.queryByText('Accepted research inference')).not.toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /include step 8/i })).not.toBeChecked()
    expect(screen.getByRole('button', { name: 'Run research analysis' })).toBeDisabled()
  })

  it('renders accepted research output without technical report metadata', async () => {
    const user = userEvent.setup()
    const result = acceptedResult()
    const analyze = vi.fn().mockResolvedValue(result)
    render(
      <App apiEndpoint="https://research.example.test/infer" analyze={analyze} checkEndpoint={readyEndpoint} />,
    )
    await uploadVideo(user)
    await uploadTimeline(user)
    await user.click(screen.getByRole('checkbox', { name: /authorized research endpoint/i }))
    await user.click(screen.getByRole('button', { name: 'Run research analysis' }))

    await waitFor(() => expect(analyze).toHaveBeenCalledTimes(1))
    expect(screen.getByText('Research report ready')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Run research analysis' })).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: /view full research report/i })).toBeVisible()
    expect(screen.getByRole('button', { name: 'Download recorded video' })).toBeEnabled()

    await user.click(screen.getByRole('link', { name: /view full research report/i }))
    expect(await screen.findByRole('heading', { name: /research movement report/i })).toHaveFocus()
    expect(screen.getByText('MEEI facial-movement classification score')).toBeInTheDocument()
    expect(screen.getByText('73 / 100')).toBeInTheDocument()
    expect(screen.getByText(/average of three model outputs/i)).toBeInTheDocument()
    expect(screen.getByText(/23 points above the fixed cutpoint of 50/i)).toBeInTheDocument()
    expect(screen.getByText('Above MEEI research cutpoint')).toBeInTheDocument()
    expect(screen.getByText('Neutral baseline + all 7 active movements')).toBeInTheDocument()
    expect(screen.getByText(/PDF includes the recorded evidence images/i)).toBeInTheDocument()
    expect(screen.getByText(/all 8 recorded steps in this session were used/i)).toBeInTheDocument()
    expect(screen.queryByText('Research use only')).not.toBeInTheDocument()
    expect(screen.queryByText('Validated response provenance')).not.toBeInTheDocument()
    expect(screen.queryByText('BLV9-009')).not.toBeInTheDocument()
    expect(screen.queryByText('Not assessed')).not.toBeInTheDocument()
    expect(screen.queryByText('Not collected')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Run research analysis' })).not.toBeInTheDocument()
    expect(document.title).toMatch(/research movement report/i)

    await user.click(screen.getByRole('link', { name: /back to session summary/i }))
    expect(screen.getByText('faces-session.webm')).toBeInTheDocument()
    expect(screen.getByText('Research report ready')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Run research analysis' })).not.toBeInTheDocument()
  })

  it('synchronously blocks rapid duplicate submissions and makes success immutable', async () => {
    const user = userEvent.setup()
    let resolveAnalysis: ((result: ResearchInferenceResult) => void) | undefined
    const analyze = vi.fn(() => new Promise<ResearchInferenceResult>((resolve) => { resolveAnalysis = resolve }))
    render(<App apiEndpoint="https://research.example.test/infer" analyze={analyze} checkEndpoint={readyEndpoint} />)
    await uploadVideo(user)
    await uploadTimeline(user)
    await user.click(screen.getByRole('checkbox', { name: /authorized research endpoint/i }))
    const run = screen.getByRole('button', { name: 'Run research analysis' })
    act(() => {
      fireEvent.click(run)
      fireEvent.click(run)
      fireEvent.keyDown(run, { key: 'Enter' })
    })
    expect(analyze).toHaveBeenCalledTimes(1)
    await act(async () => { resolveAnalysis?.(acceptedResult()) })
    expect(await screen.findByText('Research report ready')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Run research analysis' })).not.toBeInTheDocument()
    expect(analyze).toHaveBeenCalledTimes(1)
  })

  it('does not resubmit a permanently rejected capture but preserves retry for transport failures', async () => {
    const user = userEvent.setup()
    const analyze = vi.fn().mockRejectedValue(
      new InferenceContractError('The video timing did not match the guided action timeline.', false),
    )
    render(<App apiEndpoint="https://research.example.test/infer" analyze={analyze} checkEndpoint={readyEndpoint} />)
    await uploadVideo(user)
    await uploadTimeline(user)
    await user.click(screen.getByRole('checkbox', { name: /authorized research endpoint/i }))
    await user.click(screen.getByRole('button', { name: 'Run research analysis' }))
    expect(await screen.findByRole('alert')).toHaveTextContent(/timing did not match/i)
    expect(screen.getByRole('button', { name: 'New recording required' })).toBeDisabled()
    expect(screen.getByText(/cannot be resubmitted/i)).toBeInTheDocument()
    expect(analyze).toHaveBeenCalledTimes(1)
  })

  it('shows a private empty state on a direct or reloaded report route without rerunning', async () => {
    window.history.replaceState(null, '', '#research-report')
    const analyze = vi.fn()
    render(<App apiEndpoint="https://research.example.test/infer" analyze={analyze} checkEndpoint={pendingEndpoint} />)
    expect(screen.getByRole('heading', { name: /report not retained/i })).toBeInTheDocument()
    expect(screen.getByText(/recording and report stay only in this browser session/i)).toBeInTheDocument()
    expect(analyze).not.toHaveBeenCalled()
    window.history.replaceState(null, '', '#analysis')
  })

  it('does not call a configured endpoint online until readiness is verified and supports retry', async () => {
    const user = userEvent.setup()
    const checkEndpoint = vi.fn()
      .mockRejectedValueOnce(new Error('The research endpoint could not be reached.'))
      .mockResolvedValueOnce(undefined)
    render(<App apiEndpoint="https://research.example.test/infer" checkEndpoint={checkEndpoint} />)

    expect(screen.getByText(/checking analysis endpoint/i)).toBeInTheDocument()
    expect(await screen.findByText(/research endpoint unavailable/i)).toBeInTheDocument()
    expect(screen.queryByText(/analysis endpoint ready/i)).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /retry endpoint check/i }))
    expect(await screen.findByText(/analysis endpoint ready/i)).toBeInTheDocument()
    expect(checkEndpoint).toHaveBeenCalledTimes(2)
  })
})
